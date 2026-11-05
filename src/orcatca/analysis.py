"""Blocked calibration cross-fitting and joint paired operating-point analysis."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
from .calibration import negative_support
from .metrics import fpr_threshold, arrays
from .utils import output_path, atomic_json, write_csv, digest, code_digest
from . import __version__
from .result_store import CombinedStore, ReplayTrace
from .logging import configure, event, close
from .site_ranking import score_sites, write_rank_tables
from .transfer import (linked_units, crossfit_transfer, full_calibration_outputs)


def units(rows):
    """Join linked recordings and blocks so neither can cross a resampling boundary."""
    return linked_units(rows)


def support_ok(rows, settings):
    report=negative_support(rows, settings)
    # Multiple recorded block IDs can belong to one transitive component.
    # Apply the half-split minimum to the same units used by partitioning and
    # resampling, in addition to the original recording-support requirement.
    grouped=units(rows)
    minimum=int(settings.get('min_negative_blocks',2))
    required=settings.get('required_negative_subtypes',['AB','HW'])
    linked_counts={cls:sum(any(rows[i]['raw_class']==cls for i in ids)
                          for _,ids in grouped) for cls in required}
    sparse=[cls for cls,count in linked_counts.items() if count<minimum]
    for cls,count in linked_counts.items():
        report['by_subtype'][cls]['linked_units']=count
    report.update(linked_units_by_subtype=linked_counts,sparse_linked_subtypes=sparse,
                  minimum_unit_definition='Transitive recording/block components')
    if sparse:
        report['status']='unsupported'
    return report['status']=='supported_empirically' and any(r['label']==1 for r in rows), report


def crossfit_plan(rows, settings, repeats=5, seed=20261029):
    """Source labels and grouping only; scores and target observations are excluded."""
    if repeats<1: raise ValueError('At least one split repeat is required')
    grouped=units(rows)
    minimum=int(settings.get('min_negative_blocks',2))
    for cls in settings.get('required_negative_subtypes',['AB','HW']):
        count=sum(any(rows[i]['raw_class']==cls for i in ids) for _,ids in grouped)
        if count<2*minimum:
            return dict(status='not_estimable',reason='insufficient_independent_subtype_units',
                        subtype=cls,available_units=count,required_units=2*minimum,
                        original_support=negative_support(rows,settings),plans=[])
    if len(grouped)<4:
        return dict(status='not_estimable',reason='too_few_units',plans=[])
    rng=np.random.default_rng(seed)
    plans=[];seen=set();attempts=0
    while len(plans)<repeats and attempts<1000*repeats:
        attempts+=1
        chosen=set(rng.choice(len(grouped),len(grouped)//2,replace=False).tolist())
        signature=tuple(sorted(chosen))
        mirror=tuple(i for i in range(len(grouped)) if i not in chosen)
        canonical=min(signature,mirror)
        if canonical in seen: continue
        left=np.concatenate([ids for i,(_,ids) in enumerate(grouped) if i in chosen])
        right=np.concatenate([ids for i,(_,ids) in enumerate(grouped) if i not in chosen])
        ok_a,_=support_ok([rows[i] for i in left],settings)
        ok_b,_=support_ok([rows[i] for i in right],settings)
        if not(ok_a and ok_b): continue
        seen.add(canonical)
        plans.append(dict(repeat=len(plans),left=left.tolist(),right=right.tolist(),
                          left_units=[k for i,(k,_) in enumerate(grouped) if i in chosen],
                          right_units=[k for i,(k,_) in enumerate(grouped) if i not in chosen]))
    return dict(status='ready' if len(plans)==repeats else 'partial' if plans else 'not_estimable',
                reason=None if plans else 'no_supported_partition',units=len(grouped),attempts=attempts,
                source_only=True,uses_prediction_scores=False,requested_repeats=repeats,
                half_support_unit='Transitive recording/block components',
                minimum_negative_units_per_half=minimum,
                plans=plans,interpretation='Repeated partitions overlap and are not independent experiments')


def rates(rows,p,threshold):
    y,p=arrays(rows,p);pred=p>=threshold
    pos=y==1;neg=y==0
    value=dict(n=len(rows),positive=int(pos.sum()),negative=int(neg.sum()),
               tp=int((pred&pos).sum()),fp=int((pred&neg).sum()),
               tpr=float(pred[pos].mean()) if pos.any() else None,
               fpr=float(pred[neg].mean()) if neg.any() else None)
    for c in ('AB','HW'):
        ids=np.array([r['raw_class']==c for r in rows])
        value[c+'_n']=int(ids.sum())
        value[c+'_fp']=int(pred[ids].sum())
        value[c+'_fpr']=float(pred[ids].mean()) if ids.any() else None
    return value


def decomposition(source,target):
    if any(source[c+'_fpr'] is None or target[c+'_fpr'] is None for c in ('AB','HW')):
        return dict(status='not_estimable',reason='missing_subtype')
    source_n=sum(source[c+'_n'] for c in ('AB','HW'))
    target_n=sum(target[c+'_n'] for c in ('AB','HW'))
    mixture=conditional=symmetric=counter=0.0
    for c in ('AB','HW'):
        ps=source[c+'_n']/source_n;pt=target[c+'_n']/target_n
        qs=source[c+'_fpr'];qt=target[c+'_fpr']
        mixture+=(pt-ps)*qs;conditional+=pt*(qt-qs)
        symmetric+=.5*(pt-ps)*(qs+qt);counter+=pt*qs
    delta=target['fpr']-source['fpr']
    return dict(status='estimated_descriptively',delta=delta,mixture_only=counter,
                source_reference_mixture=mixture,target_weighted_conditional=conditional,
                symmetric_mixture=symmetric,symmetric_conditional=delta-symmetric,
                identity_residual=delta-mixture-conditional,
                source_error_rate_reference='held-out source check, not threshold-fitting observations')


def resolution(rows,settings):
    result=[]
    for c in ('AB','HW'):
        group=[r for r in rows if r['raw_class']==c]
        n=len(group);blocks=len({r.get('block') or r['recording'] for r in group})
        for alpha in settings['fpr_targets']:
            result.append(dict(subtype=c,alpha=alpha,windows=n,
                recordings=len({r['recording'] for r in group}),blocks=blocks,
                empirical_step=1/n if n else None,allowed_window_errors=math.floor(alpha*n),
                resolution_limited=alpha*n<1,independent_risk_certificate=False))
    return result


def joint_bootstrap(rows, probabilities, thresholds, repeats=1000, seed=20261029):
    """Pair all methods/seeds under shared block draws; thresholds stay fixed."""
    if repeats<10: raise ValueError('At least ten replicates required')
    y=np.array([r['label'] for r in rows]);probs=np.asarray(probabilities,float)
    if probs.ndim!=3 or probs.shape[-1]!=len(rows) or thresholds.shape!=probs.shape[:2]:
        raise ValueError('Expected [method, seed, sample] probabilities')
    group=units(rows);strata=defaultdict(list);counts=[]
    for j,(_,ids) in enumerate(group):
        signature=tuple(sorted({rows[i]['site'] for i in ids}));strata[signature].append(j)
        pos=y[ids]==1;neg=~pos
        pred=probs[:,:,ids]>=thresholds[:,:,None]
        counts.append(np.stack((np.broadcast_to(neg.sum(),probs.shape[:2]),
                                np.broadcast_to(pos.sum(),probs.shape[:2]),
                                (pred&neg).sum(-1),(pred&pos).sum(-1)),axis=-1))
    counts=np.asarray(counts,dtype=float)
    def calculate(w):
        total=np.einsum('g,gmsk->msk',w,counts)
        if (total[:,:,:2]<=0).any():return None
        return np.stack((total[:,:,2]/total[:,:,0],total[:,:,3]/total[:,:,1]),axis=-1).mean(1)
    original=calculate(np.ones(len(group)))
    if original is None: raise ValueError('Both binary classes required for joint comparison')
    rng=np.random.default_rng(seed);draws=[];attempts=0
    while len(draws)<repeats and attempts<repeats*10:
        attempts+=1;w=np.zeros(len(group))
        for ids in strata.values():
            w[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        value=calculate(w)
        if value is not None:draws.append(value)
    if len(draws)<repeats: raise ValueError('Insufficient valid joint bootstrap draws')
    return original,np.asarray(draws),dict(units=len(group),strata=len(strata),attempts=attempts,
                    rejected=attempts-repeats,scope='Conditional on frozen models and saved thresholds; no calibration or new-source variance')


def interval_rows(fold,rule,methods,original,draws,metadata):
    output=[]
    for m in range(1,len(methods)):
        d=original[m]-original[0];samples=draws[:,m]-draws[:,0]
        q=np.quantile(samples,[.025,.975],axis=0)
        output.append(dict(fold=fold,rule=rule,a=methods[0],b=methods[m],
            delta_fpr=float(d[0]),fpr_ci_low=float(q[0,0]),fpr_ci_high=float(q[1,0]),
            delta_recall=float(d[1]),recall_ci_low=float(q[0,1]),recall_ci_high=float(q[1,1]),
            covariance_fpr_recall=float(np.cov(samples.T,ddof=1)[0,1]),
            favorable_draw_fraction=float(((samples[:,0]<0)&(samples[:,1]>0)).mean()),
            favorable_draw_fraction_is_not_posterior_probability=True,**metadata))
    return output


def new_output(root,requested,prefix):
    relative=requested or f'result/analysis/{prefix}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
    path=output_path(root,relative)
    if not path.is_relative_to((root/'result/analysis').resolve()):
        raise ValueError('Analysis output must be below result/analysis')
    try:
        path.mkdir(parents=True,exist_ok=False)
    except FileExistsError as exc:
        detail = ' Inspect its failure.json for the earlier exception.' if (path/'failure.json').is_file() else ''
        raise FileExistsError(f'Analysis output already exists: {path}. Choose a fresh directory.{detail}') from exc
    return path


def run(root,source,output=None,methods=None,repeats=1000,crossfit_repeats=5,seed=20261029,reference=None,expect_trace=None):
    methods=methods or ['natural_mixstyle','natural_subtype_mixstyle']
    if len(methods)<2 or len(set(methods))!=len(methods):
        raise ValueError('At least two distinct methods are required')
    if repeats<10 or crossfit_repeats<1:raise ValueError('Invalid analysis repetitions')
    trace = ReplayTrace(root/expect_trace, methods, repeats, crossfit_repeats, seed) if expect_trace else None
    out=None
    try:
        with CombinedStore(root,source,reference,trace=trace) as store:
            cfg=store.config;seeds=cfg['experiment']['seeds'];settings=cfg['evaluation']
            if trace and seeds != trace.seeds:
                raise ValueError('Expected analysis trace has different training seeds')
            absent = sorted(set(methods) - set(cfg['experiment']['methods']))
            if absent:
                raise ValueError('Missing methods: ' + ', '.join(absent) +
                    '. Supply their frozen suite or selected report via --reference. '
                    'Per-fit holdout.json summaries cannot replace window predictions.')
            out=new_output(root,output,'transfer')
            configure(out/'logs','analysis')
            cf=[];decomposed=[];support=[];joint=[];plans={};audit=[];all_draws=defaultdict(list);all_values=defaultdict(list)
            gaps=[];label_rates=[];influence=[];threshold_details=[];transfer_audits=[]
            original_operating=[];original_summary=[];original_labels=[];data_flow=[];protocol_audits=[];site_ranks=[]
            for fold in range(cfg['experiment']['folds']):
                name=f'fold{fold:02d}';event('analysis_fold_started',fold=name)
                cal,cp,cals=store.predictions(fold,methods,seeds,'calibration')
                held,hp,_=store.predictions(fold,methods,seeds,'holdout')
                for key in ('id','recording','block'):
                    overlap={r.get(key) for r in cal}&{r.get(key) for r in held};overlap.discard(None)
                    if overlap:raise ValueError('Calibration/holdout overlap: '+key)
                manifest=store.protocol(fold)
                for role,observed in (('calibrate',cal),('holdout',held)):
                    lookup={r['id']:r for r in manifest if r['split']==role}
                    if set(lookup)!={r['id'] for r in observed}:raise ValueError('Prediction/protocol sample mismatch')
                    for r in observed:
                        if any(r.get(k)!=lookup[r['id']].get(k) for k in ('recording','site','block','label','raw_class')):
                            raise ValueError('Prediction/protocol metadata mismatch')
                from .splitting import validate_splits
                validate_splits(manifest)
                from .provenance import audit_protocol, minimal_manifest
                flow_rows, protocol_audit = audit_protocol(manifest, name,
                    prediction_metadata={'calibrate': cal, 'holdout': held},
                    temporal_gap_seconds=cfg.get('data',{}).get('session_gap_seconds',1800),
                    block_hours=cfg.get('data',{}).get('block_hours',24))
                data_flow.extend(flow_rows);protocol_audits.append(protocol_audit)
                atomic_json(out/f'{name}_protocol_audit.json',protocol_audit)
                # Save the failure evidence before refusing statistically invalid inputs.
                write_csv(out/'data_flow.csv',data_flow)
                atomic_json(out/'protocol_audit.json',dict(folds=protocol_audits))
                import csv, gzip
                manifest_rows = minimal_manifest(manifest, name)
                with gzip.open(out/f'{name}_window_manifest.csv.gz','wt',newline='',encoding='utf-8') as stream:
                    fields=list(dict.fromkeys(k for row in manifest_rows for k in row))
                    writer=csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader();writer.writerows(manifest_rows)
                if protocol_audit.get('protocol_valid') is False:
                    raise ValueError('Frozen protocol audit is invalid; inspect '+name+'_protocol_audit.json')
                audit.append(dict(fold=name,calibration_windows=len(cal),holdout_windows=len(held),
                                  methods=len(methods),seeds=seeds,aligned=True,protocol_verified=True))
                site_ranks.extend(score_sites(name,held,hp,methods,seeds,
                    cfg['experiment'].get('selection_max_fpr',.1)))
                support.extend([dict(fold=name,**r) for r in resolution(cal,settings)])
                plan=crossfit_plan(cal,settings,crossfit_repeats,seed+fold)
                plans[name]=dict(plan,metadata_sha256=digest(cal))
                atomic_json(out/f'{name}_crossfit_plan.json',plans[name])
                event('crossfit_plan',fold=name,status=plan['status'],partitions=len(plan['plans']))
                for alpha in settings['fpr_targets']:
                    rule=f'fpr_{alpha:g}'
                    thresholds=[];available=True;draws=None
                    for mi in range(len(methods)):
                        group=[]
                        for si in range(len(seeds)):
                            t=cals[mi][si]['operating_thresholds'].get(rule)
                            if t is None:available=False
                            group.append(t)
                        thresholds.append(group)
                    if available:
                        values,draws,meta=joint_bootstrap(held,hp,np.asarray(thresholds,float),repeats,seed+fold*101+round(alpha*10000))
                        joint.extend(interval_rows(name,rule,methods,values,draws,meta))
                        all_draws[rule].append(draws);all_values[rule].append(values)
                    else:
                        joint.append(dict(fold=name,rule=rule,status='not_estimable',reason='saved_rule_missing'))
                    original=full_calibration_outputs(name,cal,cp,held,hp,methods,seeds,
                        alpha,thresholds,target_draws=draws)
                    original_operating.extend(original['operating'])
                    original_summary.extend(original['summary'])
                    original_labels.extend(original['labels'])
                    threshold_details.extend(original['diagnostics'])
                    transfer=crossfit_transfer(name,cal,cp,held,hp,methods,seeds,plan,
                        alpha,repeats,seed+fold*101)
                    gaps.extend(transfer['summary']);label_rates.extend(transfer['labels'])
                    influence.extend(transfer['influence']);threshold_details.extend(transfer['diagnostics'])
                    transfer_audits.append(transfer['audit'])
                for planitem in plan['plans']:
                    for side in (0,1):
                        fit=np.array(planitem['left' if side==0 else 'right']);check=np.array(planitem['right' if side==0 else 'left'])
                        fitrows=[cal[i] for i in fit];checkrows=[cal[i] for i in check]
                        for mi,method in enumerate(methods):
                            for si,s in enumerate(seeds):
                                for alpha in settings['fpr_targets']:
                                    t=fpr_threshold(fitrows,cp[mi,si,fit],alpha,True)
                                    meta=dict(fold=name,method=method,seed=s,repeat=planitem['repeat'],side=side,alpha=alpha,threshold=t)
                                    rates_fit=rates(fitrows,cp[mi,si,fit],t)
                                    rates_check=rates(checkrows,cp[mi,si,check],t)
                                    rates_target=rates(held,hp[mi,si],t)
                                    for role,r in [('source_fit',rates_fit),('source_check',rates_check),('target',rates_target)]:
                                        cf.append(dict(**meta,role=role,**r))
                                    decomposed.append(dict(**meta,**decomposition(rates_check,rates_target)))
                event('analysis_fold_completed',fold=name)
            checked_trace = store.verified_trace(trace) if trace else None
            for rule,drawlist in all_draws.items():
                if len(drawlist)!=cfg['experiment']['folds']:continue
                draw=np.stack(drawlist).mean(0);value=np.stack(all_values[rule]).mean(0)
                joint.extend(interval_rows('observed_sources_equal_weight',rule,methods,value,draw,
                            dict(scope='Fixed observed sources; seeds averaged, not independent domains',valid_repeats=repeats)))
            write_rank_tables(out,site_ranks)
            write_csv(out/'joint_differences.csv',joint)
            write_csv(out/'crossfit_operating.csv',cf)
            write_csv(out/'crossfit_transfer.csv',decomposed)
            write_csv(out/'calibration_support.csv',support)
            write_csv(out/'transfer_gap_summary.csv',gaps)
            write_csv(out/'transfer_label_summary.csv',label_rates)
            write_csv(out/'transfer_original_operating.csv',original_operating)
            write_csv(out/'transfer_original_summary.csv',original_summary)
            write_csv(out/'transfer_original_labels.csv',original_labels)
            write_csv(out/'transfer_threshold_diagnostics.csv',threshold_details)
            write_csv(out/'transfer_block_influence.csv',influence)
            write_csv(out/'data_flow.csv',data_flow)
            atomic_json(out/'protocol_audit.json',dict(folds=protocol_audits))
            atomic_json(out/'transfer_bootstrap_audit.json',dict(folds=transfer_audits))
            # Averages are descriptive; repeated splits overlap.
            grouped=defaultdict(list)
            for r in cf:grouped[r['fold'],r['method'],r['alpha'],r['role']].append(r)
            summary=[]
            for (fold,method,alpha,role),rs in sorted(grouped.items()):
                summary.append(dict(fold=fold,method=method,alpha=alpha,role=role,
                    fpr=float(np.mean([r['fpr'] for r in rs])),recall_kw=float(np.mean([r['tpr'] for r in rs])),
                    rows=len(rs),independent_repeats=False))
            write_csv(out/'crossfit_summary.csv',summary)
            sources = {'primary':store.primary.provenance()}
            if store.reference: sources['reference'] = store.reference.provenance()
            atomic_json(out/'analysis.json',dict(status='completed',version=__version__,code_sha256=code_digest(root),analysis_seed=seed,crossfit_repeats=crossfit_repeats,source=str(store.path),source_sha256=store.source_hash,inputs=sources,expected_trace=checked_trace,
                used_file_sha256=store.used,folds=audit,methods=methods,seeds=seeds,
                crossfit_status={k:v['status'] for k,v in plans.items()},no_audio_read=True,no_weights_loaded=True,
                original_thresholds_modified=False,raw_data_modified=False,bootstrap_repeats=repeats,
                limits=['Cross-fit rules are new exploratory source-only rules, not replacements for saved endpoints.',
                        'Joint intervals condition on fitted models and frozen thresholds.',
                        'Transfer intervals additionally condition on the fixed overlapping partitions; shared source and target block draws preserve their dependence.',
                        'Missing-label bootstrap draws are rejected and reported. Intervals at empirical boundaries or degenerate distributions are withheld, not replaced with zero-width limits.',
                        'Zero-error samples do not establish zero population risk.',
                        'Repeated source partitions and optimization seeds are not new independent deployments.']))
            event('analysis_completed',output=str(out),crossfit_rows=len(cf))
        import zipfile
        from .utils import sha256
        archive=out/'analysis_result.zip'
        files=sorted(p for p in out.iterdir() if p.is_file() and (p.suffix in {'.csv','.json'} or p.name.endswith('.csv.gz')))
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in files:z.write(file,file.name)
            import json
            z.writestr('SHA256.json',json.dumps({p.name:sha256(p) for p in files},sort_keys=True))
        return archive
    except BaseException as exc:
        if out is not None:
            atomic_json(out/'failure.json',dict(exception=type(exc).__name__,message=str(exc)))
        raise
    finally:
        if out is not None: close()
