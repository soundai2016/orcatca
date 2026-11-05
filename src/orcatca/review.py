"""Blinded review forms with separate representative and error-enriched panels."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .analysis import new_output, units
from .result_store import CombinedStore
from .utils import atomic_json, write_csv, raw_path, digest


def create(root, source, output=None, count=240, export_audio=False, reference=None):
    if count<12: raise ValueError('At least twelve review items are required')
    out=new_output(root,output,'review')
    private=out/'private';private.mkdir()
    rng=np.random.default_rng(20261029)
    forms=[];keys=[];notes=[];retained=[]
    methods=['natural_mixstyle','natural_subtype_mixstyle']
    with CombinedStore(root,source,reference) as store:
        cfg=store.config;folds=cfg['experiment']['folds'];seed=cfg['experiment']['seeds'][0]
        budget=count//folds
        for fold in range(folds):
            rows,p,cals=store.predictions(fold,methods,[seed],'holdout')
            groups=units(rows);groupmap={k:ids for k,ids in groups}
            chosen=[];used=set();n_rep=min(budget//2,len(groups))
            for j in rng.choice(len(groups),n_rep,replace=False):
                key,ids=groups[j];index=int(rng.choice(ids));used.add(key)
                chosen.append((index,key,'representative',n_rep/len(groups)/len(ids),None))
            truth=np.asarray([r['label'] for r in rows]);decisions=[]
            for mi in range(2): decisions.append(p[mi,0]>=cals[mi][0]['threshold'])
            a,b=(decisions[0]==truth,decisions[1]==truth)
            strata=np.where(~a&b,'corrected',np.where(a&~b,'regressed',np.where(~a&~b,'shared_error','shared_correct')))
            pools=defaultdict(list)
            for key,ids in groups:
                if key in used:continue
                for label in sorted(set(strata[ids])):
                    pools[label].append((key,ids[strata[ids]==label]))
            order=['corrected','regressed','shared_error','shared_correct']
            target=min(budget,len(groups))
            while len(chosen)<target:
                changed=False
                for label in order:
                    eligible=[pair for pair in pools[label] if pair[0] not in used]
                    if not eligible or len(chosen)>=target:continue
                    key,ids=eligible[int(rng.integers(len(eligible)))];index=int(rng.choice(ids));used.add(key)
                    chosen.append((index,key,'error_enriched',None,label));changed=True
                if not changed:break
            protocol={r['id']:r for r in store.protocol(fold)}
            for index,key,panel,prob,stratum in chosen:
                r=rows[index];meta=protocol[r['id']]
                qid='R'+digest([store.source_hash,fold,r['id'],20261029])[:14];clip=qid+'.wav'
                forms.append(dict(review_id=qid,clip=clip,reviewer='',kw_present='',hw_present='',
                    other_biological='',cooccurrence='',signal_type='',masking='',clipping='',
                    band_limit='',confidence='',more_context_needed='',notes=''))
                keys.append(dict(review_id=qid,fold=f'fold{fold:02d}',sample_id=r['id'],unit=key,
                    panel=panel,stratum=stratum,inclusion_probability=prob,raw_class=r['raw_class'],
                    label=r['label'],site=r['site'],recording=r['recording'],audio=meta['audio'],
                    start_frame=meta['start_frame'],baseline_probability=float(p[0,0,index]),
                    auxiliary_probability=float(p[1,0,index]),
                    baseline_threshold=cals[0][0]['threshold'],auxiliary_threshold=cals[1][0]['threshold']))
                retained.append((clip,meta))
            notes.append(dict(fold=fold,requested=budget,exported=len(chosen),representative=n_rep,
                              unique_units=len(used),independent_acoustic_encounters_confirmed=False))
        if export_audio:
            import soundfile as sf
            records=store.json('recordings.json',True)
            clips=out/'clips';clips.mkdir()
            for name,row in retained:
                record=records[row['audio']];path=raw_path(root,row['audio']);st=path.stat()
                if (st.st_size,st.st_mtime_ns)!=(record['size'],record['mtime_ns']):
                    raise ValueError('Raw identity changed during review export')
                with sf.SoundFile(path,'r') as f:
                    if f.channels!=1 or f.samplerate!=record['rate'] or len(f)!=record['frames']:
                        raise ValueError('Audio decoder metadata differ')
                    n=round(cfg['data']['seconds']*f.samplerate);f.seek(row['start_frame'])
                    x=f.read(n,dtype='float32',always_2d=True)
                    if len(x)!=n or not np.isfinite(x).all():raise ValueError('Invalid review clip')
                    sf.write(clips/name,x,f.samplerate,subtype='FLOAT')
        # Shuffle the public presentation; identifiers carry no method or source labels.
        rng.shuffle(forms)
        write_csv(out/'review.csv',forms)
        write_csv(private/'review_key.csv',keys)
        atomic_json(private/'design.json',dict(source=str(store.path),source_sha256=store.source_hash,
                    used_file_sha256=store.used,folds=notes,seed=seed,forms=len(forms),native_audio=export_audio,
                    raw_audio_modified=False,thresholds_reselected=False,
                    limitations=['Keep private/ separate from blinded reviewers.',
                    'Representative sampling is uniform over linked units then windows, not over all windows.',
                    'Error-enriched samples do not estimate population prevalence.',
                    'Retain uncertain and co-occurring labels. Do not overwrite original annotations.']))
    return out
