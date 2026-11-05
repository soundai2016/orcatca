#!/usr/bin/env python3
"""Verify the rates, transfer contrasts and counts in a completed 36-fit analysis.

Reads the saved integer counts. Does not run the model or reconstruct bootstrap
draws. Optional table-directory and replay-check arguments verify the
LaTeX cells and the bytes of retained analysis files, respectively.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import gzip
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
from statistics import fmean as mean
import sys

ROOT = Path(__file__).resolve().parents[1]
BINARY, COARSE, THREE, FINE = METHODS = ('natural_mixstyle', 'natural_coarse_mixstyle',
                                          'natural_threeclass_mixstyle', 'natural_subtype_mixstyle')
FOLDS = {'fold00': 'SIO', 'fold01': 'DFO', 'fold02': 'Joint'}
LABELS = ('HW', 'AB')
SEEDS = ('17', '29', '43')
RULES = ('0.01', '0.05')
NAMES = dict(zip(METHODS, ('Binary', 'Coarse auxiliary', 'Three-class', 'Fine auxiliary')))
K = ('fold', 'method', 'alpha')
KEY = K + ('seed', 'repeat', 'side')
# Printed percentages in Table S3, in method order; each row: 1% FPR/recall, 5% FPR/recall.
S3 = {'SIO': [(8.99,71.21,19.74,87.53),(8.55,69.06,19.74,86.71),(12.06,75.13,24.12,89.81),(10.09,72.36,20.18,90.46)],
      'DFO': [(19.16,76.08,24.91,82.89),(17.97,76.02,22.52,81.01),(13.79,72.32,20.29,78.50),(13.05,69.51,20.88,77.89)],
      'Joint': [(4.36,68.20,11.01,83.79),(4.15,69.45,11.96,86.77),(4.89,70.27,14.22,87.12),(3.94,72.37,13.59,88.53)]}
# Source HW/AB: windows (linked units); target KW/HW/AB: windows (recordings).
T1 = {'SIO': [(12317,33),(916,52),(2539,62),(100,2),(52,17)],
      'DFO': [(14,2),(121,10),(24813,1568),(113602,5211),(9112,3046)],
      'Joint': [(11891,37),(1186,39),(12693,226),(339,286),(929,104)]}
T2 = {('SIO','0.01'):(1.79,21.71,86.3,92.4),('SIO','0.05'):(6.03,39.51,77.3,86.8),
      ('Joint','0.01'):(2.95,3.49,76.4,54.1),('Joint','0.05'):(10.90,5.81,59.4,34.8)}


def need(test, reason):
    if not test:
        raise ValueError(reason)


def close(a, b, name):
    need(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-10), f'{name}: {a} vs {b}')


def printed(values, expected, digits, name):
    """Compare rounded values at the precision of each reference table."""
    got = tuple(round(x, n) for x, n in zip(values, digits, strict=True))
    need(got == tuple(expected), f'{name}: calculated {got}, printed {expected}')
    return got


def index(rows, keys):
    out = {}
    for r in rows:
        k = tuple(r[col] for col in keys)
        need(k not in out, 'duplicate row: ' + str(k))
        out[k] = r
    return out


def grid(rows, keys, populations):
    out = index(rows, keys)
    expected = set(product(*populations))
    need(set(out) == expected, f'incomplete/unexpected grid {keys}: ' + str(sorted(expected ^ set(out))[:3]))
    return out


def tex_rows(tables_dir, label):
    text = '\n'.join(p.read_text(encoding='utf-8') for p in sorted(tables_dir.glob('*.tex')))
    bodies = [t for t in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S)
              if r'\label{' + label + '}' in t]
    need(len(bodies) == 1, f'missing/ambiguous {label} in {tables_dir}')
    rows = []
    for line in bodies[0].splitlines():
        cells = [s.strip() for s in line.replace(r'\,','').split('&')]
        if len(cells) > 1 and cells[0] in ('Scripps (SIO)', 'SIO', 'DFO','Joint','') and line.rstrip().endswith(r'\\'):
            cells[-1] = cells[-1][:-2].strip()
            if any(re.fullmatch(r'[+-]?[\d.]+(?: \([\d]+\))?',s) for s in cells[1:]):
                rows.append(cells)
    return rows


def check_tex_tables(path, table1, table2, s3):
    def read(label, keyed):
        out = {}
        fold = None
        for r in tex_rows(path, label):
            fold = (r[0].replace('Scripps (SIO)','SIO') or fold)
            key, vals = keyed(fold, r[1:])
            need(key not in out, f'duplicate table row {key}')
            out[key] = vals
        return out
    parsed = read('tab:support',lambda f,c:(f,[tuple(map(int,re.findall(r'\d+',v))) for v in c]))
    need(parsed==table1,'Table 1 LaTeX cells differ from calculated recording support')
    parsed = read('tab:labelmix',lambda f,c:((f,str(float(c[0].split('\\')[0])/100)),tuple(map(float,c[1:]))))
    need(parsed==table2,'Table 2 LaTeX cells differ from calculated attribution')
    parsed = read('tab:full',lambda f,c:((f,c[0]),tuple(map(float,c[1:]))))
    need(parsed==s3,'Table S3 LaTeX cells differ from the 48 calculated absolute rates')


def audit(base, tables_dir=None, replay=None):
    def rows(name):
        with (base/(name+'.csv')).open(encoding='utf-8', newline='') as fh:
            return list(csv.DictReader(fh))
    def obj(name):
        return json.loads((base/(name+'.json')).read_text())
    meta=obj('analysis')
    need(not (base/'failure.json').exists() and meta['status']=='completed', 'not a completed analysis')
    need(set(meta['methods'])==set(METHODS) and set(map(str,meta['seeds']))==set(SEEDS), 'different study fits')
    need((meta['analysis_seed'],meta['bootstrap_repeats'],meta['crossfit_repeats'])==(20261029,2000,5), 'unexpected resampling settings')
    need(meta['crossfit_status']=={'fold00':'ready','fold01':'not_estimable','fold02':'ready'}, 'different crossfit support')
    counts=grid(rows('transfer_original_labels'),K+('seed','role','subtype'),
                [FOLDS,METHODS,RULES,SEEDS,('source_calibration','target'),('KW',*LABELS)])
    ops=grid(rows('transfer_original_operating'),K+('seed',),[FOLDS,METHODS,RULES,SEEDS])
    full=grid(rows('transfer_original_summary'),K,[FOLDS,METHODS,RULES])
    flow=index(rows('data_flow'),('fold','role','collection','raw_class'))
    protocol=index(obj('protocol_audit')['folds'],('fold',))
    t1={}; t2={}; s3={}; actual={}; support={}
    for (f,m,a),summary in full.items():
        rates=[]
        for seed in SEEDS:
            key=f,m,a,seed
            o=ops[key]
            need(o['procedure']==summary['procedure']=='full_calibration', 'wrong full-rule provenance')
            for side,source in [('source','source_calibration'),('target','target')]:
                by={}
                for c in ('KW',*LABELS):
                    row=counts[key+(source,c)]
                    # The historical field 'errors' counts flags: a flag is a TP for KW.
                    n,e,nr,nu=(int(row[z]) for z in ('windows','errors','recordings','units'))
                    need(0<=e<=n and 0<nr<=n and 0<nu<=n,f'invalid counts at {key}/{side}/{c}')
                    if source=='source_calibration' and c in LABELS:
                        need(e<=math.floor(float(a)*n),'full source constraint violated')
                    s=support.setdefault((f,source,c),(n,nr,nu))
                    need(s==(n,nr,nu),f'support changes across seeds, rules or methods: {f}/{source}/{c}')
                    close(n,o[f'{side}_{c}_n'],'label denominator')
                    close(e,o[f'{side}_{c}_fp'],'number flagged')
                    close(e/n,row['label_rate'],'label rate')
                    by[c]=n,e
                den=sum(by[c][0] for c in LABELS); fp=sum(by[c][1] for c in LABELS)
                fpr,recall=fp/den,by['KW'][1]/by['KW'][0]
                close(den,o[side+'_negative'],'negative denominator')
                close(fp,o[side+'_fp'],'pooled negative flags')
                close(by['KW'][0],o[side+'_positive'],'KW denominator')
                close(by['KW'][1],o[side+'_tp'],'KW flags')
                close(fpr,o[side+'_fpr'],'full calibration FPR')
                close(recall,o[side+'_recall'],'full calibration recall')
                if source=='target': rates.append((fpr,recall))
        fpr,recall=map(mean,zip(*rates))
        close(fpr,summary['target_fpr'], 'full FPR mean')
        close(recall,summary['target_recall'], 'full recall mean')
        actual[f,m,a]=(fpr,recall)
    for f,col in FOLDS.items():
        p=protocol[f,]
        need(p['protocol_valid'] is True and p['prediction_metadata']['status']=='aligned'
             and not p['errors'] and p['temporal_check']['observed_cross_role_pairs']==0,'archived protocol flagged')
        table=[]
        for role,label in [('source_calibration',c) for c in LABELS]+[('target',c) for c in ('KW',*LABELS)]:
            n,nr,nu=support[f,role,label]
            role2='calibrate' if role=='source_calibration' else 'holdout'
            fr=[r for (fold,role_,_,cls),r in flow.items() if (fold,role_,cls)==(f,role2,label)]
            close(n,sum(int(r['unique_windows']) for r in fr),'manifest/score support')
            table.append((n,nu if role=='source_calibration' else nr))
        need(table==T1[col],f'printed Table 1 differs for {col}: {table}')
        t1[col]=table
        for j,m in enumerate(METHODS):
            v=[100*x for a in RULES for x in actual[f,m,a]]
            s3[col,NAMES[m]]=printed(v,S3[col][j],(2,)*4,'Table S3 '+col+' '+m)
    # Paired method contrasts are differences at each fitted model's own full cutoff.
    contrasted=[r for r in rows('joint_differences') if r['fold'] in FOLDS]
    contrasts=grid(contrasted,('fold','rule','b'),[FOLDS,('fpr_0.01','fpr_0.05'),METHODS[1:]])
    for (f,rule,m),r in contrasts.items():
        a=rule.split('_')[1];need(r['a']==BINARY and int(r['attempts'])-int(r['rejected'])==2000,'paired bootstrap design')
        for i,k in enumerate(('fpr','recall')):
            close(actual[f,m,a][i]-actual[f,BINARY,a][i],r['delta_'+k], 'paired '+k)
            need(float(r[k+'_ci_low'])<=float(r[k+'_ci_high']),'invalid paired interval ordering')
    ready=('fold00','fold02')
    cross=grid(rows('crossfit_operating'),KEY+('role',),
               [ready,METHODS,RULES,SEEDS,tuple(map(str,range(5))),('0','1'),('source_fit','source_check','target')])
    transfer=grid(rows('crossfit_transfer'),KEY,[ready,METHODS,RULES,SEEDS,tuple(map(str,range(5))),('0','1')])
    gaps=grid(rows('transfer_gap_summary'),K,[FOLDS,METHODS,RULES])
    boot=index(obj('transfer_bootstrap_audit')['folds'],('fold','alpha'))
    summary=defaultdict(list); headline={}; exceed=defaultdict(int)
    for key,tf in transfer.items():
        f,m,a=key[:3];fit,src,tgt=[cross[key+(r,)] for r in ('source_fit','source_check','target')]
        for row in (fit,src,tgt):
            close(row['threshold'],tf['threshold'],'matched cutoff')
            den=sum(int(row[c+'_n']) for c in LABELS);flags=sum(int(row[c+'_fp']) for c in LABELS)
            close(den,row['negative'],'crossfit denominator');close(flags,row['fp'],'crossfit flags')
            close(flags/den,row['fpr'],'crossfit FPR')
            for c in LABELS:
                n,e=int(row[c+'_n']),int(row[c+'_fp']);need(n>0 and 0<=e<=n,'invalid label counts')
                close(e/n,row[c+'_fpr'],'crossfit label FPR')
                if row is fit: need(e<=math.floor(float(a)*n),'source half constraint violated')
        for c in LABELS:
            close(int(fit[c+'_n'])+int(src[c+'_n']),support[f,'source_calibration',c][0],'source partition coverage')
            close(tgt[c+'_n'],support[f,'target',c][0],'target support')
        weight=[int(tgt[c+'_n'])/int(tgt['negative']) for c in LABELS]
        qs=[int(src[c+'_fp'])/int(src[c+'_n']) for c in LABELS]
        qt=[int(tgt[c+'_fp'])/int(tgt[c+'_n']) for c in LABELS]
        delta=[t-s for t,s in zip(qt,qs)]
        proj=sum(w*s for w,s in zip(weight,qs)); obs=float(tgt['fpr'])
        contrib=[w*d for w,d in zip(weight,delta)]
        close(proj,tf['mixture_only'],'mixture projection')
        close(sum(contrib),obs-proj,'gap identity')
        close(obs-proj,tf['target_weighted_conditional'],'matched gap')
        summary[f,m,a].append((proj,obs,obs-proj,max(qs),*delta,*contrib))
        exceed[f,a]+=int(obs>max(qs))
    for (f,m,a),g in gaps.items():
        b=boot[f,float(a)]
        if f=='fold01':
            need(g['status']==b['status']=='not_estimable' and b['reason']=='insufficient_independent_subtype_units'
                 and not any(g[k] for k in ('transfer_gap','gap_ci_low','gap_ci_high')), 'DFO is not a zero-gap estimate')
            continue
        vals=list(map(mean,zip(*summary[f,m,a])))
        proj,obs,gap,maximum,dh,da,ch,ca=vals
        for val,name in ((proj,'projected_fpr'),(obs,'observed_fpr'),(gap,'transfer_gap')):
            close(val,g[name],'matched mean '+name)
        need(b['bootstrap_valid']==int(g['bootstrap_valid'])==2000 and
             b['bootstrap_rejected']==int(g['bootstrap_rejected'])==(292 if f=='fold00' else 0) and
             b['bootstrap_attempts']==2000+b['bootstrap_rejected'],'bootstrap retention')
        lo,hi=[float(g[n]) for n in ('gap_ci_low','gap_ci_high')]
        need(g['gap_ci_status']=='conditional_percentile' and lo<=hi and (a!='0.01' or lo>0), 'archived gap interval')
        if m==BINARY:
            col=FOLDS[f]
            t2[col,a]=printed([100*dh,100*da,100*ca/(ch+ca),100*da/(dh+da)],T2[col,a],(2,2,1,1),'Table 2 '+col+' '+a)
            if a=='0.01':
                v=[100*x for x in (proj,obs,gap,lo,hi,maximum,obs-maximum)]
                expected=(0.65,9.25,8.61,3.31,21.95,1.25,8.01) if f=='fold00' else (1.08,4.42,3.34,1.13,5.83,1.42,3.00)
                headline[col]=dict(zip(('projected_pct','observed_pct','gap_pp','saved_low_pp','saved_high_pp','empirical_max_pct','excess_pp'),printed(v,expected,(2,)*7,'binary matched result')))
    need(exceed['fold00','0.01']==120 and exceed['fold02','0.01']==118,'empirical-maximum comparison count')
    # Other population-sensitive statements in Results and Discussion.
    source_composition={}
    for f,expected_pct in [('fold00',99.1),('fold02',99.4)]:
        dfo=sum(int(flow[f,'calibrate','DFO',c]['unique_windows']) for c in LABELS)
        total=sum(support[f,'source_calibration',c][0] for c in LABELS)
        source_composition[FOLDS[f]]=printed([100*dfo/total],[expected_pct],[1], 'DFO share of calibration negatives')[0]
    need(support['fold01','source_calibration','HW']==(14,14,2),'DFO source HW denominator')
    for c in ('KW',*LABELS):
        r=flow['fold00','holdout','SIO',c]
        close(r['unique_windows'],r['missing_timestamp_windows'],'SIO timestamp coverage')
    for f in FOLDS:
        ratios=[];hw=[];ab=[]
        for m in METHODS:
            rh,ra=[mean(float(counts[f,m,'0.01',z,'target',c]['label_rate']) for z in SEEDS) for c in LABELS]
            ratios.append(ra/rh);hw.append(rh*100);ab.append(ra*100)
        expect={'fold00':(6.12,13.85),'fold01':(.23,.30),'fold02':(1.42,2.01)}
        printed([min(ratios),max(ratios)],expect[f],[2,2],'full-calibration AB/HW rate ratios')
        if f=='fold01':
            printed([min(hw),max(hw),min(ab),max(ab)],(13.82,20.31,3.45,4.76),(2,)*4,'DFO label-specific FPR')
    influence=rows('transfer_block_influence')
    largest={}
    for f in ready:
        group=[r for r in influence if r['fold']==f and r['method']==BINARY and r['alpha']=='0.01'
               and r['role']=='target' and r['status']=='estimated']
        for r in group:
            close(float(r['observed_without_block'])-float(r['projected_without_block']),r['gap_without_block'],'block-deletion gap')
            close(float(r['gap_without_block'])-float(r['full_gap']),r['gap_change'],'block-deletion shift')
        r=max(group,key=lambda item:abs(float(item['gap_change'])))
        largest[FOLDS[f]]=round(100*abs(float(r['gap_change'])),2)
        if f=='fold00':
            need((int(r['removed_HW_windows']),int(r['removed_AB_windows']))==(85,0),'influential SIO block')
            ab_delta=mean(x[5] for x in summary[f,BINARY,'0.01'])
            new_w=support[f,'target','AB'][0]/(support[f,'target','AB'][0]+support[f,'target','HW'][0]-85)
            printed([100*float(r['gap_without_block']),100*new_w,100*new_w*ab_delta],(18.22,77.6,16.85),(2,1,2),'SIO block deletion')
        else:printed([abs(float(r['gap_change']))*100],(.99,),(2,),'Joint block deletion')
    need(all(float(contrasts['fold00','fpr_'+a,FINE]['fpr_ci_low'])<0<float(contrasts['fold00','fpr_'+a,FINE]['fpr_ci_high'])
             for a in RULES),'SIO fine-minus-binary FPR intervals')
    # pAUC is averaged within collection, then across the three collections.
    site=rows('site_metrics')
    bysite=index(site,('fold','site','method','seed'))
    pairs=sorted({k[:2] for k in bysite}); all_support={}
    for f,s in pairs:
        pop={(int(bysite[f,s,m,z]['negative_windows']),int(bysite[f,s,m,z]['positive_windows'])) for m,z in product(METHODS,SEEDS)}
        need(len(pop)==1, 'inconsistent site denominators')
        all_support[f,s]=pop.pop()
    sensitivity=grid(rows('support_sensitivity'),('fold','method','min_negative'),[FOLDS,METHODS,('1','20')])
    macro=grid(rows('main_results'),('method',),[METHODS])
    profiles={};eligible={}
    for minimum in (1,20):
        active=[(f,s) for (f,s),(neg,pos) in all_support.items() if pos>0 and neg>=minimum]
        eligible[minimum]=active;profiles[minimum]={}
        for m in METHODS:
            collection=[]
            for f in FOLDS:
                sites=[s for fold,s in active if fold==f]
                value=mean(mean(float(bysite[f,s,m,z]['pauc']) for s in sites) for z in SEEDS)
                close(value,sensitivity[f,m,str(minimum)]['macro_pauc'],'site sensitivity')
                close(len(sites),sensitivity[f,m,str(minimum)]['sites'],'eligible site count')
                collection.append(value)
            profiles[minimum][m]=mean(collection)
            if minimum==1:close(mean(collection),macro[m,]['macro_pauc'],'macro pAUC')
    need(len(eligible[1])==14 and len(eligible[20])==13 and set(eligible[1])-set(eligible[20])=={('fold00','SIO::Quin_Can')},'supports of pAUC panels')
    gains=[profiles[n][FINE]-profiles[n][BINARY] for n in (1,20)]
    printed(gains,(0.0304,0.0047),(4,4),'pAUC gain')
    printed([profiles[1][BINARY],profiles[1][FINE],profiles[20][COARSE]],(.7794,.8098,.7795),(4,4,4),'pAUC values')
    need(max(profiles[1],key=profiles[1].get)==FINE and max(profiles[20],key=profiles[20].get)==COARSE,'pAUC ordering')
    need(all_support['fold00','SIO::Quin_Can'][0]==10,'Quin_Can negative count')
    for s,v in [('SIO::Quin_Can',.1392),('SIO::Cpe_Elz',-.0152)]:
        gain=mean(float(bysite['fold00',s,FINE,z]['pauc'])-float(bysite['fold00',s,BINARY,z]['pauc']) for z in SEEDS)
        printed([gain],[v],[4],s+' pAUC gain')
    if tables_dir: check_tex_tables(tables_dir,t1,t2,s3)
    out=dict(status='passed',level='aggregate table reconciliation; prediction scores and bootstrap draws not replayed',
             source=str(base),fits=len(ops)//2,full_fit_rules=len(ops),matched_configurations=len(transfer),
             matched_configurations_at_1pct=len(transfer)//2,absolute_rate_cells_checked=4*len(s3),
             tex_tables_compared=['1','2','S3'] if tables_dir else [],
             binary_at_matched_1pct=headline,full_calibration_binary_1pct={FOLDS[f]:dict(fpr_pct=round(100*actual[f,BINARY,'0.01'][0],2),
             recall_pct=round(100*actual[f,BINARY,'0.01'][1],2)) for f in FOLDS},dfo_disjoint_check='not_estimable',
             dfo_share_of_source_negatives_pct=source_composition,largest_single_target_block_gap_change_pp=largest,
             eligible_sites={str(k):len(v) for k,v in eligible.items()},fine_minus_binary_macro_pauc=gains,
             protocol_scope='Checked stored protocol flags, support counts and recording assignments as reported; no per-window re-audit.')
    if replay:
        record=json.loads(replay.read_text()); digests=record['sha256_comparing_uncompressed_content']
        matched=[]; absent=[]
        for name,want in digests.items():
            need(Path(name).name==name, 'unsafe replay filename')
            p=base/name
            if not p.is_file(): absent.append(name); continue
            with (gzip.open(p,'rb') if p.suffix=='.gz' else p.open('rb')) as fh:
                got=hashlib.file_digest(fh,'sha256').hexdigest()
            need(got==want,f'historical replay hash differs: {name}');matched.append(name)
        need(set(record['bytes_equal_csv'])<=set(matched) and len(matched)>=17,'missing expected replay CSVs')
        out['historical_replay_hashes']={'matched':matched,'not_present':absent,'scope':'Hashes from prior prediction replay; not a new replay.'}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis',type=Path,default=ROOT/'result/analysis/transfer_review')
    ap.add_argument('--tables-dir',type=Path,help='Compare the three labelled tables found in this directory’s .tex files.')
    ap.add_argument('--replay-check',type=Path,help='Optional historical analysis_replay_check.json; compares retained hashes.')
    ap.add_argument('--output',type=Path,help='Optional new JSON file. Otherwise prints the report.')
    args=ap.parse_args()
    try:
        report=audit(args.analysis,args.tables_dir,args.replay_check)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8') as fh: json.dump(report,fh,indent=2,allow_nan=False);fh.write('\n')
        else: print(json.dumps(report,indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as exc:
        print('Audit failed:', exc,file=sys.stderr);return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
