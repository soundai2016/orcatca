"""Export site ranking from the exact reports identified by a finished analysis.

Useful for an older analysis lacking the ranking files. Writes a NEW directory;
never edits the frozen analysis, result archive, protocol, or predictions.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from orcatca.result_store import CombinedStore
from orcatca.site_ranking import write_rank_tables
from orcatca.utils import sha256, atomic_json


def recover(analysis,output,source,reference=None,root=ROOT):
    infofile=analysis/'analysis.json'; prior=json.loads(infofile.read_text()); rows=[]; input_hashes={}
    if prior['status']!='completed':raise ValueError('Completion metadata required')
    if output.exists():raise FileExistsError('Choose a new output directory')
    with CombinedStore(root,source,reference) as store:
        if any(not x.archive for x in (store.primary,store.reference) if x):raise ValueError('This recovery command requires the exact packed report ZIP(s)')
        if store.source_hash!=prior['source_sha256']:raise ValueError('Report identity differs from analysis.json')
        exp=store.config['experiment'];limit=exp['selection_max_fpr'];seeds=prior['seeds'];methods=prior['methods']
        if seeds!=exp['seeds'] or len(prior['folds'])!=exp['folds']:raise ValueError('Seed/fold mismatch')
        for i in range(exp['folds']):
            fold=f'fold{i:02d}'; foldinfo=prior['folds'][i]
            if foldinfo['fold']!=fold or not foldinfo['protocol_verified']:raise ValueError('Missing successful fold audit: '+fold)
            for method in methods:
                st=store.primary if method in store.primary.config['experiment']['methods'] else store.reference
                if st is None:raise ValueError('Unknown method: '+method)
                which='primary:' if st is store.primary else 'reference:'
                complete=st.json('complete.json',protocol=True);key=complete['folds'][i]['manifest']
                for seed in seeds:
                    prefix=f'{fold}/{method}_s{seed}/'
                    for fn in ('holdout.jsonl.gz','calibration.json'):
                        member=st.prefix+prefix+fn
                        if st.checksums[member]!=prior['used_file_sha256'][which+member]:raise ValueError('Previously analyzed member differs: '+member)
                    c=st.json(prefix+'calibration.json');h=st.json(prefix+'holdout.json')
                    if h['best_sha256']!=c['best_sha256'] or h['manifest_sha256']!=complete['files'][key]:raise ValueError('Fit report model/protocol mismatch: '+prefix)
                    metrics=h['metrics']['sites']
                    if sum(v['n'] for v in metrics.values())!=foldinfo['holdout_windows']:raise ValueError('Reported site windows differ from analyzed holdout')
                    fit=[]
                    for site,v in metrics.items():
                        neg=v['fp']+v['tn'];pos=v['tp']+v['fn']
                        if pos+neg!=v['n'] or v['pauc_max_fpr']!=limit:raise ValueError('Invalid site support or pAUC limit')
                        fit.append(dict(fold=fold,method=method,seed=seed,site=site,negative_windows=neg,
                            positive_windows=pos,pauc=v['pauc'],auc=v['auc'],pauc_max_fpr=limit))
                    valid=[r['pauc'] for r in fit if r['positive_windows'] and r['negative_windows']]
                    if valid and not math.isclose(sum(valid)/len(valid),h['metrics']['macro_pauc'],abs_tol=1e-12,rel_tol=0):raise ValueError('Saved macro pAUC mismatch')
                    rows.extend(fit)
        input_hashes=store.used
    output.mkdir(parents=True,exist_ok=False);write_rank_tables(output,rows)
    atomic_json(output/'ranking_provenance.json',dict(source_sha256=prior['source_sha256'],analysis_json_sha256=sha256(infofile),
        used_file_sha256=input_hashes,source='metrics.sites in matching frozen holdout reports',seeds=seeds,methods=methods,
        pauc_max_fpr=limit,weighting='equal eligible site within fit, equal seed within fold, equal fold'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--source',required=True);p.add_argument('--reference');a=p.parse_args()
    recover(a.analysis,a.output,a.source,a.reference,a.root)
