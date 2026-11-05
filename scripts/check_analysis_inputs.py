"""Read-only inventory of frozen inputs. No model loading, inference or training."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from orcatca.result_store import CombinedStore
from orcatca.splitting import validate_splits
from orcatca.provenance import audit_protocol
from orcatca.utils import digest

DEFAULT = ['natural_mixstyle','natural_coarse_mixstyle','natural_threeclass_mixstyle','natural_subtype_mixstyle']
ERROR = (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, EOFError, zipfile.BadZipFile)


def present(store, key):
    if store.archive:
        name = store.prefix + key
        return name in store.archive.NameToInfo and name in store.checksums
    path = store.path / key
    return path.resolve().is_relative_to(store.root) and not path.is_symlink() and path.is_file()


def finite(v):
    return not isinstance(v, bool) and math.isfinite(float(v))


def rate(v):
    try:
        return finite(v) and 0 <= float(v) <= 1
    except (TypeError, ValueError):
        return False


def inspect(root, source, reference=None, methods=None, verify=False):
    methods = DEFAULT if methods is None else methods
    if len(set(methods)) != len(methods) or len(methods)<2:
        raise ValueError('The paired analysis requires at least two distinct methods')
    out = dict(mode='verify' if verify else 'inventory', folds=[], fits=[],
               scope='Input availability only. Cross-fit support and interval estimability are decided by analyze.')
    with CombinedStore(Path(root),source,reference) as combo:
        cfg = combo.config; settings = cfg['evaluation']; exp = cfg['experiment']
        rules = [f'fpr_{x:g}' for x in settings['fpr_targets']]
        if not exp['folds'] or not exp['seeds'] or not rules: raise ValueError('No folds, seeds or FPR targets')
        summaries = {}
        for st in (combo.primary,combo.reference):
            if not st: continue
            try:
                rows = csv.DictReader(io.StringIO(st.bytes('summary.csv').decode()))
                summaries[st] = {(r['fold'],r['run']): r for r in rows}
            except ERROR: summaries[st] = {}
        for f in range(exp['folds']):
            name=f'fold{f:02d}'; manifest=None; expected={}; fingerprints={}
            fold=dict(fold=name,protocol='unavailable',issues=[]);out['folds'].append(fold)
            try:
                if not verify:
                    for st in (combo.primary,combo.reference):
                        if not st:continue
                        complete=st.json('complete.json',True);key=complete['folds'][f]['manifest']
                        if hashlib.sha256(st.bytes(key,True)).hexdigest()!=complete['files'][key]:
                            raise ValueError('protocol manifest checksum mismatch: '+key)
                    fold['protocol']='manifest_present_content_unverified'
                else:
                    manifest=combo.protocol(f)
                    validate_splits(manifest)
                    data=cfg.get('data',{})
                    _, audit=audit_protocol(manifest,name,temporal_gap_seconds=data.get('session_gap_seconds',1800),block_hours=data.get('block_hours',24))
                    fold['protocol']=audit['status']
                    if audit['protocol_valid'] is not True: raise ValueError('protocol audit invalid: '+str(audit['errors'][:3]))
                    expected={role:{r['id']:r for r in manifest if r['split']==role} for role in ('calibrate','holdout')}
            except ERROR as e: fold['issues'].append(str(e))
            for m in methods:
                st=combo.primary if m in combo.primary.config['experiment']['methods'] else combo.reference
                for seed in exp['seeds']:
                    run=f'{m}_s{seed}'; prefix=f'{name}/{run}/'; errors=[]
                    row=dict(fold=name,method=m,seed=seed,issues=errors,archived_recall={},checkpoint_present=False);out['fits'].append(row)
                    if st is None or m not in st.config['experiment']['methods']:
                        errors.append('method absent from frozen configurations');continue
                    row['checkpoint_present']=present(st,prefix+'best.pt')
                    # Independently expose summary-only recall even if window files are lost.
                    try: h=st.json(prefix+'holdout.json') if present(st,prefix+'holdout.json') else {}
                    except ERROR: h={}
                    for rule in rules:
                        v=h.get('operating',{}).get(rule,{}).get('overall',{}).get('recall_kw')
                        if rate(v): row['archived_recall'][rule]=dict(value=float(v),source='holdout.json');continue
                        v=summaries[st].get((name,run),{}).get('holdout_'+rule+'_recall')
                        if rate(v):row['archived_recall'][rule]=dict(value=float(v),source='summary.csv')
                    try:
                        if st.json(prefix+'status.json').get('status')!='completed':raise ValueError('fit incomplete')
                    except ERROR as e: errors.append('status.json: '+str(e))
                    try:
                        c=st.json(prefix+'calibration.json')['operating_thresholds']
                        for key in rules:
                            if key not in c or c[key] is None or not finite(c[key]):raise ValueError('missing or invalid '+key)
                    except ERROR as e:errors.append('calibration.json: '+str(e))
                    for file_role,role in [('calibration','calibrate'),('holdout','holdout')]:
                        file=file_role+'.jsonl.gz'
                        try:
                            if not present(st,prefix+file):raise ValueError('missing file or unlisted report member')
                            if not verify:continue
                            rows,_,_=st.predictions(f,[m],[seed],file_role)
                            if expected:
                                lookup=expected[role]
                                if len(rows)!=len(lookup) or any(r['id'] not in lookup or any(r.get(k)!=lookup[r['id']].get(k) for k in ('recording','site','group','block','label','raw_class')) for r in rows):raise ValueError('prediction/protocol mismatch')
                            mark=digest(rows)
                            if mark!=fingerprints.setdefault(role,mark):raise ValueError('prediction metadata differ between fits')
                        except ERROR as e: errors.append(file+': '+str(e))
    ok=all(not x['issues'] for x in out['folds']+out['fits'])
    out['status']=('inputs_verified' if verify else 'inputs_present_unverified') if ok else 'incomplete_or_invalid'
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    choice=p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--source',help='Frozen suite directory or report ZIP')
    choice.add_argument('--layout',action='store_true',help='Inspect the migrated run tree; with --verify check original inputs and compare holdout summaries with retained results')
    p.add_argument('--root',type=Path,default=ROOT,help='Project root containing runs and state/protocol')
    p.add_argument('--reference');p.add_argument('--methods',nargs='+',default=DEFAULT)
    p.add_argument('--verify',action='store_true',help='With --source, align window scores. With --layout, hash direct suite inputs and reconcile saved holdout summaries.')
    a=p.parse_args()
    if a.layout and (a.reference or a.methods!=DEFAULT):
        p.error('Use --reference and --methods with --source; --layout inspects the recorded run locations')
    try:
        if a.layout:
            from run_layout import layout
            result=layout(a.root,a.verify)
        else: result=inspect(a.root,a.source,a.reference,a.methods,a.verify)
    except ERROR as e:
        if a.layout:
            result=dict(status='cannot_check_layout', error=str(e), hint='Use the project root with its original result/, runs/ and state/ directories.')
        else:
            result=dict(status='cannot_open_frozen_suite',error=str(e),hint='Supply a frozen suite directory or a packed report ZIP, not the source-code ZIP or figure_data directory.')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return int(result['status'] not in ('inputs_present_unverified','inputs_verified','layout_inventory','available_server_evidence_verified'))

if __name__=='__main__':raise SystemExit(main())
