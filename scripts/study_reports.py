#!/usr/bin/env python3
"""Pack or restore only the archived members needed to replay the transfer analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'orcatca-selected-prediction-inputs-v1'
ROLES = ('primary', 'reference')
METHODS = {'primary': ('natural_coarse_mixstyle', 'natural_threeclass_mixstyle'),
           'reference': ('natural_mixstyle', 'natural_subtype_mixstyle')}
SEEDS = (17, 29, 43)
FOLDS = ('fold00', 'fold01', 'fold02')
TAILS = ('status.json', 'calibration.json', 'calibration.jsonl.gz', 'holdout.jsonl.gz')
HEX = re.compile(r'[0-9a-f]{64}\Z')
README = """# Prediction inputs for the killer-whale detector transfer study

This is a lossless selection from the two original training reports. It contains
the saved scores and window metadata, thresholds and fit identities, frozen
configurations, and the exact protocol shared by the two suites. No model
weights, waveforms, training logs or software environments are included.
The protocol covers the entire split, including the training assignments.
A checkpoint hash identifies a fitted model; it is not a set of weights.

Use the matching source-code release. From its project root, after installing
the numerical analysis dependencies listed in its README:

    python scripts/study_reports.py restore --bundle prediction_inputs.zip --output result/replay_inputs
    PYTHONPATH=src python -m orcatca analyze --source result/replay_inputs/primary_selected.zip --reference result/replay_inputs/reference_selected.zip --methods natural_mixstyle natural_coarse_mixstyle natural_threeclass_mixstyle natural_subtype_mixstyle --repeats 2000 --crossfit-repeats 5 --seed 20261029 --output result/analysis/independent_replay
    python scripts/plot.py --analysis result/analysis/independent_replay --figure both

Set --bundle to the companion's actual path. The restore directory must be new;
it will never overwrite an archive. Restoring uses only Python's standard
library. The same restore script is also included at the top of this ZIP.
There is no need to unpack the hashed objects manually or access raw audio.

The manifest maps original member paths to their SHA-256 values under the
primary (new control models) and reference (binary and fine auxiliary) reports.
The 36 completed fits each have one calibration score stream and one holdout
stream. Original report-level SHA-256 values and their checksums-file hashes
identify the parent archives. Each selected member was checked against both its
original report's checksums and the retained analysis read trace. The restored
selected reports retain the original member bytes and paths, together with
a new REPORT_SHA256.json for those selected members. Their ZIP hashes differ
from the originals. The analysis thus reports new source paths/container
hashes; fitted probabilities, thresholds, protocol and numerical results do
not change. Files are selected-only derived reports, not training suites.

For example, compare transfer_gap_summary.csv, joint_differences.csv,
transfer_original_summary.csv, main_results.csv, site_metrics.csv and
protocol_audit.json with the retained analysis. Re-running produces
new logs and packaging metadata. Bytes of gzip wrappers can vary, so compare
decompressed contents when comparing output window manifests.
"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def allowed(role, prefix):
    allowed_names={prefix+'config.yaml', prefix+'config.json','state/protocol/complete.json'}
    for fold in FOLDS:
        allowed_names.add('state/protocol/'+fold+'.jsonl.gz')
        for method in METHODS[role]:
            for seed in SEEDS:
                for tail in TAILS:
                    allowed_names.add(f'{prefix}{fold}/{method}_s{seed}/{tail}')
    return allowed_names


def validate(report, role):
    keys=set(report['members'])
    prefixes=[p for p in ('runs/','result/runs/') if p+'config.yaml' in keys]
    if len(prefixes)!=1 or keys!=allowed(role,prefixes[0]):
        raise ValueError('Unexpected or incomplete run member set: '+role)
    if not all(HEX.fullmatch(h) for h in report['members'].values()):
        raise ValueError('Malformed SHA-256 digest in manifest')
    return prefixes[0]


def output_path(path):
    path=path.resolve()
    if 'dclde_2027_killer_whales' in path.parts:
        raise ValueError('Output must be outside the raw audio corpus')
    if path.exists(): raise FileExistsError(f'Output must be new: {path}')
    return path


def put(archive,name,data):
    # Fixed timestamps and exact bytes keep the derived containers reproducible.
    info=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
    info.compress_type=zipfile.ZIP_STORED if name.endswith('.gz') else zipfile.ZIP_DEFLATED
    info.external_attr=0o100644 << 16
    archive.writestr(info,data)


def encode(obj):
    return (json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def pack(primary,reference,trace_path,output):
    output=output_path(output)
    trace=json.loads(trace_path.read_bytes())
    if trace.get('status')!='completed': raise ValueError('Expected a completed analysis trace')
    manifest={'format':SCHEMA,'original_combined_sha256':trace['source_sha256'],
              'methods':trace['methods'],'seeds':trace['seeds'],'reports':{}}
    sources={'primary':primary,'reference':reference}
    for role,path in sources.items():
        report={'original_report_filename':path.name, 'original_report_sha256':file_hash(path),
                'members':{k.split(':',1)[1]:v for k,v in trace['used_file_sha256'].items()
                           if k.startswith(role+':')}}
        validate(report,role);manifest['reports'][role]=report
    joined=''.join(manifest['reports'][r]['original_report_sha256'] for r in ROLES)
    if digest(joined.encode())!=manifest['original_combined_sha256']:
        raise ValueError('Supplied source archives do not match canonical run (including role order)')
    output.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    with zipfile.ZipFile(output,'x',allowZip64=True) as target:
        for role,path in sources.items():
            report=manifest['reports'][role];prefix=validate(report,role)
            with zipfile.ZipFile(path) as src:
                names=src.namelist()
                if len(names)!=len(set(names)): raise ValueError('Duplicate original member names')
                checks=src.read('REPORT_SHA256.json');expected=json.loads(checks)
                report['original_checksums_file_sha256']=digest(checks);report['fits']={}
                for name,sha in sorted(report['members'].items()):
                    raw=src.read(name)
                    if digest(raw)!=sha or expected.get(name)!=sha:
                        raise ValueError('Original/checksums/analysis mismatch for '+name)
                    if sha not in seen:
                        put(target,'objects/'+sha,raw);seen.add(sha)
                    if name==prefix+'config.json': report['configuration_identity']=json.loads(raw)['identity']
                    if name.endswith('/calibration.json'):
                        cal=json.loads(raw)
                        report['fits'][name[len(prefix):].rsplit('/',1)[0]]={
                            'identity':cal['identity'],'checkpoint_sha256':cal['best_sha256'],
                            'checkpoint_epoch':cal['checkpoint_epoch']}
        put(target,'manifest.json',encode(manifest));put(target,'README.md',README.encode())
        put(target,'study_reports.py',Path(__file__).read_bytes())
    print(f'Created {output} ({len(seen)} unique objects, {output.stat().st_size/2**20:.2f} MiB)')


def restore(bundle,output):
    output=output_path(output)
    with zipfile.ZipFile(bundle) as src:
        names=src.namelist()
        if len(names)!=len(set(names)): raise ValueError('Duplicate companion member names')
        manifest=json.loads(src.read('manifest.json'))
        reports=manifest['reports']
        if manifest['format']!=SCHEMA or set(reports)!=set(ROLES):
            raise ValueError('Unrecognized prediction input manifest')
        source_hashes=set()
        for role in ROLES:
            validate(reports[role],role);source_hashes.update(reports[role]['members'].values())
        if set(names)!={'objects/'+s for s in source_hashes}|{'manifest.json','README.md','study_reports.py'}:
            raise ValueError('Missing or unexpected companion objects')
        # Verify before creating outputs, then write only controlled archive paths.
        for h in source_hashes:
            if digest(src.read('objects/'+h))!=h: raise ValueError('Companion object mismatch: '+h)
        output.mkdir(parents=True,exist_ok=False)
        for role in ROLES:
            report=reports[role];dest=output/(role+'_selected.zip')
            with zipfile.ZipFile(dest,'x',allowZip64=True) as z:
                for n,h in sorted(report['members'].items()): put(z,n,src.read('objects/'+h))
                put(z,'REPORT_SHA256.json',encode(report['members']))
            print(f'{role}: {len(report["members"])} original members; {dest}; SHA-256 {file_hash(dest)}')


def main():
    p=argparse.ArgumentParser(description=__doc__);cmd=p.add_subparsers(dest='cmd',required=True)
    pack_cmd=cmd.add_parser('pack',help='Build from original, hash-verified reports and a completed analysis')
    pack_cmd.add_argument('--primary',type=Path,required=True);pack_cmd.add_argument('--reference',type=Path,required=True)
    pack_cmd.add_argument('--trace',type=Path,default=ROOT/'result/analysis/transfer_review/analysis.json')
    pack_cmd.add_argument('--output',type=Path,required=True)
    rest=cmd.add_parser('restore',help='Create two fresh ResultStore-compatible derived report archives')
    rest.add_argument('--bundle',type=Path,required=True);rest.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.cmd=='pack': pack(a.primary,a.reference,a.trace,a.output)
    else: restore(a.bundle,a.output)


if __name__=='__main__': main()
