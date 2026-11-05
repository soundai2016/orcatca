"""Inspect the article's migrated run, report and summary locations without writing."""
import csv
import hashlib
import json
import math
from pathlib import Path

from orcatca.result_store import ResultStore
from study_reports import FOLDS, METHODS, SEEDS, allowed

SUITE = {'primary': 'semantic230_20261029T160625Z_9e1b11a8',
         'reference': 'jel220_20261026T021531Z_ac09b0f3'}
ARTICLE = Path('result/analysis/transfer_review')
FIGURE_INPUTS = ('analysis.json', 'transfer_gap_summary.csv', 'transfer_label_summary.csv',
                 'transfer_original_labels.csv', 'transfer_original_summary.csv',
                 'crossfit_operating.csv', 'crossfit_transfer.csv', 'site_metrics.csv',
                 'joint_differences.csv')


def fits(role):
    for fold in FOLDS:
        for method in METHODS[role]:
            for seed in SEEDS:
                yield fold, method, str(seed)


def index_rows(file, fields):
    with file.open(newline='') as stream:
        data = {}
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in fields)
            if key in data:
                raise ValueError(f'Duplicate key in {file.name}: {key}')
            data[key] = row
    return data


def compare_holdout(holdout, endpoints, fitted, manifest_hash, checkpoint_hash):
    """Compare counts and rates for one completed fit at both original cutoffs."""
    if holdout['manifest_sha256'] != manifest_hash or holdout['best_sha256'] != checkpoint_hash:
        raise ValueError('Frozen manifest or checkpoint identity differs')
    def equal(a, b, field):
        if (not math.isfinite(float(a)) or not math.isfinite(float(b)) or
                not math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=0)):
            raise ValueError(f'{field}: saved {a}, analysis {b}')
    equal(holdout['metrics']['macro_pauc'], fitted['macro_pauc'], 'site-mean pAUC')
    for alpha in ('0.01', '0.05'):
        actual = holdout['operating']['fpr_'+alpha]['overall']
        article = endpoints[alpha]
        if article['procedure'] != 'full_calibration' or article['status'] != 'estimated':
            raise ValueError('Expected an estimated full-calibration endpoint')
        for field, column in [('threshold', 'threshold'), ('fp', 'target_fp'), ('tp', 'target_tp'),
                              ('false_positive_rate', 'target_fpr'), ('recall_kw', 'target_recall')]:
            equal(actual[field], article[column], alpha+' '+field)
        equal(actual['fp'] + actual['tn'], article['target_negative'], alpha+' negative windows')
        equal(actual['tp'] + actual['fn'], article['target_positive'], alpha+' positive windows')


def check_run(root, role, run, trace):
    expected = {k.split(':', 1)[1]: v for k, v in trace.items() if k.startswith(role+':')}
    prefixes = [p for p in ('runs/', 'result/runs/') if p+'config.yaml' in expected]
    if len(prefixes) != 1 or set(expected) != allowed(role, prefixes[0]):
        raise ValueError(role+': article read trace has unexpected member coverage')
    with ResultStore(root, str(run)) as store:
        if store.name != SUITE[role]:
            raise ValueError('Unexpected frozen suite: '+store.name)
        for key, wanted in sorted(expected.items()):
            protocol = key.startswith('state/protocol/')
            prefix = 'state/protocol/' if protocol else prefixes[0]
            value = store.bytes(key[len(prefix):], protocol)
            if hashlib.sha256(value).hexdigest() != wanted:
                raise ValueError('Saved member differs from article: '+role+':'+key)
    return len(expected)


def layout(root, verify=False):
    root = Path(root).resolve()
    article = root / ARTICLE
    def safe(p):
        q = p.resolve()
        if not q.is_relative_to(root) or q.is_relative_to(root/'dclde_2027_killer_whales') or p.is_symlink():
            raise ValueError('Refusing redirected data path: '+str(p))
        return p
    missing = [n for n in FIGURE_INPUTS if not safe(article/n).is_file()]
    info = json.loads((article/'analysis.json').read_bytes()) if 'analysis.json' not in missing else {}
    out = dict(status='layout_inventory', figure_input=str(ARTICLE),
               figure_files_present=not missing and info.get('status') == 'completed',
               missing_figure_inputs=missing, runs={}, summary_files={}, original_report_candidates={},
               checked_trace_members={}, verified_holdout_fits={}, issues=[])
    failed = safe(root/'result/analysis/reanalysis/failure.json')
    if failed.exists():
        err = json.loads(failed.read_bytes())
        out['previous_failed_analysis'] = dict(path=str(failed.relative_to(root)),
            exception=err.get('exception'), message=err.get('message'))
    roots = {}
    for role, suite in SUITE.items():
        run = safe(root/'runs'/suite)
        complete = sum(all(safe(run/fold/f'{m}_s{s}'/(r+'.jsonl.gz')).is_file()
                           for r in ('calibration', 'holdout')) for fold, m, s in fits(role))
        out['runs'][role] = dict(path=str(run.relative_to(root)), directory_present=run.is_dir(),
                                 window_score_pairs=complete, expected_fits=18)
        report = safe(root/'state'/(suite+'_report.zip'))
        if report.is_file(): out['original_report_candidates'][role] = str(report.relative_to(root))
        if run.is_dir():
            roots[role+'_run'] = (role, run)
            if verify:
                try: out['checked_trace_members'][role] = check_run(root, role, run, info['used_file_sha256'])
                except (OSError, ValueError, KeyError, TypeError) as e: out['issues'].append(str(e))
    cached = safe(root/'result/analysis/reference/holdouts')
    if cached.is_dir(): roots['reference_holdout_cache'] = ('reference', cached)
    for name, (role, base) in roots.items():
        out['summary_files'][name] = sum(safe(base/f/f'{m}_s{s}'/'holdout.json').is_file()
                                         for f, m, s in fits(role))
    candidates = [root/'prediction_inputs.zip', root/'result/replay_inputs/primary_selected.zip',
                  root/'result/replay_inputs/reference_selected.zip']
    out['prediction_archive_candidates'] = [str(p.relative_to(root)) for p in candidates if safe(p).is_file()]
    possible = {role: (out['runs'][role]['window_score_pairs'] == 18 or
                       role in out['original_report_candidates'] or
                       f'result/replay_inputs/{role}_selected.zip' in out['prediction_archive_candidates'])
                for role in SUITE}
    missing_roles = [role for role in SUITE if not possible[role]]
    if not missing_roles:
        out['full_replay'] = 'Both input candidates located; analyze --expect-trace checks identity.'
    elif 'prediction_inputs.zip' in out['prediction_archive_candidates']:
        out['full_replay'] = 'Prediction companion located; restore and validate its selected reports.'
    else:
        out['full_replay'] = ('Missing window inputs in inspected locations: '+', '.join(missing_roles)+
                              '. Supply the corresponding original report or the prediction companion.')
    out['scope'] = ('Only named paths were inspected. Per-fit holdout summaries support absolute rates '
                    'and site-mean pAUC; they contain no window predictions for a new paired resample. '
                    'Report candidates require the full trace check during replay.')
    if verify and roots:
        fit_key = ('fold', 'method', 'seed')
        trace = info.get('used_file_sha256', {})
        op = index_rows(safe(article/'transfer_original_operating.csv'), fit_key+('alpha',))
        fit_stats = index_rows(safe(article/'fit_metrics.csv'), fit_key)
        training = index_rows(safe(root/'result/training_records/training_fits.csv'), fit_key)
        for name, (role, base) in roots.items():
            passed = 0
            for fit in fits(role):
                fold, method, seed = fit
                file = safe(base/fold/f'{method}_s{seed}'/'holdout.json')
                try:
                    h = json.loads(file.read_bytes())
                    if training[fit]['suite'] != SUITE[role]:
                        raise ValueError('Training record has the wrong suite')
                    compare_holdout(h, {a: op[fit+(a,)] for a in ('0.01', '0.05')}, fit_stats[fit],
                                    trace[role+':state/protocol/'+fold+'.jsonl.gz'], training[fit]['best_pt_sha256'])
                    passed += 1
                except (OSError, ValueError, KeyError, TypeError) as e:
                    out['issues'].append(name+' '+str(fit)+': '+str(e))
            out['verified_holdout_fits'][name] = passed
    if verify:
        out['status'] = ('layout_mismatch' if out['issues'] else 'available_server_evidence_verified'
                         if roots else 'no_server_run_or_holdout_cache_to_verify')
    return out
