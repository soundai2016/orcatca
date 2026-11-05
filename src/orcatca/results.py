"""Fixed-rule evaluation, paired uncertainty and compact experiment reports."""
from __future__ import annotations
from collections import defaultdict
import gzip
import json
from pathlib import Path
import zipfile
import numpy as np
from .utils import read_json, read_rows, output_path, atomic_json, write_csv, sha256, code_digest, digest
from .metrics import report, evaluate, calibrate, fpr_threshold
from .calibration import operating_rules, block_error_rates

def summarize(root, name):
    suite = output_path(root, f'runs/{name}')
    entries = []
    for path in sorted(suite.glob('fold*/*/calibration.json')):
        value = read_json(path)
        m = value['calibration_metrics']['overall']
        row = dict(fold=path.parent.parent.name, run=path.parent.name, selected_epoch=value['checkpoint_epoch'], selection_metric=value['selection_metric'], selection_score=value['selection_score'], calibration_auc=m['auc'], calibration_pauc=m['pauc'], calibration_ba=m['balanced_accuracy'], calibration_hw_fpr=m['by_class'].get('HW', {}).get('predicted_kw'), threshold=value['threshold'], n=m['n'], recordings=m['independent_recordings'])
        row['calibration_macro_auc'] = value['calibration_metrics']['macro_auc']
        row['calibration_macro_pauc'] = value['calibration_metrics']['macro_pauc']
        held = path.parent / 'holdout.json'
        if held.exists():
            h = read_json(held)
            o = h['metrics']['overall']
            row.update(holdout_auc=o['auc'], holdout_pauc=o['pauc'], holdout_ba=o['balanced_accuracy'], holdout_hw_fpr=o['by_class'].get('HW', {}).get('predicted_kw'), holdout_recall=o['recall_kw'])
            row['holdout_macro_auc'] = h['metrics']['macro_auc']
            row['holdout_macro_pauc'] = h['metrics']['macro_pauc']
            for key, point in h['operating'].items():
                row['holdout_' + key + '_recall'] = point['overall']['recall_kw']
                row['holdout_' + key + '_actual_fpr'] = point['overall']['false_positive_rate']
        entries.append(row)
    write_csv(suite / 'summary.csv', entries)
    grouped = defaultdict(list)
    for row in entries:
        grouped[row['fold'], row['run'].rsplit('_s', 1)[0]].append(row)
    means = []
    for (fold, method), group in sorted(grouped.items()):
        item = dict(fold=fold, method=method, seed_runs=len(group))
        for key in ('calibration_auc', 'calibration_ba', 'holdout_auc', 'holdout_pauc', 'holdout_macro_auc', 'holdout_macro_pauc', 'holdout_ba', 'holdout_hw_fpr'):
            values = [r[key] for r in group if r.get(key) is not None]
            if values:
                item[key + '_mean'] = float(np.mean(values))
                item[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        means.append(item)
    write_csv(suite / 'seed_summary.csv', means)
    atomic_json(suite / 'summary.json', dict(runs=entries, by_fold_method=means, interpretation='Seeds are optimization replicates, not independent deployments. Source calibration is not test performance.'))
    return entries

def check_suite(root, name):
    from .config import load
    suite = output_path(root, f'runs/{name}')
    cfg = load(suite / 'config.yaml')
    frozen = read_json(suite / 'config.json')
    if digest([cfg, code_digest(root)]) != frozen['identity']:
        raise ValueError('Frozen code/config changed')
    return (suite, cfg)

def holdout(root, name, fold, method, seed, confirmed=False):
    if not confirmed:
        raise ValueError('Explicit --confirm-holdout is required')
    from .config import METHODS
    from .objective import output_contract
    from .runtime import Runtime
    from .dataset import AudioDataset
    from .model import AcousticNet
    from .train import predict, write_predictions, protocol_inputs
    from .logging import configure, event, close
    import torch
    suite, cfg = check_suite(root, name)
    if method not in cfg['experiment']['methods'] or seed not in cfg['experiment']['seeds']:
        raise ValueError('Undeclared evaluation')
    out = suite / f'fold{fold:02d}' / f'{method}_s{seed}'
    if read_json(out / 'status.json')['status'] != 'completed':
        raise ValueError('Run is incomplete')
    cal = read_json(out / 'calibration.json')
    best = out / 'best.pt'
    if sha256(best) != cal['best_sha256']:
        raise ValueError('Model changed after calibration')
    if (out / 'holdout.json').exists():
        previous = read_json(out / 'holdout.json')
        if previous['best_sha256'] != cal['best_sha256']:
            raise ValueError('Existing holdout has a different model')
        return previous
    proto, manifest, all_rows, records, cache_path, _ = protocol_inputs(root, cfg, fold)
    rows = [r for r in all_rows if r['split'] == 'holdout']
    configure(out / 'logs', 'holdout')
    runtime = dict(cfg['runtime'], nproc=1)
    local_cfg = dict(cfg, runtime=runtime)
    rt = Runtime(runtime)
    try:
        saved = torch.load(best, map_location='cpu', weights_only=True)
        if saved['format'] != 'orcatca230' or saved['identity'] != cal['identity'] or saved.get('output_contract') != output_contract(METHODS[method]):
            raise ValueError('Foreign checkpoint')
        model = AcousticNet(saved['width'], saved['dropout']).to(rt.device)
        model.load_state_dict(saved['model'])
        p = predict(model, AudioDataset(root, rows, records, local_cfg, cache_path=cache_path), local_cfg, rt, out, 'holdout', method=METHODS[method])
        m = report(rows, p, cal['threshold'], cfg['experiment']['selection_max_fpr'])
        operating = {k: report(rows, p, t, cfg['experiment']['selection_max_fpr']) for k, t in cal['operating_thresholds'].items()}
        result = dict(metrics=m, operating=operating, operating_support=cal.get('operating_support'), block_errors={k: block_error_rates(rows, p, t) for k, t in cal['operating_thresholds'].items()}, best_sha256=cal['best_sha256'], manifest_sha256=sha256(manifest), threshold_source='source calibration only', scope='Outer contributor-held-out development evaluation; corpus previously used for research')
        write_predictions(out / 'holdout.jsonl.gz', rows, p)
        atomic_json(out / 'holdout.json', result)
        event('holdout_evaluated', fold=fold, method=method, seed=seed, metrics=m['overall'])
        return result
    finally:
        rt.close()
        close()

def aligned_predictions(paths):
    values = [read_rows(p) for p in paths]
    first = values[0]
    ids = [r['id'] for r in first]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction ids')
    matrix = []
    for rows in values:
        lookup = {r['id']: r for r in rows}
        if len(lookup) != len(rows) or set(lookup) != set(ids):
            raise ValueError('Prediction sets differ')
        for row in first:
            match = lookup[row['id']]
            if any((row.get(k) != match.get(k) for k in ('label', 'raw_class', 'site', 'recording', 'block'))):
                raise ValueError('Prediction metadata differ')
        matrix.append([lookup[key]['p'] for key in ids])
    return (first, np.asarray(matrix, dtype=np.float64))

def ensemble(root, name, fold, method):
    suite, cfg = check_suite(root, name)
    seeds = cfg['experiment']['seeds']
    if len(seeds) < 2:
        return None
    runs = [suite / f'fold{fold:02d}' / f'{method}_s{seed}' for seed in seeds]
    if not all(((r / 'calibration.json').exists() for r in runs)):
        raise ValueError('Ensemble requires every predeclared seed')
    rows, matrix = aligned_predictions([r / 'calibration.jsonl.gz' for r in runs])
    p = matrix.mean(0)
    threshold = calibrate(rows, p)
    operating, support_report = operating_rules(rows, p, cfg['evaluation'])
    result = dict(operating_support=support_report, seeds=seeds, aggregation='arithmetic mean of all predeclared seed probabilities', threshold=threshold, calibration=report(rows, p, threshold, cfg['experiment']['selection_max_fpr']), holdout=None)
    if all(((r / 'holdout.json').exists() for r in runs)):
        held, hp = aligned_predictions([r / 'holdout.jsonl.gz' for r in runs])
        hp = hp.mean(0)
        result['holdout'] = report(held, hp, threshold, cfg['experiment']['selection_max_fpr'])
        result['operating'] = {k: report(held, hp, t, cfg['experiment']['selection_max_fpr']) for k, t in operating.items()}
    atomic_json(suite / f'fold{fold:02d}' / f'ensemble_{method}.json', result)
    return result

def evaluate_all(root, name, confirmed=False):
    if not confirmed:
        raise ValueError('Explicit evaluation confirmation is required')
    suite, cfg = check_suite(root, name)
    if not all(((suite / f'fold{f:02d}' / f'{m}_s{s}' / 'calibration.json').exists() for f in range(cfg['experiment']['folds']) for m in cfg['experiment']['methods'] for s in cfg['experiment']['seeds'])):
        raise ValueError('Finish every frozen fit before outer evaluation')
    for fold in range(cfg['experiment']['folds']):
        for method in cfg['experiment']['methods']:
            for seed in cfg['experiment']['seeds']:
                holdout(root, name, fold, method, seed, True)
            ensemble(root, name, fold, method)
    return summarize(root, name)

def compare(root, a, b, repeats=2000, role='holdout', unit='block', metric='macro_pauc', max_fpr=0.1):
    from sklearn.metrics import roc_auc_score
    if role not in ('holdout', 'calibration', 'select_best') or unit not in ('block', 'recording'):
        raise ValueError('Invalid comparison role/unit')
    if metric not in ('auc', 'macro_auc', 'macro_pauc') or repeats < 10 or not 0 < max_fpr <= 1:
        raise ValueError('Invalid comparison statistic')
    paths = [output_path(root, r) / (role + '.jsonl.gz') for r in (a, b)]
    rows, probs = aligned_predictions(paths)
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row.get(unit, row['recording'])].append(i)
    strata = defaultdict(list)
    for indices in clusters.values():
        signature = tuple(sorted({rows[i]['site'] for i in indices}))
        strata[signature].append(np.asarray(indices))
    y = np.array([r['label'] for r in rows])
    sites = np.array([r['site'] for r in rows])

    fixed_sites = [s for s in np.unique(sites) if len(np.unique(y[sites == s])) == 2]

    def statistic(ids, scores):
        if metric == 'auc':
            return roc_auc_score(y[ids], scores[ids]) if len(np.unique(y[ids])) == 2 else None
        values = []
        for site in fixed_sites:
            chosen = ids[sites[ids] == site]
            if len(np.unique(y[chosen])) != 2:
                return None
            if len(np.unique(y[chosen])) == 2:
                values.append(roc_auc_score(y[chosen], scores[chosen],
                              max_fpr=max_fpr if metric == 'macro_pauc' else None))
        return float(np.mean(values)) if values else None

    rng = np.random.default_rng(2021)
    full = np.arange(len(rows))
    original = [statistic(full, p) for p in probs]
    if None in original:
        raise ValueError('No supported dual-class comparison')
    values = []
    attempts = 0
    while len(values) < repeats and attempts < repeats * 5:
        attempts += 1
        ids = np.concatenate([group[j] for group in strata.values()
                              for j in rng.integers(0, len(group), len(group))])
        pair = [statistic(ids, p) for p in probs]
        if None not in pair:
            values.append(pair[1] - pair[0])
    if not values:
        raise ValueError('No valid bootstrap replicate')
    return dict(role=role, metric=metric, max_fpr=max_fpr, unit=unit,
                clusters=len(clusters), source_strata=len(strata), valid_repeats=len(values),
                attempted_repeats=attempts, rejected_repeats=attempts-len(values),
                fixed_dual_class_sites=fixed_sites,
                difference=float(original[1] - original[0]), values=original,
                interval=np.quantile(values, [0.025, 0.975]).tolist(),
                prediction_hashes=[sha256(p) for p in paths],
                scope='Conditional paired site-stratified cluster interval; not seed/domain variance or selection correction')

def pack(root, name):
    from .references import protocol_directory
    import yaml

    suite = output_path(root, f'runs/{name}')
    pack_cfg = yaml.safe_load((suite / 'config.yaml').read_text(encoding='utf-8'))
    # Resolve the suite's frozen protocol before opening an existing report ZIP.
    # The completion record and manifests must describe the same protocol.
    protocol_dir = protocol_directory(root, pack_cfg, allow_relocated=True)
    summarize(root, name)
    path = output_path(root, f'state/{name}_report.zip')
    checksums = {}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(suite.rglob('*')):
            relative = file.relative_to(suite)
            if not file.is_file() or file.is_symlink() or 'logs' in relative.parts:
                continue
            if file.suffix in ('.json', '.csv', '.yaml') or file.name in ('select_best.jsonl.gz', 'calibration.jsonl.gz', 'holdout.jsonl.gz'):
                name_ = 'runs/' + str(relative)
                archive.write(file, name_)
                checksums[name_] = sha256(file)
        for name_ in ('inventory.json', 'status.json', 'environment.rank0.json', 'smoke.rank0.json'):
            file = root / 'state' / name_
            if file.is_file():
                archive.write(file, 'state/' + name_)
                checksums['state/' + name_] = sha256(file)
        complete = protocol_dir / 'complete.json'
        if complete.is_file():
            archive.write(complete, 'state/protocol/complete.json')
            checksums['state/protocol/complete.json'] = sha256(complete)
        for file in sorted(protocol_dir.glob('*')):
            if file.is_file() and not file.is_symlink() and file.name != 'complete.json' and (file.name.endswith('.jsonl.gz') or file.suffix in {'.json', '.csv'}):
                member = 'state/protocol/' + file.name
                archive.write(file, member)
                checksums[member] = sha256(file)
        bootstrap = root / 'state/bootstrap.log'
        if bootstrap.is_file():
            with bootstrap.open('rb') as stream:
                stream.seek(max(0, bootstrap.stat().st_size - 1024 ** 2))
                archive.writestr('state/bootstrap_tail.log', stream.read())
        archive.writestr('REPORT_INDEX.json', json.dumps(dict(included_files=len(checksums), size_limit_skips=[], models_included=False, raw_audio_included=False)))
        archive.writestr('REPORT_SHA256.json', json.dumps(checksums, sort_keys=True, indent=2))
    return path
