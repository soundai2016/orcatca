"""Bind an incremental suite without loading any earlier model weights."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml
from .config import validate
from .references import verify_reference, checked_input
from .utils import read_json, sha256, output_path, atomic_json, code_digest

METHODS = ['natural_coarse_mixstyle', 'natural_threeclass_mixstyle']
BASELINES = ['natural_mixstyle', 'natural_subtype_mixstyle']


def create(root: Path, source: str, pilot: bool = False, with_baselines: bool = False) -> Path:
    root = root.resolve()
    relative = Path(source)
    suite = (root / relative).resolve()
    raw = (root / 'dclde_2027_killer_whales').resolve()
    if relative.is_absolute() or '..' in relative.parts or not suite.is_relative_to(root) or suite.is_relative_to(raw):
        raise ValueError('Source suite must be project-relative, for example runs/<suite>')
    config = suite / 'config.yaml'
    if not config.is_file():
        raise FileNotFoundError('Missing source config.yaml')
    cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
    frozen = read_json(suite / 'config.json')
    if frozen.get('config') != cfg:
        raise ValueError('Frozen source configuration differs from config.yaml')
    for fold in range(cfg['experiment']['folds']):
        for method in BASELINES:
            for seed in cfg['experiment']['seeds']:
                path = suite / f'fold{fold:02d}/{method}_s{seed}'
                status = read_json(path / 'status.json')
                if status.get('status') != 'completed':
                    raise ValueError('Complete both source baselines before supplementary fitting')
                if not (path / 'calibration.jsonl.gz').is_file():
                    raise ValueError('Source baseline predictions are missing')
    # Resolve metadata only. No checkpoint is opened or reused.
    source_protocol = (cfg.get('input_reference') or {}).get('protocol_complete', 'state/protocol/complete.json')
    protocol = checked_input(root, source_protocol)
    ref = dict(source_suite=str(suite.relative_to(root)), source_config=str(config.relative_to(root)),
               source_config_sha256=sha256(config), protocol_complete=str(protocol.relative_to(root)),
               protocol_sha256=sha256(protocol), reuse_weights=False)
    if cfg['cache']['enabled']:
        pointer = read_json(suite / 'cache.json')
        path = Path(pointer['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Cache must remain inside this project')
        checked_input(root, str(path.relative_to(root)))
        if sha256(path) != pointer['summary_sha256']:
            raise ValueError('Cache pointer checksum differs')
        ref.update(cache_complete=str(path.relative_to(root)), cache_sha256=pointer['summary_sha256'])
    new = copy.deepcopy(cfg)
    new['input_reference'] = ref
    new['name'] = 'semantic_pilot' if pilot else 'semantic'
    new['experiment']['methods'] = (BASELINES if with_baselines else []) + METHODS
    if pilot:
        new['experiment']['seeds'] = cfg['experiment']['seeds'][:1]
    validate(new)
    verify_reference(root, new)
    from .control import fresh_name
    name = fresh_name(new['name'])
    new['name'] = name
    path = output_path(root, f'state/requests/{name}.yaml')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(yaml.safe_dump(new, sort_keys=False))
    atomic_json(path.with_suffix('.inputs.json'), dict(
        source_config_sha256=ref['source_config_sha256'], protocol_sha256=ref['protocol_sha256'],
        code_sha256=code_digest(root), random_initialization=True, weights_loaded=False,
        source_results_used_for_initialization=False,
        baseline_strategy='rerun' if with_baselines else 'existing results; bridge checks required',
        planned_fits=len(new['experiment']['methods'])*len(new['experiment']['seeds'])*new['experiment']['folds'],
        exploratory=True))
    return path
