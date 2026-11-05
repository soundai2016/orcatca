"""Explicit, read-only reuse of frozen input protocols and unaugmented caches."""
from __future__ import annotations
from pathlib import Path
from .utils import read_json, sha256, raw_path


def _metadata_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    rel = Path(relative)
    path = (root / rel).resolve()
    raw = (root / 'dclde_2027_killer_whales').resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root) or path.is_relative_to(raw):
        raise ValueError('Input reference must be a project-relative metadata path')
    if (root / rel).is_symlink():
        raise ValueError('Symbolic input reference: ' + relative)
    return path


def checked_input(root: Path, relative: str) -> Path:
    path = _metadata_path(root, relative)
    if not path.is_file():
        raise ValueError('Missing or symbolic input reference: ' + relative)
    return path


def protocol_directory(root: Path, cfg: dict, *, allow_relocated=False) -> Path:
    ref = cfg.get('input_reference')
    if not ref:
        return root / 'state/protocol'
    relative = ref['protocol_complete']
    path = _metadata_path(root, relative)
    if Path(relative).name != 'complete.json':
        raise ValueError('Expected a protocol completion record named complete.json')
    if allow_relocated and not path.exists():
        # A migrated run may have lost its old parent directory. Only the
        # original completion bytes (and their bound manifests) are portable.
        relative = 'state/protocol/complete.json'
    path = checked_input(root, relative)
    if sha256(path) != ref['protocol_sha256']:
        raise ValueError('Referenced protocol changed: ' + relative)
    return path.parent


def verify_reference(root: Path, cfg: dict, raw_identity: bool = True) -> dict:
    ref = cfg['input_reference']
    original = checked_input(root, ref['source_config'])
    if sha256(original) != ref['source_config_sha256']:
        raise ValueError('Source configuration changed after binding')
    import yaml
    old = yaml.safe_load(original.read_text(encoding='utf-8'))
    if old['data'] != cfg['data'] or old['cache'] != cfg['cache']:
        raise ValueError('Supplement must use the same data and cache representation')
    for key, value in old['experiment'].items():
        if key not in {'methods', 'seeds', 'folds'} and cfg['experiment'].get(key) != value:
            raise ValueError('Training recipe changed: ' + key)
    if cfg['runtime'] != old['runtime'] or cfg['augmentation'] != old['augmentation']:
        raise ValueError('Runtime or augmentation changed from the input suite')
    if cfg['evaluation'] != old['evaluation']:
        raise ValueError('Evaluation policy changed from the input suite')
    if cfg['experiment']['folds'] > old['experiment']['folds']:
        raise ValueError('Supplement requested undeclared folds')
    if set(cfg['experiment']['seeds']) - set(old['experiment']['seeds']):
        raise ValueError('Supplement requested undeclared seeds')
    directory = protocol_directory(root, cfg)
    info = read_json(directory / 'complete.json')
    for name, expected in info['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe protocol member')
        p = directory / rel
        if not p.is_file() or p.is_symlink() or sha256(p) != expected:
            raise ValueError('Referenced protocol file changed: ' + name)
    if raw_identity:
        for relative, item in read_json(directory / 'recordings.json').items():
            p = raw_path(root, relative)
            st = p.stat()
            if (st.st_size, st.st_mtime_ns) != (item['size'], item['mtime_ns']):
                raise ValueError('Referenced raw recording changed: ' + relative)
    return info


def reference_cache(root: Path, cfg: dict, verify_content: bool = True):
    if not cfg['cache']['enabled']:
        return None
    ref = cfg['input_reference']
    path = checked_input(root, ref['cache_complete'])
    if sha256(path) != ref['cache_sha256']:
        raise ValueError('Referenced cache summary changed')
    from .cache import verify
    info = verify(path, verify_content)
    protocol = read_json(protocol_directory(root, cfg) / 'complete.json')
    if info['protocol_identity'] != protocol['identity']:
        raise ValueError('Reference cache belongs to a different protocol')
    if info['dtype'] != cfg['cache']['dtype']:
        raise ValueError('Reference cache dtype changed')
    return info | {'path': str(path)}


def initialization_bridge(root: Path, cfg: dict, fold: int, seed: int, observed_hash: str) -> dict:
    ref=cfg.get('input_reference')
    if not ref:
        return dict(status='not_applicable',historical_weights_loaded=False)
    source=checked_input(root, f"{ref['source_suite']}/fold{fold:02d}/natural_mixstyle_s{seed}/initialization.json")
    value=read_json(source)
    matched=value.get('source')=='random' and value.get('loaded_checkpoint') is None and value.get('parameters_sha256')==observed_hash
    rerun={'natural_mixstyle','natural_subtype_mixstyle'} <= set(cfg['experiment']['methods'])
    result=dict(status='matched' if matched else 'different_rerun_references' if rerun else 'mismatch',
                source_record_sha256=sha256(source),initial_parameters_match=matched,
                historical_weights_loaded=False,baseline_rerun=rerun,
                limits='Initialization matching and CPU gradient checks do not certify identical NPU trajectories')
    if not matched and not rerun:
        raise ValueError('Initialization bridge differs; rerun reference methods with --with-baselines')
    return result
