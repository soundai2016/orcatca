"""Source-only perturbation checks before training."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from .cache import FeatureStore
from .dataset import AudioDataset
from .transforms import transform
from .utils import output_path, read_json, read_rows, atomic_json, write_csv, digest
from .logging import event

def run(root, cfg):
    suite = root / 'runs' / cfg['name']
    from .references import protocol_directory
    directory = protocol_directory(root, cfg)
    proto = read_json(directory / 'complete.json')
    rows = read_rows(directory / proto['folds'][0]['manifest'])
    train = [r for r in rows if r['split'] == 'train']
    groups = defaultdict(list)
    for r in train:
        groups[r['site'], r['raw_class']].append(r)
    selected = []
    for group, items in sorted(groups.items()):
        selected.extend(sorted(items, key=lambda r: digest(['audit', r['id']]))[:16])
    selected = sorted(selected, key=lambda r: digest(['audit-order', r['id']]))[:512]
    records = read_json(directory / 'recordings.json')
    cache = read_json(suite / 'cache.json')['path'] if cfg['cache']['enabled'] else None
    ds = AudioDataset(root, selected, records, cfg, cache_path=cache)
    diagnostics = []
    settings = dict(cfg['augmentation'], probability=1.0)
    from .config import METHODS
    kinds = sorted({METHODS[m]['transform'] for m in cfg['experiment']['methods']})
    for start in range(0, len(ds), 16):
        batch = [ds[i] for i in range(start, min(start + 16, len(ds)))]
        x = torch.stack([v[0] for v in batch])
        sites = np.array([v[4] for v in batch])
        recs = np.array([v[5] for v in batch])
        for kind in kinds:
            out, info = transform(x, kind, settings, sites, recs, 17, 0, start)
            difference = (out - x).detach().numpy()
            temporal = (out[:, :, :, 1:] - out[:, :, :, :-1] - (x[:, :, :, 1:] - x[:, :, :, :-1])).abs().amax((1, 2, 3)).numpy() * 40.0
            for j, item in enumerate(batch):
                r = selected[start + j]
                diagnostics.append(dict(id=r['id'], site=r['site'], raw_class=r['raw_class'], transform=kind, feature_rms_db=float(np.sqrt(np.mean(difference[j] ** 2)) * 40), temporal_difference_error_db=float(temporal[j]), rms_match_gap_db=float(info['rms_match_gap_db'][j].mean()) if 'rms_match_gap_db' in info else None))
    out = output_path(root, f"runs/{cfg['name']}/audit")
    write_csv(out / 'spectral.csv', diagnostics)
    stats = {kind: dict(n=sum((d['transform'] == kind for d in diagnostics)), max_temporal_difference_error_db=max((d['temporal_difference_error_db'] for d in diagnostics if d['transform'] == kind))) for kind in kinds}
    atomic_json(out / 'summary.json', dict(samples=len(selected), statistics=stats, scope='Stratified source-training diagnostic, not a population estimate or waveform simulation', hypothesis='Source-only augmentation; confuser-tail utility requires controlled training comparisons', original_data_modified=False))
    event('spectral_audit_complete', samples=len(selected), statistics=stats)
