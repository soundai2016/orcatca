"""Resumable, bounded shards of unaugmented spectra."""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import shutil
import numpy as np
from .audio import load_wave, feature
from .logging import event
from .utils import atomic_json, digest, output_path, read_json, read_rows, sha256, raw_path

def build(root, cfg, protocol):
    policy = cfg['cache']
    if not policy['enabled']:
        return None
    rows = {}
    directory = root / 'state/protocol'
    for fold in protocol['folds'][:cfg['experiment']['folds']]:
        for row in read_rows(directory / fold['manifest']):
            if row['split'] != 'holdout':
                rows[row['id']] = row
    ordered = sorted(rows.values(), key=lambda r: r['id'])
    records = read_json(directory / 'recordings.json')
    identity = digest([cfg['data'], policy['dtype'], policy['shard_size'], protocol['identity'], [r['id'] for r in ordered]])
    out = output_path(root, 'state/cache/' + identity[:20])
    out.mkdir(parents=True, exist_ok=True)
    complete = out / 'complete.json'
    if complete.exists():
        summary = verify(complete, policy['verify_sha256'])
        event('cache_reused', windows=len(ordered), path=str(out), dtype=policy['dtype'])
        return summary | {'path': str(complete)}
    if not ordered:
        raise ValueError('No source windows for cache')
    probe = feature(load_wave(root, ordered[0], records[ordered[0]['audio']], cfg['data']), cfg['data'])
    shape = probe.shape
    dtype = np.dtype(policy['dtype'])
    expected = len(ordered) * math.prod(shape) * dtype.itemsize
    if expected > policy['budget_gib'] * 1024 ** 3:
        raise RuntimeError('Feature cache exceeds configured budget; reduce dtype/size or disable cache')
    shards = []
    lookup = {}
    max_error = 0.0

    def convert(row):
        x = feature(load_wave(root, row, records[row['audio']], cfg['data']), cfg['data'])
        value = x.astype(dtype)
        return (value, float(np.max(np.abs(x - value.astype(np.float32)))))
    for number, start in enumerate(range(0, len(ordered), policy['shard_size'])):
        batch = ordered[start:start + policy['shard_size']]
        path = out / f'shard_{number:05d}.npy'
        info_path = path.with_suffix('.json')
        batch_hash = digest([r['id'] for r in batch])
        reused = False
        if path.is_file() and info_path.is_file():
            info = read_json(info_path)
            reused = info.get('sample_hash') == batch_hash and info.get('sha256') == sha256(path)
        if not reused:
            needed = len(batch) * math.prod(shape) * dtype.itemsize + 4096
            if shutil.disk_usage(out).free - needed < policy['reserve_gib'] * 1024 ** 3:
                raise RuntimeError('Insufficient free space for next cache shard; no raw data are deleted')
            tmp = path.with_suffix('.tmp.npy')
            mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=dtype, shape=(len(batch), *shape))
            error = 0.0
            try:
                with ThreadPoolExecutor(max_workers=cfg['data']['decode_workers']) as pool:
                    for offset, (array, difference) in enumerate(pool.map(convert, batch)):
                        mm[offset] = array
                        error = max(error, difference)
                mm.flush()
                del mm
                mm = None
                os.replace(tmp, path)
            except BaseException:
                if mm is not None:
                    del mm
                raise
            stat = path.stat()
            info = dict(file=path.name, sha256=sha256(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, samples=len(batch), sample_hash=batch_hash, max_quantization_error_db=error)
            atomic_json(info_path, info)
        max_error = max(max_error, info['max_quantization_error_db'])
        shards.append(info)
        for offset, row in enumerate(batch):
            lookup[row['id']] = [number, offset]
        event('cache_progress', shard=number + 1, windows=min(start + len(batch), len(ordered)), total=len(ordered), reused=reused)
    index = out / 'index.json'
    atomic_json(index, lookup)
    summary = dict(format='orcatca230-spectra', identity=identity, protocol_identity=protocol['identity'], shape=list(shape), dtype=policy['dtype'], windows=len(ordered), shards=shards, index_sha256=sha256(index), max_quantization_error_db=max_error, scope='Unaugmented fixed frontend for source roles; no fitted corpus statistics')
    atomic_json(complete, summary)
    return summary | {'path': str(complete)}

def verify(path, hash_content=True):
    path = Path(path)
    summary = read_json(path)
    if summary['format'] not in {'orcatca230-spectra', 'orcatca220-spectra'}:
        raise ValueError('Foreign feature cache')
    if sha256(path.parent / 'index.json') != summary['index_sha256']:
        raise ValueError('Cache index changed')
    for item in summary['shards']:
        shard = path.parent / item['file']
        stat = shard.stat()
        if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
            raise ValueError('Cache shard identity changed: ' + str(shard))
        if hash_content and sha256(shard) != item['sha256']:
            raise ValueError('Cache checksum mismatch')
    return summary

class FeatureStore:

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = verify(self.path, False)
        self.index = read_json(self.path.parent / 'index.json')
        self.opened = OrderedDict()

    def __getstate__(self):
        return dict(path=self.path, metadata=self.metadata, index=self.index, opened=OrderedDict())

    def read(self, key):
        if key not in self.index:
            return None
        shard, offset = self.index[key]
        info = self.metadata['shards'][shard]
        path = self.path.parent / info['file']
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (info['size'], info['mtime_ns']):
            raise RuntimeError('Feature cache changed during training')
        if shard not in self.opened:
            self.opened[shard] = np.load(path, mmap_mode='r', allow_pickle=False)
            while len(self.opened) > 8:
                self.opened.popitem(last=False)
        self.opened.move_to_end(shard)
        value = np.array(self.opened[shard][offset], dtype=np.float32, copy=True)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite cached features')
        return value
