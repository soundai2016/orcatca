"""Read-only audio features with an optional shared shard cache."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from .audio import load_wave, feature
from .cache import FeatureStore
from .objective import SUBTYPE
from .utils import raw_path

class AudioDataset(Dataset):

    def __init__(self, root, rows, records, cfg, weights=None, cache_path=None):
        self.root = root
        self.rows = rows
        self.records = records
        self.cfg = cfg
        self.weights = np.ones(len(rows), np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        self.cache = FeatureStore(cache_path) if cache_path else None
        self.sites = {s: i for i, s in enumerate(sorted({r['site'] for r in rows}))}
        self.recording_ids = {s: i for i, s in enumerate(sorted({r['recording'] for r in rows}))}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        record = self.records[row['audio']]
        path = raw_path(self.root, row['audio'])
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record['size'], record['mtime_ns']):
            raise RuntimeError('Raw recording changed: ' + row['audio'])
        try:
            x = self.cache.read(row['id']) if self.cache else None
            if x is None:
                x = feature(load_wave(self.root, row, record, self.cfg['data']), self.cfg['data'])
            x = np.ascontiguousarray((x + 60.0) / 40.0, dtype=np.float32)
            return (torch.from_numpy(x), row['label'], SUBTYPE[row['raw_class']], self.weights[index], self.sites[row['site']], self.recording_ids[row['recording']], index)
        except Exception as exc:
            raise RuntimeError(f"Failed sample {row['id']}: {exc}") from exc

def init_worker(worker_id):
    import os
    import sys
    if sys.platform == 'linux':
        import ctypes
        import signal
        parent = os.getppid()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot set worker parent-death signal')
        if parent == 1 or os.getppid() != parent:
            raise RuntimeError('DataLoader parent exited during worker startup')
    torch.set_num_threads(1)
    np.random.seed(torch.initial_seed() % 2 ** 32)
