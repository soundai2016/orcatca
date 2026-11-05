"""Read-only CPU waveform/STFT frontend; no peak normalization or channel averaging."""
from __future__ import annotations
from functools import lru_cache
from math import gcd
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from scipy.fft import rfft
import soundfile as sf
from .utils import raw_path

def load_wave(root, row, record, cfg):
    p = raw_path(Path(root), row['audio'])
    st = p.stat()
    if st.st_size != record['size'] or st.st_mtime_ns != record['mtime_ns']:
        raise RuntimeError(f"Raw recording changed: {row['audio']}")
    rate = record['rate']
    frames = round(cfg['seconds'] * rate)
    start = row['start_frame']
    if rate < cfg['sample_rate'] or record['channels'] != 1 or start < 0 or (start + frames > record['frames']):
        raise ValueError('Ineligible clip geometry; no upsampling, channel mixing or padding')
    with sf.SoundFile(str(p), 'r') as f:
        if (f.samplerate, f.channels, len(f)) != (rate, 1, record['frames']):
            raise ValueError('Decoder identity changed')
        f.seek(start)
        x = f.read(frames, dtype='float32', always_2d=True)[:, 0]
    if len(x) != frames or not np.isfinite(x).all():
        raise ValueError('Short or nonfinite waveform')
    if rate != cfg['sample_rate']:
        divisor = gcd(rate, cfg['sample_rate'])
        x = resample_poly(x, cfg['sample_rate'] // divisor, rate // divisor).astype(np.float32)
    target = round(cfg['seconds'] * cfg['sample_rate'])
    if len(x) < target:
        raise ValueError('Resampled clip too short')
    return np.ascontiguousarray(x[:target])

@lru_cache(maxsize=16)
def window(size):
    return np.hanning(size + 1)[:-1].astype(np.float32)

def feature(x, cfg):
    frames = np.lib.stride_tricks.sliding_window_view(x, cfg['n_fft'])[::cfg['hop']]
    w = window(cfg['n_fft'])
    z = rfft(frames * w, axis=1, workers=1)
    power = (z.real ** 2 + z.imag ** 2) / float(w.sum() ** 2)
    db = 10 * np.log10(np.maximum(power, 1e-12))
    hz = np.fft.rfftfreq(cfg['n_fft'], 1 / cfg['sample_rate'])
    keep = (hz >= cfg['min_hz']) & (hz <= cfg['max_hz'])
    return np.ascontiguousarray(np.clip(db[:, keep].T, -120.0, 40.0)[None], dtype=np.float32)
