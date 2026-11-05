"""Source-only spectral perturbations; time differences are preserved by shifts."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from .utils import digest

def stream(seed, epoch, step, purpose):
    return np.random.default_rng(int(digest([seed, epoch, step, purpose])[:16], 16))

def donors(sites, recordings, rng):
    sites, recordings = (np.asarray(sites), np.asarray(recordings))
    selected = []
    cross = []
    for i in range(len(sites)):
        choices = np.flatnonzero((sites != sites[i]) & (recordings != recordings[i]))
        if not len(choices):
            choices = np.flatnonzero(recordings != recordings[i])
        j = int(rng.choice(choices)) if len(choices) else i
        selected.append(j)
        cross.append(bool(sites[i] != sites[j]))
    return (np.array(selected, dtype=np.int64), np.array(cross))

def smooth_curve(curve, knots):
    small = F.interpolate(curve, size=min(knots, curve.shape[-1]), mode='linear', align_corners=True)
    return F.interpolate(small, size=curve.shape[-1], mode='linear', align_corners=True)

def transform(x, kind, cfg, sites, recordings, seed, epoch, step):
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError('Expected [B,1,F,T] features')
    rng = stream(seed, epoch, step, 'spectral')
    b, _, f, _ = x.shape
    active = rng.random(b) < cfg['probability']
    index, cross = donors(sites, recordings, rng)
    perm = torch.as_tensor(index, device=x.device)
    enabled = torch.as_tensor(active, device=x.device, dtype=x.dtype).view(b, 1, 1, 1)
    diagnostic = {'active': int(active.sum()), 'cross_site': int((cross & active).sum()), 'batch': b}
    if kind == 'none':
        return (x, {**diagnostic, 'active': 0, 'cross_site': 0})
    mu = x.float().mean(-1, keepdim=True).detach()
    alpha = torch.as_tensor(rng.beta(cfg['mixstyle_alpha'], cfg['mixstyle_alpha'], (b, 1, 1, 1)), device=x.device, dtype=torch.float32)
    if kind == 'mixstyle':
        std = (x.float().var(-1, unbiased=False, keepdim=True) + 1e-06).sqrt().detach()
        mixed = (x - mu) / std * (alpha * std + (1 - alpha) * std[perm]) + alpha * mu + (1 - alpha) * mu[perm]
        return (x + enabled * (mixed - x), diagnostic)
    bound = cfg['filter_db'] / 40.0
    if kind == 'filter':
        knots = rng.uniform(-bound, bound, (b, 1, cfg['knots'])).astype(np.float32)
        offset = F.interpolate(torch.as_tensor(knots, device=x.device), size=f, mode='linear', align_corners=True)
    else:
        if kind not in ('shift', 'random_matched'):
            raise ValueError('Unknown spectral transform')
        offset = smooth_curve((mu[perm] - mu).squeeze(-1), cfg['knots'])
        offset = (1 - alpha.squeeze(-1)) * offset.clamp(-bound, bound)
        if kind == 'random_matched':
            noise = torch.as_tensor(rng.normal(size=(b, 1, cfg['knots'])).astype(np.float32), device=x.device)
            noise = F.interpolate(noise, size=f, mode='linear', align_corners=True)
            scale = offset.square().mean(-1, keepdim=True).sqrt() / noise.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-08)
            random_offset = (noise * scale).clamp(-bound, bound)
            diagnostic['rms_match_gap_db'] = ((random_offset.square().mean(-1).sqrt() - offset.square().mean(-1).sqrt()).abs() * 40.0).detach()
            offset = random_offset
    offset = offset.unsqueeze(-1) * enabled
    diagnostic['offset_rms_db'] = (offset.square().mean((1, 2, 3)).sqrt() * 40.0).detach()
    return (x + offset.to(x.dtype), diagnostic)
