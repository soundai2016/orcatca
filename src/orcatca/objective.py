"""Recording-aware supervised risk and paired-view consistency."""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.nn import functional as F
SUBTYPE = {'AB': 0, 'HW': 1, 'KW': 2}

def training_weights(rows, strength=0.75, cap=10.0):
    """Mix natural risk with class/site/recording-balanced risk; mean weight is one."""
    if not rows or not 0 <= strength <= 1 or cap < 1:
        raise ValueError('Invalid balancing inputs')
    sites, recordings = (defaultdict(set), defaultdict(set))
    counts = Counter()
    for r in rows:
        label, site, rec = (r['raw_class'], r['site'], r['recording'])
        sites[label].add(site)
        recordings[label, site].add(rec)
        counts[label, site, rec] += 1
    present = set(sites)
    if 'KW' not in present or not present & {'AB', 'HW'}:
        raise ValueError('Training requires KW and supported negatives')
    negatives = present & {'AB', 'HW'}
    target = {'KW': 0.5, **{label: 0.5 / len(negatives) for label in negatives}}
    balanced = np.array([len(rows) * target[r['raw_class']] / (len(sites[r['raw_class']]) * len(recordings[r['raw_class'], r['site']]) * counts[r['raw_class'], r['site'], r['recording']]) for r in rows], dtype=np.float64)
    weights = 1 - strength + strength * balanced
    clipped = int((weights > cap).sum())
    weights = np.minimum(weights, cap)
    weights /= weights.mean()
    mass = {label: float(weights[[r['raw_class'] == label for r in rows]].sum() / weights.sum()) for label in sorted(present)}
    return (weights.astype(np.float32), dict(strength=strength, cap_before_normalization=cap, clipped_windows=clipped, effective_class_mass=mass, minimum=float(weights.min()), maximum=float(weights.max()), effective_sample_size=float(weights.sum() ** 2 / (weights ** 2).sum()), interpretation='Weighted empirical risk; no replacement sampling; limited DDP padding is reported'))

def js_divergence(a, b):
    pa, pb = (F.softmax(a.float(), -1), F.softmax(b.float(), -1))
    mean = (pa + pb) * 0.5
    return 0.5 * ((pa * (pa.clamp_min(1e-08).log() - mean.clamp_min(1e-08).log())).sum(-1) + (pb * (pb.clamp_min(1e-08).log() - mean.clamp_min(1e-08).log())).sum(-1))

def objective_mode(method):
    return method.get('objective', 'fine_aux' if method['subtype'] else 'binary')


def score_probability(logits, method):
    """Binary columns are [nonKW, KW]; subtype columns are [AB, HW, KW]."""
    if logits.ndim != 2 or logits.shape[1] != 5:
        raise ValueError('Expected two binary and three subtype logits')
    if objective_mode(method) == 'direct3':
        return F.softmax(logits[:, 2:].float(), dim=1)[:, SUBTYPE['KW']]
    return F.softmax(logits[:, :2].float(), dim=1)[:, 1]


def output_contract(method):
    return dict(objective=objective_mode(method), binary_order=['nonKW', 'KW'],
                subtype_order=['AB', 'HW', 'KW'],
                score_head='subtype_KW' if objective_mode(method) == 'direct3' else 'binary_KW')


def coarse_auxiliary(logits, subtype, smoothing):
    """Aggregate the original three-class smoothing mass before binary supervision."""
    logits = logits.float()
    q = (1.0 - smoothing) * F.one_hot(subtype, 3).to(logits.dtype) + smoothing / 3.0
    grouped_logits = torch.stack((torch.logsumexp(logits[:, :2], dim=1), logits[:, 2]), dim=1)
    grouped_target = torch.stack((q[:, :2].sum(dim=1), q[:, 2]), dim=1)
    return -(grouped_target * F.log_softmax(grouped_logits, dim=1)).sum(dim=1)


def loss_per_sample(original, augmented, target, subtype, config, method, epoch):
    smoothing = config['label_smoothing']
    # Keep the two existing recipes algebraically and operationally unchanged.
    supervised = 0.5 * (F.cross_entropy(original[:, :2].float(), target, reduction='none', label_smoothing=smoothing) + F.cross_entropy(augmented[:, :2].float(), target, reduction='none', label_smoothing=smoothing))
    fine = 0.5 * (F.cross_entropy(original[:, 2:].float(), subtype, reduction='none', label_smoothing=smoothing) + F.cross_entropy(augmented[:, 2:].float(), subtype, reduction='none', label_smoothing=smoothing))
    divergence = js_divergence(original[:, :2], augmented[:, :2])
    mode = objective_mode(method)
    if mode == 'coarse_aux':
        aux = 0.5 * (coarse_auxiliary(original[:, 2:], subtype, smoothing) + coarse_auxiliary(augmented[:, 2:], subtype, smoothing))
    else:
        aux = fine
    if mode == 'direct3':
        zero = (original[:, :2].float().sum(1) + augmented[:, :2].float().sum(1)) * 0.0
        return fine + zero, fine, fine.detach(), divergence.detach()
    aux_weight = config['subtype_weight'] if method['subtype'] else 0.0
    ramp = min(1.0, (epoch + 1) / max(1, config['consistency_warmup']))
    js_weight = config['consistency_weight'] * ramp if method['consistency'] else 0.0
    return supervised + aux_weight * aux + js_weight * divergence, supervised, aux, divergence
