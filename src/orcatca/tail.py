"""Recording-capped negative-tail regularization on source minibatches."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F
from .transforms import stream


def negative_tail(original, augmented, labels, subtypes, recordings, cfg, mode,
                  seed, epoch, step):
    """Return a rank-local surrogate, not a full-dataset pAUC estimator."""
    zero = (original[:, :2].sum() + augmented[:, :2].sum()) * 0.0
    info = dict(eligible_groups=0, negative_recordings=0, selected_recordings=0,
                sparse_groups=0, hw_groups=0, ab_groups=0, scope='rank_local_microbatch')
    if mode is None:
        return zero, info
    if mode not in {'hard', 'random', 'all'}:
        raise ValueError('Unknown tail policy')
    y = np.asarray(labels, dtype=np.int64)
    subtype = np.asarray(subtypes, dtype=np.int64)
    rec = np.asarray(recordings)
    if y.shape != (len(original),) or subtype.shape != y.shape or rec.shape != y.shape:
        raise ValueError('Tail metadata must align with logits')
    if np.any(~np.isin(y, [0, 1])) or np.any(~np.isin(subtype, [0, 1, 2])) or np.any((subtype == 2) != (y == 1)):
        raise ValueError('Binary labels and subtypes disagree')
    if original.shape != augmented.shape or original.shape[1] < 2:
        raise ValueError('Invalid paired logits')
    fraction = float(cfg.get('tail_fraction', .25))
    minimum = int(cfg.get('tail_min_recordings', 2))
    if not 0 < fraction <= 1 or minimum < 2:
        raise ValueError('Invalid tail fraction or support')
    # A negative example should have a small KW-vs-non-KW logit margin.
    losses = .5 * (F.softplus(original[:, 1].float() - original[:, 0].float())
                   + F.softplus(augmented[:, 1].float() - augmented[:, 0].float()))
    rng = stream(seed, epoch, step, 'negative_tail')
    groups = []
    for cls, name in ((0, 'ab_groups'), (1, 'hw_groups')):
        ids = np.flatnonzero((y == 0) & (subtype == cls))
        unique = np.unique(rec[ids])
        info['negative_recordings'] += len(unique)
        if len(unique) < minimum:
            info['sparse_groups'] += int(len(unique) > 0)
            continue
        values = torch.stack([losses[torch.as_tensor(ids[rec[ids] == r], device=losses.device)].mean()
                              for r in unique])
        count = max(1, int(math.ceil(fraction * len(unique))))
        if mode == 'all':
            selected = values
        elif mode == 'hard':
            selected = torch.topk(values, count, largest=True, sorted=False).values
        else:
            chosen = rng.choice(len(unique), count, replace=False)
            selected = values[torch.as_tensor(chosen, device=losses.device)]
        groups.append(selected.mean())
        info['eligible_groups'] += 1
        info['selected_recordings'] += len(selected)
        info[name] += 1
    if not groups:
        return zero, info
    ramp = min(1.0, (epoch + 1) / max(1, int(cfg.get('tail_warmup_epochs', 5))))
    return float(cfg.get('tail_weight', .1)) * ramp * torch.stack(groups).mean(), info
