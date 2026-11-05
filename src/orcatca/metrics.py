"""Exact threshold selection and fixed-rule evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

def arrays(rows, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray([r['label'] for r in rows], dtype=np.int64)
    if not len(y) or p.shape != y.shape or (not np.isfinite(p).all()) or np.any((p < 0) | (p > 1)) or (not set(y) <= {0, 1}):
        raise ValueError('Invalid binary predictions')
    return (y, p)

def calibrate(rows, probabilities):
    """Maximize empirical BA over all distinct decisions; keep tied scores together."""
    y, p = arrays(rows, probabilities)
    positive, negative = (int(y.sum()), int((1 - y).sum()))
    if not positive or not negative:
        raise ValueError('Calibration requires both classes')
    order = np.argsort(-p, kind='stable')
    score, label = (p[order], y[order])
    end = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    tp = np.r_[0, np.cumsum(label)[end]]
    fp = np.r_[0, np.cumsum(1 - label)[end]]
    thresholds = np.r_[np.nextafter(score[0], np.inf), score[end]]
    ba = 0.5 * (tp / positive + 1 - fp / negative)
    candidates = np.flatnonzero(np.isclose(ba, ba.max(), rtol=0, atol=1e-12))
    best = min(candidates, key=lambda i: (abs(thresholds[i] - 0.5), thresholds[i]))
    return float(thresholds[best])

def fpr_threshold(rows, probabilities, target, by_subtype=True):
    """Smallest empirical threshold meeting each available negative subtype target."""
    y, p = arrays(rows, probabilities)
    if not 0 < target < 1 or not np.any(y == 0):
        raise ValueError('FPR calibration needs negative samples and target in (0, 1)')
    groups = sorted({r['raw_class'] for r in rows if r['label'] == 0}) if by_subtype else ['all']
    thresholds = []
    for group in groups:
        keep = np.array([r['label'] == 0 and (group == 'all' or r['raw_class'] == group) for r in rows])
        scores = np.sort(p[keep])[::-1]
        allowance = int(np.floor(target * len(scores)))
        thresholds.append(float(np.nextafter(scores[allowance], np.inf)))
    return max(thresholds)

def evaluate(rows, probabilities, threshold=0.5, max_fpr=0.1):
    y, p = arrays(rows, probabilities)
    pred = p >= threshold
    pos, neg = (y == 1, y == 0)
    tp, tn = (int((pred & pos).sum()), int((~pred & neg).sum()))
    fp, fn = (int((pred & neg).sum()), int((~pred & pos).sum()))
    dual = bool(pos.any() and neg.any())
    tpr = tp / pos.sum() if pos.any() else None
    tnr = tn / neg.sum() if neg.any() else None
    q = np.clip(p, 1e-07, 1 - 1e-07)
    nll = -(y * np.log(q) + (1 - y) * np.log1p(-q))
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        idx = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if idx.any():
            ece += idx.mean() * abs(float(p[idx].mean() - y[idx].mean()))
    classes = {}
    for label in sorted({r['raw_class'] for r in rows}):
        keep = np.array([r['raw_class'] == label for r in rows])
        classes[label] = dict(n=int(keep.sum()), predicted_kw=float(pred[keep].mean()), recordings=len({r['recording'] for r in rows if r['raw_class'] == label}))
    return dict(n=len(y), threshold=float(threshold), tp=tp, tn=tn, fp=fp, fn=fn, recall_kw=float(tpr) if tpr is not None else None, specificity=float(tnr) if tnr is not None else None, false_positive_rate=float(1 - tnr) if tnr is not None else None, precision_kw=tp / (tp + fp) if tp + fp else None, balanced_accuracy=float((tpr + tnr) / 2) if dual else None, macro_f1=float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)), auc=float(roc_auc_score(y, p)) if dual else None, pauc=float(roc_auc_score(y, p, max_fpr=max_fpr)) if dual else None, pauc_max_fpr=max_fpr, ap_kw=float(average_precision_score(y, p)) if pos.any() else None, ap_nonkw=float(average_precision_score(1 - y, 1 - p)) if neg.any() else None, nll=float(nll.mean()), balanced_nll=float((nll[pos].mean() + nll[neg].mean()) / 2) if dual else None, brier=float(((p - y) ** 2).mean()), ece=float(ece), by_class=classes, independent_recordings=len({r['recording'] for r in rows}), temporal_blocks=len({r.get('block', r['recording']) for r in rows}))

def selection_score(rows, probabilities, metric='macro_auc', max_fpr=0.1):
    _, p = arrays(rows, probabilities)
    values = []
    for site in sorted({r['site'] for r in rows}):
        index = np.array([r['site'] == site for r in rows])
        y = np.array([r['label'] for r in rows])[index]
        if len(np.unique(y)) == 2:
            values.append(roc_auc_score(y, p[index], max_fpr=max_fpr if metric == 'macro_pauc' else None))
    if not values:
        raise ValueError('No dual-class selection site')
    return float(np.mean(values))

def report(rows, probabilities, threshold=0.5, max_fpr=0.1):
    p = np.asarray(probabilities)
    sites = {}
    for site in sorted({r['site'] for r in rows}):
        ids = [i for i, r in enumerate(rows) if r['site'] == site]
        sites[site] = evaluate([rows[i] for i in ids], p[ids], threshold, max_fpr)
    valid = [m for m in sites.values() if m['auc'] is not None]
    confusers = {}
    for subtype in ('HW', 'AB'):
        ids = [i for i, r in enumerate(rows) if r['label'] == 1 or r['raw_class'] == subtype]
        if ids:
            confusers[subtype] = evaluate([rows[i] for i in ids], p[ids], threshold, max_fpr)
    negative_sites = [m['false_positive_rate'] for m in sites.values() if m['false_positive_rate'] is not None]
    return dict(confuser_discrimination=confusers,
                worst_site_fpr=max(negative_sites) if negative_sites else None,
                overall=evaluate(rows, p, threshold, max_fpr), sites=sites, macro_auc=float(np.mean([m['auc'] for m in valid])) if valid else None, macro_pauc=float(np.mean([m['pauc'] for m in valid])) if valid else None, excluded_single_class_sites=[s for s, m in sites.items() if m['auc'] is None])
