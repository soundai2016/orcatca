"""Source-only operating rules with explicit negative-support limits."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .metrics import arrays, fpr_threshold


def negative_support(rows, settings):
    required = settings.get('required_negative_subtypes', ['AB', 'HW'])
    summary = {}
    for cls in required:
        group = [r for r in rows if r['label'] == 0 and r['raw_class'] == cls]
        summary[cls] = dict(windows=len(group), recordings=len({r['recording'] for r in group}),
                            blocks=len({r.get('block', r['recording']) for r in group}),
                            subtype_present=bool(group),
                            empirical_step=1/len(group) if group else None,
                            resolution_limited={f'{alpha:g}': bool(alpha*len(group)<1) for alpha in settings.get('fpr_targets',[.01,.05])})
    missing = [c for c, s in summary.items() if s['windows'] == 0]
    sparse = [c for c, s in summary.items() if s['windows'] and
              (s['recordings'] < settings.get('min_negative_recordings', 2)
               or s['blocks'] < settings.get('min_negative_blocks', 2))]
    return dict(status='supported_empirically' if not missing and not sparse else 'unsupported',
                by_subtype=summary, missing_subtypes=missing, sparse_subtypes=sparse,
                guarantee=False, interpretation='Observed source support only; not a target-domain risk certificate')


def weighted_fpr_threshold(scores, weights, target):
    scores, weights = np.asarray(scores, float), np.asarray(weights, float)
    if not len(scores) or weights.shape != scores.shape or not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError('Invalid weighted negative scores')
    if np.any((scores < 0) | (scores > 1)) or np.any(weights <= 0) or not 0 < target < 1:
        raise ValueError('Invalid score, weight or target')
    order = np.argsort(-scores, kind='stable')
    s, w = scores[order], weights[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    # Exclude the entire first tie group that would exceed the allowed mass.
    j = ends[np.flatnonzero(np.cumsum(w)[ends] > target*w.sum())[0]]
    return float(np.nextafter(s[j], np.inf))


def operating_rules(rows, probabilities, settings):
    _, p = arrays(rows, probabilities)
    support = negative_support(rows, settings)
    if support['status'] != 'supported_empirically':
        return {}, dict(support, rules_withheld=True)
    cells = defaultdict(list)
    for i, row in enumerate(rows):
        if row['label'] == 0:
            cells[row.get('group', row['site']), row['raw_class']].append(i)
    rules, diagnostics = {}, []
    for alpha in settings['fpr_targets']:
        rules[f'fpr_{alpha:g}'] = fpr_threshold(rows, p, alpha, True)
        window, blocked = [], []
        for (source, subtype), ids in sorted(cells.items()):
            counts = defaultdict(int)
            for i in ids: counts[rows[i].get('block', rows[i]['recording'])] += 1
            weights = np.array([1.0/counts[rows[i].get('block', rows[i]['recording'])] for i in ids])
            tw = weighted_fpr_threshold(p[ids], np.ones(len(ids)), alpha)
            tb = weighted_fpr_threshold(p[ids], weights, alpha)
            window.append(tw); blocked.append(tb)
            diagnostics.append(dict(target=alpha, source=source, subtype=subtype,
                                    windows=len(ids), blocks=len(counts), window_threshold=tw,
                                    block_weighted_threshold=tb,
                                    tail_window_allowance=int(np.floor(alpha*len(ids)))))
        rules[f'source_fpr_{alpha:g}'] = max(window)
        rules[f'block_fpr_{alpha:g}'] = max(blocked)
    return rules, dict(support, rules_withheld=False, cells=diagnostics,
                       block_estimand='Equal-block mean of negative-window FPR; not event FPR',
                       envelope_scope='Maximum source/subtype empirical threshold; no unseen-source guarantee')


def block_error_rates(rows, probabilities, threshold):
    y, p = arrays(rows, probabilities)
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r.get('block', r['recording'])].append(i)
    fp, tp = [], []
    for ids in groups.values():
        ids = np.asarray(ids);neg=ids[y[ids] == 0];pos=ids[y[ids] == 1]
        if len(neg): fp.append(float((p[neg] >= threshold).mean()))
        if len(pos): tp.append(float((p[pos] >= threshold).mean()))
    return dict(negative_blocks=len(fp), positive_blocks=len(tp),
                block_mean_fpr=float(np.mean(fp)) if fp else None,
                block_mean_recall=float(np.mean(tp)) if tp else None)
