"""Conditional block uncertainty for the fixed-threshold composition contrast.

The sampling units are connected components of recordings and recorded blocks.
One source draw is reused in every overlapping check partition; one target draw
is reused for every method, seed, and threshold. Models, partitions, and cutoffs
are fixed. This is not an uncertainty estimate for retraining, threshold fitting,
new deployments, or errors absent from the empirical sample.
"""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

LABELS = ('AB', 'HW', 'KW')
SCOPE = ('Conditional on frozen models, fixed source-only partitions and their '
         'selected thresholds; shared linked-block draws for source checks and '
         'target observations; no threshold-selection or new-source uncertainty')


def linked_units(rows):
    """Link a recording to all its blocks, including cross-site components."""
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        join('r:' + str(row['recording']),
             'b:' + str(row.get('block') or row['recording']))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find('r:' + str(row['recording']))].append(i)
    return [(key, np.asarray(ids, dtype=np.int64))
            for key, ids in sorted(groups.items())]


def _validate(rows, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if not rows or p.shape[-1] != len(rows) or not np.isfinite(p).all():
        raise ValueError('Empty, nonfinite, or misaligned predictions')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Prediction scores must lie in [0, 1]')
    for row in rows:
        if row['raw_class'] not in LABELS or row['label'] != int(row['raw_class'] == 'KW'):
            raise ValueError('AB/HW/KW must form an exclusive partition')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample identifiers')
    return p


def _layout(rows):
    groups = linked_units(rows)
    index = np.empty(len(rows), dtype=np.int64)
    strata = defaultdict(list)
    for i, (_, ids) in enumerate(groups):
        index[ids] = i
        strata[tuple(sorted({str(rows[j]['site']) for j in ids}))].append(i)
    labels = np.asarray([LABELS.index(r['raw_class']) for r in rows])
    counts = np.stack([np.bincount(index[labels == c], minlength=len(groups))
                       for c in range(3)], axis=-1).astype(float)
    return groups, index, labels, dict(strata), counts


def label_support(rows):
    groups = linked_units(rows)
    return {label: dict(windows=sum(r['raw_class'] == label for r in rows),
                       recordings=len({r['recording'] for r in rows
                                       if r['raw_class'] == label}),
                       units=sum(any(rows[i]['raw_class'] == label for i in ids)
                                 for _, ids in groups)) for label in LABELS}


def threshold_diagnostics(rows, probabilities, alpha, selected_threshold=None):
    """Exact inclusive-score rule; the first disallowed tie group is excluded.

    For n negatives and allowance floor(alpha*n), let b be descending score
    number allowance+1. The class cutoff is nextafter(b, +inf), so an entire
    boundary tie is excluded. The maximum of the AB and HW cutoffs is used.
    """
    p = _validate(rows, probabilities)
    if p.ndim != 1 or not 0 < alpha < 1:
        raise ValueError('Expected one score vector and alpha in (0, 1)')
    support = label_support(rows)
    diagnostics, cutoffs = [], []
    for label in LABELS[:2]:
        mask = np.asarray([r['raw_class'] == label for r in rows])
        scores = np.sort(p[mask])[::-1]
        if not len(scores):
            raise ValueError('Missing required negative label: ' + label)
        allowance = math.floor(alpha * len(scores))
        boundary = float(scores[allowance])
        cutoff = float(np.nextafter(boundary, np.inf))
        cutoffs.append(cutoff)
        diagnostics.append(dict(subtype=label, **support[label],
            allowed_errors=allowance, boundary_score=boundary,
            boundary_ties=int((scores == boundary).sum()), label_cutoff=cutoff,
            errors_at_label_cutoff=int((scores >= cutoff).sum()),
            empirical_step=1 / len(scores), zero_allowed_errors=allowance == 0))
    recomputed = max(cutoffs)
    selected = recomputed if selected_threshold is None else float(selected_threshold)
    if not np.isfinite(selected):
        raise ValueError('Threshold must be finite')
    for row in diagnostics:
        mask = np.asarray([r['raw_class'] == row['subtype'] for r in rows])
        errors = int((p[mask] >= selected).sum())
        row.update(selected_threshold=selected, recomputed_threshold=recomputed,
                   saved_matches_recomputed=selected == recomputed,
                   binding_label=row['label_cutoff'] == recomputed,
                   selected_errors=errors, selected_fpr=errors / row['windows'],
                   empirical_target_met=errors <= row['allowed_errors'],
                   score_ties_positive=True, threshold_above_one=selected > 1,
                   independent_risk_certificate=False)
    return selected, diagnostics


def operating_rates(rows, probabilities, threshold):
    p = _validate(rows, probabilities)
    if p.ndim != 1:
        raise ValueError('Expected one score vector')
    prediction = p >= threshold
    label = np.asarray([r['raw_class'] for r in rows])
    value = {}
    for cls in LABELS:
        mask = label == cls
        n, errors = int(mask.sum()), int(prediction[mask].sum())
        value.update({cls + '_n': n, cls + '_fp': errors,
                      cls + '_fpr': errors / n if n else None})
    negative = value['AB_n'] + value['HW_n']
    fp = value['AB_fp'] + value['HW_fp']
    value.update(negative=negative, positive=value['KW_n'], fp=fp,
                 tp=value['KW_fp'], fpr=fp / negative if negative else None,
                 recall=value['KW_fpr'])
    return value


def safe_interval(samples, point, boundary=False):
    """Do not present a boundary or degenerate bootstrap as a risk certificate."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if boundary:
        return None, None, 'boundary_empirical_distribution'
    if point is None or not np.isfinite(point) or len(values) < 10:
        return None, None, 'insufficient_valid_draws'
    if np.ptp(values) <= 1e-14:
        return None, None, 'degenerate_empirical_distribution'
    low, high = np.quantile(values, [.025, .975])
    if high - low <= 1e-14:
        return None, None, 'degenerate_percentile_bounds'
    return float(low), float(high), 'conditional_percentile'


def full_calibration_outputs(fold, cal, cp, target, tp, methods, seeds,
                             alpha, saved_thresholds, target_draws=None):
    """Keep the saved full-pool operating points separate from cross-fitting."""
    raw, summaries, labels, diagnostics = [], [], [], []
    support = {'source_calibration': label_support(cal), 'target': label_support(target)}
    for m, method in enumerate(methods):
        member = []
        for s, seed in enumerate(seeds):
            threshold = saved_thresholds[m][s]
            meta = dict(fold=fold, method=method, seed=seed, alpha=alpha,
                        procedure='full_calibration')
            if threshold is None:
                raw.append(dict(**meta, status='not_estimable', reason='saved_rule_missing'))
                continue
            _, detail = threshold_diagnostics(cal, cp[m, s], alpha, threshold)
            diagnostics.extend(dict(**meta, repeat=None, side=None, **d) for d in detail)
            source_rate = operating_rates(cal, cp[m, s], threshold)
            target_rate = operating_rates(target, tp[m, s], threshold)
            row = dict(**meta, status='estimated', threshold=float(threshold),
                       **{'source_' + k: v for k, v in source_rate.items()},
                       **{'target_' + k: v for k, v in target_rate.items()})
            raw.append(row)
            member.append(row)
            for role, rates in [('source_calibration', source_rate), ('target', target_rate)]:
                for cls in LABELS:
                    labels.append(dict(**meta, role=role, subtype=cls,
                        **support[role][cls], errors=rates[cls + '_fp'],
                        label_rate=rates[cls + '_fpr']))
        summary = dict(fold=fold, method=method, alpha=alpha,
                       procedure='full_calibration', seeds=len(member),
                       status='estimated' if len(member) == len(seeds) else 'not_estimable')
        if len(member) == len(seeds):
            for key in ('source_fpr', 'target_fpr', 'target_recall'):
                vals = [r[key] for r in member]
                summary[key] = float(np.mean(vals)) if all(v is not None for v in vals) else None
            summary.update(target_negative=member[0]['target_negative'],
                           target_positive=member[0]['target_positive'])
            if target_draws is not None:
                for key, column in [('target_fpr', 0), ('target_recall', 1)]:
                    point = summary[key]
                    low, high, status = safe_interval(target_draws[:, m, column], point,
                                                     point in (0, 1))
                    summary.update({key + '_ci_low': low, key + '_ci_high': high,
                                    key + '_ci_status': status})
        summaries.append(summary)
    return dict(operating=raw, summary=summaries, labels=labels, diagnostics=diagnostics)


def _calculate(sd, se, td, te):
    """Evaluate totals; seeds were averaged within each fixed threshold/check."""
    if np.any(sd[:, :2] <= 0) or np.any(td[:2] <= 0):
        return None
    source_labels = np.divide(se, sd[:, None, :], out=np.full_like(se, np.nan),
                              where=sd[:, None, :] > 0).mean(axis=0)
    target_labels = np.divide(te, td[None, :], out=np.full_like(te, np.nan),
                              where=td[None, :] > 0)
    source_fpr = (se[:, :, :2].sum(-1) / sd[:, :2].sum(-1)[:, None]).mean(0)
    target_fpr = te[:, :2].sum(-1) / td[:2].sum()
    projected = (source_labels[:, :2] * (td[:2] / td[:2].sum())[None, :]).sum(-1)
    return dict(source_labels=source_labels, target_labels=target_labels,
                source_check_fpr=source_fpr, projected_fpr=projected,
                observed_fpr=target_fpr, transfer_gap=target_fpr - projected,
                target_recall=target_labels[:, 2])


def _draw_weights(rng, strata, count):
    w = np.zeros(count, dtype=float)
    for ids in strata.values():
        w[ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
    return w


def crossfit_transfer(fold, cal, cp, target, tp, methods, seeds, plan, alpha,
                      repeats=1000, seed=20261029):
    """Joint conditional uncertainty and leave-one-unit influence of the gap.

    Thresholds are fitted once from the stored plan. Each bootstrap realization
    draws the complete calibration population once, then restricts its weights
    to each check half. This keeps observations reused in different partitions
    dependent. The shared target draw jointly changes target proportions and
    observed errors. Missing required labels invalidate the whole paired draw.
    """
    if repeats < 10:
        raise ValueError('At least ten bootstrap draws are required')
    cp, tp = _validate(cal, cp), _validate(target, tp)
    expected = (len(methods), len(seeds))
    if cp.ndim != 3 or tp.ndim != 3 or cp.shape[:2] != expected or tp.shape[:2] != expected:
        raise ValueError('Expected aligned [method, seed, window] score arrays')
    cg, ci, cl, cs, cc = _layout(cal)
    tg, ti, tl, ts, tc = _layout(target)
    base = dict(fold=fold, alpha=alpha, procedure='crossfit_disjoint_source_check',
                source_units=len(cg), target_units=len(tg), seeds=len(seeds),
                partitions=len(plan.get('plans', [])), conditional_scope=SCOPE)
    empty = dict(summary=[], labels=[], influence=[], diagnostics=[], audit={})
    if not plan.get('plans') or np.any(tc[:, :2].sum(0) <= 0):
        reason = plan.get('reason') or 'missing_required_target_label'
        empty['summary'] = [dict(**base, method=m, status='not_estimable', reason=reason,
                                 bootstrap_status='not_run') for m in methods]
        empty['audit'] = dict(**base, status='not_estimable', reason=reason)
        return empty
    checks, thresholds, diagnostics, check_support = [], [], [], []
    for item in plan['plans']:
        for side in (0, 1):
            fit = np.asarray(item['left' if side == 0 else 'right'], dtype=int)
            check = np.asarray(item['right' if side == 0 else 'left'], dtype=int)
            if len(np.intersect1d(fit, check)) or set(np.r_[fit, check]) != set(range(len(cal))):
                raise ValueError('Cross-fit plan is not an exhaustive disjoint partition')
            fit_units = set(ci[fit]); check_units = set(ci[check])
            if fit_units & check_units:
                raise ValueError('A linked source unit crosses fit/check halves')
            fitrows, checkrows = [cal[i] for i in fit], [cal[i] for i in check]
            cut = np.empty(expected)
            for m, method in enumerate(methods):
                for s, source_seed in enumerate(seeds):
                    cut[m, s], detail = threshold_diagnostics(fitrows, cp[m, s, fit], alpha)
                    diagnostics.extend(dict(fold=fold, method=method, seed=source_seed,
                        alpha=alpha, procedure=base['procedure'], repeat=item['repeat'],
                        side=side, **d) for d in detail)
            checks.append(check)
            thresholds.append(cut)
            check_support.append(label_support(checkrows))
    thresholds = np.asarray(thresholds)
    r_count, m_count, s_count = thresholds.shape
    sd = np.zeros((len(cg), r_count, 3), dtype=float)
    se = np.zeros((len(cg), r_count, m_count, 3), dtype=float)
    te = np.zeros((len(tg), m_count, 3), dtype=float)
    for r, check in enumerate(checks):
        for c in range(3):
            idx = check[cl[check] == c]
            sd[:, r, c] = np.bincount(ci[idx], minlength=len(cg))
        for m in range(m_count):
            pred = (cp[m][:, check] >= thresholds[r, m, :, None]).mean(0)
            for c in range(3):
                mask = cl[check] == c
                se[:, r, m, c] = np.bincount(ci[check[mask]], weights=pred[mask], minlength=len(cg))
    for m in range(m_count):
        prediction_mean = np.zeros(len(target), dtype=float)
        for r in range(r_count):
            prediction_mean += (tp[m] >= thresholds[r, m, :, None]).sum(0)
        prediction_mean /= r_count * s_count
        for c in range(3):
            mask = tl == c
            te[:, m, c] = np.bincount(ti[mask], weights=prediction_mean[mask], minlength=len(tg))
    sd0, se0, td0, te0 = sd.sum(0), se.sum(0), tc.sum(0), te.sum(0)
    point = _calculate(sd0, se0, td0, te0)
    if point is None:
        empty['summary'] = [dict(**base, method=m, status='not_estimable',
            reason='missing_required_source_check_label', bootstrap_status='not_run') for m in methods]
        empty['audit'] = dict(**base, status='not_estimable',
                             reason='missing_required_source_check_label')
        empty['diagnostics'] = diagnostics
        return empty
    rng = np.random.default_rng(seed)
    draws, attempts = [], 0
    # Batched contractions avoid repeatedly scanning every window.
    while len(draws) < repeats and attempts < repeats * 10:
        batch = min(32, repeats - len(draws), repeats * 10 - attempts)
        sw = np.asarray([_draw_weights(rng, cs, len(cg)) for _ in range(batch)])
        tw = np.asarray([_draw_weights(rng, ts, len(tg)) for _ in range(batch)])
        sds = (sw @ sd.reshape(len(cg), -1)).reshape(batch, r_count, 3)
        ses = (sw @ se.reshape(len(cg), -1)).reshape(batch, r_count, m_count, 3)
        tds = tw @ tc
        tes = (tw @ te.reshape(len(tg), -1)).reshape(batch, m_count, 3)
        for i in range(batch):
            attempts += 1
            value = _calculate(sds[i], ses[i], tds[i], tes[i])
            if value is not None:
                draws.append(value)
    samples = {key: np.asarray([d[key] for d in draws]) for key in point}
    labels, summary = [], []
    target_support = label_support(target)
    complete = len(draws) == repeats
    for m, method in enumerate(methods):
        source_boundary = [LABELS[c] for c in range(2)
                           if point['source_labels'][m, c] <= 1e-14
                           or point['source_labels'][m, c] >= 1 - 1e-14]
        target_boundary = [LABELS[c] for c in range(2)
                           if point['target_labels'][m, c] <= 1e-14
                           or point['target_labels'][m, c] >= 1 - 1e-14]
        row = dict(**base, method=method, status='estimated',
                   bootstrap_valid=len(draws), bootstrap_requested=repeats,
                   bootstrap_rejected=attempts - len(draws),
                   bootstrap_rejection_fraction=(attempts - len(draws)) / attempts,
                   source_boundary_labels='|'.join(source_boundary),
                   target_boundary_labels='|'.join(target_boundary),
                   target_negative_windows=int(td0[:2].sum()),
                   source_negative_windows=int(cc[:, :2].sum()),
                   check_directions=r_count, repetitions_are_independent=False)
        for key in ('source_check_fpr', 'projected_fpr', 'observed_fpr', 'transfer_gap', 'target_recall'):
            value = float(point[key][m])
            row[key] = value if np.isfinite(value) else None
        for key, prefix in [('projected_fpr', 'projected'), ('observed_fpr', 'observed'),
                            ('transfer_gap', 'gap')]:
            boundary = bool(source_boundary) if key == 'projected_fpr' else (
                bool(source_boundary or target_boundary) if key == 'transfer_gap'
                else bool(target_boundary))
            interval = samples[key][:, m] if complete else []
            low, high, status = safe_interval(interval, row[key], boundary)
            row.update({prefix + '_ci_low': low, prefix + '_ci_high': high,
                        prefix + '_ci_status': status})
        row['bootstrap_status'] = row['gap_ci_status']
        summary.append(row)
        for role in ('source_check', 'target'):
            for c, cls in enumerate(LABELS):
                if role == 'source_check':
                    support = {key: float(np.mean([x[cls][key] for x in check_support]))
                               for key in ('windows', 'recordings', 'units')}
                    errors = float(se0[:, m, c].mean())
                    point_rate = float(point['source_labels'][m, c])
                    values = samples['source_labels'][:, m, c] if complete else []
                else:
                    support = target_support[cls]
                    errors = float(te0[m, c])
                    point_rate = float(point['target_labels'][m, c])
                    values = samples['target_labels'][:, m, c] if complete else []
                low, high, status = safe_interval(values, point_rate,
                    np.isfinite(point_rate) and (point_rate <= 1e-14 or point_rate >= 1 - 1e-14))
                labels.append(dict(fold=fold, method=method, alpha=alpha,
                    procedure=base['procedure'], role=role, subtype=cls,
                    windows_mean=support['windows'], errors_mean=errors,
                    recordings_mean=support['recordings'], units_mean=support['units'],
                    label_rate=point_rate if np.isfinite(point_rate) else None,
                    rate_ci_low=low, rate_ci_high=high, rate_ci_status=status,
                    count_interpretation='Means across fixed evaluations, not replicated independent windows'))
    influence = []
    maxima = {role: np.zeros(m_count) for role in ('source_check', 'target')}
    valid_removals = {role: np.zeros(m_count, dtype=int) for role in maxima}
    invalid_removals = {role: np.zeros(m_count, dtype=int) for role in maxima}
    for role, groups, counts, rows in [('source_check', cg, cc, cal), ('target', tg, tc, target)]:
        for g, (key, ids) in enumerate(groups):
            if counts[g, :2].sum() == 0:
                continue
            value = (_calculate(sd0 - sd[g], se0 - se[g], td0, te0)
                     if role == 'source_check' else
                     _calculate(sd0, se0, td0 - tc[g], te0 - te[g]))
            for m, method in enumerate(methods):
                item = dict(fold=fold, method=method, alpha=alpha,
                    procedure=base['procedure'], role=role, block=key,
                    sites='|'.join(sorted({str(rows[i]['site']) for i in ids})),
                    removed_windows=len(ids), removed_negative_windows=int(counts[g, :2].sum()),
                    removed_AB_windows=int(counts[g, 0]), removed_HW_windows=int(counts[g, 1]),
                    full_gap=float(point['transfer_gap'][m]))
                if value is None:
                    item.update(status='not_estimable_after_removal',
                                reason='required_label_support_lost')
                    invalid_removals[role][m] += 1
                else:
                    change = float(value['transfer_gap'][m] - point['transfer_gap'][m])
                    item.update(status='estimated', gap_without_block=float(value['transfer_gap'][m]),
                                gap_change=change, projected_without_block=float(value['projected_fpr'][m]),
                                observed_without_block=float(value['observed_fpr'][m]))
                    maxima[role][m] = max(maxima[role][m], abs(change))
                    valid_removals[role][m] += 1
                influence.append(item)
    for m, row in enumerate(summary):
        for role, prefix in [('source_check','source'),('target','target')]:
            row.update({f'max_abs_{prefix}_block_gap_change':
                        float(maxima[role][m]) if valid_removals[role][m] else None,
                        f'{prefix}_block_removals_valid':int(valid_removals[role][m]),
                        f'{prefix}_block_removals_not_estimable':int(invalid_removals[role][m])})
    audit = dict(**base, status='completed', bootstrap_requested=repeats,
        bootstrap_valid=len(draws), bootstrap_attempts=attempts,
        bootstrap_rejected=attempts - len(draws), complete_bootstrap=complete,
        bootstrap_rejection_fraction=(attempts - len(draws)) / attempts,
        source_strata=len(cs), target_strata=len(ts),
        source_singleton_strata=sum(len(x) == 1 for x in cs.values()),
        target_singleton_strata=sum(len(x) == 1 for x in ts.values()),
        source_draw_reused_across_partitions=True, target_proportions_recomputed=True,
        seeds_resampled=False, thresholds_refitted=False,
        invalid_draw_rule='Reject paired draw if either negative label is absent from any source check or target',
        boundary_rule='Withhold affected intervals at zero/all-error label boundaries or degenerate empirical bounds',
        block_influence_scope='Leave one observed linked unit out with thresholds and partitions fixed; no refitting',
        rejection_caveat='Intervals, when reported, condition additionally on required-label support surviving resampling')
    return dict(summary=summary, labels=labels, influence=influence,
                diagnostics=diagnostics, audit=audit)
