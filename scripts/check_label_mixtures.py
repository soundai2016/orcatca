"""Compare observed FPR with the empirical upper bound over HW/AB mixtures.

Rates/weights are fractions; "_pp" denotes percentage points and "_pct" percent.
The companion table reports AB’s share of the mean observed-mixture gap and
of the hypothetical gap giving both negative labels equal weight. These are
ratios of mean contributions, not means of threshold-specific ratios.
For each fitted seed, partition direction and nominal rule the two label-specific
source-check rates and the target FPR use exactly the same threshold. Their
maximum is taken BEFORE averaging. It bounds mixtures of the *measured* source
rates; it is not a confidence bound for future recordings. Matched evaluations
reuse windows: their counts are descriptive, not independent trials.
"""
import argparse
import csv
import math
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('HW', 'AB')
ROLES = {'source_fit', 'source_check', 'target'}


def key(r):
    return (r['fold'], r['method'], int(r['seed']), int(r['repeat']),
            int(r['side']), float(r['alpha']))


def same(a, b, label):
    if not math.isclose(a, b, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f'{label}: {a!r} != {b!r}')


def rates(row):
    counts = [int(row[c + '_n']) for c in LABELS]
    errors = [int(row[c + '_fp']) for c in LABELS]
    if any(n < 1 or not 0 <= e <= n for n, e in zip(counts, errors)):
        raise ValueError('Missing negative-label support or invalid error count')
    if sum(counts) != int(row['negative']) or sum(errors) != int(row['fp']):
        raise ValueError('Negative labels do not partition overall counts')
    q = [e / n for e, n in zip(errors, counts)]
    for c, rate in zip(LABELS, q):
        same(rate, float(row[c + '_fpr']), c + ' FPR')
    overall = sum(errors) / sum(counts)
    same(overall, float(row['fpr']), 'overall FPR')
    return q, [n / sum(counts) for n in counts], overall


def contribution_shares(hw_pp, ab_pp, weight_ab):
    """AB share of nonnegative mean changes; do not turn cancellation into shares."""
    if not 0 < weight_ab < 1 or not all(map(math.isfinite, (hw_pp, ab_pp))):
        raise ValueError('Need two supported negative labels and finite mean changes')
    if min(hw_pp, ab_pp) < 0:
        return None, None, 'withheld_negative_mean_change'
    observed_gap = (1 - weight_ab) * hw_pp + weight_ab * ab_pp
    balanced_gap = (hw_pp + ab_pp) / 2
    if min(observed_gap, balanced_gap) <= 1e-12:  # in percentage points
        return None, None, 'withheld_zero_or_near_zero_gap'
    return 100 * weight_ab * ab_pp / observed_gap, 50 * ab_pp / balanced_gap, 'estimated'


def summarize_tables(operating, transfer, expected_keys):
    op = defaultdict(dict)
    for row in operating:
        k, role = key(row), row['role']
        if role not in ROLES or role in op[k]:
            raise ValueError(f'Duplicate or unexpected operating role: {k}, {role}')
        op[k][role] = row
    tr = {}
    for row in transfer:
        k = key(row)
        if k in tr or row['status'] != 'estimated_descriptively':
            raise ValueError(f'Duplicate or unsupported decomposition: {k}')
        tr[k] = row
    if set(op) != expected_keys or set(tr) != expected_keys:
        raise ValueError('Operating/decomposition rows do not match the expected evaluation grid')

    groups = defaultdict(list)
    target_counts, changes = {}, defaultdict(list)
    for k in sorted(expected_keys):
        roles, t = op[k], tr[k]
        if set(roles) != ROLES:
            raise ValueError(f'Missing operating role: {k}')
        thresholds = [float(r['threshold']) for r in [t, *roles.values()]]
        if len(set(thresholds)) != 1 or not math.isfinite(thresholds[0]):
            raise ValueError(f'Paired thresholds differ or are non-finite: {k}')
        qs, _, source_fpr = rates(roles['source_check'])
        qt, wt, observed = rates(roles['target'])
        counts = tuple(int(roles['target'][label + '_n']) for label in LABELS)
        if target_counts.setdefault(k[0], counts) != counts:
            raise ValueError(f'Target HW/AB counts change across matched settings in {k[0]}')
        changes[k[0], k[1], k[-1]].append(tuple(100 * (y - x) for y, x in zip(qt, qs)))
        projection = sum(w * q for w, q in zip(wt, qs))
        movement = projection - source_fpr
        contributions = [w * (y - x) for w, y, x in zip(wt, qt, qs)]
        gap = observed - projection
        for value, column in ((projection, 'mixture_only'), (movement, 'source_reference_mixture'),
                              (gap, 'target_weighted_conditional'), (observed - source_fpr, 'delta')):
            same(value, float(t[column]), f'{k} {column}')
        same(sum(contributions), gap, 'additive label contributions')
        ceiling = max(qs)  # the harder label can change between matched evaluations
        groups[k[0], k[1], k[-1]].append(dict(
            source_check_fpr=source_fpr, target_mixture_projection=projection,
            target_observed_fpr=observed, empirical_mixture_ceiling=ceiling,
            composition_movement_pp=100 * movement, actual_weight_gap_pp=100 * gap,
            hw_contribution_pp=100 * contributions[0], ab_contribution_pp=100 * contributions[1],
            excess_over_ceiling_pp=100 * (observed - ceiling)))
    output, sensitivity = [], []
    for (fold, method, alpha), values in sorted(groups.items()):
        excess = [v['excess_over_ceiling_pp'] for v in values]
        means = {'mean_' + name: fmean(v[name] for v in values) for name in values[0]}
        output.append(dict(fold=fold, method=method, alpha=alpha,
            n_matched_evaluations=len(values), n_observed_above_ceiling=sum(x > 0 for x in excess),
            **means, min_excess_over_ceiling_pp=min(excess), max_excess_over_ceiling_pp=max(excess)))
        n_hw, n_ab = target_counts[fold]
        weight_ab = n_ab / (n_ab + n_hw)
        change_hw, change_ab = map(fmean, zip(*changes[fold, method, alpha]))
        same((1-weight_ab)*change_hw, means['mean_hw_contribution_pp'], 'weighted HW change')
        same(weight_ab*change_ab, means['mean_ab_contribution_pp'], 'weighted AB change')
        observed_pct, balanced_pct, status = contribution_shares(change_hw, change_ab, weight_ab)
        sensitivity.append(dict(fold=fold, method=method, alpha=alpha,
            procedure='crossfit_disjoint_source_check', n_matched_evaluations=len(values),
            target_hw_windows=n_hw, target_ab_windows=n_ab, observed_ab_weight=weight_ab,
            mean_hw_rate_change_pp=change_hw, mean_ab_rate_change_pp=change_ab,
            ab_share_observed_gap_pct=observed_pct, ab_share_balanced_gap_pct=balanced_pct,
            share_status=status))
    return output, sensitivity


def summarize(operating, transfer, expected_keys):
    """Return only the existing empirical-ceiling table, as used by plot.py."""
    return summarize_tables(operating, transfer, expected_keys)[0]


def write_rows(output, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(directory, output, sensitivity_output=None):
    if sensitivity_output is None:
        sensitivity_output = output.with_name('label_mix_sensitivity.csv')
    if output.resolve() == sensitivity_output.resolve():
        raise ValueError('Provide separate destinations for the two tables')
    info = json.loads((directory / 'analysis.json').read_text())
    if info['status'] != 'completed':
        raise ValueError('A completed transfer analysis is required')
    folds = [f for f, status in info['crossfit_status'].items() if status == 'ready']
    expected = set(product(folds, info['methods'], info['seeds'], range(info['crossfit_repeats']),
                           (0, 1), (0.01, 0.05)))
    if not expected:
        raise ValueError('No supported transfer evaluations')
    with (directory / 'crossfit_operating.csv').open(newline='') as op, \
         (directory / 'crossfit_transfer.csv').open(newline='') as tr:
        bounds, sensitivity = summarize_tables(csv.DictReader(op), csv.DictReader(tr), expected)
    write_rows(output, bounds)
    write_rows(sensitivity_output, sensitivity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'result/analysis/transfer_review')
    parser.add_argument('--output', type=Path, default=ROOT / 'result/figure_data/derived/mixture_bounds.csv')
    parser.add_argument('--sensitivity-output', type=Path, help='Default: label_mix_sensitivity.csv beside --output')
    args = parser.parse_args()
    run(args.analysis, args.output, args.sensitivity_output)
