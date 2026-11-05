#!/usr/bin/env python3
"""Redraw reported figures from archived tables or one completed analysis."""
from __future__ import annotations
import argparse
import csv
from itertools import product
import io
import sys
import hashlib
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_label_mixtures as mixture_check
import acoustic_panels
import check_figures

BLUE, RUST, TEAL, INK = '#165B86', '#BB4B16', '#007A70', '#142B38'
METHODS = {'natural_mixstyle': 'Binary', 'natural_coarse_mixstyle': 'Coarse auxiliary',
           'natural_threeclass_mixstyle': 'Three-class', 'natural_subtype_mixstyle': 'Fine auxiliary'}
FOLDS = [('fold00', 'Scripps'), ('fold01', 'DFO'), ('fold02', 'Joint')]
ARCHIVE = ('false_positive_profile', 'fit_metrics', 'joint_differences', 'main_results',
           'site_metrics', 'source_window_counts', 'support_sensitivity', 'threshold_transfer')
RICH = ('transfer_gap_summary', 'transfer_original_labels', 'transfer_original_summary')


def resolved(root, value, default):
    return (default if value is None else root / value).expanduser().resolve()


def safe_output(root, value):
    path, raw = value.resolve(), (root / 'dclde_2027_killer_whales').resolve()
    if path.is_relative_to(raw) or raw.is_relative_to(path):
        raise ValueError('Output must not be the raw corpus or one of its parent directories')
    return path


def analysis_source(root, supplied):
    base = resolved(root, supplied, root / 'result/analysis')
    if (base / 'analysis.json').is_file():
        candidates = [base]
    else:
        candidates = [p.parent for p in base.rglob('analysis.json')
                      if json.loads(p.read_text()).get('status') == 'completed'
                      and all((p.parent / (n+'.csv')).is_file() for n in RICH)]
    if len(candidates) != 1:
        raise ValueError(f'Choose one completed analysis explicitly; found {len(candidates)} under {base}: {candidates}')
    path = candidates[0]
    if json.loads((path / 'analysis.json').read_text()).get('status') != 'completed':
        raise ValueError(f'Analysis is not completed: {path}')
    if not all((path / (name+'.csv')).is_file() for name in RICH):
        raise FileNotFoundError(f'Completed analysis lacks required transfer tables: {path}')
    return path



def unique_index(frame, keys):
    """Build a unique index without the deprecated Pandas verification option."""
    result = frame.set_index(keys)
    if not result.index.is_unique:
        duplicates = result.index[result.index.duplicated()].unique()
        raise ValueError(f'Duplicate index keys for {keys}: {list(duplicates[:5])}')
    return result

def decompose_gap(gap, rates):
    """Derive label contributions only from matched half-calibration summaries."""
    fields=['fold','method','alpha','procedure','subtype','target_windows','target_weight',
            'source_check_rate','target_rate','gap_contribution','transfer_gap']
    records=[]
    if rates.empty: return pd.DataFrame(columns=fields)
    matched=unique_index(rates[rates.procedure.eq('crossfit_disjoint_source_check') &
                  rates.subtype.isin(['HW','AB'])], ['fold','method','alpha','role','subtype'])
    for r in gap.dropna(subset=['transfer_gap']).itertuples():
        key=(r.fold,r.method,r.alpha)
        if any((*key,role,c) not in matched.index for role in ('source_check','target') for c in ('HW','AB')):
            continue
        cells=[]
        for c in ('HW','AB'):
            src=matched.loc[(*key,'source_check',c)];tar=matched.loc[(*key,'target',c)]
            cells.append([c,tar.windows_mean,src.label_rate,tar.label_rate])
        values=np.array([v[1:] for v in cells],dtype=float)
        if not np.isfinite(values).all() or (values[:,0]<=0).any(): continue
        if (values[:,1:]<0).any() or (values[:,1:]>1).any(): raise ValueError('Invalid cross-fit label rate')
        n=values[:,0].sum(); rows=[]
        for c,nc,qsrc,qtgt in cells:
            w=nc/n
            rows.append([*key,'crossfit_disjoint_source_check',c,nc,w,qsrc,qtgt,w*(qtgt-qsrc),r.transfer_gap])
        if not np.isclose(sum(row[-2] for row in rows),r.transfer_gap,atol=1e-10,rtol=1e-8):
            raise ValueError('Label contributions do not sum to the matched transfer gap')
        records.extend(rows)
    return pd.DataFrame(records, columns=fields)


def matched_ceiling(source, info, alpha, gap, hashes):
    """For one rule, derive maxima from matched checks in the selected run."""
    required = ['crossfit_operating.csv', 'crossfit_transfer.csv']
    if any(not (source / name).is_file() for name in required):
        raise FileNotFoundError('Empirical mixture ceiling requires crossfit_operating.csv '
                                'and crossfit_transfer.csv in the selected analysis')
    matched = []
    for name in required:
        raw = (source / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        matched.append([row for row in csv.DictReader(io.StringIO(raw.decode('utf-8')))
                        if float(row['alpha']) == alpha])
    ready = [f for f, status in info['crossfit_status'].items() if status == 'ready']
    expected = set(product(ready, info['methods'], info['seeds'],
                           range(info['crossfit_repeats']), (0, 1), (alpha,)))
    if not expected:
        if gap.transfer_gap.notna().any():
            raise ValueError('Transfer estimates present without a supported evaluation grid')
        return pd.DataFrame()
    ceiling = pd.DataFrame(mixture_check.summarize(*matched, expected))
    key = ['fold','method','alpha']
    a = unique_index(gap.dropna(subset=['transfer_gap']), key).sort_index()
    b = unique_index(ceiling, key).sort_index()
    if not a.index.equals(b.index):
        raise ValueError('Empirical ceiling and transfer summary do not have the same cells')
    for src, dst, scale in [('projected_fpr','mean_target_mixture_projection',1),
                            ('observed_fpr','mean_target_observed_fpr',1),
                            ('transfer_gap','mean_actual_weight_gap_pp',100)]:
        if not np.allclose(scale*a[src], b[dst], rtol=0, atol=1e-10):
            raise ValueError('Empirical ceiling is paired with different transfer rates: '+src)
    return ceiling


def transfer_configurations(operating, transfers, alpha, summary):
    """Retain one matched threshold evaluation per seed, split and direction.

    These rows reuse recordings; their spread is descriptive, not a sampling
    distribution. Source label rates are weighted by target negative counts.
    """
    keys = ['fold', 'method', 'seed', 'repeat', 'side', 'alpha']
    columns = keys + ['threshold', 'projected_fpr', 'observed_fpr',
                     'transfer_gap', 'source_max_fpr', 'HW_contribution',
                     'AB_contribution']
    if operating.empty and transfers.empty:
        return pd.DataFrame(columns=columns)
    op = operating.loc[np.isclose(operating.alpha, alpha)].copy()
    tr = transfers.loc[np.isclose(transfers.alpha, alpha)].copy()
    required = keys + ['threshold', 'role', 'HW_n', 'AB_n', 'HW_fpr', 'AB_fpr', 'fpr']
    if not set(required).issubset(op) or not set(keys + ['threshold', 'target_weighted_conditional']).issubset(tr):
        raise ValueError('Matched configuration rows lack source/target label rates')
    if op.duplicated(keys + ['role']).any() or tr.duplicated(keys).any():
        raise ValueError('Duplicate matched configuration')
    source = op.loc[op.role.eq('source_check')].set_index(keys).sort_index()
    target = op.loc[op.role.eq('target')].set_index(keys).sort_index()
    reported = tr.set_index(keys).sort_index()
    if not source.index.equals(target.index) or not source.index.equals(reported.index):
        raise ValueError('Configuration source checks, targets and transfer rows do not match')
    if source.empty:
        return pd.DataFrame(columns=columns)
    for other in (target, reported):
        if not np.allclose(source.threshold, other.threshold, rtol=0, atol=1e-12):
            raise ValueError('Configuration rows use different thresholds')
    for frame in (source, target):
        q = frame[['HW_fpr', 'AB_fpr', 'fpr']].to_numpy(float)
        n = frame[['HW_n', 'AB_n']].to_numpy(float)
        if not np.isfinite(q).all() or ((q < 0) | (q > 1)).any() or not np.isfinite(n).all() or (n <= 0).any():
            raise ValueError('Invalid label support or rates in matched configuration')
    weights = target[['HW_n', 'AB_n']].div(target.HW_n + target.AB_n, axis=0)
    projected = weights.HW_n * source.HW_fpr + weights.AB_n * source.AB_fpr
    observed = weights.HW_n * target.HW_fpr + weights.AB_n * target.AB_fpr
    if not np.allclose(observed, target.fpr, rtol=0, atol=1e-12):
        raise ValueError('Target label rates do not recover the configuration FPR')
    gap = observed - projected
    if not np.allclose(gap, reported.target_weighted_conditional, rtol=0, atol=1e-12):
        raise ValueError('Configuration gap differs from the target-weighted transfer contrast')
    out = pd.DataFrame({'threshold': target.threshold, 'projected_fpr': projected,
        'observed_fpr': observed, 'transfer_gap': gap,
        'source_max_fpr': source[['HW_fpr', 'AB_fpr']].max(axis=1),
        'HW_contribution': weights.HW_n * (target.HW_fpr - source.HW_fpr),
        'AB_contribution': weights.AB_n * (target.AB_fpr - source.AB_fpr)}).reset_index()
    identity = out.HW_contribution + out.AB_contribution
    if not np.allclose(identity, out.transfer_gap, rtol=0, atol=1e-12):
        raise ValueError('Configuration label changes do not sum to the transfer gap')
    group = ['fold', 'method', 'alpha']
    fields = ['projected_fpr', 'observed_fpr', 'transfer_gap']
    actual = out.groupby(group)[fields].mean().sort_index()
    expected = summary.dropna(subset=['transfer_gap']).set_index(group)[fields].sort_index()
    if not actual.index.equals(expected.index) or not np.allclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError('Configuration means do not match the saved transfer summary')
    return out[columns].sort_values(keys).reset_index(drop=True)


def load_tables(root, tables=None, analysis=None, alpha=.01):
    if not 0 < alpha < 1 or (tables is not None and analysis is not None):
        raise ValueError('Use alpha in (0,1), and select either --tables or --analysis')
    rich = analysis is not None
    source = analysis_source(root, analysis) if rich else resolved(root, tables, root / 'result/figure_data')
    hashes = {}

    def read(name, optional=False):
        path = source / (name+'.csv')
        if optional and not path.exists():
            return pd.DataFrame()
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def selected(frame):
        if 'alpha' not in frame or not np.isclose(frame.alpha, alpha).any():
            raise ValueError(f'No alpha={alpha:g} results in selected table set')
        return frame[np.isclose(frame.alpha, alpha)].copy()

    labels, original = [], []
    half_labels=pd.DataFrame()
    if rich:
        hashes['analysis.json'] = hashlib.sha256((source/'analysis.json').read_bytes()).hexdigest()
        gap, raw, operating = [selected(read(n)) for n in RICH]
        for (fold, method, role, subtype), group in raw.groupby(['fold','method','role','subtype']):
            if subtype not in ('HW','AB'):
                continue
            if group.seed.duplicated().any() or any(group[k].nunique(dropna=False) != 1 for k in ('windows','recordings','units')):
                raise ValueError('Seed metadata do not share one unique support population')
            n, errors = int(group.windows.sum()), float(group.errors.sum())
            if n <= 0 or errors < 0 or errors > n:
                raise ValueError('Invalid full-calibration label counts')
            labels.append(dict(fold=fold, method=method, role=role, raw_class=subtype,
                alpha=alpha, procedure='full_calibration', windows=int(group.windows.iloc[0]),
                recordings=int(group.recordings.iloc[0]), units=int(group.units.iloc[0]),
                prediction_count=n, errors_pooled=errors, seeds=group.seed.nunique(), label_rate=errors/n,
                recordings_provenance='saved_prediction_metadata'))
        original = operating.to_dict('records')
        supplementary = {'main_results': read('main_results', True),
                         'support_sensitivity': read('support_sensitivity', True),
                         'joint_differences': read('joint_differences', True)}
        site = read('site_metrics', True)
        supplementary['site_metrics'] = site
        if supplementary['main_results'].empty or supplementary['support_sensitivity'].empty:
            if not site.empty:
                a,b = site_sensitivity(site)
                supplementary.update(main_results=a, support_sensitivity=b)
        source_labels = read('transfer_label_summary', True)
        if not source_labels.empty:
            half_labels=selected(source_labels)
            source_labels=half_labels[half_labels.role.eq('source_check')]
    else:
        if not np.isclose(alpha, .01, rtol=0, atol=1e-12):
            raise ValueError('Archived figure tables only support alpha=0.01; use a completed --analysis for other rules')
        archived = {name: read(name) for name in ARCHIVE}
        if archived['false_positive_profile'].duplicated(['fold','method']).any():
            raise ValueError('Duplicate pooled false-positive profiles')
        counts = unique_index(archived['source_window_counts'], 'fold')
        support = read('recording_support', True)
        support = unique_index(support, ['fold','raw_class']) if not support.empty else support
        for _, r in archived['false_positive_profile'].iterrows():
            seeds = int(r.seeds)
            if seeds < 1 or seeds != r.seeds:
                raise ValueError('Invalid pooled seed count')
            for cls, errors in [('HW', r.HW_fp), ('AB', r.fp-r.HW_fp)]:
                n, unique = float(r[cls+'_n']), int(counts.loc[r.fold, cls])
                if n <= 0 or n != unique*seeds or not 0 <= errors <= n:
                    raise ValueError('Pooled prediction denominator disagrees with unique-window support')
                record, provenance = np.nan, 'not_available_in_archived_tables'
                if not support.empty and (r.fold, cls) in support.index:
                    s = support.loc[(r.fold, cls)]
                    if int(s.windows) != unique:
                        raise ValueError('Reported recording support has a different window population')
                    record, provenance = int(s.recordings), 'retained_support_table'
                labels.append(dict(fold=r.fold, method=r.method, role='target', raw_class=cls,
                    alpha=alpha, procedure='full_calibration', windows=unique, recordings=record,
                    units=np.nan, prediction_count=int(n), errors_pooled=float(errors), seeds=seeds,
                    label_rate=float(errors/n), recordings_provenance=provenance))
            original.append(dict(fold=r.fold, method=r.method, alpha=alpha, procedure='full_calibration',
                status='reported_aggregate', target_fpr=float(r.fp/(r.HW_n+r.AB_n)), target_recall=np.nan,
                target_negative=int(counts.loc[r.fold,'HW']+counts.loc[r.fold,'AB']),
                target_positive=int(counts.loc[r.fold,'KW']), recall_status='not_available_in_archived_tables'))
        gap = archived['threshold_transfer'].rename(columns={'mixture_only':'projected_fpr','fpr':'observed_fpr'}).copy()
        gap['transfer_gap'] = gap.observed_fpr-gap.projected_fpr
        gap['alpha'], gap['status'] = alpha, 'reported_aggregate'
        gap['procedure'] = 'crossfit_disjoint_source_check'
        gap['gap_ci_status'] = 'not_available_from_archived_aggregates'
        supplementary = {name: archived[name] for name in ('main_results','support_sensitivity','joint_differences','site_metrics')}
        source_labels = pd.DataFrame()
    for column in ('projected_fpr','observed_fpr','transfer_gap','gap_ci_low','gap_ci_high'):
        if column not in gap:
            gap[column] = np.nan
    if gap.duplicated(['fold','method']).any():
        raise ValueError('Transfer table must contain one paired estimate per fold and method')
    valid = gap[['observed_fpr','projected_fpr','transfer_gap']].notna().all(axis=1)
    if not np.allclose(gap.loc[valid,'transfer_gap'], gap.loc[valid,'observed_fpr']-gap.loc[valid,'projected_fpr']):
        raise ValueError('Reported transfer gap does not match the paired rates')
    ceiling=(matched_ceiling(source, json.loads((source/'analysis.json').read_text()),
                             alpha, gap, hashes) if rich else pd.DataFrame())
    configurations = (transfer_configurations(read('crossfit_operating'),
                       read('crossfit_transfer'), alpha, gap) if rich else
                       pd.DataFrame(columns=['fold','method','seed','repeat','side','alpha',
                       'threshold','projected_fpr','observed_fpr','transfer_gap',
                       'source_max_fpr','HW_contribution','AB_contribution']))
    contrib=decompose_gap(gap,half_labels)
    stack_ok=(valid.any() and len(contrib)==2*valid.sum() and contrib.gap_contribution.ge(0).all())
    available_ci = gap[['gap_ci_low','gap_ci_high']].notna().all(axis=1)
    if (gap.loc[available_ci,'gap_ci_low'] > gap.loc[available_ci,'gap_ci_high']).any():
        raise ValueError('Inverted gap confidence bounds')
    for fold, _ in FOLDS:
        if fold not in set(gap.fold):
            gap = pd.concat([gap, pd.DataFrame([dict(fold=fold, method=m, alpha=alpha,
                status='not_estimable', reason='no_disjoint_source_check_in_selected_inputs') for m in METHODS])], ignore_index=True)
    evidence = dict(mode='completed_analysis' if rich else 'archived_aggregates',
        input_directory=source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source),
        input_path_basis='project_root' if source.is_relative_to(root) else 'absolute',
        input_sha256=hashes, alpha=alpha, gap_ci_available=bool(available_ci.any()),
        gap_ci_rows=int(available_ci.sum()), gap_estimate_rows=int(valid.sum()),
        nonnegative_label_bridge_available=bool(stack_ok),
        matched_empirical_ceiling_rows=len(ceiling),
        matched_configuration_rows=len(configurations),
        configuration_rows_are_independent=False,
        site_metric_rows=len(supplementary['site_metrics']),
        intervals='Conditional where supplied; absent intervals are not reconstructed from aggregates',
        repeated_seed_predictions_are_independent_windows=False,
        absolute_recall_available=any(pd.notna(r.get('target_recall')) for r in original),
        recording_support='saved_prediction_metadata' if rich else 'retained_support_table, when supplied; not recomputed',
        full_calibration_and_crossfit_thresholds_differ=True, original_audio_loaded=False,
        comparison_status=gap[['fold','method','status']].to_dict('records'))
    return dict(gap=gap, labels=pd.DataFrame(labels), original=pd.DataFrame(original), ceiling=ceiling,
                configuration_points=configurations,
                source_labels=source_labels, contributions=contrib, evidence=evidence, **supplementary)


def full_label_contrasts(labels):
    """Pair target HW and AB rates at each full-pool rule, before rounding.

    The denominator for a rate is the number of scored predictions. Windows
    and recordings describe the distinct target observations, counted once.
    """
    ids = ['fold', 'method', 'alpha']
    quantities = ['label_rate', 'windows', 'recordings']
    schema = ids + ['hw_rate', 'ab_rate', 'ab_to_hw_ratio', 'ratio_status',
                    'hw_windows', 'ab_windows', 'hw_recordings', 'ab_recordings']
    if labels.empty:
        return pd.DataFrame(columns=schema)
    target = labels.loc[labels.role.eq('target') & labels.procedure.eq('full_calibration')
                        & labels.raw_class.isin(['HW', 'AB'])]
    if target.empty:
        return pd.DataFrame(columns=schema)
    if target.duplicated(ids + ['raw_class']).any():
        raise ValueError('Duplicate full-calibration target rate')
    if target.groupby(['fold', 'alpha', 'raw_class'])[['windows', 'recordings']].nunique(
            dropna=False).gt(1).any().any():
        raise ValueError('Objectives have different target support; a shared denominator is invalid')
    pairs = []
    for tag in ['HW', 'AB']:
        rows = target.loc[target.raw_class.eq(tag), ids + quantities].set_index(ids)
        pairs.append(rows.rename(columns={'label_rate':'rate'}).add_prefix(tag.lower() + '_'))
    out = pd.concat(pairs, axis=1).reset_index()
    a, h = out.ab_rate, out.hw_rate
    out['ab_to_hw_ratio'] = a.div(h.where(h.gt(0)))
    out['ratio_status'] = np.select(
        [a.isna() | h.isna(), a.eq(0) & h.eq(0), h.eq(0)],
        ['missing_label', 'both_zero', 'zero_hw'], default='finite')
    out.loc[out.ratio_status.eq('zero_hw'), 'ab_to_hw_ratio'] = np.inf
    return out[schema].sort_values(ids).reset_index(drop=True)


def pauc_gain(scores):
    """Fine-auxiliary minus binary standardized pAUC on its original scale."""
    cols = ['all_sites', 'at_least_20']
    return scores.loc['natural_subtype_mixstyle', cols] - scores.loc['natural_mixstyle', cols]


def appearance():
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans') if f in installed), 'DejaVu Sans')
    plt.rcParams.update({'font.family':family,'font.weight':'normal','axes.titleweight':'normal','axes.labelweight':'normal','font.size':8, 'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})


def panel_title(fig, x, y, letter, title):
    """Use the same type hierarchy for both figures."""
    style = dict(ha='left', va='center', color=INK)
    fig.text(x, y, letter, fontsize=10, fontweight='bold', **style)
    fig.text(x+.035, y, title, fontsize=9, fontweight='normal', **style)


def save(fig, out, stem):
    from matplotlib.text import Text
    from tempfile import TemporaryDirectory
    # The figures are printed at their native width. Smaller type would fail
    # the 8-point minimum even if a screen preview looked legible.
    fig.canvas.draw()
    small = [t.get_text() for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < 8]
    if small:
        raise ValueError('Figure text below 8 pt: ' + repr(small))
    out.mkdir(parents=True, exist_ok=True)
    # Exclude the volatile creation time so a redraw in the same environment
    # gives the same vector file as the one used by LaTeX.
    # Replace only complete files; an interrupted render keeps the previous
    # figure instead of leaving a truncated PDF in the results directory.
    with TemporaryDirectory(prefix='.render-', dir=out) as temporary:
        staging = Path(temporary)
        fig.savefig(staging/(stem+'.pdf'), dpi=400, metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(staging/(stem+'.png'),dpi=400)
        for suffix in ('.pdf', '.png'):
            (staging/(stem+suffix)).replace(out/(stem+suffix))
    plt.close(fig)


def figure1(tables, out, alpha, acoustic=None):
    """Show measured configuration spread, conditional intervals, and attribution."""
    from matplotlib.ticker import MaxNLocator

    ink, muted, connector = INK, '#53636B', '#BAC5CB'
    methods = list(METHODS)
    groups = [('fold00', 'Scripps'), ('fold02', 'Joint')]
    gaps = unique_index(tables['gap'], ['fold', 'method'])
    configurations = tables.get('configuration_points', pd.DataFrame())
    contributions = tables.get('contributions', pd.DataFrame())
    ceiling = tables.get('ceiling', pd.DataFrame())
    fig = plt.figure(figsize=(6.25, 4.40))

    def frame(ax, left=False):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_visible(left)
        ax.spines[['bottom', 'left']].set_color(muted)
        ax.spines[['bottom', 'left']].set_linewidth(.65)
        ax.tick_params(axis='both', labelsize=8, length=2.5, width=.65,
                       pad=3, color=muted, labelcolor=ink)
        ax.grid(False)

    has_points = not configurations.empty
    handles = []
    if has_points:
        counts = configurations.groupby(['fold','method']).size().reindex(
            pd.MultiIndex.from_product([[fold for fold, _ in groups], methods]))
        label = '30 matched evaluations' if counts.eq(30).all() else 'Matched evaluations'
        handles.append(Line2D([], [], marker='o', ls='none', ms=3.2,
                              mec=BLUE, mfc=BLUE, label=label))
    has_intervals = gaps[['gap_ci_low','gap_ci_high']].notna().all(axis=1).any()
    handles.append(Line2D([], [], marker='o', color=ink, lw=1.1, ms=3.6,
                          ls='-' if has_intervals else 'none',
                          label='Mean and 95% CI' if has_intervals else 'Mean'))
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.582, .918),
               ncol=2, frameon=False, fontsize=8, labelcolor=ink,
               handlelength=1.7, handletextpad=.5, columnspacing=1.3)

    for j, (fold, name) in enumerate(groups):
        panel_title(fig, .014 if j == 0 else .615, .970, 'ab'[j], name)
        ax = fig.add_axes([.205+.433*j, .542, .345, .330], label='gap_'+fold)
        frame(ax)
        ax.set_ylim(-.65, 3.57)
        ax.set_yticks(np.arange(4)[::-1]-.09,
                      [METHODS[m] for m in methods] if j == 0 else [])
        ax.tick_params(axis='y', length=0, pad=5)
        ax.set_xlabel('Observed − projected (pp)', fontsize=8, color=ink, labelpad=4)
        ax.axvline(0, color=connector, lw=.65, zorder=0)
        extent = [0., 1.]
        for i, method in enumerate(methods):
            key, y = (fold, method), 3-i
            if key not in gaps.index:
                continue
            r = gaps.loc[key]
            if has_points:
                points = configurations.loc[
                    configurations.fold.eq(fold) & configurations.method.eq(method)]
                values = 100*points.sort_values(['seed', 'repeat', 'side']).transfer_gap.to_numpy(float)
                values = values[np.isfinite(values)]
                if len(values):
                    extent.extend(values)
                    # Smooth only the saved repeated evaluations. The separate
                    # block interval is never used to manufacture a density.
                    if len(values) >= 3 and np.ptp(values) > 1e-10:
                        sd = values.std(ddof=1)
                        iqr = np.subtract(*np.percentile(values, [75, 25]))/1.349
                        scale = min(sd, iqr) if iqr > 0 else sd
                        bandwidth = max(.9*scale*len(values)**(-.2), np.ptp(values)/100)
                        grid = np.linspace(values.min(), values.max(), 160)
                        density = np.exp(-.5*((grid[:, None]-values)/bandwidth)**2).mean(axis=1)
                        cloud = ax.fill_between(grid, y+.035,
                            y+.035+.29*density/density.max(), color=BLUE,
                            alpha=.26, linewidth=0, zorder=1)
                        cloud.set_gid(f'configuration_density:{fold}:{method}')
                    jitter = (.61803398875*np.arange(len(values)) % 1-.5)*.13
                    rain = ax.scatter(values, y-.105+jitter, s=5.2, c=BLUE,
                                      linewidths=0, alpha=.83, zorder=3)
                    rain.set_gid(f'configuration:{fold}:{method}')
            mean, low, high = 100*np.array([r.transfer_gap, r.gap_ci_low, r.gap_ci_high], dtype=float)
            extent.extend(v for v in (mean, low, high) if np.isfinite(v))
            if not np.isfinite(mean):
                continue
            level = y-.34 if has_points else y-.09
            if np.isfinite(low) and np.isfinite(high):
                interval = ax.hlines(level, low, high, color=ink, lw=1.1, zorder=2)
                interval.set_gid(f'transfer:{fold}:{method}:ci')
            point, = ax.plot(mean, level, marker='o', ls='none', ms=3.6,
                             mec=ink, mfc=ink, zorder=4)
            point.set_gid(f'transfer:{fold}:{method}:point')
        ticks = MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(min(extent), max(extent))
        spread = max(1., ticks[-1]-ticks[0])
        ax.set_xlim(ticks[0]-.025*spread, ticks[-1]+.025*spread)
        ax.set_xticks(ticks)

    panel_title(fig, .014, .408, 'c', 'Binary FPR: projection and excess')
    ax = fig.add_axes([.115, .120, .870, .225], label='binary_decomposition')
    frame(ax, left=True)
    ax.set_ylabel('Window FPR (%)', fontsize=8, color=ink, labelpad=4)
    tick_positions, tick_labels, heights = [], [], [0., 1.]
    showed_ceiling = False
    for j, (fold, name) in enumerate(groups):
        positions = np.arange(4)+5*j
        method = methods[0]
        key = fold, method
        fig.text(.115+.870*(positions.mean()+.6)/9.2, .023, name,
                 ha='center', va='center', fontsize=8, fontweight='bold', color=ink)
        if key not in gaps.index or not np.isfinite(gaps.loc[key].observed_fpr):
            continue
        r = gaps.loc[key]
        projection, observed = 100*np.array([r.projected_fpr, r.observed_fpr], dtype=float)
        part = (contributions.loc[contributions.fold.eq(fold) &
                contributions.method.eq(method)].set_index('subtype')
                if not contributions.empty else pd.DataFrame())
        complete = not part.empty and {'HW','AB'}.issubset(part.index)
        if complete:
            hw, ab = 100*part.loc[['HW','AB'], 'gap_contribution'].to_numpy(float)
            values = [projection, hw, ab, observed]
            starts = [0., projection, projection+hw, 0.]
            names = ['Projection', 'HW rise', 'AB rise', 'Observed']
            colors = [BLUE, TEAL, RUST, '#53636B']
            ids = [f'projected:{fold}:{method}', f'component:{fold}:HW',
                   f'component:{fold}:AB', f'observed:{fold}:{method}']
            for k in range(3):
                level = starts[k]+values[k]
                ax.plot([positions[k]+.32, positions[k+1]-.32], [level, level],
                        color=connector, lw=.75, zorder=1)
        else:
            positions = positions[[0, 3]]
            values, starts = [projection, observed], [0., 0.]
            names, colors = ['Projection','Observed'], [BLUE, '#53636B']
            ids = [f'projected:{fold}:{method}', f'observed:{fold}:{method}']
        for x, value, bottom, label, color, gid in zip(positions, values, starts, names, colors, ids):
            bar, = ax.bar(x, value, bottom=bottom, width=.64, color=color,
                          linewidth=0, zorder=3)
            bar.set_gid(gid)
            top = bottom+value
            heights.extend([bottom, top])
            prefix = '+' if label in ('HW rise','AB rise') and value >= 0 else ''
            ax.annotate(prefix+f'{value:.2f}'.replace('-', '−'), (x, max(bottom, top)),
                        xytext=(0, 3), textcoords='offset points', ha='center',
                        va='bottom', fontsize=8, color=ink)
        tick_positions.extend(positions)
        tick_labels.extend(names)
        if not ceiling.empty:
            q = ceiling.loc[ceiling.fold.eq(fold) & ceiling.method.eq(method)]
            if len(q) == 1:
                maximum = 100*float(q.iloc[0].mean_empirical_mixture_ceiling)
                mark, = ax.plot([5*j-.38, 5*j+3.38], [maximum, maximum],
                                color=ink, lw=.8, ls=(0, (2.4, 2.1)), zorder=4)
                mark.set_gid('source_max:'+fold)
                heights.append(maximum)
                showed_ceiling = True
    lower, upper = min(heights), max(heights)
    span = max(1., upper-lower)
    ax.set(xlim=(-.6, 8.6), ylim=(lower-.025*span, upper+.28*span))
    ax.set_xticks(tick_positions, tick_labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    if showed_ceiling:
        line = Line2D([], [], color=ink, lw=.8, ls=(0, (2.4, 2.1)), label='Max. source rate')
        fig.legend(handles=[line], loc='center right', bbox_to_anchor=(.990, .408),
                   frameon=False, fontsize=8, labelcolor=ink,
                   handlelength=1.8, handletextpad=.5, borderaxespad=0)
    acoustic_panels.add_band(fig, 1, acoustic)
    save(fig, out, 'Fig1')


def site_sensitivity(site):
    """Matched run: average sites within fit, seeds within fold, folds equally."""
    if site.duplicated(['fold','method','seed','site']).any():
        raise ValueError('Duplicate site/fit pAUC input')
    valid=site.dropna(subset=['pauc']); blocks=[]
    for minimum in [1,20]:
        df=valid[valid.negative_windows.ge(minimum)]
        fit=df.groupby(['fold','method','seed']).pauc.mean().rename('macro_pauc')
        q=fit.groupby(['fold','method']).mean().reset_index();q['min_negative']=minimum;blocks.append(q)
    sensitivity=pd.concat(blocks,ignore_index=True)
    return blocks[0].groupby('method').macro_pauc.mean().reset_index(),sensitivity


def figure2_data(tables):
    """Only plot complete matched inputs from one source for both rules."""
    main,sens,changes=[tables[k] for k in ('main_results','support_sensitivity','joint_differences')]
    msg='Figure 2 requires matching pAUC sensitivity and paired differences at 1% and 5% in the selected input directory. Use --figure 1 for transfer-only runs.'
    if any(t.empty for t in (main,sens,changes)): raise ValueError(msg)
    ranked=unique_index(main, 'method').macro_pauc.reindex(METHODS)
    folds=[f for f,_ in FOLDS]; sub=sens[sens.min_negative.eq(20)&sens.fold.isin(folds)].pivot(index='fold',columns='method',values='macro_pauc').reindex(index=folds,columns=METHODS)
    paired=unique_index(changes[changes.a.eq('natural_mixstyle')&changes.b.eq('natural_subtype_mixstyle')&changes.fold.isin(folds)], ['fold','rule']).reindex(pd.MultiIndex.from_product([folds,['fpr_0.01','fpr_0.05']],names=['fold','rule']))
    fields=['delta_fpr','fpr_ci_low','fpr_ci_high','delta_recall','recall_ci_low','recall_ci_high']
    if ranked.isna().any() or sub.isna().any().any() or paired[fields].isna().any().any(): raise ValueError(msg)
    for k in ('fpr','recall'):
        if (paired[k+'_ci_low']>paired[k+'_ci_high']).any(): raise ValueError('Inverted paired difference interval')
    return pd.DataFrame({'all_sites':ranked,'at_least_20':sub.mean()}),paired[fields]


def paired_site_gains(site_metrics):
    """Average matched seeds within sites before contrasting each objective.

    Site points describe site-to-site variation. They do not supply a pooled
    site mean, and seed repetitions do not create additional sites.
    """
    fields = ['fold', 'site', 'method', 'seeds_count', 'negative_windows',
              'positive_windows', 'binary_pauc', 'objective_pauc',
              'delta_pauc', 'meets_20_negatives']
    if site_metrics is None or site_metrics.empty:
        return pd.DataFrame(columns=fields)
    required = {'fold', 'site', 'method', 'seed', 'negative_windows',
                'positive_windows', 'pauc'}
    if not required.issubset(site_metrics.columns):
        raise ValueError('Site gains require complete fit/site pAUC inputs')
    raw = site_metrics.loc[site_metrics.method.isin(METHODS) &
                           site_metrics.fold.isin([f for f, _ in FOLDS])].copy()
    if raw.empty:
        return pd.DataFrame(columns=fields)
    if raw[['fold', 'site', 'method', 'seed']].isna().any().any():
        raise ValueError('Missing identifiers in fit/site pAUC inputs')
    if raw.duplicated(['fold', 'method', 'seed', 'site']).any():
        raise ValueError('Duplicate fit/site pAUC input')
    rows = []
    baseline = next(iter(METHODS))
    for (fold, site), block in raw.groupby(['fold', 'site'], sort=True):
        if set(block.method) != set(METHODS):
            raise ValueError('Site gains require all objectives at each matched site')
        seeds = {m: set(block.loc[block.method.eq(m), 'seed']) for m in METHODS}
        if not seeds[baseline] or any(s != seeds[baseline] for s in seeds.values()):
            raise ValueError('Objectives have unequal seed sets at a matched site')
        for count in ('negative_windows', 'positive_windows'):
            values = block[count].to_numpy(dtype=float)
            if (not np.isfinite(values).all() or (values < 0).any() or
                    (values != np.floor(values)).any() or block[count].nunique() != 1):
                raise ValueError('Objectives or seeds have different site window support')
        negative = int(block.negative_windows.iloc[0])
        positive = int(block.positive_windows.iloc[0])
        if negative == 0 or positive == 0:
            if block.pauc.notna().any():
                raise ValueError('A single-class site cannot supply a pAUC value')
            continue
        values = block.pauc.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Eligible matched site has missing or invalid pAUC')
        means = block.groupby('method').pauc.mean()
        for method in list(METHODS)[1:]:
            rows.append(dict(fold=fold, site=site, method=method,
                seeds_count=len(seeds[baseline]), negative_windows=negative,
                positive_windows=positive, binary_pauc=float(means[baseline]),
                objective_pauc=float(means[method]),
                delta_pauc=float(means[method] - means[baseline]),
                meets_20_negatives=negative >= 20))
    return pd.DataFrame(rows, columns=fields)


def _site_density(values, grid, bandwidth):
    """A shared Gaussian bandwidth gives comparable descriptive silhouettes."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.ptp(values) < 1e-12 or bandwidth <= 0:
        return np.zeros_like(grid)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-.5*z*z).mean(axis=1) / (bandwidth*np.sqrt(2*np.pi))


def figure2(data, out, sites=None, acoustic=None):
    """Site-level gains, macro ranking, and paired operating-point effects."""
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    scores, paired = data
    ink, muted = INK, '#53636B'
    fig = plt.figure(figsize=(6.25, 4.10))

    def frame(ax, left=True):
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_visible(left)
        ax.spines[['bottom', 'left']].set_linewidth(.65)
        ax.spines[['bottom', 'left']].set_color(muted)
        ax.tick_params(axis='both', width=.65, length=2.5, pad=3,
                       labelsize=8, color=muted, labelcolor=ink)
        ax.grid(False)

    panel_title(fig, .015, .966, 'a', 'Site-level gains')
    panel_title(fig, .635, .966, 'b', 'Sensitivity to site support')
    panel_title(fig, .015, .470, 'c', 'False-positive rate')
    panel_title(fig, .535, .470, 'd', 'Killer whale recall')

    a = fig.add_axes([.205, .615, .350, .235], label='site_gain')
    frame(a, left=False)
    methods = list(METHODS)[1:]
    rows = np.arange(len(methods))[::-1]
    a.set(ylim=(-.45, 2.48), yticks=rows)
    a.set_yticklabels([METHODS[m] for m in methods])
    a.tick_params(axis='y', length=0, pad=5)
    a.set_xlabel('ΔpAUC (vs binary)', fontsize=8, color=ink, labelpad=4)
    a.axvline(0, color='#A2AFB5', linewidth=.65, zorder=0)
    if sites is not None and not sites.empty:
        expected = set(methods)
        if (not expected.issubset(set(sites.method)) or
                sites.duplicated(['fold', 'site', 'method']).any()):
            raise ValueError('Site-gain figure needs one paired value per site and objective')
        selected = sites.loc[sites.method.isin(methods)].copy()
        values = selected.delta_pauc.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite paired site gain')
        lo = min(-.05, float(values.min())-.012)
        hi = max(.16, float(values.max())+.012)
        a.set_xlim(lo, hi)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        a.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        grid = np.linspace(lo, hi, 256)
        n_sites = selected[['fold', 'site']].drop_duplicates().shape[0]
        bandwidth = float(np.std(values, ddof=1)) * max(2, n_sites)**(-.2)
        for y, method in zip(rows, methods):
            block = selected.loc[selected.method.eq(method)].sort_values(['fold', 'site'])
            v = block.delta_pauc.to_numpy(dtype=float)
            support = np.unique(np.r_[v.min(), grid[(grid > v.min()) & (grid < v.max())], v.max()])
            density = _site_density(v, support, bandwidth)
            if density.max() > 0:
                shape = .31*density/density.max()
                cloud = a.fill_between(support, y+.045, y+.045+shape,
                                       color=TEAL, alpha=.23, linewidth=0, zorder=1)
                cloud.set_gid('site_gain_cloud:' + method)
                a.plot(support, y+.045+shape, color=TEAL, lw=.65, zorder=2)
            offsets = []
            for r in block.itertuples():
                digest = hashlib.sha256((str(r.fold)+'|'+str(r.site)).encode()).digest()
                offsets.append(-.09 - .14*int.from_bytes(digest[:4], 'big')/(2**32-1))
            y_points = y + np.array(offsets)
            points = a.scatter(v, y_points, s=8.0, facecolor=ink, edgecolor='white',
                               linewidth=.5, alpha=.8, zorder=3)
            points.set_gid('site_gain:' + method)
            low_support = ~block.meets_20_negatives.to_numpy(dtype=bool)
            if low_support.any():
                highlight = a.scatter(v[low_support], y_points[low_support], s=24,
                                      marker='D', facecolor=RUST, edgecolor='white',
                                      linewidth=.6, zorder=4)
                highlight.set_gid('site_gain_low_support:' + method)
        key = [Line2D([], [], marker='o', ls='none', ms=3.0, mec='white',
                      mew=.5, mfc=ink, label=f'{n_sites} paired sites')]
        excluded = selected.loc[~selected.meets_20_negatives,
                                ['fold', 'site', 'negative_windows']].drop_duplicates()
        if len(excluded) == 1:
            r = excluded.iloc[0]
            label = f'{str(r.site).split("::")[-1]} ({int(r.negative_windows)} negatives)'
        else:
            label = '<20 negatives/site'
        if not excluded.empty:
            key.append(Line2D([], [], marker='D', ls='none', ms=4.3,
                              mec=RUST, mfc=RUST, label=label))
        fig.legend(handles=key, loc='center left', bbox_to_anchor=(.105, .906),
                   frameon=False, ncol=2, fontsize=8, labelcolor=ink,
                   handlelength=.8, handletextpad=.4, columnspacing=.9, borderaxespad=0)
    else:
        a.set_xlim(-.05, .15)
        a.set_xticks([-.05, 0, .05, .10, .15])
        a.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # Archived aggregate-only inputs cannot supply a distribution.
        a.text(.50, .50, 'Site-level pairs unavailable', transform=a.transAxes,
               ha='center', va='center', fontsize=8, color=ink)

    b = fig.add_axes([.685, .615, .290, .235], label='site_pauc')
    frame(b)
    fields = [('all_sites', BLUE, 'o', BLUE, 'All eligible sites'),
              ('at_least_20', RUST, 'D', 'white', '≥20 negatives/site')]
    vals = scores.reindex(METHODS)
    lower, upper = float(vals.min().min()), float(vals.max().max())
    span = max(.01, upper-lower)
    b.set(xlim=(-.34, 3.34), ylim=(lower-.25*span, upper+.27*span))
    b.set_xticks(range(4), ['Binary', 'Coarse\nauxiliary', 'Three-\nclass', 'Fine\nauxiliary'])
    b.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    b.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    b.set_ylabel('Standardized pAUC', fontsize=8, color=ink, labelpad=3)
    for field, color, marker, fill, name in fields:
        line, = b.plot(range(4), vals[field].to_numpy(), lw=1.15, color=color,
                       marker=marker, ms=4.1, mfc=fill, mec=color, mew=1.0,
                       label=name, zorder=3)
        line.set_gid('pauc:' + field)
    b.legend(loc='lower left', bbox_to_anchor=(-.01, 1.045), borderaxespad=0,
             frameon=False, fontsize=8, labelcolor=ink, handlelength=1.3,
             handletextpad=.45, labelspacing=.4)

    styles = [('fpr_0.01', .14, BLUE, 'o', 'white', '1% rule'),
              ('fpr_0.05', -.14, RUST, 'D', RUST, '5% rule')]
    key = [Line2D([], [], ls='none', marker=marker, ms=4.0, mec=color,
                  mfc=fill, mew=1.0, label=label)
           for _, _, color, marker, fill, label in styles]
    fig.legend(handles=key, loc='center', bbox_to_anchor=(.560, .423),
               frameon=False, ncol=2, fontsize=8, labelcolor=ink,
               handlelength=.8, handletextpad=.45, columnspacing=1.4)
    bound = max(8., float(np.ceil(100*np.abs(paired.to_numpy()).max())))
    rows = np.arange(len(FOLDS))[::-1]
    for j, metric in enumerate(['fpr', 'recall']):
        ax = fig.add_axes([.155+.505*j, .145, .325-.010*j, .250],
                          label='delta_' + metric)
        frame(ax, left=False)
        ax.set(xlim=(-1.08*bound, 1.08*bound), ylim=(-.5, 2.5),
               xticks=[-bound, -bound/2, 0, bound/2, bound], yticks=rows)
        ax.set_yticklabels([name for _, name in FOLDS])
        ax.tick_params(axis='y', length=0, pad=5)
        ax.set_xlabel('Change (pp)', fontsize=8, color=ink, labelpad=4)
        ax.axvline(0, color='#A2AFB5', linewidth=.65, zorder=1)
        for yy, (fold, _) in zip(rows, FOLDS):
            for rule, offset, color, marker, fill, _ in styles:
                row = 100*paired.loc[fold, rule]
                y = yy+offset
                low, high = row[metric+'_ci_low'], row[metric+'_ci_high']
                ci = ax.hlines(y, low, high, color=color, linewidth=1.05, zorder=2)
                ci.set_gid(f'{metric}:{fold}:{rule}:ci')
                caps = ax.vlines([low, high], y-.080, y+.080,
                                 color=color, linewidth=.8, zorder=2)
                caps.set_gid(f'{metric}:{fold}:{rule}:caps')
                point, = ax.plot(row['delta_'+metric], y, marker=marker, ls='none',
                                 ms=3.8, mec=color, mfc=fill, mew=1.0, zorder=3)
                point.set_gid(f'{metric}:{fold}:{rule}:point')
    fig.text(.560, .033, 'Fine auxiliary − binary', ha='center', va='center',
             fontsize=8, color=ink)
    acoustic_panels.add_band(fig, 2, acoustic)
    save(fig, out, 'Fig2')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tables',type=Path,help='Archived CSV input directory; selects archived aggregates explicitly')
    parser.add_argument('--analysis',type=Path,help='One completed analysis directory or an unambiguous parent')
    parser.add_argument('--figure',choices=['1','2','both','all'],default='both')
    parser.add_argument('--figure-data',type=Path,help='Derived CSV output directory')
    parser.add_argument('--output',type=Path,help='PDF/PNG output directory (standard pair: result/)')
    parser.add_argument('--alpha',type=float,default=.01)
    choose=parser.add_mutually_exclusive_group()
    choose.add_argument('--acoustic-dir',type=Path,help='Verified acoustic features from selected recording windows')
    choose.add_argument('--stats-only',action='store_true',help='Omit illustrative acoustic panels')
    args=parser.parse_args(argv); root=args.root.expanduser().resolve()
    print(f'Renderer: {Path(__file__).resolve()}\nProject root: {root}')
    base=analysis_source(root,args.analysis) if args.analysis is not None else None
    archive_explicit=args.tables is not None
    selected_analysis=base
    retained=root/'result/analysis/transfer_review'
    if args.analysis is None and args.tables is None:
        if not retained.exists():
            raise FileNotFoundError(f'Fixed retained analysis not found: {retained}. Select --tables explicitly for archived aggregates.')
        if not (retained/'analysis.json').is_file():
            raise ValueError(f'Fixed retained run lacks its completion manifest: {retained}. Select --tables explicitly for archived aggregates.')
        selected_analysis=retained
    result=root/'result'
    fixed=selected_analysis==retained
    standard=fixed and not archive_explicit and args.figure in ('both','all') and args.alpha==.01 and not args.stats_only
    if standard:
        figure_default=result/'figures'; data_default=result/'figure_data/derived'
    elif archive_explicit:
        figure_default=result/'figures'; data_default=result/'figure_data/archive_derived'
    elif not fixed and not args.stats_only:
        figure_default=result/'figures'; data_default=selected_analysis/'figure_data'
    else:
        variant=('stats_only' if args.stats_only else 'single_figures' if args.figure in ('1','2') else 'alternate_rule')
        figure_default=result/'figures'
        data_default=result/'figure_data'/variant/'derived'
    out=safe_output(root,resolved(root,args.output,figure_default))
    # A custom export keeps its data and digest record with that export.
    if args.output is not None and out!=figure_default.resolve():
        data_default=out/'figure_data/derived'
    destination=safe_output(root,resolved(root,args.figure_data,data_default))
    if not standard and (out==result.resolve() or destination==(result/'figure_data/derived').resolve()):
        raise ValueError('Reserve the standard output paths for both figures from the fixed transfer analysis')
    acoustic,context=acoustic_panels.context(root,args.acoustic_dir,args.stats_only,selected_analysis)
    if standard and context.get('mode') != 'measured':
        raise ValueError('The standard pair requires verified KW, HW and AB features. '
                         'Install the current code and assets with '
                         '`python orcatca_code_update.zip --target /path/to/orcatca`, '
                         'then rerun this script. Use --stats-only for a separate statistical export.')
    tables=load_tables(root,args.tables,selected_analysis,args.alpha)
    tables['evidence']['acoustic_context']=context
    code_root=Path(__file__).resolve().parents[1]
    tables['evidence']['renderer']={
        'layout':check_figures.LAYOUT if acoustic is not None else 'statistics-only',
        'output_directory':out.relative_to(root).as_posix() if out.is_relative_to(root) else str(out),
        'code_sha256':{n:hashlib.sha256((code_root/n).read_bytes()).hexdigest()
                       for n in check_figures.CODE_FILES},
        'environment':{'python':platform.python_version(),'matplotlib':matplotlib.__version__,
                       'numpy':np.__version__,'pandas':pd.__version__}}
    fig2=figure2_data(tables) if args.figure in ('2','both','all') else None
    site_gains = paired_site_gains(tables['site_metrics']) if fig2 is not None else None
    destination.mkdir(parents=True,exist_ok=True)
    for name,key in [('label_error_rates','labels'),('transfer_gap_estimates','gap'),('original_operating_summary','original'),('label_gap_contributions','contributions')]:
        tables[key].to_csv(destination/(name+'.csv'),index=False)
    if not tables['source_labels'].empty: tables['source_labels'].to_csv(destination/'source_check_label_rates.csv',index=False)
    appearance()
    if args.figure in ('1','both','all'):
        full_label_contrasts(tables['labels']).to_csv(destination/'full_label_contrasts.csv',index=False)
        tables['configuration_points'].to_csv(destination/'transfer_configuration_points.csv',index=False)
        figure1(tables,out,args.alpha,acoustic=acoustic)
    if fig2 is not None:
        fig2[0].to_csv(destination/'ranking_sensitivity.csv'); fig2[1].to_csv(destination/'paired_operating_differences.csv')
        gain = pauc_gain(fig2[0]).rename_axis('site_selection').rename('fine_minus_binary_pauc').to_frame()
        gain['pauc_difference_times_100'] = 100 * gain.fine_minus_binary_pauc
        gain.to_csv(destination/'pauc_gain.csv')
        site_gains.to_csv(destination/'paired_site_gains.csv',index=False)
        figure2(fig2,out,sites=site_gains,acoustic=acoustic)
    numbers = (1,2) if args.figure in ('both','all') else (int(args.figure),)
    tables['evidence']['figure_sha256']={f'Fig{n}.{ext}': hashlib.sha256((out/f'Fig{n}.{ext}').read_bytes()).hexdigest()
                                        for n in numbers for ext in ('pdf','png')}
    (destination/'available_evidence.json').write_text(json.dumps(tables['evidence'],indent=2,allow_nan=False)+'\n')
    print(f'Inputs: {tables["evidence"]["input_directory"]}\nDerived tables: {destination}')
    print('Layout:',tables['evidence']['renderer']['layout'])
    if acoustic is not None:
        print('Acoustic source:',acoustic['site'])
        print('Acoustic assets:',(root/context['asset_directory']).resolve())
        if context.get('previous_generated_bundle_skipped'):
            print('Using updated bundled features; the recognized earlier local features were left in place.')
    for n in numbers:
        for ext in ('pdf','png'):
            print('Written:',out/f'Fig{n}.{ext}')
    if out==(result/'figures').resolve():
        print('Check: python scripts/check_figures.py --root "'+str(root)+'"')


if __name__=='__main__': main()
