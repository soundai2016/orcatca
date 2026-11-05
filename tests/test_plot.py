"""Figure inputs cannot create independent windows or missing uncertainty."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('orcatca_plot', ROOT/'scripts/plot.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def rich_fixture(folder, alpha=.05, completed=True):
    """Small fixture for the aggregated-input path."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'analysis.json').write_text(json.dumps({'status':'completed' if completed else 'running',
        'crossfit_status':{'fold00':'ready'},'methods':['natural_mixstyle'],'seeds':[17,29],
        'crossfit_repeats':1}))
    op=[]; tr=[]
    for seed in [17,29]:
        for side in [0,1]:
            fields=dict(fold='fold00',method='natural_mixstyle',seed=seed,repeat=0,side=side,
                        alpha=alpha,threshold=.4+.1*side)
            for role,h,a in [('source_fit',0,0),('source_check',0,2),('target',5,5)]:
                op.append(dict(**fields,role=role,HW_n=50,AB_n=50,HW_fp=h,AB_fp=a,
                               HW_fpr=h/50,AB_fpr=a/50,fp=h+a,negative=100,fpr=(h+a)/100))
            tr.append(dict(**fields,status='estimated_descriptively',delta=.08,mixture_only=.02,
                           source_reference_mixture=0,target_weighted_conditional=.08))
    pd.DataFrame(op).to_csv(folder/'crossfit_operating.csv',index=False)
    pd.DataFrame(tr).to_csv(folder/'crossfit_transfer.csv',index=False)
    gap = [dict(fold='fold00',method='natural_mixstyle',alpha=alpha,status='estimated',
                projected_fpr=.02,observed_fpr=.10,transfer_gap=.08,
                gap_ci_low=.04,gap_ci_high=.12,gap_ci_status='conditional_percentile')]
    pd.DataFrame(gap).to_csv(folder/'transfer_gap_summary.csv',index=False)
    labels=[]
    for seed in (17,29):
        for cls,n,e in [('HW',10,1),('AB',20,3),('KW',30,24)]:
            labels.append(dict(fold='fold00',method='natural_mixstyle',alpha=alpha,
                procedure='full_calibration',seed=seed,role='target',subtype=cls,
                windows=n,errors=e,recordings=2,units=2,label_rate=e/n))
    pd.DataFrame(labels).to_csv(folder/'transfer_original_labels.csv',index=False)
    pd.DataFrame([dict(fold='fold00',method='natural_mixstyle',alpha=alpha,
        procedure='full_calibration',status='estimated',target_fpr=4/30,
        target_recall=.8)]).to_csv(folder/'transfer_original_summary.csv',index=False)


class PlotInputTests(unittest.TestCase):
    def test_archived_seed_denominators_are_not_unique_windows(self):
        result = plot.load_tables(ROOT)
        data = result['labels']
        r = data[(data.fold=='fold00') & (data.method=='natural_mixstyle') & (data.raw_class=='HW')].iloc[0]
        self.assertEqual(r.windows,100)
        self.assertEqual(r.prediction_count,300)
        self.assertEqual(r.recordings,2)
        self.assertAlmostEqual(r.label_rate,5/300)
        a = data[(data.fold=='fold00') & (data.method=='natural_mixstyle') & (data.raw_class=='AB')].iloc[0]
        self.assertAlmostEqual(a.label_rate,(41-5)/156)
        self.assertFalse(result['evidence']['repeated_seed_predictions_are_independent_windows'])

    def test_archive_gap_is_paired_and_intervals_stay_missing(self):
        result=plot.load_tables(ROOT)
        data=result['gap']; valid=data.transfer_gap.notna()
        np.testing.assert_allclose(data.loc[valid,'transfer_gap'],
                                   data.loc[valid,'observed_fpr']-data.loc[valid,'projected_fpr'])
        self.assertTrue(data.gap_ci_low.isna().all())
        self.assertFalse(result['evidence']['gap_ci_available'])
        self.assertTrue(result['ceiling'].empty)
        self.assertTrue(data[data.fold=='fold01'].status.eq('not_estimable').all())

    def test_absolute_recall_is_not_invented_from_paired_differences(self):
        result=plot.load_tables(ROOT)
        self.assertTrue(result['original'].target_recall.isna().all())
        self.assertFalse(result['evidence']['absolute_recall_available'])

    def test_a_single_extra_pooled_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'tables'
            shutil.copytree(ROOT/'result/figure_data',folder)
            path=folder/'false_positive_profile.csv'; table=pd.read_csv(path)
            i=table.index[table.fold.eq('fold01')][0]
            table.loc[i,'HW_n']+=1; table.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'denominator'):
                plot.load_tables(ROOT,tables=folder)

    def test_archive_only_supports_reported_one_percent_rule(self):
        with self.assertRaisesRegex(ValueError,'alpha=0.01'):
            plot.load_tables(ROOT,alpha=.05)

    def test_completed_rich_analysis_is_isolated_and_uses_its_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'analysis'; rich_fixture(folder)
            result=plot.load_tables(ROOT,analysis=folder,alpha=.05)
            self.assertTrue(result['main_results'].empty)
            self.assertTrue(result['support_sensitivity'].empty)
            self.assertTrue(result['joint_differences'].empty)
            self.assertTrue(result['evidence']['gap_ci_available'])
            self.assertTrue(result['evidence']['absolute_recall_available'])
            self.assertEqual(set(result['evidence']['input_sha256']),
                             {'analysis.json','crossfit_operating.csv','crossfit_transfer.csv', *(n+'.csv' for n in plot.RICH)})
            self.assertAlmostEqual(result['ceiling'].iloc[0].mean_empirical_mixture_ceiling,.04)
            hw=result['labels'][result['labels'].raw_class.eq('HW')].iloc[0]
            self.assertEqual(hw.windows,10)
            self.assertEqual(hw.prediction_count,20)
            self.assertAlmostEqual(hw.label_rate,.1)

    def test_summary_must_match_paired_ceiling_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); rich_fixture(folder)
            path=folder/'transfer_gap_summary.csv'; g=pd.read_csv(path)
            g['observed_fpr']+=.01; g['transfer_gap']+=.01; g.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'different transfer rates'):
                plot.load_tables(ROOT,analysis=folder,alpha=.05)
            (folder/'crossfit_operating.csv').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'Empirical mixture ceiling'):
                plot.load_tables(ROOT,analysis=folder,alpha=.05)

    def test_two_completed_analyses_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); rich_fixture(base/'a'); rich_fixture(base/'b')
            with self.assertRaisesRegex(ValueError,'found 2'):
                plot.analysis_source(ROOT,base)
            self.assertEqual(plot.analysis_source(ROOT,base/'a'),base/'a')

    def test_incomplete_analysis_does_not_fall_back_to_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); rich_fixture(folder,completed=False)
            with self.assertRaisesRegex(ValueError,'not completed'):
                plot.load_tables(ROOT,analysis=folder,alpha=.05)

    def test_missing_rich_table_does_not_fall_back_to_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); rich_fixture(folder)
            (folder/'transfer_original_labels.csv').unlink()
            with self.assertRaisesRegex(FileNotFoundError,'lacks required'):
                plot.load_tables(ROOT,analysis=folder,alpha=.05)

    def test_rich_gap_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); rich_fixture(folder)
            path=folder/'transfer_gap_summary.csv'; data=pd.read_csv(path)
            data['transfer_gap']=.9; data.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'paired rates'):
                plot.load_tables(ROOT,analysis=folder,alpha=.05)

    def test_source_modes_cannot_be_combined(self):
        with self.assertRaisesRegex(ValueError,'either'):
            plot.load_tables(ROOT,tables=Path('a'),analysis=Path('b'))

    def test_output_cannot_write_raw_audio_or_through_a_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'dclde_2027_killer_whales'; raw.mkdir()
            link=root/'unsafe'; link.symlink_to(raw,target_is_directory=True)
            for path in (raw,raw/'figures',root,link/'output'):
                with self.assertRaisesRegex(ValueError,'raw corpus'):
                    plot.safe_output(root,path)
            self.assertEqual(plot.safe_output(root,root/'result'),root/'result')


class MainFigureDesignTests(unittest.TestCase):
    def test_full_label_ratio_uses_unrounded_target_rates_and_unique_support(self):
        rates = plot.load_tables(ROOT)['labels']
        contrasts = plot.full_label_contrasts(rates).set_index(['fold', 'method'])
        sio = contrasts.loc[('fold00', 'natural_mixstyle')]
        # The same 100 HW and 52 AB windows were each scored three times.
        self.assertAlmostEqual(sio.ab_to_hw_ratio, (36/156)/(5/300))
        self.assertEqual((sio.hw_windows, sio.ab_windows, sio.hw_recordings,
                          sio.ab_recordings), (100, 52, 2, 17))
        self.assertEqual(sio.ratio_status, 'finite')

    def test_zero_and_missing_hw_are_not_treated_as_large_finite_ratios(self):
        rates = plot.load_tables(ROOT)['labels']
        rates = rates.loc[rates.fold.eq('fold00') & rates.method.eq('natural_mixstyle')].copy()
        hw, ab = rates.raw_class.eq('HW'), rates.raw_class.eq('AB')
        rates.loc[hw,'label_rate'] = 0
        divided = plot.full_label_contrasts(rates).iloc[0]
        self.assertTrue(np.isposinf(divided.ab_to_hw_ratio))
        self.assertEqual(divided.ratio_status, 'zero_hw')
        rates.loc[ab,'label_rate'] = 0
        self.assertEqual(plot.full_label_contrasts(rates).iloc[0].ratio_status, 'both_zero')
        absent = plot.full_label_contrasts(rates[ab]).iloc[0]
        self.assertTrue(np.isnan(absent.ab_to_hw_ratio))
        self.assertEqual(absent.ratio_status, 'missing_label')

    def test_contrasts_reject_shared_support_when_recordings_disagree_by_method(self):
        rates = plot.load_tables(ROOT)['labels'].copy()
        i = rates.index[rates.raw_class.eq('AB') & rates.fold.eq('fold00')][0]
        rates.loc[i,'recordings'] += 1
        with self.assertRaisesRegex(ValueError, 'different target support'):
            plot.full_label_contrasts(rates)

    def test_sensitivity_can_be_rebuilt_from_matching_fit_level_sites(self):
        archived=plot.load_tables(ROOT); a,b=plot.figure2_data(archived)
        site=pd.read_csv(ROOT/'result/figure_data/site_metrics.csv')
        main,sens=plot.site_sensitivity(site)
        same,_=plot.figure2_data({**archived,'main_results':main,'support_sensitivity':sens})
        np.testing.assert_allclose(same,a,rtol=0,atol=1e-12)
        self.assertEqual(len(b),6)
        self.assertLess(b.loc[('fold02','fpr_0.01'),'delta_fpr'],0)
        self.assertGreater(b.loc[('fold02','fpr_0.05'),'fpr_ci_low'],0)

    def test_rich_only_transfer_does_not_create_partial_or_mixed_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);rich_fixture(directory)
            with self.assertRaisesRegex(ValueError,'Use --figure 1'):
                plot.main(['--analysis',str(directory),'--alpha','.05','--figure','both'])
            self.assertFalse((directory/'figures').exists())
            plot.main(['--analysis',str(directory),'--alpha','.05','--figure','1'])
            self.assertTrue((directory/'figures/Fig1.pdf').is_file())
            self.assertFalse((directory/'figures/Fig2.pdf').is_file())

    def test_run_default_prefers_fixed_run_and_preserves_its_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'result/analysis/transfer_review'
            shutil.copytree(ROOT/'result/analysis/transfer_review',run)
            shutil.copytree(ROOT/'result/figure_data',root/'result/figure_data')
            shutil.copytree(ROOT/'assets',root/'assets')
            plot.main(['--root',str(root)])
            result=root/'result'
            data=result/'figure_data/derived/available_evidence.json'
            metadata=json.loads(data.read_text())
            self.assertEqual(metadata['input_directory'],'result/analysis/transfer_review')
            names={f'Fig{i}.{ext}' for i in (1,2) for ext in ('pdf','png')}
            self.assertEqual(set(metadata['figure_sha256']),names)
            self.assertTrue(all((result/n).is_file() for n in names))
            self.assertFalse((result/'figures').exists())
            # Explicitly naming the same run has the same destination.
            plot.main(['--root',str(root),'--analysis',str(run)])
            self.assertEqual(json.loads(data.read_text()),metadata)
            self.assertFalse((run/'figures').exists())
            retained={n:(result/n).stat().st_mtime_ns for n in names}
            for options,folder in [(['--figure','1'],'single_figures'),
                                   (['--stats-only'],'stats_only'),
                                   (['--tables','result/figure_data','--figure','1'],'archive_figures')]:
                plot.main(['--root',str(root),*options])
                self.assertTrue((result/folder/'Fig1.pdf').is_file())
                self.assertEqual(json.loads(data.read_text()),metadata)
                self.assertEqual({n:(result/n).stat().st_mtime_ns for n in names},retained)
            with self.assertRaisesRegex(ValueError,'Reserve'):
                plot.main(['--root',str(root),'--stats-only','--output','result'])

    def test_missing_default_run_requires_explicit_archive_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            shutil.copytree(ROOT/'result/figure_data',root/'result/figure_data')
            with self.assertRaisesRegex(FileNotFoundError,'Select --tables explicitly'):
                plot.main(['--root',str(root),'--figure','1'])

    def test_missing_acoustic_features_do_not_replace_the_standard_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=root/'result'
            shutil.copytree(ROOT/'result/analysis/transfer_review',result/'analysis/transfer_review')
            names=[f'Fig{i}.{ext}' for i in (1,2) for ext in ('pdf','png')]
            for name in names:
                (result/name).write_bytes(b'previous complete export')
            with self.assertRaisesRegex(ValueError,'verified KW, HW and AB'):
                plot.main(['--root',str(root)])
            self.assertTrue(all((result/n).read_bytes()==b'previous complete export' for n in names))
            self.assertFalse((result/'figure_data').exists())

    def test_partial_fixed_run_fails_even_if_archive_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'result/analysis/transfer_review';run.mkdir(parents=True)
            shutil.copytree(ROOT/'result/figure_data',root/'result/figure_data')
            with self.assertRaisesRegex(ValueError,'lacks its completion manifest'):
                plot.main(['--root',str(root),'--figure','1'])
            rich_fixture(run,completed=False)
            with self.assertRaisesRegex(ValueError,'not completed'):
                plot.main(['--root',str(root),'--figure','1'])

    def test_figure2_groups_marginal_intervals_by_metric_and_collection(self):
        # Changing the panel mapping must not swap FPR/recall, source rules,
        # or their marginal CI endpoints in a figure without printed bounds.
        run = ROOT/'result/analysis/transfer_review'
        tables = plot.load_tables(ROOT, analysis=run)
        scores, paired = plot.figure2_data(tables)
        sites = plot.paired_site_gains(tables['site_metrics'])
        with patch.object(plot, 'save') as save:
            plot.figure2((scores, paired), Path('.'), sites=sites)
        fig = save.call_args.args[0]
        try:
            artists = {a.get_gid(): a for a in fig.findobj() if a.get_gid()}
            for method in list(plot.METHODS)[1:]:
                cloud = artists['site_gain:'+method].get_offsets()
                expected = sites[sites.method.eq(method)].sort_values(['fold','site'])
                np.testing.assert_allclose(cloud[:,0],expected.delta_pauc)
            for metric in ['fpr', 'recall']:
                ax, = [a for a in fig.axes if a.get_label() == 'delta_' + metric]
                ci_keys = [g for g in artists if g.startswith(metric + ':') and g.endswith(':ci')]
                self.assertEqual(len(ci_keys), 6)
                for fold,_ in plot.FOLDS:
                    for j,rule in enumerate(['fpr_0.01','fpr_0.05']):
                        r = paired.loc[fold, rule]
                        key = f'{metric}:{fold}:{rule}:'
                        interval = artists[key+'ci']
                        self.assertIs(interval.axes, ax)
                        segment, = interval.get_segments()
                        np.testing.assert_allclose(segment[:,0], 100*r[[metric+'_ci_low',metric+'_ci_high']])
                        self.assertAlmostEqual(segment[0,1], segment[1,1])
                        symbol = 'o' if j == 0 else 'D'
                        pt = artists[key+'point']
                        self.assertEqual(pt.get_marker(), symbol)
                        self.assertAlmostEqual(pt.get_xdata()[0], 100*r['delta_'+metric])
                        self.assertAlmostEqual(pt.get_ydata()[0], segment[0,1])
        finally:
            plot.plt.close(fig)

    def test_transfer_figure_keeps_half_calibration_rates_and_paired_gap_together(self):
        # Original full-pool rates are different: Scripps binary is 8.99% there,
        # versus 9.25% at the matched half-pool thresholds used by this figure.
        run = ROOT/'result/analysis/transfer_review'
        expected = pd.read_csv(run/'transfer_gap_summary.csv')
        expected = expected.loc[np.isclose(expected.alpha, .01)].dropna(subset=['transfer_gap'])
        tables = plot.load_tables(ROOT, analysis=run)
        with patch.object(plot, 'save') as save:
            plot.figure1(tables, Path('.'), .01)
        fig = save.call_args.args[0]
        try:
            artists = {a.get_gid(): a for a in fig.findobj() if a.get_gid()}
            self.assertEqual(len([g for g in artists if g.startswith('transfer:') and g.endswith(':ci')]), 8)
            for r in expected.itertuples():
                suffix = f'{r.fold}:{r.method}'
                if r.method == 'natural_mixstyle':
                    for kind, value in [('projected', r.projected_fpr), ('observed', r.observed_fpr)]:
                        self.assertAlmostEqual(artists[kind+':'+suffix].get_height(), 100*value)
                        self.assertAlmostEqual(artists[kind+':'+suffix].get_y(), 0)
                segment, = artists['transfer:'+suffix+':ci'].get_segments()
                np.testing.assert_allclose(segment[:,0], 100*np.array([r.gap_ci_low, r.gap_ci_high]))
                self.assertAlmostEqual(artists['transfer:'+suffix+':point'].get_xdata()[0], 100*r.transfer_gap)
                points = tables['configuration_points']
                expected_cloud = points[points.fold.eq(r.fold) & points.method.eq(r.method)]
                cloud = artists['configuration:'+suffix].get_offsets()
                np.testing.assert_allclose(np.sort(cloud[:,0]), np.sort(100*expected_cloud.transfer_gap))
        finally:
            plot.plt.close(fig)

    def test_one_missing_rule_in_one_collection_is_not_a_complete_figure(self):
        data=plot.load_tables(ROOT);d=data['joint_differences'];
        data['joint_differences']=d[~(d.fold.eq('fold02') & d.rule.eq('fpr_0.05'))]
        with self.assertRaisesRegex(ValueError,'1% and 5%'): plot.figure2_data(data)

    def test_configuration_cloud_uses_target_weighted_gaps_and_matched_grid(self):
        run = ROOT/'result/analysis/transfer_review'
        data = plot.load_tables(ROOT, analysis=run)
        rows = data['configuration_points']
        self.assertEqual(len(rows), 240)
        self.assertTrue(rows.groupby(['fold','method']).size().eq(30).all())
        self.assertFalse(data['evidence']['configuration_rows_are_independent'])
        raw = pd.read_csv(run/'crossfit_transfer.csv')
        raw = raw[np.isclose(raw.alpha,.01)]
        key = ['fold','method','seed','repeat','side','alpha']
        joined = rows.merge(raw, on=key, validate='one_to_one')
        np.testing.assert_allclose(joined.transfer_gap, joined.target_weighted_conditional, atol=1e-12)
        self.assertFalse(np.allclose(joined.transfer_gap, joined.delta))
        np.testing.assert_allclose(rows.HW_contribution+rows.AB_contribution,rows.transfer_gap,atol=1e-12)

    def test_configuration_cloud_rejects_unweighted_substitution_and_mispaired_cutoff(self):
        run = ROOT/'result/analysis/transfer_review'
        op = pd.read_csv(run/'crossfit_operating.csv')
        tr = pd.read_csv(run/'crossfit_transfer.csv')
        summary = pd.read_csv(run/'transfer_gap_summary.csv')
        summary = summary[np.isclose(summary.alpha,.01)]
        wrong = tr.copy();wrong['target_weighted_conditional'] = wrong.delta
        with self.assertRaisesRegex(ValueError,'target-weighted'):
            plot.transfer_configurations(op,wrong,.01,summary)
        wrong = tr.copy();wrong.loc[wrong.index[0],'threshold'] += .01
        with self.assertRaisesRegex(ValueError,'different thresholds'):
            plot.transfer_configurations(op,wrong,.01,summary)

    def test_site_cloud_uses_paired_seed_means_and_preserves_macro_weighting(self):
        run = ROOT/'result/analysis/transfer_review'
        data = plot.load_tables(ROOT, analysis=run)
        sites = plot.paired_site_gains(data['site_metrics'])
        self.assertEqual(len(sites),42)
        self.assertTrue(sites.groupby('method').size().eq(14).all())
        self.assertTrue(sites.seeds_count.eq(3).all())
        self.assertEqual(set(sites.loc[~sites.meets_20_negatives,'site']),{'SIO::Quin_Can'})
        fine = sites[(sites.method=='natural_subtype_mixstyle') & (sites.site=='SIO::Quin_Can')].iloc[0]
        self.assertAlmostEqual(fine.delta_pauc,.13921901528013592,places=10)
        scores,_ = plot.figure2_data(data)
        for method in sites.method.unique():
            for field,selected in [('all_sites',sites),('at_least_20',sites[sites.meets_20_negatives])]:
                balanced = selected[selected.method.eq(method)].groupby('fold').delta_pauc.mean().mean()
                self.assertAlmostEqual(balanced,scores.loc[method,field]-scores.loc['natural_mixstyle',field])

    def test_site_cloud_rejects_mismatched_seed_and_window_support(self):
        raw = pd.read_csv(ROOT/'result/analysis/transfer_review/site_metrics.csv')
        eligible = raw.positive_windows.gt(0) & raw.negative_windows.gt(0)
        first = raw.index[eligible][0]
        with self.assertRaises(ValueError):
            plot.paired_site_gains(raw.drop(index=first))
        wrong = raw.copy();wrong.loc[first,'negative_windows'] += 1
        with self.assertRaises(ValueError):
            plot.paired_site_gains(wrong)


if __name__=='__main__': unittest.main()
