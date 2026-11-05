"""Numerical checks of dependence, identities, ties, and unsupported tails.

Synthetic inputs exercise calibration and resampling behavior.
Run: PYTHONPATH=src python -m unittest discover -s tests -p 'test_transfer.py'
"""
import unittest
import numpy as np
import csv
import gzip
import json
from pathlib import Path
import tempfile
import zipfile
import yaml

from orcatca.transfer import (crossfit_transfer, full_calibration_outputs,
    threshold_diagnostics, operating_rates, safe_interval, linked_units)


def fixture():
    cal, target = [], []
    # Every linked unit contains both negative labels. A source half has two
    # units, and its maximum scores differ from those of the opposite half.
    for prefix, rows, n in [('s', cal, 8), ('t', target, 6)]:
        for block in range(n):
            for label in ('AB', 'HW', 'KW'):
                for window in range(3):
                    rows.append(dict(id=f'{prefix}{block}-{label}-{window}',
                        raw_class=label, label=int(label == 'KW'),
                        site=prefix+'site', recording=f'{prefix}rec{block}',
                        block=f'{prefix}block{block}'))
    rng = np.random.default_rng(83)
    cp = rng.uniform(.05, .95, (2, 2, len(cal)))
    tp = rng.uniform(.1, 1., (2, 2, len(target)))
    plans = []
    for repeat, left_blocks in enumerate(({0, 1, 2, 3}, {0, 2, 4, 6})):
        left = [i for i, r in enumerate(cal) if int(r['block'].replace('sblock','')) in left_blocks]
        right = sorted(set(range(len(cal))) - set(left))
        plans.append(dict(repeat=repeat, left=left, right=right))
    plan = dict(status='ready', plans=plans)
    return cal, cp, target, tp, plan


class ThresholdTests(unittest.TestCase):
    def test_tie_group_and_zero_allowance(self):
        rows = [dict(id=str(i), recording='r'+str(i), block='b'+str(i),
                     site='s', label=0, raw_class=c)
                for i, c in enumerate(['AB']*4 + ['HW']*4)]
        scores = np.asarray([.9, .9, .4, .1, .8, .7, .6, .2])
        threshold, detail = threshold_diagnostics(rows, scores, .25)
        self.assertEqual(threshold, np.nextafter(.9, np.inf))
        self.assertEqual(detail[0]['boundary_ties'], 2)
        self.assertEqual(detail[0]['allowed_errors'], 1)
        self.assertEqual(detail[0]['selected_errors'], 0)
        threshold, detail = threshold_diagnostics(rows, scores, .01)
        self.assertTrue(all(d['zero_allowed_errors'] for d in detail))
        self.assertEqual(operating_rates(rows, scores, threshold)['fp'], 0)

    def test_threshold_can_exceed_one_without_clipping(self):
        rows = [dict(id=c, recording=c, site='s', label=0, raw_class=c)
                for c in ('AB', 'HW')]
        threshold, detail = threshold_diagnostics(rows, np.ones(2), .01)
        self.assertGreater(threshold, 1)
        self.assertTrue(all(d['selected_errors'] == 0 for d in detail))

    def test_linked_recordings_union_different_blocks(self):
        rows = [dict(recording='r1', block='b1'), dict(recording='r1', block='b2'),
                dict(recording='r2', block='b2')]
        self.assertEqual(len(linked_units(rows)), 1)


class TransferTests(unittest.TestCase):
    def run_analysis(self, fixture_data=None, repeats=80):
        cal, cp, target, tp, plan = fixture_data or fixture()
        return crossfit_transfer('fold00', cal, cp, target, tp,
            ['binary', 'auxiliary'], [17, 29], plan, .2, repeats=repeats, seed=71)

    def test_paired_decomposition_identity_against_direct_calculation(self):
        cal, cp, target, tp, plan = fixture()
        result = self.run_analysis((cal, cp, target, tp, plan))
        for m, summary in enumerate(result['summary']):
            projected, observed = [], []
            for item in plan['plans']:
                for side in (0, 1):
                    fit = item['left' if side == 0 else 'right']
                    check = item['right' if side == 0 else 'left']
                    for s in range(2):
                        cut, _ = threshold_diagnostics([cal[i] for i in fit], cp[m, s, fit], .2)
                        sr = operating_rates([cal[i] for i in check], cp[m, s, check], cut)
                        tr = operating_rates(target, tp[m, s], cut)
                        projection = sum(tr[c+'_n'] / tr['negative'] * sr[c+'_fpr']
                                         for c in ('AB', 'HW'))
                        conditional = sum(tr[c+'_n'] / tr['negative'] * (tr[c+'_fpr'] - sr[c+'_fpr'])
                                          for c in ('AB', 'HW'))
                        self.assertAlmostEqual(tr['fpr'] - projection, conditional, places=13)
                        projected.append(projection); observed.append(tr['fpr'])
            self.assertAlmostEqual(summary['projected_fpr'], np.mean(projected), places=13)
            self.assertAlmostEqual(summary['observed_fpr'], np.mean(observed), places=13)
            self.assertAlmostEqual(summary['transfer_gap'], np.mean(np.asarray(observed)-projected), places=13)
        self.assertTrue(result['audit']['source_draw_reused_across_partitions'])
        self.assertFalse(result['audit']['seeds_resampled'])

    def test_repeating_partitions_does_not_narrow_intervals(self):
        data = fixture()
        first = self.run_analysis(data)
        cal, cp, target, tp, plan = data
        duplicated = dict(plan, plans=plan['plans'] + plan['plans'])
        second = self.run_analysis((cal, cp, target, tp, duplicated))
        for a, b in zip(first['summary'], second['summary']):
            self.assertEqual(a['source_units'], b['source_units'])
            self.assertEqual(a['target_units'], b['target_units'])
            for key in ('projected_fpr', 'observed_fpr', 'transfer_gap', 'gap_ci_low', 'gap_ci_high'):
                self.assertAlmostEqual(a[key], b[key], places=12)

    def test_identical_methods_use_shared_draws(self):
        cal, cp, target, tp, plan = fixture()
        cp[1] = cp[0]; tp[1] = tp[0]
        result = self.run_analysis((cal, cp, target, tp, plan))
        a, b = result['summary']
        for key in ('projected_fpr', 'observed_fpr', 'transfer_gap', 'gap_ci_low', 'gap_ci_high'):
            self.assertEqual(a[key], b[key])

    def test_no_plan_and_missing_target_label_are_not_estimable(self):
        cal, cp, target, tp, plan = fixture()
        none = self.run_analysis((cal, cp, target, tp, dict(plans=[], reason='insufficient_units')))
        self.assertTrue(all(r['status'] == 'not_estimable' for r in none['summary']))
        keep = [i for i, row in enumerate(target) if row['raw_class'] != 'HW']
        result = self.run_analysis((cal, cp, [target[i] for i in keep], tp[:, :, keep], plan))
        self.assertEqual(result['summary'][0]['reason'], 'missing_required_target_label')
        self.assertNotIn('gap_ci_low', result['summary'][0])

    def test_boundary_does_not_emit_zero_width_interval(self):
        cal, cp, target, tp, plan = fixture()
        cp[:] = .5
        result = self.run_analysis((cal, cp, target, tp, plan))
        for summary in result['summary']:
            self.assertEqual(summary['projected_fpr'], 0)
            self.assertIsNone(summary['gap_ci_low'])
            self.assertIsNone(summary['projected_ci_high'])
            self.assertEqual(summary['gap_ci_status'], 'boundary_empirical_distribution')
        self.assertEqual(safe_interval([.3]*50, .3)[2], 'degenerate_empirical_distribution')

    def test_single_zero_half_does_not_hide_estimable_interval(self):
        result = self.run_analysis()
        # In this fixture some individual source checks have no AB errors,
        # while the method-level source AB error distribution is non-boundary.
        self.assertTrue(all(row['gap_ci_status'] == 'conditional_percentile'
                            for row in result['summary']))

    def test_leave_one_unit_matches_manual_removal(self):
        cal, cp, target, tp, plan = fixture()
        result = self.run_analysis((cal, cp, target, tp, plan))
        self.assertTrue(result['influence'])
        for row in result['influence']:
            if row['status'] == 'estimated':
                self.assertAlmostEqual(row['gap_without_block']-row['full_gap'], row['gap_change'])
        self.assertEqual({r['role'] for r in result['influence']}, {'source_check', 'target'})
        for role in ('source_check', 'target'):
            selected = next(row for row in result['influence']
                            if row['role'] == role and row['method'] == 'binary'
                            and row['status'] == 'estimated')
            population = cal if role == 'source_check' else target
            deleted = set(next(ids.tolist() for key, ids in linked_units(population)
                               if key == selected['block']))
            gaps = []
            for item in plan['plans']:
                for side in (0, 1):
                    fit = item['left' if side == 0 else 'right']
                    check = item['right' if side == 0 else 'left']
                    remaining_check = [i for i in check if role != 'source_check' or i not in deleted]
                    remaining_target = [i for i in range(len(target)) if role != 'target' or i not in deleted]
                    for s in range(2):
                        cut, _ = threshold_diagnostics([cal[i] for i in fit], cp[0, s, fit], .2)
                        sr = operating_rates([cal[i] for i in remaining_check], cp[0, s, remaining_check], cut)
                        tr = operating_rates([target[i] for i in remaining_target], tp[0, s, remaining_target], cut)
                        projection = sum(tr[c+'_n']/tr['negative'] * sr[c+'_fpr'] for c in ('AB','HW'))
                        gaps.append(tr['fpr'] - projection)
            self.assertAlmostEqual(selected['gap_without_block'], np.mean(gaps), places=13)

    def test_missing_label_bootstrap_draws_are_rejected_and_reported(self):
        cal, cp, target, tp, plan = fixture()
        keep = [i for i, row in enumerate(target)
                if (row['block'] == 'tblock0' and row['raw_class'] == 'AB')
                or (row['block'] == 'tblock1' and row['raw_class'] == 'HW')
                or (row['block'] in ('tblock2','tblock3') and row['raw_class'] == 'KW')]
        result = self.run_analysis((cal, cp, [target[i] for i in keep], tp[:, :, keep], plan))
        self.assertEqual(result['audit']['bootstrap_valid'], 80)
        self.assertGreater(result['audit']['bootstrap_rejected'], 0)
        self.assertTrue(any(row['status'] == 'not_estimable_after_removal'
                            for row in result['influence'] if row['role'] == 'target'))

    def test_no_valid_block_removal_has_missing_influence_maximum(self):
        cal, cp, target, tp, plan = fixture()
        keep = [i for i, row in enumerate(target)
                if (row['block'] == 'tblock0' and row['raw_class'] == 'AB')
                or (row['block'] == 'tblock1' and row['raw_class'] == 'HW')
                or (row['block'] == 'tblock2' and row['raw_class'] == 'KW')]
        result = self.run_analysis((cal, cp, [target[i] for i in keep], tp[:, :, keep], plan))
        for row in result['summary']:
            self.assertIsNone(row['max_abs_target_block_gap_change'])
            self.assertEqual(row['target_block_removals_valid'],0)
            self.assertEqual(row['target_block_removals_not_estimable'],2)
            self.assertEqual(row['source_block_removals_valid'],8)
            self.assertEqual(row['source_block_removals_not_estimable'],0)
            self.assertIsNotNone(row['max_abs_source_block_gap_change'])

    def test_half_support_counts_transitive_components_not_block_ids(self):
        from orcatca.analysis import crossfit_plan, support_ok
        from orcatca.calibration import negative_support
        rows=[]
        def add(recording,block,label):
            rows.append(dict(id=str(len(rows)),recording=recording,block=block,
                             raw_class=label,label=int(label=='KW'),site='source'))
        # Component a0 has two recording IDs and two block IDs but is one
        # independent resampling unit through the recording/block bridge.
        add('r0','a0','AB');add('r01','a0','AB');add('r01','a00','AB')
        add('r01','a00','HW');add('r01','a00','KW')
        for i in range(1,8):
            if i<4:add(f'r{i}',f'a{i}','AB')
            add(f'r{i}',f'a{i}','HW');add(f'r{i}',f'a{i}','KW')
        settings=dict(required_negative_subtypes=['AB','HW'],min_negative_blocks=2,
                      min_negative_recordings=2,fpr_targets=[.01])
        bad_half=[row for row in rows if row['block'] in {'a0','a00','a4','a5','a6'}]
        self.assertEqual(negative_support(bad_half,settings)['status'],'supported_empirically')
        ok,report=support_ok(bad_half,settings)
        self.assertFalse(ok)
        self.assertEqual(report['linked_units_by_subtype']['AB'],1)
        plan=crossfit_plan(rows,settings,repeats=5,seed=0)
        self.assertEqual(plan['status'],'ready')
        for item in plan['plans']:
            for side in ('left','right'):
                half=[rows[i] for i in item[side]]
                for label in ('AB','HW'):
                    count=sum(any(half[i]['raw_class']==label for i in ids)
                              for _,ids in linked_units(half))
                    self.assertGreaterEqual(count,2)

    def test_original_operating_points_retain_saved_thresholds(self):
        cal, cp, target, tp, _ = fixture()
        saved = np.full((2, 2), .73)
        result = full_calibration_outputs('fold00', cal, cp, target, tp,
            ['binary', 'auxiliary'], [17, 29], .2, saved)
        self.assertTrue(all(r['threshold'] == .73 for r in result['operating']))
        for row in result['operating']:
            self.assertEqual(row['target_HW_n'], 18)
            self.assertEqual(row['target_AB_n'], 18)
            self.assertEqual(row['target_positive'], 18)
        self.assertTrue(any(not r['saved_matches_recomputed'] for r in result['diagnostics']))
        self.assertTrue(all(r['procedure'] == 'full_calibration' for r in result['labels']))


class FrozenStoreIntegrationTests(unittest.TestCase):
    def test_three_folds_four_methods_full_analysis_and_archive(self):
        from orcatca.analysis import run
        from orcatca.utils import write_rows, atomic_json, sha256
        methods, seeds = ['binary', 'auxiliary', 'coarse', 'threeclass'], [17, 29]
        settings = dict(fpr_targets=[.2], required_negative_subtypes=['AB','HW'],
                        min_negative_recordings=2, min_negative_blocks=2)
        config = dict(name='synthetic_fixture', data=dict(session_gap_seconds=1800, block_hours=24),
            experiment=dict(folds=3, methods=methods, seeds=seeds), evaluation=settings)
        with tempfile.TemporaryDirectory(prefix='orcatca_fixture_') as temporary:
            root = Path(temporary)
            store = root / 'runs'; store.mkdir()
            protocol = root / 'state' / 'protocol'; protocol.mkdir(parents=True)
            (store / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            atomic_json(store / 'config.json', dict(config=config))
            complete = dict(folds=[], files={})
            for fold in range(3):
                cal, cp, target, tp, _ = fixture()
                for role, rows in [('calibrate',cal),('holdout',target)]:
                    for i, row in enumerate(rows):
                        row.update(id=f'f{fold}-'+row['id'],
                            recording=f'f{fold}-'+row['recording'],
                            block=f'f{fold}-'+row['block'])
                        row.update(group='source' if role=='calibrate' else 'target', split=role,
                            audio=row['recording']+'.wav', start_frame=(i%9)*80000,
                            time_metadata_available=False, recording_utc=None, block_utc=None,
                            recording_duration=45.)
                manifest = cal + target
                name = f'fold{fold:02d}.jsonl.gz'
                write_rows(protocol / name, manifest)
                complete['folds'].append(dict(manifest=name))
                complete['files'][name] = sha256(protocol / name)
                cp = np.concatenate([cp, np.clip(cp+.02,0,1)], axis=0)
                tp = np.concatenate([tp, np.clip(tp+.02,0,1)], axis=0)
                for m, method in enumerate(methods):
                    for s, seed in enumerate(seeds):
                        fit = store / f'fold{fold:02d}' / f'{method}_s{seed}'
                        fit.mkdir(parents=True)
                        atomic_json(fit / 'status.json', dict(status='completed'))
                        threshold, _ = threshold_diagnostics(cal, cp[m,s], .2)
                        atomic_json(fit / 'calibration.json',
                                    dict(operating_thresholds={'fpr_0.2':threshold}))
                        for role, rows, scores in [('calibration',cal,cp[m,s]),('holdout',target,tp[m,s])]:
                            write_rows(fit / (role+'.jsonl.gz'),
                                       [dict(row,p=float(score)) for row,score in zip(rows,scores)])
            atomic_json(protocol / 'complete.json', complete)
            hashes_before = {str(p.relative_to(root)): sha256(p)
                             for p in root.rglob('*') if p.is_file()}
            archive = run(root,'runs',output='result/analysis/integration_fixture',
                          methods=methods,repeats=20,crossfit_repeats=2,seed=193)
            self.assertTrue(archive.is_file())
            result = archive.parent
            with (result/'transfer_gap_summary.csv').open() as stream:
                gaps = list(csv.DictReader(stream))
            self.assertEqual(len(gaps),12)
            self.assertTrue(all(row['status']=='estimated' for row in gaps))
            with (result/'transfer_original_labels.csv').open() as stream:
                original = list(csv.DictReader(stream))
            self.assertEqual(len(original),3*4*2*2*3)
            self.assertEqual({row['role'] for row in original},{'source_calibration','target'})
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn('transfer_gap_summary.csv',zipped.namelist())
                self.assertIn('protocol_audit.json',zipped.namelist())
                for fold in range(3):
                    name=f'fold{fold:02d}_window_manifest.csv.gz'
                    self.assertIn(name,zipped.namelist())
                    self.assertIn(b'raw_class',gzip.decompress(zipped.read(name)))
                audit=json.loads(zipped.read('analysis.json'))
                self.assertEqual(audit['status'],'completed')
                self.assertFalse(audit['raw_data_modified'])
            for name, digest in hashes_before.items():
                self.assertEqual(sha256(root/name),digest)
            # A distinct ID at the same content/start is an invalid manifest,
            # even if the historical split validator does not reject it.
            first_name=complete['folds'][0]['manifest']
            with gzip.open(protocol/first_name,'rt') as stream:
                invalid=[json.loads(line) for line in stream]
            invalid[1]['start_frame']=invalid[0]['start_frame']
            write_rows(protocol/first_name,invalid)
            complete['files'][first_name]=sha256(protocol/first_name)
            atomic_json(protocol/'complete.json',complete)
            with self.assertRaisesRegex(ValueError,'Frozen protocol audit is invalid'):
                run(root,'runs',output='result/analysis/invalid_fixture',
                    methods=methods,repeats=20,crossfit_repeats=2,seed=193)
            failed=root/'result/analysis/invalid_fixture'
            self.assertTrue((failed/'failure.json').is_file())
            self.assertFalse((failed/'transfer_gap_summary.csv').exists())
            self.assertFalse((failed/'analysis.json').exists())
            self.assertEqual(json.loads((failed/'fold00_protocol_audit.json').read_text())['status'],'invalid')


if __name__ == '__main__':
    unittest.main()
