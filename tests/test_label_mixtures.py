import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[1] / 'scripts' / 'check_label_mixtures.py'
spec = importlib.util.spec_from_file_location('mixture_check', script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def example():
    operating, decomposed = [], []
    for side, (source_hw, source_ab) in enumerate(((6, 0), (0, 8))):
        fields = dict(fold='F', method='M', seed=17, repeat=0, side=side, alpha=0.01, threshold=.5 + side / 10)
        for role, h, hn, a, an in [('source_fit', 0, 10, 0, 10),
                                   ('source_check', source_hw, 10, source_ab, 10),
                                   ('target', 4, 5, 14, 15)]:
            operating.append(dict(**fields, role=role, HW_fp=h, HW_n=hn, HW_fpr=h / hn,
                                  AB_fp=a, AB_n=an, AB_fpr=a / an, negative=hn+an,
                                  fp=h+a, fpr=(h+a)/(hn+an)))
        proj = .25 * source_hw/10 + .75 * source_ab/10
        source = (source_hw + source_ab)/20
        decomposed.append(dict(**fields, status='estimated_descriptively', mixture_only=proj,
                               source_reference_mixture=proj-source,
                               target_weighted_conditional=.9-proj, delta=.9-source))
    return operating, decomposed, {mod.key(r) for r in decomposed}


class MixtureBound(unittest.TestCase):
    def test_harder_label_can_switch_and_input_order_does_not_matter(self):
        op, tr, grid = example()
        row, = mod.summarize(reversed(op), tr, grid)
        # Midpoints of source rates have a ceiling of .4; average of actual
        # matched ceilings (.6 and .8) is .7, so the observed .9 exceeds it by .2.
        self.assertAlmostEqual(row['mean_empirical_mixture_ceiling'], .7)
        self.assertAlmostEqual(row['mean_excess_over_ceiling_pp'], 20)
        self.assertAlmostEqual(row['mean_composition_movement_pp'], 2.5)
        self.assertEqual(row['n_observed_above_ceiling'], 2)

    def test_observed_and_balanced_shares_use_mean_paired_changes(self):
        op, tr, grid = example()
        _, (row,) = mod.summarize_tables(op, tr, grid)
        # Higher weights for AB (15 of 20 windows) versus equal weights. The
        # two matched thresholds have very different class-specific changes.
        self.assertEqual(row['share_status'], 'estimated')
        self.assertAlmostEqual(row['observed_ab_weight'], .75)
        self.assertAlmostEqual(row['mean_hw_rate_change_pp'], 50)
        self.assertAlmostEqual(row['mean_ab_rate_change_pp'], 100 * 8/15)
        self.assertAlmostEqual(row['ab_share_observed_gap_pct'], 100*40/52.5)
        self.assertAlmostEqual(row['ab_share_balanced_gap_pct'], 100*(8/15)/(.5+8/15))

    def test_changed_target_mix_and_signed_or_zero_gap(self):
        op, tr, grid = example()
        op[-1].update(HW_n=6, AB_n=14, HW_fpr=4/6, AB_fpr=1)
        with self.assertRaisesRegex(ValueError, 'counts change'):
            mod.summarize_tables(op, tr, grid)
        for dh, da in [(-1, 1), (-2, -1), (0, 0), (1e-14, 1e-14)]:
            a, b, status = mod.contribution_shares(dh, da, .9)
            self.assertIsNone(a)
            self.assertIsNone(b)
            self.assertTrue(status.startswith('withheld_'))
        self.assertEqual(mod.contribution_shares(0, 2, .5), (100, 100, 'estimated'))

    def test_mispaired_rows_rejected(self):
        op, tr, grid = example()
        op[1]['threshold'] = .55
        with self.assertRaisesRegex(ValueError, 'thresholds differ'):
            mod.summarize(op, tr, grid)
        with self.assertRaisesRegex(ValueError, 'evaluation grid'):
            mod.summarize(op, tr[:-1], grid)
