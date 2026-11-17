import unittest

import pandas as pd

from motorcycle_surface.publication_policy import (
    POLICY_VERSION,
    PublicationConfig,
    apply_publication,
    unsupported_young_plateau_curves,
)


class AdjacentYearPlateauPolicyTests(unittest.TestCase):
    def make_surface(self, *, young_n120=14, young_n30=5, use_price_mid=False,
                     mismatch_bucket=None):
        rows = []
        for age, n120, n30 in ((3, young_n120, young_n30), (4, 687, 167)):
            for bucket in range(6):
                mid = 700_000_000 - bucket * 10_000_000
                if age == 3 and mismatch_bucket == bucket:
                    mid *= 1.02
                row = {
                    'motorcycle_model_id': '203',
                    'real_brand': 'Honda',
                    'real_model': 'Click 150',
                    'is_copy': 0,
                    'model_variant': 'DEFAULT',
                    'age': age,
                    'condition_tier': 1,
                    'mileage_bucket': bucket,
                    'price_center': mid,
                    'price_low': mid * 0.90,
                    'price_high': mid * 1.10,
                    'n120': n120,
                    'n30': n30,
                    'publication_allowed': True,
                    'is_reliable': True,
                    'max_grid_bucket': 5,
                    'n_group_all': 10,
                    'motorcycle_n30': 8,
                    'group_n30': 8,
                    'is_factory': False,
                    'adjusted_price': mid,
                    'reference_price': mid,
                    'production_year': 1405 - age,
                    'isotonic_change_pct': 0.0,
                    'asof': '2026-11-16',
                }
                if use_price_mid:
                    row['price_mid'] = 700_000_000 - bucket * 10_000_000
                    row['price_center'] = row['price_mid'] * (1.20 if age == 3 else 0.80)
                    row['adjusted_price'] = row['price_center']
                    row['reference_price'] = row['price_center']
                rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def curve_key(age=3):
        return ('203', 'Honda', 'Click 150', 0, 'DEFAULT', age, 1)

    def test_boundary_support_n30_five_holds_younger_duplicate_curve(self):
        surface = self.make_surface(young_n120=14, young_n30=5)
        held = unsupported_young_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertIn(self.curve_key(3), held)
        self.assertNotIn(self.curve_key(4), held)

    def test_sufficient_30_day_support_preserves_curve(self):
        surface = self.make_surface(young_n120=14, young_n30=6)
        held = unsupported_young_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertNotIn(self.curve_key(3), held)

    def test_only_middle_price_is_used_when_price_mid_exists(self):
        surface = self.make_surface(use_price_mid=True)
        held = unsupported_young_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertIn(self.curve_key(3), held)

    def test_one_nonmatching_bucket_breaks_ninety_percent_tie_rule(self):
        surface = self.make_surface(mismatch_bucket=0)
        held = unsupported_young_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertNotIn(self.curve_key(3), held)

    def test_good_completed_oof_evidence_exempts_curve(self):
        surface = self.make_surface()
        history_rows = []
        for origin, buckets in (('2026-10-17', (0, 1)), ('2026-10-24', (2,))):
            for bucket in buckets:
                mid = 700_000_000 - bucket * 10_000_000
                history_rows.append({
                    'motorcycle_model_id': '203',
                    'real_brand': 'Honda',
                    'real_model': 'Click 150',
                    'is_copy': 0,
                    'model_variant': 'DEFAULT',
                    'age': 3,
                    'condition_tier': 1,
                    'mileage_bucket': bucket,
                    'asof': origin,
                    'price_center': mid,
                    'actual_independent': mid,
                    'future_unique_ads': 3,
                    'future_observed_days': 2,
                    'future_iqr_pct': 10,
                })
        history = pd.DataFrame(history_rows)
        held = unsupported_young_plateau_curves(surface, history, '2026-11-16')
        self.assertNotIn(self.curve_key(3), held)

    def test_apply_publication_holds_entire_younger_curve_only(self):
        surface = self.make_surface()
        published = apply_publication(surface, '2026-11-16', pd.DataFrame())
        young = published[published.age.eq(3)]
        older = published[published.age.eq(4)]
        self.assertTrue(young['age_plateau_held'].all())
        self.assertFalse(young['publication_allowed'].any())
        self.assertTrue(older['publication_allowed'].all())
        self.assertEqual(published['policy_version'].unique().tolist(), [POLICY_VERSION])

    def test_default_parameters_match_data_calibrated_boundary(self):
        cfg = PublicationConfig()
        self.assertEqual(cfg.age_plateau_n120_min, 20)
        self.assertEqual(cfg.age_plateau_n30_min, 6)
        self.assertEqual(cfg.age_plateau_price_rtol, 0.001)
        self.assertEqual(cfg.age_plateau_min_matching_buckets, 3)
        self.assertEqual(cfg.age_plateau_min_pair_tie_share, 0.90)


if __name__ == '__main__':
    unittest.main()
