import unittest
from unittest.mock import patch

import pandas as pd

from motorcycle_surface.publication_policy import (
    POLICY_VERSION,
    PublicationConfig,
    apply_publication,
    unsupported_adjacent_plateau_curves,
)


class AllAgeDuplicateQualityPolicyTests(unittest.TestCase):
    def make_surface(self, ages=(2, 4), supports=None, *, mismatch_bucket=None):
        supports = supports or {age: (10, 3) for age in ages}
        rows = []
        for age in ages:
            n120, n30 = supports[age]
            for bucket in range(6):
                mid = 700_000_000 - bucket * 10_000_000
                if age == ages[0] and mismatch_bucket == bucket:
                    mid *= 1.02
                rows.append({
                    'motorcycle_model_id': '203', 'real_brand': 'Honda',
                    'real_model': 'Click 150', 'is_copy': 0,
                    'model_variant': 'DEFAULT', 'age': age,
                    'condition_tier': 1, 'mileage_bucket': bucket,
                    'price_center': mid, 'n120': n120, 'n30': n30,
                    'publication_allowed': True, 'is_reliable': True,
                    'max_grid_bucket': 5, 'n_group_all': 10,
                    'motorcycle_n30': 8, 'group_n30': 8,
                    'is_factory': False, 'adjusted_price': mid,
                    'reference_price': mid, 'production_year': 1405-age,
                    'isotonic_change_pct': 0.0, 'asof': '2026-11-16',
                })
        return pd.DataFrame(rows)

    @staticmethod
    def key(age):
        return ('203', 'Honda', 'Click 150', 0, 'DEFAULT', age, 1)

    def test_non_adjacent_duplicate_is_detected(self):
        surface = self.make_surface(ages=(2, 5))
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(5)})

    def test_clear_market_dominance_holds_weaker_curve(self):
        surface = self.make_surface(ages=(2, 4), supports={2:(40,12), 4:(8,2)})
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(4)})

    def test_indistinguishable_market_evidence_holds_both(self):
        surface = self.make_surface(ages=(1, 4), supports={1:(10,3), 4:(10,3)})
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(1), self.key(4)})

    def test_only_price_center_is_used_and_nonmatching_curve_is_not_held(self):
        surface = self.make_surface(ages=(2, 4))
        surface['price_low'] = surface['price_center'] * 0.5
        surface['price_high'] = surface['price_center'] * 1.5
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(4)})

    def test_insufficient_shared_buckets_are_not_compared(self):
        surface = self.make_surface(ages=(2, 4))
        surface.loc[surface.age.eq(4) & surface.mileage_bucket.ge(3), 'mileage_bucket'] += 10
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, set())

    def test_exact_age_oof_winner_is_kept(self):
        surface = self.make_surface(ages=(2, 4), supports={2:(8,2), 4:(40,12)})
        history_rows = []
        # Age 2 has qualifying, low-error OOF; age 4 has no OOF certificate.
        for origin, buckets in (('2026-10-17',(0,1)), ('2026-10-24',(2,))):
            for bucket in buckets:
                mid = 700_000_000 - bucket * 10_000_000
                history_rows.append({
                    'motorcycle_model_id':'203','real_brand':'Honda',
                    'real_model':'Click 150','is_copy':0,'model_variant':'DEFAULT',
                    'age':2,'condition_tier':1,'mileage_bucket':bucket,
                    'asof':origin,'price_center':mid,'actual_independent':mid,
                    'future_unique_ads':3,'future_observed_days':2,'future_iqr_pct':10,
                })
        held = unsupported_adjacent_plateau_curves(
            surface, pd.DataFrame(history_rows), '2026-11-16')
        self.assertEqual(held, {self.key(4)})

    def test_apply_publication_holds_full_curves_and_versions_policy(self):
        surface = self.make_surface(ages=(2, 4))
        published = apply_publication(surface, '2026-11-16', pd.DataFrame())
        self.assertTrue(published['age_plateau_held'].all())
        self.assertFalse(published['publication_allowed'].any())
        self.assertEqual(published['policy_version'].unique().tolist(),
                         ['USED_RELEASE_FINAL_6_ALL_AGE_DUPLICATE_QUALITY_GUARD_2PCT'])

    def test_all_distinct_ages_are_compared(self):
        surface = self.make_surface(ages=(0, 3, 8))
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(0), self.key(3), self.key(8)})


    def test_one_vs_zero_ads_cannot_create_a_winner(self):
        surface = self.make_surface(ages=(2, 4), supports={2:(1,0), 4:(0,0)})
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(4)})

    def test_market_winner_must_clear_absolute_support_floors(self):
        surface = self.make_surface(ages=(2, 4), supports={2:(19,5), 4:(1,0)})
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(4)})

    def test_market_winner_with_absolute_and_relative_support_is_kept(self):
        surface = self.make_surface(ages=(2, 4), supports={2:(40,12), 4:(8,2)})
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(4)})

    def test_default_similarity_threshold_is_two_percent(self):
        import inspect
        parameter = inspect.signature(unsupported_adjacent_plateau_curves).parameters['price_rtol']
        self.assertEqual(parameter.default, 0.02)

    def test_similarity_threshold_is_two_percent(self):
        surface = self.make_surface(ages=(2, 4))
        surface.loc[surface.age.eq(2), 'price_center'] *= 1.015
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(4)})

    def test_difference_above_two_percent_is_not_a_duplicate(self):
        surface = self.make_surface(ages=(2, 4))
        surface.loc[surface.age.eq(2), 'price_center'] *= 1.025
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, set())

    def test_similarity_threshold_is_symmetric(self):
        surface = self.make_surface(ages=(2, 4))
        # The older curve is 2.01% above the younger curve. A one-sided
        # younger/older ratio would incorrectly call this less than 2%.
        surface.loc[surface.age.eq(4), 'price_center'] *= 1.0201
        held = unsupported_adjacent_plateau_curves(
            surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, set())

    def test_apply_publication_fails_closed_if_a_duplicate_survives(self):
        surface = self.make_surface(ages=(2, 4))
        with patch(
            'motorcycle_surface.publication_policy.unsupported_adjacent_plateau_curves',
            side_effect=[set(), {self.key(2), self.key(4)}],
        ):
            with self.assertRaisesRegex(
                AssertionError, 'Similar age curves remain publishable'
            ):
                apply_publication(surface, '2026-11-16', pd.DataFrame())

    def test_two_oof_curves_with_unacceptable_extreme_errors_are_both_held(self):
        surface = self.make_surface(ages=(2, 4), supports={2:(40,12), 4:(8,2)})
        history_rows = []
        for age in (2, 4):
            for origin, buckets in (('2026-10-17',(0,1)), ('2026-10-24',(2,))):
                for bucket in buckets:
                    mid = 700_000_000 - bucket * 10_000_000
                    history_rows.append({
                        'motorcycle_model_id':'203','real_brand':'Honda',
                        'real_model':'Click 150','is_copy':0,'model_variant':'DEFAULT',
                        'age':age,'condition_tier':1,'mileage_bucket':bucket,
                        'asof':origin,'price_center':mid,
                        'actual_independent':mid * 1.30,
                        'future_unique_ads':3,'future_observed_days':2,'future_iqr_pct':10,
                    })
        held = unsupported_adjacent_plateau_curves(
            surface, pd.DataFrame(history_rows), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(4)})

    def test_config_has_only_price_similarity_thresholds(self):
        cfg = PublicationConfig()
        self.assertEqual(cfg.age_plateau_price_rtol, 0.02)
        self.assertEqual(cfg.age_plateau_min_matching_buckets, 3)
        self.assertEqual(cfg.age_plateau_min_pair_tie_share, 0.90)
        self.assertEqual(cfg.age_plateau_market_dominance_ratio, 1.25)
        self.assertEqual(cfg.age_plateau_winner_min_n120, 20.0)
        self.assertEqual(cfg.age_plateau_winner_min_n30, 6.0)


if __name__ == '__main__':
    unittest.main()
