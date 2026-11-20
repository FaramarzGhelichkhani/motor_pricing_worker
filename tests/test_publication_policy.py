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
        surface.loc[surface.age.eq(4) & surface.mileage_bucket.ge(2), 'mileage_bucket'] += 10
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
                         ['USED_RELEASE_FINAL_7_ALL_AGE_DUPLICATE_2PCT_CURRENT_YEAR_EVIDENCE'])

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
        surface['price_center'] = surface['price_center'].astype(float)
        surface.loc[surface.age.eq(2), 'price_center'] *= 1.015
        held = unsupported_adjacent_plateau_curves(surface, pd.DataFrame(), '2026-11-16')
        self.assertEqual(held, {self.key(2), self.key(4)})

    def test_difference_above_two_percent_is_not_a_duplicate(self):
        surface = self.make_surface(ages=(2, 4))
        surface['price_center'] = surface['price_center'].astype(float)
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

    def test_disabling_quality_selection_cannot_bypass_duplicate_invariant(self):
        surface = self.make_surface(ages=(2, 4))
        cfg = PublicationConfig(age_plateau_guard_enabled=False)
        with self.assertRaisesRegex(
            AssertionError, 'Similar age curves remain publishable'
        ):
            apply_publication(surface, '2026-11-16', pd.DataFrame(), cfg)

    def test_config_has_only_price_similarity_thresholds(self):
        cfg = PublicationConfig()
        self.assertEqual(cfg.age_plateau_price_rtol, 0.02)
        self.assertEqual(cfg.age_plateau_min_matching_buckets, 3)
        self.assertEqual(cfg.age_plateau_min_pair_tie_share, 0.90)
        self.assertEqual(cfg.age_plateau_market_dominance_ratio, 1.25)
        self.assertEqual(cfg.age_plateau_winner_min_n120, 20.0)
        self.assertEqual(cfg.age_plateau_winner_min_n30, 6.0)

    def test_current_year_guard_holds_all_mileage_and_condition_rows(self):
        surface = self.make_surface(ages=(0, 1), supports={0: (4, 0), 1: (4, 0)})
        surface['price_center'] = surface['price_center'].astype(float)
        older = surface.age.eq(1)
        surface.loc[older, 'price_center'] *= 0.90
        surface.loc[older, 'adjusted_price'] = surface.loc[older, 'price_center']
        extra_condition = surface[surface.age.eq(0)].copy()
        extra_condition['condition_tier'] = 2
        extra_condition = extra_condition[extra_condition.mileage_bucket.ge(1)]
        surface = pd.concat([surface, extra_condition], ignore_index=True)
        result = apply_publication(surface, '2026-11-16', pd.DataFrame())
        current = result[result.age.eq(0)]
        older = result[result.age.eq(1)]
        self.assertEqual(len(current), 11)
        self.assertFalse(current.publication_allowed.any())
        self.assertFalse(current.is_reliable.any())
        self.assertTrue(current.publication_reason_code.eq(
            'HOLD_CURRENT_YEAR_INSUFFICIENT_EVIDENCE').all())
        self.assertTrue(older.publication_allowed.all())

    def test_current_year_guard_uses_both_exact_support_thresholds(self):
        for n120, n30, expected_publish in ((4, 0, False), (5, 0, True),
                                              (4, 1, True), (1, 2, True)):
            with self.subTest(n120=n120, n30=n30):
                surface = self.make_surface(ages=(0,), supports={0: (n120, n30)})
                result = apply_publication(surface, '2026-11-16', pd.DataFrame())
                self.assertEqual(result.publication_allowed.tolist(),
                                 [expected_publish] * len(result))

    def test_current_year_guard_keeps_other_variants_and_preexisting_holds(self):
        surface = self.make_surface(ages=(0,), supports={0: (4, 0)})
        other_variant = surface.copy()
        other_variant['model_variant'] = 'ABS'
        other_variant[['n120', 'n30']] = [8, 1]
        surface = pd.concat([surface, other_variant], ignore_index=True)
        # The current-year guard must not overwrite an existing model hold.
        surface.loc[surface.model_variant.eq('DEFAULT'), 'motorcycle_n30'] = 0
        result = apply_publication(surface, '2026-11-16', pd.DataFrame())
        self.assertTrue(result.loc[result.model_variant.eq('DEFAULT'),
                                   'publication_status'].eq('HOLD_WEAK_MODEL_MARKET').all())
        self.assertTrue(result.loc[result.model_variant.eq('ABS'),
                                   'publication_allowed'].all())

    def test_current_year_guard_cannot_hold_part_of_a_variant_year(self):
        surface = self.make_surface(ages=(0,), supports={0: (4, 0)})
        # One inconsistent support row cannot leave an otherwise published
        # mileage bucket of the same year/variant exposed.
        surface.loc[surface.mileage_bucket.eq(0), ['n30', 'n120']] = [1, 5]
        result = apply_publication(surface, '2026-11-16', pd.DataFrame())
        self.assertFalse(result.publication_allowed.any())

    def test_current_year_guard_can_be_disabled(self):
        surface = self.make_surface(ages=(0,), supports={0: (4, 0)})
        cfg = PublicationConfig(current_year_evidence_guard_enabled=False)
        result = apply_publication(surface, '2026-11-16', pd.DataFrame(), cfg)
        self.assertTrue(result.publication_allowed.all())


if __name__ == '__main__':
    unittest.main()
