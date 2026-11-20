"""Observed market guards keep existing overprice rules and add safe underprice vetoes."""
import unittest

import pandas as pd

from motorcycle_surface.recent_market_guard import (
    apply_recent_market_guard,
    detect_observed_over20,
    detect_observed_under20,
)


class RecentMarketUnderpriceGuardTests(unittest.TestCase):
    ASOF = '2026-11-16'

    @staticmethod
    def surface(*, age=1, variant='DEFAULT', mid=203, published=True):
        rows = []
        for bucket in range(4):
            rows.append({
                'motorcycle_model_id': mid, 'real_brand': 'Honda',
                'real_model': 'Click 150', 'is_copy': 0,
                'model_variant': variant, 'age': age, 'condition_tier': 1,
                'mileage_bucket': bucket, 'price_center': 100.0 - bucket * 5,
                'publication_allowed': published, 'is_reliable': published,
                'publication_status': ('PUBLISH_OOF_CHECKED' if published else 'HOLD_OOF_SEVERE'),
                'publication_reason_code': ('PUBLISH_OOF_CHECKED' if published else 'HOLD_OOF_SEVERE'),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def quote(index, *, price=200.0, bucket=0, age=1, variant='DEFAULT',
              mid=203, date=None, title=None, description=None, suspicious=0):
        return {
            'motorcycle_model_id': mid, 'real_brand': 'Honda',
            'real_model': 'Click 150', 'is_copy': 0,
            'model_variant': variant, 'age': age, 'condition_tier': 1,
            'mileage_bucket': bucket, 'token': f'ad-{index}',
            'date': date or f'2026-11-{10+index:02d}', 'price': price,
            'source': 'ads', 'title': title or f'Independent listing {index}',
            'description': (description if description is not None
                            else f'Independent motorcycle description {index}'),
            'city': 'tehran', 'district': 'district-a',
            'seller_type': 'private', 'mileage': 12000,
            'copy_label_suspicious': suspicious,
        }

    def test_under20_blocks_complete_curve_and_preserves_other_ages_and_variants(self):
        s = pd.concat([
            self.surface(age=1), self.surface(age=2),
            self.surface(age=1, variant='ABS'),
        ], ignore_index=True)
        ads = pd.DataFrame([self.quote(i) for i in (1, 2, 3)])
        out = apply_recent_market_guard(s, ads, self.ASOF)
        held = out.age.eq(1) & out.model_variant.eq('DEFAULT')
        self.assertEqual(int(held.sum()), 4)
        self.assertFalse(out.loc[held, 'publication_allowed'].any())
        self.assertFalse(out.loc[held, 'is_reliable'].any())
        self.assertTrue(out.loc[held, 'publication_reason_code'].eq(
            'HOLD_OBSERVED_UNDER20').all())
        self.assertTrue(out.loc[~held, 'publication_allowed'].all())

    def test_new_token_relistings_count_as_one_independent_ad(self):
        ads = pd.DataFrame([
            self.quote(i, title='Kawasaki Z300 one motorcycle',
                       description='Identical dealer listing', date=f'2026-11-{10+i:02d}')
            for i in (1, 2, 3)
        ])
        held, evidence = detect_observed_under20(self.surface(), ads, self.ASOF)
        self.assertEqual(held, set())
        self.assertTrue(evidence.empty)

    def test_repriced_relisting_is_still_one_ad(self):
        ads = pd.DataFrame([
            self.quote(i, price=190.0 + i, title='Same motorcycle',
                       description='Same motorcycle and seller')
            for i in (1, 2, 3)
        ])
        held, _ = detect_observed_under20(self.surface(), ads, self.ASOF)
        self.assertEqual(held, set())

    def test_under20_requires_lower_quartile_confirmation(self):
        # Median 130 implies >20% underpricing, but q25=115 does not.
        ads = pd.DataFrame([self.quote(i, price=v) for i, v in
                            enumerate((100, 130, 140), start=1)])
        held, _ = detect_observed_under20(self.surface(), ads, self.ASOF)
        self.assertEqual(held, set())

    def test_under20_requires_three_ads_on_two_days(self):
        two_ads = pd.DataFrame([self.quote(i) for i in (1, 2)])
        held, _ = detect_observed_under20(self.surface(), two_ads, self.ASOF)
        self.assertEqual(held, set())
        same_day = pd.DataFrame([self.quote(i, date='2026-11-14') for i in (1, 2, 3)])
        held, _ = detect_observed_under20(self.surface(), same_day, self.ASOF)
        self.assertEqual(held, set())

    def test_under20_does_not_use_future_factory_or_suspicious_ads(self):
        ads = pd.DataFrame([self.quote(i) for i in (1, 2, 3)])
        future = ads.copy()
        future['date'] = '2026-11-16'
        factory = ads.copy()
        factory['source'] = 'factory'
        suspicious = ads.copy()
        suspicious['copy_label_suspicious'] = 1
        for invalid in (future, factory, suspicious):
            held, _ = detect_observed_under20(self.surface(), invalid, self.ASOF)
            self.assertEqual(held, set())

    def test_under20_pooled_proof_normalizes_each_mileage_bucket(self):
        surface = self.surface()
        ads = []
        for bucket, prediction in ((0, 100), (1, 95), (2, 90)):
            for i in (1, 2):
                row = self.quote(2 * bucket + i, bucket=bucket,
                                 price=prediction / .7,
                                 date=f'2026-11-{10+bucket:02d}')
                ads.append(row)
        held, evidence = detect_observed_under20(surface, pd.DataFrame(ads), self.ASOF)
        self.assertEqual(len(held), 1)
        self.assertEqual(evidence.reason.tolist(), ['MULTI_BUCKET_CURVE_UNDER20'])
        out = apply_recent_market_guard(surface, pd.DataFrame(ads), self.ASOF)
        self.assertFalse(out.publication_allowed.any())

    def test_old_overprice_guard_conditions_are_unchanged(self):
        # The existing token-only overprice proof must still block this case.
        ads = pd.DataFrame([
            self.quote(i, price=50.0, title='Identical overprice case',
                       description='Reposted cheap ad') for i in (1, 2, 3)
        ])
        over, _ = detect_observed_over20(self.surface(), ads, self.ASOF)
        self.assertEqual(len(over), 1)
        out = apply_recent_market_guard(self.surface(), ads, self.ASOF)
        self.assertFalse(out.publication_allowed.any())
        self.assertTrue(out.publication_reason_code.eq('HOLD_OBSERVED_OVER20').all())

    def test_preexisting_holds_keep_original_reason(self):
        surface = self.surface(published=False)
        ads = pd.DataFrame([self.quote(i) for i in (1, 2, 3)])
        out = apply_recent_market_guard(surface, ads, self.ASOF)
        self.assertFalse(out.publication_allowed.any())
        self.assertTrue(out.publication_reason_code.eq('HOLD_OOF_SEVERE').all())

    def test_under20_skips_undocumented_ad_identity(self):
        # An upstream caller without listing text cannot establish independence.
        ads = pd.DataFrame([self.quote(i) for i in (1, 2, 3)]).drop(
            columns=['title', 'description'])
        held, _ = detect_observed_under20(self.surface(), ads, self.ASOF)
        self.assertEqual(held, set())

    def test_under20_does_not_count_blank_text_as_independent_evidence(self):
        ads = pd.DataFrame([self.quote(i, title=' ', description='  ')
                            for i in (1, 2, 3)])
        held, _ = detect_observed_under20(self.surface(), ads, self.ASOF)
        self.assertEqual(held, set())


if __name__ == '__main__':
    unittest.main()
