"""Regression tests for the systemic variant/model publication veto.

The tests intentionally avoid price errors above 20%: local severe protection
is a separate authority and must remain unchanged.
"""
import inspect
import pandas as pd
from motorcycle_surface.hierarchical_risk_guard import (
    apply_hierarchical_oof_guard, HierarchicalRiskConfig)
from motorcycle_surface.release_gate import validate_recent_oof_release


def fixture_rows(variants=('BASE',), ages=(2,3,4), dates=(
        '2026-10-10','2026-10-17','2026-10-24','2026-10-31',
        '2026-11-07','2026-11-14'), bad_variants=(), bad_ages=None):
    surface, history = [], []
    for variant in variants:
        for age in ages:
            coords=dict(motorcycle_model_id=353,real_brand='Bajaj',
                        real_model='Pulsar 180',is_copy=0,
                        model_variant=variant,age=age,condition_tier=1)
            for bucket in (1,2,3):
                surface.append(dict(coords,mileage_bucket=bucket,
                                    publication_allowed=True,is_reliable=True,
                                    publication_status='PUBLISH_OOF_CHECKED',
                                    publication_reason_code='PUBLISH_OOF_CHECKED'))
            for origin in dates:
                severe = variant in bad_variants and (
                    bad_ages is None or age in bad_ages)
                history.append(dict(coords,mileage_bucket=2,asof=origin,
                                    price_center=118.0 if severe else 102.0,
                                    actual_independent=100.,
                                    future_unique_ads=3,
                                    future_observed_days=2,
                                    future_iqr_pct=8.))
    return pd.DataFrame(surface),pd.DataFrame(history)


def test_one_bad_curve_cannot_veto_healthy_variant():
    s,h=fixture_rows(bad_variants=('BASE',),bad_ages=(2,))
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert got.publication_allowed.all()


def test_systemic_variant_holds_only_its_variant():
    s,h=fixture_rows(variants=('BASE','ABS'),bad_variants=('BASE',))
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert not got.loc[got.model_variant.eq('BASE'),'publication_allowed'].any()
    assert got.loc[got.model_variant.eq('ABS'),'publication_allowed'].all()
    assert got.loc[got.model_variant.eq('BASE'),'publication_status'].eq(
        'HOLD_OOF_VARIANT_SYSTEMIC').all()


def test_systemic_multi_variant_model_blocks_when_variants_individually_unevaluable():
    s,h=fixture_rows(variants=('BASE','ABS'),ages=(2,3),
                     bad_variants=('BASE','ABS'))
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert not got.publication_allowed.any()
    assert got.publication_status.eq('HOLD_OOF_MODEL_SYSTEMIC').all()


def test_prior_locally_blocked_curve_does_not_blacklist_other_curves():
    s,h=fixture_rows(bad_variants=('BASE',),bad_ages=(2,))
    blocked=s.age.eq(2)
    s.loc[blocked,'publication_allowed']=False
    s.loc[blocked,'is_reliable']=False
    s.loc[blocked,'publication_status']='HOLD_OOF_SINGLE_CONFIRMED_OVER20'
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert got.loc[got.age.ne(2),'publication_allowed'].all()
    assert got.loc[blocked,'publication_status'].eq(
        'HOLD_OOF_SINGLE_CONFIRMED_OVER20').all()


def test_future_labels_cannot_change_asof_model_risk_decision():
    s,h=fixture_rows(dates=('2026-11-14','2026-11-21'),
                     bad_variants=('BASE',))
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert got.publication_allowed.all()


def test_recent_AND_historical_is_required():
    s,h=fixture_rows(bad_variants=('BASE',))
    # Current recent origins are good; all historical older origins are bad.
    h.loc[h['asof'].isin(['2026-11-07','2026-11-14']),'price_center']=102.
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert got.publication_allowed.all()


def test_threshold_is_four_percent_without_relaxing_fifteen_percent():
    signature=inspect.signature(validate_recent_oof_release)
    assert signature.parameters['max_tail20_upper_pct'].default==4.0
    assert signature.parameters['max_tail15_upper_pct'].default==5.0


def test_curve_level_20pct_rule_remains_independent():
    from motorcycle_surface.severe_oof_guard import held_severe_curve_reasons
    _,h=fixture_rows(dates=('2026-11-14',))
    h.loc[h.age.eq(2),'price_center']=123.
    rejected,_=held_severe_curve_reasons(h,'2026-11-22')
    assert len(rejected)==1
    assert list(rejected.values())==['HOLD_OOF_SINGLE_CONFIRMED_OVER20']

def test_pulsar_like_two_of_thirteen_is_not_proof_of_systemic_model_risk():
    s,h=fixture_rows(ages=(10,11,12,13),dates=(
        '2026-10-24','2026-10-31','2026-11-07','2026-11-14'))
    # 13 independent curve/weeks, 4 origins, 4 curves; only two >15%.
    drop=(h['age'].eq(13) & h['asof'].eq('2026-10-24')) | (
          h['age'].eq(13) & h['asof'].eq('2026-11-07')) | (
          h['age'].eq(12) & h['asof'].eq('2026-11-14'))
    h=h.loc[~drop].copy()
    assert len(h)==13
    h['price_center']=103.
    h.loc[(h['age'].eq(10) & h['asof'].eq('2026-10-24')) |\n          (h['age'].eq(11) & h['asof'].eq('2026-10-31')), 'price_center']=118.
    got=apply_hierarchical_oof_guard(s,h,'2026-11-22')
    assert got.publication_allowed.all()
    assert not got.oof_variant_risk_state.eq('MODEL_PROVEN_UNSAFE').any()
