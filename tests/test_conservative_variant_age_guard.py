"""Regression for conservative variant-age publication decisions."""
import numpy as np
import pandas as pd
from motorcycle_surface.conservative_variant_age_guard import (
    apply_conservative_variant_age_guard, ConservativeAgeConfig)


def surface(ages=(1,2,3), supports=(6,0,6), prices=(110,100,90),
            conditions=(1,2), buckets=(1,2,3)):
    rows=[]
    for a,n,price in zip(ages,supports,prices):
        for cond in conditions:
            for km in buckets:
                rows.append(dict(motorcycle_model_id='203',real_brand='Honda',
                    real_model='Click 150',model_variant='DEFAULT',is_copy='0',
                    age=a,condition_tier=cond,mileage_bucket=km,n120=n,n30=0,
                    price_center=price-3*(km-1)-4*(cond-1),
                    publication_allowed=True,is_reliable=True,
                    publication_status='PUBLISH_OOF_CHECKED',
                    publication_reason_code='PUBLISH_OOF_CHECKED'))
    return pd.DataFrame(rows)


def target(age=3,origin='2026-11-07',bucket=1,condition=1,
           model_variant='DEFAULT'):
    return dict(motorcycle_model_id=203,real_brand='Honda',
        real_model='Click 150',model_variant=model_variant,is_copy=0,age=age,
        condition_tier=condition,mileage_bucket=bucket,asof=origin,
        price_center=100.,actual_independent=100.,future_unique_ads=3,
        future_observed_days=2,future_iqr_pct=10.)


def test_bracket_supports_age_without_own_ad_or_oof():
    s=surface()
    r=apply_conservative_variant_age_guard(s,None,'2026-11-22')
    assert r.publication_allowed.all()
    assert r.loc[r.age.eq(2),'age_support_bracketed'].all()
    assert r.is_copy.dtype==s.is_copy.dtype and r.is_copy.eq('0').all()


def test_unbracketed_zero_age_holds_every_condition_and_mileage_cell():
    s=surface(supports=(0,0,20))
    r=apply_conservative_variant_age_guard(s,None,'2026-11-22')
    mask=r.age.eq(2)
    assert not r.loc[mask,'publication_allowed'].any()
    assert r.loc[mask,'publication_reason_code'].eq('HOLD_AGE_ZERO_UNSUPPORTED').all()
    assert r.loc[mask,'age_evidence_source'].eq('UNSUPPORTED_AGE_EXTRAPOLATION').all()
    assert r.loc[r.age.eq(3),'publication_allowed'].all()


def test_one_year_extrapolation_needs_ten_oof_targets_AND_two_weeks():
    s=surface(supports=(0,0,20))
    h=pd.DataFrame([target(age=3,origin=week,bucket=km) for week in
        ('2026-11-07','2026-11-14') for km in range(1,6)])
    r=apply_conservative_variant_age_guard(s,h,'2026-11-22')
    assert r.loc[r.age.eq(2),'publication_allowed'].all()
    assert r.loc[r.age.eq(2),'age_extrapolation_protected'].all()
    assert r.loc[r.age.eq(2),'variant_oof_n'].eq(10).all()
    assert r.loc[r.age.eq(2),'variant_oof_weeks'].eq(2).all()


def test_single_oof_week_not_enough_for_safe_extrapolation():
    s=surface(supports=(0,0,20))
    h=pd.DataFrame([target(age=3,origin='2026-11-07',bucket=km)
                    for km in range(1,11)])
    r=apply_conservative_variant_age_guard(s,h,'2026-11-22')
    assert not r.loc[r.age.eq(2),'publication_allowed'].any()


def test_sparse_sharp_jump_holds_only_affected_age():
    s=surface(ages=(2,3),supports=(3,20),prices=(145,90))
    r=apply_conservative_variant_age_guard(s,None,'2026-11-22')
    assert r.loc[r.age.eq(2),'publication_reason_code'].eq(
        'HOLD_AGE_SPARSE_SHARP_JUMP').all()
    assert r.loc[r.age.eq(3),'publication_allowed'].all()


def test_direct_variant_age_oof_overrides_missing_market_support():
    s=surface(supports=(0,0,20))
    h=pd.DataFrame([target(age=2,origin='2026-11-14')])
    r=apply_conservative_variant_age_guard(s,h,'2026-11-22')
    assert r.loc[r.age.eq(2),'publication_allowed'].all()
    assert r.loc[r.age.eq(2),'variant_age_oof_n'].eq(1).all()
    assert r.loc[r.age.eq(2)&r.condition_tier.eq(2),
                 'age_evidence_source'].eq('DIRECT_VARIANT_AGE_OOF').all()
    assert r.loc[r.age.eq(2)&r.condition_tier.eq(1),
                 'curve_oof_n'].eq(1).all()


def test_future_incomplete_oof_cannot_justify_missing_age():
    s=surface(supports=(0,0,20))
    h=pd.DataFrame([target(age=2,origin='2026-11-21')])
    r=apply_conservative_variant_age_guard(s,h,'2026-11-22')
    assert not r.loc[r.age.eq(2),'publication_allowed'].any()


def test_weak_oof_cannot_justify_missing_age():
    s=surface(supports=(0,0,20))
    h=pd.DataFrame([target(age=2,origin='2026-11-14')])
    h.loc[0,'future_unique_ads']=2
    r=apply_conservative_variant_age_guard(s,h,'2026-11-22')
    assert not r.loc[r.age.eq(2),'publication_allowed'].any()


def test_upstream_severe_hold_and_prices_preserved():
    s=surface(supports=(0,0,20))
    s.loc[s.age.eq(3),['publication_allowed','is_reliable']]=False
    s.loc[s.age.eq(3),'publication_status']='HOLD_OOF_SINGLE_CONFIRMED_OVER20'
    original=s.copy(deep=True)
    r=apply_conservative_variant_age_guard(s,None,'2026-11-22')
    assert not r.loc[r.age.eq(3),'publication_allowed'].any()
    assert r.loc[r.age.eq(3),'publication_status'].eq(
        'HOLD_OOF_SINGLE_CONFIRMED_OVER20').all()
    assert np.array_equal(s.price_center.to_numpy(),r.price_center.to_numpy())
    pd.testing.assert_frame_equal(s,original)


def test_already_rejected_bad_price_does_not_crash_age_shape_guard():
    s=surface(supports=(0,0,20))
    row=s.age.eq(1)
    s.loc[row,'price_center']=float('nan')
    s.loc[row,['publication_allowed','is_reliable']]=False
    s.loc[row,'publication_status']='HOLD_INVALID_CURVE'
    r=apply_conservative_variant_age_guard(s,None,'2026-11-22')
    assert not r.loc[row,'publication_allowed'].any()
    assert r.loc[row,'publication_status'].eq('HOLD_INVALID_CURVE').all()


def test_audit_mode_flags_but_cannot_veto():
    s=surface(supports=(0,0,20))
    r=apply_conservative_variant_age_guard(s,None,'2026-11-22',
                                            ConservativeAgeConfig(mode='audit'))
    assert r.publication_allowed.all()
    assert r.loc[r.age.eq(2),'age_conservative_reason'].eq(
        'HOLD_AGE_ZERO_UNSUPPORTED').all()


def test_parent_model_oof_cannot_masquerade_as_variant_age_oof():
    s=surface(supports=(0,0,20))
    s['model_variant']='ABS'
    h=pd.DataFrame([target(age=2,origin='2026-11-14',model_variant='DEFAULT')])
    r=apply_conservative_variant_age_guard(s,h,'2026-11-22')
    assert not r.loc[r.age.eq(2),'publication_allowed'].any()
    assert r.loc[r.age.eq(2),'age_evidence_source'].eq(
        'PARENT_MODEL_OOF_ONLY').all()


def test_local_input_string_is_copy_stays_string_through_hierarchical_guard():
    from motorcycle_surface.hierarchical_risk_guard import apply_hierarchical_oof_guard
    s=surface()
    r=apply_hierarchical_oof_guard(s,None,'2026-11-22')
    assert r.is_copy.eq('0').all()
    assert r.is_copy.dtype==s.is_copy.dtype
