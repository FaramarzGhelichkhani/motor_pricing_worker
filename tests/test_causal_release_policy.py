"""Regressions for chronological severe OOF veto and unbiased release decision."""
import pandas as pd
import pytest
from motorcycle_surface.severe_oof_guard import (
    audit_severe_oof_curves, held_severe_curve_reasons, CURVE)
from motorcycle_surface.release_gate import ReleaseRiskError,validate_recent_oof_release

def target(asof,price,actual=100.,bucket=1,ads=3,days=2,iqr=10.,mid=1):
    return dict(motorcycle_model_id=mid,real_brand='Brand',real_model='Model',
        is_copy=0,model_variant='DEFAULT',age=2,condition_tier=1,
        mileage_bucket=bucket,asof=asof,price_center=price,
        actual_independent=actual,future_unique_ads=ads,
        future_observed_days=days,future_iqr_pct=iqr)

def test_single_confirmed_over_20_holds_only_when_week_complete():
    h=pd.DataFrame([target('2026-11-14',123)])
    assert audit_severe_oof_curves(h,'2026-11-20').empty
    held,audit=held_severe_curve_reasons(h,'2026-11-22')
    assert len(held)==1
    assert list(held.values())==['HOLD_OOF_SINGLE_CONFIRMED_OVER20']
    assert audit.iloc[0].severe_origins==1

def test_under20_requires_independent_ads():
    h=pd.DataFrame([target('2026-11-14',75),target('2026-11-07',130,ads=2)])
    held,_=held_severe_curve_reasons(h,'2026-11-22')
    assert list(held.values())==['HOLD_OOF_SINGLE_CONFIRMED_UNDER20']
    assert audit_severe_oof_curves(pd.DataFrame([target('2026-11-14',130,days=1)]),'2026-11-22').empty
    assert audit_severe_oof_curves(pd.DataFrame([target('2026-11-14',130,iqr=21)]),'2026-11-22').empty

def test_two_cells_one_origin_do_not_count_as_two_recovery_weeks():
    h=pd.DataFrame([target('2026-10-24',125),
                    target('2026-10-31',108,bucket=1),
                    target('2026-10-31',109,bucket=2)])
    held,audit=held_severe_curve_reasons(h,'2026-11-10')
    assert len(held)==1
    assert audit.iloc[0].recovery_clean_weeks==1

def test_recovery_after_two_clean_completed_origins():
    h=pd.DataFrame([target('2026-10-24',125),
                    target('2026-10-31',110),target('2026-11-07',111)])
    held,audit=held_severe_curve_reasons(h,'2026-11-14')
    assert not held and not bool(audit.iloc[0].held)
    h=pd.concat([h,pd.DataFrame([target('2026-11-14',128)])],ignore_index=True)
    held,_=held_severe_curve_reasons(h,'2026-11-22')
    assert len(held)==1

def test_candidate_hindsight_cannot_override_global_risk():
    rows=[]
    for date in ['2026-11-07','2026-11-14']:
        for mid in range(1,5):
            for j in range(30):
                row=target(date,125 if mid==1 and j<2 else 102,
                           bucket=j%9,mid=mid)
                row['fold_publication_allowed']=True
                rows.append(row)
    b=pd.DataFrame(rows)
    current=b[CURVE].drop_duplicates().copy()
    current['publication_allowed']=current.motorcycle_model_id.ne(1)
    class Builder: pass
    builder=Builder()
    builder.backtest_cells=b
    builder._surface_df=current
    with pytest.raises(ReleaseRiskError) as error:
        validate_recent_oof_release(builder,min_motorcycles=2,
                                    min_targets=40,repeats=300,
                                    max_tail20_upper_pct=0.5)
    report=error.value.report
    assert report['historical_policy_risk']['bad20']>0
    assert report['current_candidate_risk_audit']['bad20']==0
