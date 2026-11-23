"""Causal single-confirmed OOF veto with evidence-based curve recovery.

Only independent, stable, fully completed next-week targets count.
A held curve is reinstated after two subsequent qualifying clean origins.
No future labels are available to the decision at its asof.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CURVE = ['motorcycle_model_id','real_brand','real_model','is_copy',
         'model_variant','age','condition_tier']
CELL = CURVE + ['mileage_bucket']

def audit_severe_oof_curves(history, asof, *, error_pct=20.0,
                             recovery_error_pct=15.0,
                             recovery_weeks=2, min_ads=3, min_days=2,
                             max_iqr_pct=20.0):
    """Return curve-level holds/recoveries, never counting two cells as two weeks."""
    cols = CELL + ['asof','price_center','actual_independent',
                   'future_unique_ads','future_observed_days','future_iqr_pct']
    audit_cols = CURVE + ['held','reason','last_severe_origin',
                          'last_severe_error_pct','severe_origins',
                          'recovery_clean_weeks','eligible_origins']
    if history is None or history.empty:
        return pd.DataFrame(columns=audit_cols)
    missing = set(cols) - set(history.columns)
    if missing:
        raise ValueError(f'Missing OOF severe guard columns: {sorted(missing)}')
    h = history[cols].copy()
    h['motorcycle_model_id'] = (pd.to_numeric(h['motorcycle_model_id'], errors='coerce')
                                .astype('Int64').astype(str))
    h['is_copy'] = pd.to_numeric(h['is_copy'], errors='coerce').fillna(0).astype(int)
    for c in ['real_brand','real_model','model_variant']:
        h[c] = h[c].fillna('').astype(str)
    for c in ['age','condition_tier','mileage_bucket']:
        h[c] = pd.to_numeric(h[c], errors='coerce')
    h['_origin'] = pd.to_datetime(h['asof'], errors='coerce').dt.normalize()
    cutoff = pd.Timestamp(asof).normalize() - pd.Timedelta(days=7)
    h = h[h['_origin'].notna() & h['_origin'].le(cutoff)].copy()
    for c in ['price_center','actual_independent','future_unique_ads',
              'future_observed_days','future_iqr_pct']:
        h[c] = pd.to_numeric(h[c], errors='coerce')
    h = h.loc[(h.price_center.gt(0) & np.isfinite(h.price_center) &
               h.actual_independent.gt(0) & np.isfinite(h.actual_independent) &
               h.future_unique_ads.ge(min_ads) &
               h.future_observed_days.ge(min_days) &
               h.future_iqr_pct.between(0,max_iqr_pct))].copy()
    if h.empty:
        return pd.DataFrame(columns=audit_cols)
    h = h.sort_values('_origin').drop_duplicates(['_origin']+CELL, keep='last')
    h['_signed_error'] = 100.0*(h.price_center/h.actual_independent-1.0)
    h['_abs_error'] = h['_signed_error'].abs()
    h['_severe'] = h['_abs_error'].gt(error_pct)
    h['_clean'] = h['_abs_error'].le(recovery_error_pct)
    weekly = (h.groupby(CURVE+['_origin'], dropna=False, sort=True)
              .agg(severe=('_severe','max'),clean=('_clean','all'),
                   severe_signed_error=('_signed_error',
                       lambda s: float(s.loc[s.abs().idxmax()])))
              .reset_index())
    records=[]
    for key,g in weekly.groupby(CURVE, dropna=False, sort=False):
        g = g.sort_values('_origin')
        severe_idx = np.flatnonzero(g['severe'].to_numpy())
        if not len(severe_idx):
            continue
        last_idx = int(severe_idx[-1])
        clean_count=0
        for is_clean in g.iloc[last_idx+1:]['clean'].to_numpy():
            clean_count = clean_count+1 if bool(is_clean) else 0
        recovered = clean_count >= recovery_weeks
        signed=float(g.iloc[last_idx]['severe_signed_error'])
        reason = ('RELEASE_OOF_SEVERE_RECOVERED' if recovered else
                  ('HOLD_OOF_SINGLE_CONFIRMED_OVER20' if signed>0 else
                   'HOLD_OOF_SINGLE_CONFIRMED_UNDER20'))
        records.append(dict(zip(CURVE,key)) | dict(
            held=not recovered,reason=reason,
            last_severe_origin=str(g.iloc[last_idx]['_origin'].date()),
            last_severe_error_pct=round(signed,4),
            severe_origins=int(len(severe_idx)),
            recovery_clean_weeks=int(clean_count),
            eligible_origins=int(len(g))))
    return pd.DataFrame(records,columns=audit_cols)

def held_severe_curve_reasons(history, asof, **kwargs):
    audit = audit_severe_oof_curves(history, asof, **kwargs)
    if audit.empty:
        return {},audit
    held=audit.loc[audit.held]
    return dict(zip(held[CURVE].itertuples(index=False,name=None),
                    held['reason'])),audit
