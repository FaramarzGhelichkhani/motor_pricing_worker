"""One publication authority for USED motorcycle prices.

The guard is selective, not a guarantee about unobserved future sale prices.
All source history is required to have completed BEFORE the asof snapshot.
One decision per complete (model, copy, variant, age, condition) curve.
No exact mileage-cell ads are required. Factory quotes never participate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

POLICY_VERSION = 'USED_RELEASE_FINAL_1'
IDENTITY = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy']
GROUP = IDENTITY + ['model_variant']
CURVE = GROUP + ['age', 'condition_tier']
CELL = CURVE + ['mileage_bucket']


@dataclass(frozen=True)
class PublicationConfig:
    # Decision parameters are empirical guardrails, not a per-price accuracy warranty.
    # 3 unique ads in >=2 days reduces false model holds from unstable targets.
    # Never require direct examples for every mileage bucket.
    min_model_recent_ads: int = 8
    min_variant_recent_ads: int = 3
    min_parent_ads: int = 3
    min_oof_cells_to_reject: int = 5
    min_oof_weeks_to_reject: int = 2
    max_oof_median_abs_pct: float = 8.0
    severe_error_pct: float = 15.0
    min_repeated_severe: int = 2
    max_severe_share: float = 0.15
    # Structural guard, not a statistical forecast error certificate.
    max_curve_isotonic_pct: float = 15.0
    max_mileage_bucket: int = 8
    min_future_ads: int = 3
    min_future_days: int = 2
    max_future_iqr_pct: float = 20.0


def _norm(df):
    d = df.copy()
    d['motorcycle_model_id'] = pd.to_numeric(d['motorcycle_model_id'], errors='coerce').astype('Int64').astype(str)
    d['is_copy'] = pd.to_numeric(d['is_copy'], errors='coerce').fillna(0).astype(int)
    for name in ['real_brand', 'real_model', 'model_variant']:
        d[name] = d[name].fillna('').astype(str)
    return d


def build_evidence(history, asof, cfg: PublicationConfig):
    """Historic future-week outcomes. Never includes outcomes unavailable by asof."""
    cols = CELL + ['asof', 'price_center', 'actual_independent',
                   'future_unique_ads', 'future_observed_days', 'future_iqr_pct']
    if history is None or history.empty:
        return pd.DataFrame(columns=IDENTITY + ['oof_n','oof_weeks','oof_median_abs_pct',
                                                'oof_bad15','oof_over15','oof_under15','oof_rejected'])
    if missing := (set(cols) - set(history.columns)):
        raise ValueError(f'Missing backtest evidence fields: {sorted(missing)}')
    h = _norm(history)
    h['_origin'] = pd.to_datetime(h['asof'], errors='coerce').dt.normalize()
    max_completed = pd.Timestamp(asof).normalize() - pd.Timedelta(days=7)
    h = h[h['_origin'].notna() & h['_origin'].le(max_completed)].copy()
    for col in ['actual_independent','price_center','future_unique_ads','future_observed_days','future_iqr_pct']:
        h[col] = pd.to_numeric(h[col], errors='coerce')
    h = h.loc[(h.actual_independent.gt(0) & h.price_center.gt(0) &
               np.isfinite(h.actual_independent) & np.isfinite(h.price_center) &
               h.future_unique_ads.ge(cfg.min_future_ads) &
               h.future_observed_days.ge(cfg.min_future_days) &
               h.future_iqr_pct.between(0, cfg.max_future_iqr_pct))].copy()
    h = h.sort_values('_origin').drop_duplicates(['_origin'] + CELL, keep='last')
    if h.empty:
        return pd.DataFrame(columns=IDENTITY + ['oof_n','oof_weeks','oof_median_abs_pct',
                                                'oof_bad15','oof_over15','oof_under15','oof_rejected'])
    h['_err'] = 100 * (h.price_center / h.actual_independent - 1)
    h['_abs'] = h._err.abs()
    ev = h.groupby(IDENTITY, dropna=False).agg(
        oof_n=('_err','size'), oof_weeks=('_origin','nunique'),
        oof_median_abs_pct=('_abs','median'),
        oof_bad15=('_abs',lambda z: int((z>cfg.severe_error_pct).sum())),
        oof_over15=('_err',lambda z: int((z>cfg.severe_error_pct).sum())),
        oof_under15=('_err',lambda z: int((z< -cfg.severe_error_pct).sum())),
    ).reset_index()
    qualified = ev.oof_n.ge(cfg.min_oof_cells_to_reject) & ev.oof_weeks.ge(cfg.min_oof_weeks_to_reject)
    ev['oof_rejected'] = (qualified & (
        ev.oof_median_abs_pct.gt(cfg.max_oof_median_abs_pct) |
        (ev.oof_bad15.ge(cfg.min_repeated_severe) &
         (ev.oof_bad15 / ev.oof_n).gt(cfg.max_severe_share))))
    return ev


def confirmed_20pct_curves(history, asof, cfg: PublicationConfig):
    """Find historically confirmed bad price curves, without using future labels.

    A curve is blocked only after its independent next-week targets show the
    SAME direction of error >20% in at least two *completed* weekly origins.
    Evidence must have >=3 unique ads, >=2 observed days and IQR <=20%.
    No direct ads are required for other mileage buckets of the curve.

    This CANNOT identify an error in a future week which was never previously
    observed; historical evidence is necessarily a lagging signal.
    """
    if history is None or history.empty:
        return set()
    required = set(CURVE + ['asof', 'price_center', 'actual_independent',
                            'future_unique_ads', 'future_observed_days',
                            'future_iqr_pct'])
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f'Missing historical price evidence: {sorted(missing)}')
    h = _norm(history)
    h['_origin'] = pd.to_datetime(h['asof'], errors='coerce').dt.normalize()
    completed_through = pd.Timestamp(asof).normalize() - pd.Timedelta(days=7)
    h = h.loc[h['_origin'].notna() & h['_origin'].le(completed_through)].copy()
    for col in ['price_center', 'actual_independent', 'future_unique_ads',
                'future_observed_days', 'future_iqr_pct']:
        h[col] = pd.to_numeric(h[col], errors='coerce')
    h = h.loc[
        h.price_center.gt(0) & np.isfinite(h.price_center) &
        h.actual_independent.gt(0) & np.isfinite(h.actual_independent) &
        h.future_unique_ads.ge(3) & h.future_observed_days.ge(2) &
        h.future_iqr_pct.between(0, 20)
    ].copy()
    if h.empty:
        return set()
    # Only one observation per cell and forecast origin.  Two mileage buckets
    # from a single week must not masquerade as two independent weeks.
    h = h.sort_values('_origin').drop_duplicates(['_origin'] + CELL, keep='last')
    error = 100.0 * (h.price_center / h.actual_independent - 1.0)
    h['_side'] = np.select([error > 20.0, error < -20.0],
                           ['OVER20', 'UNDER20'], default='OK')
    h = h[h['_side'].ne('OK')]
    if h.empty:
        return set()
    recurring = (h.groupby(CURVE + ['_side'], dropna=False)['_origin']
                  .nunique().reset_index(name='_weeks'))
    recurring = recurring[recurring['_weeks'].ge(2)]
    return set(map(tuple, recurring[CURVE].itertuples(index=False, name=None)))


def apply_publication(surface, asof, history, cfg: PublicationConfig = PublicationConfig()):
    if surface is None or surface.empty:
        raise ValueError('Cannot publish an empty price grid')
    needed = set(CELL + ['max_grid_bucket','price_center','n_group_all',
                         'motorcycle_n30','group_n30','is_factory'])
    if missing := (needed - set(surface.columns)):
        raise ValueError(f'Missing required surface fields: {sorted(missing)}')
    d = _norm(surface)
    if d.is_factory.fillna(False).astype(bool).any():
        raise ValueError('Factory data cannot enter used-price publication')
    if d.duplicated(CELL).any():
        raise ValueError('Duplicate price coordinates')
    if 'asof' in d:
        times = pd.to_datetime(d['asof'], errors='coerce').dt.normalize()
        if times.isna().any() or times.nunique()!=1 or times.iloc[0]!=pd.Timestamp(asof).normalize():
            raise ValueError('Mixed or incompatible snapshot asof')
    hist = build_evidence(history, asof, cfg)
    bad = set(map(tuple, hist.loc[hist.oof_rejected, IDENTITY].itertuples(index=False,name=None)))
    risk_map = hist.set_index(IDENTITY).to_dict('index') if len(hist) else {}
    midkey = d[IDENTITY].apply(tuple,axis=1)
    curvekey = d[CURVE].apply(tuple,axis=1)
    confirmed_severe = curvekey.isin(confirmed_20pct_curves(history, asof, cfg))
    d['oof_n'] = midkey.map(lambda k: risk_map.get(k,{}).get('oof_n',0)).fillna(0).astype(int)
    d['oof_weeks'] = midkey.map(lambda k: risk_map.get(k,{}).get('oof_weeks',0)).fillna(0).astype(int)
    d['oof_median_abs_pct'] = midkey.map(lambda k: risk_map.get(k,{}).get('oof_median_abs_pct',np.nan))
    d['oof_over15'] = midkey.map(lambda k: risk_map.get(k,{}).get('oof_over15',0)).fillna(0).astype(int)
    d['oof_under15'] = midkey.map(lambda k: risk_map.get(k,{}).get('oof_under15',0)).fillna(0).astype(int)
    badmodel = midkey.isin(bad)
    n30 = pd.to_numeric(d.motorcycle_n30,errors='coerce').fillna(0)
    n30variant = pd.to_numeric(d.group_n30,errors='coerce').fillna(0)
    variant_support = pd.to_numeric(d.n_group_all,errors='coerce').fillna(0)
    model_weak = n30.lt(cfg.min_model_recent_ads)
    variant_weak = n30variant.lt(cfg.min_variant_recent_ads) | variant_support.lt(cfg.min_parent_ads)
    price = pd.to_numeric(d.price_center,errors='coerce')
    adjusted = pd.to_numeric(d.adjusted_price,errors='coerce') if 'adjusted_price' in d else price
    ref = (pd.to_numeric(d.reference_price,errors='coerce') if 'reference_price' in d
           else np.exp(pd.to_numeric(d.age_ref_log,errors='coerce')) if 'age_ref_log' in d
           else pd.Series(np.nan,index=d.index))
    year = pd.to_numeric(d.production_year,errors='coerce') if 'production_year' in d else pd.Series(1400,index=d.index)
    invalid = (~np.isfinite(price)|price.le(0)|~np.isfinite(adjusted)|
               ~np.isclose(price,adjusted,rtol=1e-8,atol=1e-6)|
               ~np.isfinite(ref)|ref.le(0)|
               ~year.between(1300,1600)|~year.mod(1).eq(0)|
               d.motorcycle_model_id.isin(['<NA>', 'UNKNOWN']))
    iso = pd.to_numeric(d.isotonic_change_pct,errors='coerce').abs() if 'isotonic_change_pct' in d else pd.Series(0.,index=d.index)
    invalid |= iso.gt(cfg.max_curve_isotonic_pct)
    invalid_curves = set(curvekey[invalid])
    incomplete = set()
    for key, frame in d.groupby(CURVE,dropna=False,sort=False):
        caps = pd.to_numeric(frame.max_grid_bucket,errors='coerce').dropna().unique()
        if len(caps)!=1 or not float(caps[0]).is_integer() or not (0<=caps[0]<=cfg.max_mileage_bucket):
            incomplete.add(key);continue
        start = 0 if int(frame.condition_tier.iloc[0])==1 else 1
        expected = set(range(start,int(caps[0])+1))
        got = pd.to_numeric(frame.mileage_bucket,errors='coerce')
        if got.isna().any() or not got.mod(1).eq(0).all() or set(got.astype(int))!=expected:
            incomplete.add(key);continue
        vals=frame.sort_values('mileage_bucket').price_center.to_numpy(dtype=float)
        if np.any(np.diff(vals)>1e-5):
            invalid_curves.add(key)
    curve_invalid = curvekey.isin(invalid_curves)
    curve_incomplete = curvekey.isin(incomplete)
    # Never publish a partial curve.
    d['publication_allowed'] = (~badmodel & ~model_weak & ~variant_weak &
                                ~curve_invalid & ~curve_incomplete & ~confirmed_severe).astype(bool)
    d['is_reliable']=d['publication_allowed'].copy()
    d['publication_status'] = np.select(
        [curve_invalid,curve_incomplete,badmodel,confirmed_severe,model_weak,variant_weak],
        ['HOLD_INVALID_CURVE','HOLD_INCOMPLETE_CURVE','HOLD_OOF_SEVERE','HOLD_CONFIRMED_OVER_UNDER_20',
         'HOLD_WEAK_MODEL_MARKET','HOLD_WEAK_VARIANT_MARKET'],
        default='PUBLISH_OOF_CHECKED')
    unseen=d.oof_n.eq(0) & d.publication_allowed
    d.loc[unseen,'publication_status']='PUBLISH_MARKET_SUPPORTED_UNTESTED'
    d['publication_reason_code']=d['publication_status']
    d['policy_version']=POLICY_VERSION
    # Critical: label untested models transparently rather than claiming historical validation.
    d['publication_evidence_source']=np.where(d.oof_n.gt(0),'TEMPORAL_OOF','MARKET_SUPPORT_ONLY')
    if d.groupby(CURVE,dropna=False).publication_allowed.nunique().max()>1:
        raise AssertionError('Partial-mileage publication forbidden')
    d['is_copy'] = surface['is_copy'].to_numpy()
    return d
