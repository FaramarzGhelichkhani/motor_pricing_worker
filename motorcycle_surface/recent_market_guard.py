"""Post-surface market-evidence veto for USED motorcycle prices.

A price with *strong observed overpricing evidence* is not published. All
mileage buckets of the same (identity, variant, age, condition) curve receive
one decision. No direct ad is ever required for an unobserved bucket: the
upstream Stage-1 age reference + Stage-2 residual remain its prediction, and
upstream time-completed OOF model gates remain its validation.

'Observed' means discrepancies to standardized asking prices before `asof`,
not a guarantee about unobserved future transaction prices. All thresholds
refer to prediction/asking-price ratios (positive => overprice).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CELL = ['motorcycle_model_id','real_brand','real_model','is_copy',
        'model_variant','age','condition_tier','mileage_bucket']
CURVE = CELL[:-1]
LOOKBACK_DAYS = 30
MIN_ADS = 3
MIN_DAYS = 2
MAX_PRICE_IQR_PCT = 20.0
OVERPRICE_RATIO = 1.20
# Mixed-mileage evidence needs more observations than an exact cell.
MIN_CURVE_ADS = 6
MIN_CURVE_DAYS = 3
MAX_CURVE_RATIO_IQR_PCT = 30.0


def _normalize(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    absent = set(CELL) - set(x.columns)
    if absent:
        raise ValueError(f'Missing price coordinates: {sorted(absent)}')
    x['motorcycle_model_id'] = pd.to_numeric(x.motorcycle_model_id, errors='coerce').astype('Int64').astype(str)
    x['is_copy'] = pd.to_numeric(x.is_copy, errors='coerce').fillna(0).astype(int)
    for col in ('real_brand','real_model','model_variant'):
        x[col] = x[col].fillna('').astype(str)
    for col in ('age','condition_tier','mileage_bucket'):
        x[col] = pd.to_numeric(x[col],errors='coerce')
    return x


def detect_observed_over20(surface: pd.DataFrame, ads: pd.DataFrame, asof):
    """Return {(curve key)} and transparent evidence stats.

    Exact cell proof: >=3 unique ads on >=2 days, stable quote IQR <=20%,
    price > 1.20 times BOTH the sample median and its 75th percentile.
    This checks that the high-price part of the observed market is ALSO
    cheaper than prediction, mitigating rejection from one cheap listing.

    Pooled curve proof: >=6 unique ads on >=3 days aligned to each ad's OWN
    mileage-bucket prediction; 25th percentile ratio >1.20 and normalized
    ratio IQR <=30%. This allows evidence across multiple mileage buckets.
    """
    if surface is None or surface.empty or ads is None or ads.empty:
        return set(), pd.DataFrame()
    if not {'date','token','price'}.issubset(ads):
        raise ValueError('Missing date/token/price in cleaned used ads')
    if 'price_center' not in surface:
        raise ValueError('Missing final price_center')
    s = _normalize(surface)
    if s.duplicated(CELL).any():
        raise ValueError('Duplicate surface cell coordinates')
    s['price_center'] = pd.to_numeric(s.price_center,errors='coerce')
    s = s[s.price_center.gt(0) & np.isfinite(s.price_center)]
    a = _normalize(ads)
    if 'source' in a:
        a = a[a.source.astype('string').str.lower().str.strip().eq('ads')]
    a['date'] = pd.to_datetime(a.date,errors='coerce').dt.normalize()
    a['price'] = pd.to_numeric(a.price,errors='coerce')
    end = pd.Timestamp(asof).normalize()
    a = a[a.date.ge(end - pd.Timedelta(days=LOOKBACK_DAYS)) & a.date.lt(end) &
          a.price.gt(0) & np.isfinite(a.price) & a.token.notna()].copy()
    # A relisted token is ONE independent observation; latest quote wins.
    a = a.sort_values('date').drop_duplicates('token', keep='last')
    if a.empty or s.empty:
        return set(), pd.DataFrame()
    m = a.merge(s[CELL+['price_center']],on=CELL,how='inner',validate='many_to_one')
    if m.empty:
        return set(), pd.DataFrame()
    m['ratio'] = m.price_center/m.price
    # Exact-cell comparison. No distinction between new and used factory quote:
    # only source=ads rows reach this point.
    exact = m.groupby(CELL,dropna=False).agg(
        ads_n=('token','nunique'), days_n=('date','nunique'),
        median_ad=('price','median'), q25_ad=('price',lambda z:z.quantile(.25)),
        q75_ad=('price',lambda z:z.quantile(.75)),
        prediction=('price_center','first'),
    ).reset_index()
    exact['iqr_pct'] = 100*(exact.q75_ad-exact.q25_ad)/exact.median_ad
    exact['proven'] = (exact.ads_n.ge(MIN_ADS) & exact.days_n.ge(MIN_DAYS) &
                       exact.iqr_pct.le(MAX_PRICE_IQR_PCT) &
                       exact.prediction.gt(OVERPRICE_RATIO*exact.median_ad) &
                       exact.prediction.gt(OVERPRICE_RATIO*exact.q75_ad))
    exact['reason'] = 'EXACT_CELL'
    # Pool matched quotes across the curve AFTER normalization by the price
    # predicted for their respective mileage bucket (never pool raw prices).
    pooled = m.groupby(CURVE,dropna=False).agg(
        ads_n=('token','nunique'), days_n=('date','nunique'),
        median_ratio=('ratio','median'),
        q25_ratio=('ratio',lambda z:z.quantile(.25)),
        q75_ratio=('ratio',lambda z:z.quantile(.75)),
    ).reset_index()
    pooled['ratio_iqr_pct'] = 100*(pooled.q75_ratio-pooled.q25_ratio)/pooled.median_ratio
    pooled['proven'] = (pooled.ads_n.ge(MIN_CURVE_ADS) &
                        pooled.days_n.ge(MIN_CURVE_DAYS) &
                        pooled.median_ratio.gt(OVERPRICE_RATIO) &
                        pooled.q25_ratio.gt(OVERPRICE_RATIO) &
                        pooled.ratio_iqr_pct.le(MAX_CURVE_RATIO_IQR_PCT))
    pooled['reason'] = 'MULTI_BUCKET_CURVE'
    ex = exact.loc[exact.proven,CELL+['ads_n','days_n','iqr_pct','reason']]
    pp = pooled.loc[pooled.proven,CURVE+['ads_n','days_n','ratio_iqr_pct','reason']]
    rejected = pd.concat([ex[CURVE], pp[CURVE]],ignore_index=True).drop_duplicates()
    evidence = pd.concat([
        ex[CURVE+['reason','ads_n','days_n']].assign(scope='cell'),
        pp[CURVE+['reason','ads_n','days_n']].assign(scope='curve')
    ],ignore_index=True)
    log.info('Observed market over20: asof=%s matched=%s exact=%s pooled=%s curves=%s',
             end.date(),len(m),len(ex),len(pp),len(rejected))
    return set(map(tuple,rejected[CURVE].itertuples(index=False,name=None))),evidence


def apply_recent_market_guard(surface: pd.DataFrame, ads: pd.DataFrame, asof) -> pd.DataFrame:
    """Veto confirmed bad complete curves; preserve upstream reliability/OOT checks."""
    needed = {'is_reliable','publication_allowed','publication_status','publication_reason_code'}
    if surface is None or surface.empty:
        raise ValueError('Cannot publish empty surface')
    if missing := needed - set(surface.columns):
        raise ValueError(f'Publication policy must run first: {sorted(missing)}')
    bad,_ = detect_observed_over20(surface,ads,asof)
    out=surface.copy()
    if not bad:
        return out
    normalized=_normalize(surface)
    held=normalized[CURVE].apply(tuple,axis=1).isin(bad).to_numpy()
    newly=held & surface.publication_allowed.astype(bool).to_numpy()
    out.loc[held,'publication_allowed']=False
    out.loc[held,'is_reliable']=False
    out.loc[newly,'publication_status']='HOLD_OBSERVED_OVER20'
    out.loc[newly,'publication_reason_code']='HOLD_OBSERVED_OVER20'
    if not (out.is_reliable.astype(bool)==out.publication_allowed.astype(bool)).all():
        raise AssertionError('is_reliable differs from publication_allowed')
    if out.groupby(CURVE,dropna=False).publication_allowed.nunique().gt(1).any():
        raise AssertionError('Partial mileage curve in publication')
    log.info('Observed market over20 blocked %d extra cells / %d curves',newly.sum(),len(bad))
    return out

# Public helper kept for scripts built against the prior guard module.
def recent_overpriced_curves(surface: pd.DataFrame, ads: pd.DataFrame, asof) -> set[tuple]:
    return detect_observed_over20(surface, ads, asof)[0]
