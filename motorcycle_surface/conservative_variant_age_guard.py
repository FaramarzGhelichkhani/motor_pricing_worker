"""Causal conservative model/variant/age publication guard.

Never replace local severe or systemic vetoes. This layer only removes an
already-approved complete variant-age grid when the same age lacks direct
market/OOF evidence and neither interpolation nor single-step extrapolation
is adequately supported, or when a low-sample age has a sharp price jump.

Evidence only from completed earlier OOF origins (asof <= decision date-7).
Metadata distinguishes direct variant-age OOF from parent/variant evidence.
Original surface keys, source data and predicted prices are never modified.
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
IDENTITY = ['motorcycle_model_id','real_brand','real_model','is_copy']
VARIANT = IDENTITY + ['model_variant']
AGE = VARIANT + ['age']
CURVE = AGE + ['condition_tier']
CELL = CURVE + ['mileage_bucket']

@dataclass(frozen=True)
class ConservativeAgeConfig:
    min_anchor_ads120: int = 6
    bracket_distance_years: int = 2
    max_sparse_ads120: int = 6
    sharp_jump_pct: float = 25.0
    min_one_step_ads120: int = 20
    max_one_step_jump_pct: float = 15.0
    min_variant_oof_targets: int = 10
    min_variant_oof_weeks: int = 2
    min_future_ads: int = 3
    min_future_days: int = 2
    max_future_iqr_pct: float = 20.0
    mode: str = 'enforce' # 'enforce' | 'audit'


def _keys(x):
    d = x.copy()
    d['motorcycle_model_id'] = (pd.to_numeric(d.motorcycle_model_id, errors='coerce')
                                .astype('Int64').astype(str))
    copy = pd.to_numeric(d.is_copy, errors='coerce')
    if copy.isna().any() or not copy.isin([0,1]).all():
        raise ValueError('Invalid is_copy in conservative age guard')
    d['is_copy'] = copy.astype('int8')
    for name in ('real_brand','real_model','model_variant'):
        d[name] = d[name].fillna('').astype(str)
    for name in ('age','condition_tier','mileage_bucket'):
        if name in d:
            d[name] = pd.to_numeric(d[name], errors='coerce')
    return d


def _completed_history(history, asof, cfg):
    cols = CELL + ['asof','price_center','actual_independent',
                   'future_unique_ads','future_observed_days','future_iqr_pct']
    if history is None or history.empty:
        return pd.DataFrame(columns=CELL + ['_origin'])
    missing = set(cols)-set(history.columns)
    if missing:
        raise ValueError(f'Missing age OOF evidence columns: {sorted(missing)}')
    h = _keys(history[cols])
    h['_origin'] = pd.to_datetime(h['asof'],errors='coerce').dt.normalize()
    latest = pd.Timestamp(asof).normalize() - pd.Timedelta(days=7)
    h = h[h['_origin'].notna() & h['_origin'].le(latest)].copy()
    for name in ('price_center','actual_independent','future_unique_ads',
                 'future_observed_days','future_iqr_pct'):
        h[name] = pd.to_numeric(h[name],errors='coerce')
    h = h.loc[h.price_center.gt(0) & np.isfinite(h.price_center) &
              h.actual_independent.gt(0) & np.isfinite(h.actual_independent) &
              h.future_unique_ads.ge(cfg.min_future_ads) &
              h.future_observed_days.ge(cfg.min_future_days) &
              h.future_iqr_pct.between(0,cfg.max_future_iqr_pct)].copy()
    return h.sort_values('_origin').drop_duplicates(['_origin']+CELL,keep='last')


def _counts(h, keys, stem):
    if h.empty:
        return pd.DataFrame(columns=keys+[stem+'_n',stem+'_weeks'])
    return (h.groupby(keys,as_index=False,dropna=False)
            .agg(**{stem+'_n':('_origin','size'),
                    stem+'_weeks':('_origin','nunique')}))


def _neighbor_support(summary, distance, floor):
    """Two-sided strong anchors at +/- distance, inside the same variant."""
    supported={}
    for direction,sgn in [('older',1),('younger',-1)]:
        yes=np.zeros(len(summary),dtype=bool)
        for step in range(1,distance+1):
            r=summary[AGE+['age_n120']].copy()
            r['age'] = r['age'] - sgn*step
            r=r.rename(columns={'age_n120':'_neighbor_support'})
            m=summary[AGE].merge(r,on=AGE,how='left',validate='one_to_one')
            yes |= m['_neighbor_support'].fillna(0).ge(floor).to_numpy()
        supported[direction]=yes
    return supported['older'] & supported['younger']


def _neighbor_one_step(summary):
    younger=summary[AGE+['age_n120']].copy()
    younger['age'] += 1
    younger=younger.rename(columns={'age_n120':'younger_n120'})
    older=summary[AGE+['age_n120']].copy()
    older['age'] -= 1
    older=older.rename(columns={'age_n120':'older_n120'})
    return (summary[AGE].merge(older,on=AGE,how='left',validate='one_to_one')
            .merge(younger,on=AGE,how='left',validate='one_to_one'))


def _adjacent_price_jumps(d):
    """Median price jumps across matching mileage and condition coordinates."""
    pts=d[CELL+['price_center']].drop_duplicates(CELL)
    if not pts.price_center.gt(0).all():
        raise ValueError('Invalid price_center in age-price shape check')
    base=pts.copy()
    older=base.copy()
    older['age'] -= 1
    older=older.rename(columns={'price_center':'older_price'})
    younger=base.copy()
    younger['age'] += 1
    younger=younger.rename(columns={'price_center':'younger_price'})
    p=base.merge(older,on=CELL,how='left',validate='one_to_one')
    p=p.merge(younger,on=CELL,how='left',validate='one_to_one')
    p['jump_older_pct']=100*(p.price_center/p.older_price-1)
    p['jump_younger_pct']=100*(p.younger_price/p.price_center-1)
    return (p.groupby(AGE,as_index=False,dropna=False)
            .agg(jump_older_pct=('jump_older_pct','median'),
                 jump_younger_pct=('jump_younger_pct','median')))


def apply_conservative_variant_age_guard(surface, history, asof,
                                        cfg: ConservativeAgeConfig = ConservativeAgeConfig()):
    if surface is None or surface.empty:
        raise ValueError('Cannot guard empty price surface')
    if cfg.mode not in ('audit','enforce'):
        raise ValueError('Unknown conservative age guard mode')
    needed = set(CELL+['n120','n30','price_center','publication_allowed',
                       'is_reliable','publication_status','publication_reason_code'])
    missing = needed-set(surface.columns)
    if missing:
        raise ValueError(f'Conservative age guard missing {sorted(missing)}')
    out=surface.copy()
    d=_keys(surface)
    for name in ('n120','n30','price_center'):
        d[name] = pd.to_numeric(d[name],errors='coerce')
    if d[['n120','n30']].isna().any().any():
        raise ValueError('Missing age-level market support')
    if d.duplicated(CELL).any():
        raise ValueError('Duplicate cells in conservative age guard')
    if d['age'].isna().any():
        raise ValueError('Missing/invalid age in conservative age guard')
    h=_completed_history(history,asof,cfg)
    summary=(d.groupby(AGE,as_index=False,dropna=False)
             .agg(age_n120=('n120','max'),age_n30=('n30','max')))
    for keys,stem in [(IDENTITY,'age_parent_oof'),(VARIANT,'variant_oof'),
                      (AGE,'variant_age_oof')]:
        if h.empty:
            summary[stem+'_n'] = 0
            summary[stem+'_weeks'] = 0
        else:
            evidence=_counts(h,keys,stem)
            summary=summary.merge(evidence,on=keys,how='left',validate='many_to_one')
            for col in (stem+'_n',stem+'_weeks'):
                summary[col]=summary[col].fillna(0).astype(int)
    bracket=_neighbor_support(summary,cfg.bracket_distance_years,cfg.min_anchor_ads120)
    adjacent=_neighbor_one_step(summary)
    jumps=_adjacent_price_jumps(d)
    summary=summary.merge(adjacent[AGE+['older_n120','younger_n120']],on=AGE,how='left',validate='one_to_one')
    summary=summary.merge(jumps,on=AGE,how='left',validate='one_to_one')
    max_jump=summary[['jump_older_pct','jump_younger_pct']].max(axis=1)
    strong_one_step=((summary.older_n120.ge(cfg.min_one_step_ads120) &
                      summary.jump_older_pct.between(-1e-8,cfg.max_one_step_jump_pct)) |
                     (summary.younger_n120.ge(cfg.min_one_step_ads120) &
                      summary.jump_younger_pct.between(-1e-8,cfg.max_one_step_jump_pct)))
    valid_variant_oof=(summary.variant_oof_n.ge(cfg.min_variant_oof_targets) &
                       summary.variant_oof_weeks.ge(cfg.min_variant_oof_weeks))
    extrapolation_protected=strong_one_step & valid_variant_oof
    no_direct=summary.variant_age_oof_n.eq(0)
    no_market=no_direct & summary.age_n120.fillna(0).le(0)
    unsupported=no_market & ~bracket & ~extrapolation_protected
    sharp_jump=(no_direct & summary.age_n120.fillna(0).lt(cfg.max_sparse_ads120) &
                max_jump.gt(cfg.sharp_jump_pct) & ~bracket)
    summary['age_support_bracketed']=bracket
    summary['age_extrapolation_protected']=extrapolation_protected
    summary['age_neighbor_jump_pct']=max_jump
    summary['age_conservative_reason']=np.select(
        [sharp_jump,unsupported],
        ['HOLD_AGE_SPARSE_SHARP_JUMP','HOLD_AGE_ZERO_UNSUPPORTED'],default='')
    summary['age_evidence_source']=np.select([
        summary.variant_age_oof_n.gt(0),summary.age_n120.gt(0),
        bracket,extrapolation_protected,summary.variant_oof_n.gt(0),
        summary.age_parent_oof_n.gt(0)],
        ['DIRECT_VARIANT_AGE_OOF','MARKET_AGE_SUPPORTED_NO_DIRECT_OOF',
         'SUPPORTED_AGE_INTERPOLATION','SUPPORTED_ONE_YEAR_EXTRAPOLATION',
         'VARIANT_OOF_OTHER_AGE_ONLY','PARENT_MODEL_OOF_ONLY'],
        default='UNSUPPORTED_AGE_EXTRAPOLATION')
    fields=['age_n120','age_n30','age_parent_oof_n','age_parent_oof_weeks',
            'variant_oof_n','variant_oof_weeks','variant_age_oof_n',
            'variant_age_oof_weeks','age_support_bracketed',
            'age_extrapolation_protected','age_neighbor_jump_pct',
            'age_conservative_reason','age_evidence_source']
    annotated=d[AGE].merge(summary[AGE+fields],on=AGE,
                            how='left',validate='many_to_one',sort=False)
    if len(annotated)!=len(out) or annotated['age_evidence_source'].isna().any():
        raise AssertionError('Variant-age evidence join was incomplete')
    for name in fields:
        out[name]=annotated[name].to_numpy()
    if h.empty:
        out['curve_oof_n'] = 0
        out['curve_oof_weeks'] = 0
    else:
        cv=_counts(h,CURVE,'curve_oof')
        c=d[CURVE].merge(cv,on=CURVE,how='left',validate='many_to_one',sort=False)
        out['curve_oof_n']=c['curve_oof_n'].fillna(0).astype(int).to_numpy()
        out['curve_oof_weeks']=c['curve_oof_weeks'].fillna(0).astype(int).to_numpy()
    out.loc[out['curve_oof_n'].gt(0),'age_evidence_source']='DIRECT_CURVE_OOF'
    hold = out['age_conservative_reason'].ne('') & out.publication_allowed.astype(bool)
    if cfg.mode=='enforce' and hold.any():
        out.loc[hold,['publication_allowed','is_reliable']]=False
        out.loc[hold,'publication_status']=out.loc[hold,'age_conservative_reason']
        out.loc[hold,'publication_reason_code']=out.loc[hold,'age_conservative_reason']
    original = surface.publication_allowed.astype(bool).to_numpy()
    current = out.publication_allowed.astype(bool).to_numpy()
    newly = original & ~current
    if cfg.mode=='enforce' and np.any(newly != hold.to_numpy()):
        raise AssertionError('Unexpected conservative age hold selection')
    if cfg.mode=='audit' and not np.array_equal(original,current):
        raise AssertionError('Age audit mutated publication decisions')
    counts = d[AGE].assign(_approved=original.astype(int),_held=newly.astype(int))
    coverage = counts.groupby(AGE,dropna=False)[['_approved','_held']].sum()
    if (coverage._held.gt(0) & coverage._held.lt(coverage._approved)).any():
        raise AssertionError('Partial variant-age publication veto')
    if out.groupby([d[col] for col in CURVE],dropna=False).publication_allowed.nunique().gt(1).any():
        raise AssertionError('Partial mileage curve in conservative age guard')
    if not out.publication_allowed.astype(bool).eq(out.is_reliable.astype(bool)).all():
        raise AssertionError('Reliability flag mismatch in conservative age guard')
    log.info('Conservative variant-age guard: mode=%s proposed=%s newly_held=%s ages=%s',
             cfg.mode,int(out.age_conservative_reason.ne('').sum()),
             int(hold.sum()) if cfg.mode=='enforce' else 0,
             len(summary))
    return out
