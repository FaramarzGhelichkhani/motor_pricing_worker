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

POLICY_VERSION = 'USED_RELEASE_FINAL_5_ALL_AGE_DUPLICATE_QUALITY_GUARD'
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
    age_plateau_guard_enabled: bool = True
    # All distinct production ages are compared; quality decides which full
    # curve is held, and indistinguishable pairs are both held.
    age_plateau_price_rtol: float = 0.01
    age_plateau_min_matching_buckets: int = 3
    age_plateau_min_pair_tie_share: float = 0.90
    age_plateau_market_dominance_ratio: float = 1.25
    # A market-supported winner must clear both absolute support floors.
    # Relative ratios alone are unsafe when the competing curve has zero ads.
    age_plateau_winner_min_n120: float = 20.0
    age_plateau_winner_min_n30: float = 6.0


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



def _mid_price_column(frame: pd.DataFrame) -> str:
    """Return the sole point-price field used by adjacent-year comparisons.

    Newer schemas may call it ``price_mid``; the current production surface
    calls the same central estimate ``price_center``. Bounds are never used.
    """
    if 'price_mid' in frame.columns:
        return 'price_mid'
    if 'price_center' in frame.columns:
        return 'price_center'
    raise ValueError('Adjacent-year plateau guard requires price_mid/price_center')


def _plateau_curve_quality(surface: pd.DataFrame, history: pd.DataFrame, asof):
    """Build independent quality evidence for each complete age/condition curve.

    OOF evidence is exact to model, variant, age, condition and mileage cell.
    Market support is summarized per curve so every curve is compared on the
    same evidence dimensions. No information about the duplicate partner enters
    this quality calculation.
    """
    metrics = {}
    base = surface.copy()
    base['n120'] = pd.to_numeric(base['n120'], errors='coerce').fillna(0)
    base['n30'] = pd.to_numeric(base['n30'], errors='coerce').fillna(0)
    support = (base.groupby(CURVE, dropna=False)
               .agg(n120=('n120', 'median'), n30=('n30', 'median'),
                    bucket_count=('mileage_bucket', 'nunique'))
               .reset_index())
    for row in support.itertuples(index=False, name=None):
        key = tuple(row[:len(CURVE)])
        metrics[key] = {
            'n120': float(row[-3]), 'n30': float(row[-2]),
            'bucket_count': int(row[-1]),
            'oof_evaluable': False, 'oof_valid': False,
            'oof_n': 0, 'oof_weeks': 0,
            'oof_mdape': np.nan, 'oof_max_ape': np.nan,
        }

    if history is None or history.empty:
        return metrics

    mid_col = _mid_price_column(history)
    required = set(CURVE + ['asof', mid_col, 'actual_independent',
                            'future_unique_ads', 'future_observed_days',
                            'future_iqr_pct'])
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f'Missing exact-age OOF fields for duplicate-age quality: {sorted(missing)}')

    hist = _norm(history)
    hist['_origin'] = pd.to_datetime(hist['asof'], errors='coerce').dt.normalize()
    cutoff = pd.Timestamp(asof).normalize() - pd.Timedelta(days=7)
    hist = hist.loc[hist['_origin'].notna() & hist['_origin'].le(cutoff)].copy()
    for col in [mid_col, 'actual_independent', 'future_unique_ads',
                'future_observed_days', 'future_iqr_pct']:
        hist[col] = pd.to_numeric(hist[col], errors='coerce')
    hist = hist.loc[
        hist[mid_col].gt(0) & hist['actual_independent'].gt(0) &
        np.isfinite(hist[mid_col]) & np.isfinite(hist['actual_independent']) &
        hist['future_unique_ads'].ge(3) & hist['future_observed_days'].ge(2) &
        hist['future_iqr_pct'].between(0, 20)
    ].copy()
    if hist.empty:
        return metrics

    hist = hist.sort_values('_origin').drop_duplicates(['_origin'] + CELL, keep='last')
    hist['_ape'] = 100.0 * (hist[mid_col] / hist['actual_independent'] - 1.0).abs()
    evidence = (hist.groupby(CURVE, dropna=False)
                .agg(oof_n=('_ape', 'size'), oof_weeks=('_origin', 'nunique'),
                     oof_mdape=('_ape', 'median'), oof_max_ape=('_ape', 'max'))
                .reset_index())
    for row in evidence.itertuples(index=False, name=None):
        key = tuple(row[:len(CURVE)])
        if key not in metrics:
            continue
        n, weeks, mdape, max_ape = row[-4:]
        metrics[key].update({
            'oof_n': int(n), 'oof_weeks': int(weeks),
            'oof_mdape': float(mdape), 'oof_max_ape': float(max_ape),
            'oof_evaluable': bool(n >= 3 and weeks >= 2),
            'oof_valid': bool(n >= 3 and weeks >= 2 and mdape <= 8 and max_ape <= 20),
        })
    return metrics


def _compare_duplicate_curve_quality(
    a: dict,
    b: dict,
    market_dominance_ratio: float = 1.25,
    winner_min_n120: float = 20.0,
    winner_min_n30: float = 6.0,
) -> int:
    """Return 1 if a is stronger, -1 if b is stronger, 0 if indistinguishable.

    Quality is decided independently of similarity. Exact-age OOF evidence has
    priority. Market evidence can decide only when one curve clears absolute
    support floors AND dominates the other in both time windows. This avoids
    false wins caused by a zero denominator (e.g. 1 ad versus 0 ads).
    """
    av, bv = bool(a.get('oof_valid', False)), bool(b.get('oof_valid', False))
    ae, be = bool(a.get('oof_evaluable', False)), bool(b.get('oof_evaluable', False))

    # A curve with a qualifying exact-age OOF certificate wins over one without.
    if av != bv:
        return 1 if av else -1

    def oof_better(x, y):
        """Material OOF advantage without a worse extreme-error tail."""
        if not np.isfinite(x.get('oof_mdape', np.nan)) or not np.isfinite(y.get('oof_mdape', np.nan)):
            return False
        mdape_better = x['oof_mdape'] + 0.5 < y['oof_mdape']
        max_not_worse = (
            np.isfinite(x.get('oof_max_ape', np.nan)) and
            np.isfinite(y.get('oof_max_ape', np.nan)) and
            x['oof_max_ape'] <= y['oof_max_ape'] + 2.0
        )
        return bool(mdape_better and max_not_worse and x['oof_max_ape'] <= 20.0)

    if ae and be:
        # Two independently evaluable curves: use OOF only for a material,
        # robust advantage. If both show >20% tail error, neither is a safe
        # winner and both must be held.
        a_max_bad = a['oof_max_ape'] > 20.0
        b_max_bad = b['oof_max_ape'] > 20.0
        if a_max_bad and b_max_bad:
            return 0
        if a_max_bad != b_max_bad:
            return -1 if a_max_bad else 1
        if oof_better(a, b):
            return 1
        if oof_better(b, a):
            return -1
    elif ae != be:
        # A poor, independently evaluated curve cannot win merely on market
        # counts. The unevaluated curve may win only with strong market proof.
        evaluated = a if ae else b
        if evaluated.get('oof_max_ape', np.inf) > 20.0:
            pass  # let the market test select only the unevaluated side
        elif evaluated.get('oof_mdape', np.inf) > 8.0:
            pass
        else:
            # Evidence exists but does not meet the certificate criteria;
            # absent a material comparison, do not infer a winner.
            pass

    # Market winner: require absolute recent and historical support first,
    # then require a relative advantage in BOTH windows. The absolute floors
    # make 1-vs-0 and other tiny-denominator comparisons non-decisive.
    a_qualified = a['n120'] >= winner_min_n120 and a['n30'] >= winner_min_n30
    b_qualified = b['n120'] >= winner_min_n120 and b['n30'] >= winner_min_n30
    a_dom = (
        a_qualified and
        a['n120'] >= b['n120'] * market_dominance_ratio and
        a['n30'] >= b['n30'] * market_dominance_ratio and
        (a['n120'] > b['n120'] or a['n30'] > b['n30'])
    )
    b_dom = (
        b_qualified and
        b['n120'] >= a['n120'] * market_dominance_ratio and
        b['n30'] >= a['n30'] * market_dominance_ratio and
        (b['n120'] > a['n120'] or b['n30'] > a['n30'])
    )

    # If one curve has evaluable but failed OOF evidence, do not let that same
    # curve win on market counts; only its unevaluated competitor can be kept.
    if ae and not av:
        a_dom = False
    if be and not bv:
        b_dom = False

    if a_dom and not b_dom:
        return 1
    if b_dom and not a_dom:
        return -1
    return 0

def unsupported_adjacent_plateau_curves(
    d: pd.DataFrame,
    history: pd.DataFrame,
    asof,
    *,
    price_rtol: float = 0.01,
    min_matching_buckets: int = 3,
    min_pair_tie_share: float = 0.90,
    market_dominance_ratio: float = 1.25,
    winner_min_n120: float = 20.0,
    winner_min_n30: float = 6.0,
) -> set[tuple]:
    """Hold weaker or indistinguishable curves whenever ANY two ages plateau.

    Similarity only identifies pairs. It never decides which age is weak.
    Quality is assessed independently for each complete curve using exact-age
    OOF evidence first, then per-curve recent/120-day market support.

    All age combinations are compared (not only adjacent ages). For every
    similar pair, the weaker curve is held; if evidence cannot distinguish the
    pair, BOTH are held. The returned keys are complete CURVE keys, so the
    caller suppresses every mileage bucket of each held curve.
    """
    mid_col = _mid_price_column(d)
    required = set(CELL + [mid_col, 'n120', 'n30', 'publication_allowed'])
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f'All-age plateau gate requires {sorted(missing)}')

    live = d.loc[d.publication_allowed.fillna(False).astype(bool)].copy()
    if live.empty:
        return set()
    live['age'] = pd.to_numeric(live['age'], errors='coerce')
    live[mid_col] = pd.to_numeric(live[mid_col], errors='coerce')
    live['mileage_bucket'] = pd.to_numeric(live['mileage_bucket'], errors='coerce')
    live = live.loc[live['age'].notna() & live[mid_col].gt(0) &
                    np.isfinite(live[mid_col]) & live['mileage_bucket'].notna()].copy()
    if live.empty:
        return set()

    quality = _plateau_curve_quality(live, history, asof)
    held = set()
    compare_groups = GROUP + ['condition_tier']

    # Grouping by model/variant/copy/condition prevents cross-variant or
    # cross-condition comparisons. Every distinct age pair is examined.
    for _, group in live.groupby(compare_groups, dropna=False, sort=False):
        age_frames = {
            age: frame[['mileage_bucket', mid_col]].rename(columns={mid_col: 'mid'})
            for age, frame in group.groupby('age', sort=False)
        }
        ages = sorted(age_frames)
        for i, age_a in enumerate(ages):
            for age_b in ages[i + 1:]:
                a = age_frames[age_a]
                b = age_frames[age_b]
                shared = a.merge(b, on='mileage_bucket', how='inner',
                                 suffixes=('_a', '_b'), validate='one_to_one')
                if len(shared) < min_matching_buckets:
                    continue
                near = (shared['mid_a'] / shared['mid_b'] - 1.0).abs().le(price_rtol)
                if float(near.mean()) < min_pair_tie_share:
                    continue

                first = tuple(group.iloc[0][GROUP].tolist()) + (age_a, group.iloc[0]['condition_tier'])
                second = tuple(group.iloc[0][GROUP].tolist()) + (age_b, group.iloc[0]['condition_tier'])
                qa = quality.get(first)
                qb = quality.get(second)
                if qa is None or qb is None:
                    # Missing quality evidence cannot justify keeping either
                    # member of a duplicate pair.
                    held.update([first, second])
                    continue

                decision = _compare_duplicate_curve_quality(
                    qa, qb, market_dominance_ratio,
                    winner_min_n120, winner_min_n30,
                )
                if decision > 0:
                    held.add(second)
                elif decision < 0:
                    held.add(first)
                else:
                    held.update([first, second])

    return held
# Backward-compatible alias for scripts using the earlier helper name.
unsupported_young_plateau_curves = unsupported_adjacent_plateau_curves


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
    # All-age duplicate curves are resolved by independent per-curve quality evidence.
    d['age_plateau_held'] = False
    if cfg.age_plateau_guard_enabled:
        plateau_held = unsupported_adjacent_plateau_curves(
            d, history, asof,
            price_rtol=cfg.age_plateau_price_rtol,
            min_matching_buckets=cfg.age_plateau_min_matching_buckets,
            min_pair_tie_share=cfg.age_plateau_min_pair_tie_share,
            market_dominance_ratio=cfg.age_plateau_market_dominance_ratio,
            winner_min_n120=cfg.age_plateau_winner_min_n120,
            winner_min_n30=cfg.age_plateau_winner_min_n30,
        )
        held = curvekey.isin(plateau_held)
        d.loc[held, 'age_plateau_held'] = True
        d.loc[held, ['publication_allowed', 'is_reliable']] = False
        d.loc[held, 'publication_status'] = 'HOLD_DUPLICATE_AGE_WEAKER_OR_TIED_QUALITY'
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
