"""Hierarchical, causal systemic-risk veto for otherwise publishable used-price curves.

Level order: local curve guards (upstream) -> variant -> multi-variant model.
These tests do NOT certify individual prices; uncertain evidence never overrides a
local veto, and a separate walk-forward Bootstrap controls whole-release risk.

Each OOF curve/week counts once.  The lower confidence bound is the minimum
of a one-sided Wilson bound and two marginal cluster bootstraps (weeks/curves);
the latter keep within-block dependence. Limits must be calibrated walk-forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
import numpy as np
import pandas as pd

IDENTITY = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy']
VARIANT = IDENTITY + ['model_variant']
CURVE = VARIANT + ['age', 'condition_tier']
CELL = CURVE + ['mileage_bucket']


@dataclass(frozen=True)
class HierarchicalRiskConfig:
    min_curve_weeks: int = 12
    min_weeks: int = 4
    min_curves: int = 3
    min_bad_curves: int = 2
    min_bad_weeks: int = 2
    recent_days: int = 28
    recent_bad_weeks: int = 2
    tail15_limit: float = .15
    mdape_limit_pct: float = 8.0
    block_bootstrap_repeats: int = 400
    wilson_z: float = 1.6448536269514722  # one-sided 95%
    min_ads: int = 3
    min_days: int = 2
    max_iqr_pct: float = 20.0


def _normalize(x):
    d = x.copy()
    d['motorcycle_model_id'] = (pd.to_numeric(d.motorcycle_model_id, errors='coerce')
                                .astype('Int64').astype(str))
    d['is_copy'] = pd.to_numeric(d.is_copy, errors='coerce').fillna(0).astype(int)
    for col in ('real_brand', 'real_model', 'model_variant'):
        d[col] = d[col].fillna('').astype(str)
    for col in ('age', 'condition_tier', 'mileage_bucket'):
        if col in d:
            d[col] = pd.to_numeric(d[col], errors='coerce')
    return d


def _weekly_evidence(history, asof, cfg):
    columns = CELL + ['asof', 'price_center', 'actual_independent',
                      'future_unique_ads', 'future_observed_days', 'future_iqr_pct']
    if history is None or history.empty:
        return pd.DataFrame(columns=CURVE + ['_origin', 'ape_pct'])
    missing = set(columns) - set(history)
    if missing:
        raise ValueError(f'Hierarchical OOF guard missing {sorted(missing)}')
    h = _normalize(history[columns])
    h['_origin'] = pd.to_datetime(h['asof'], errors='coerce').dt.normalize()
    cutoff = pd.Timestamp(asof).normalize() - pd.Timedelta(days=7)
    h = h.loc[h['_origin'].notna() & h['_origin'].le(cutoff)].copy()
    for col in ('price_center', 'actual_independent', 'future_unique_ads',
                'future_observed_days', 'future_iqr_pct'):
        h[col] = pd.to_numeric(h[col], errors='coerce')
    h = h.loc[h.price_center.gt(0) & np.isfinite(h.price_center) &
              h.actual_independent.gt(0) & np.isfinite(h.actual_independent) &
              h.future_unique_ads.ge(cfg.min_ads) &
              h.future_observed_days.ge(cfg.min_days) &
              h.future_iqr_pct.between(0, cfg.max_iqr_pct)].copy()
    if h.empty:
        return pd.DataFrame(columns=CURVE + ['_origin', 'ape_pct'])
    h = h.sort_values('_origin').drop_duplicates(['_origin'] + CELL, keep='last')
    h['ape_pct'] = 100.0 * (h.price_center / h.actual_independent - 1).abs()
    # A curve/week is one correlated observation irrespective of mileage buckets.
    return (h.groupby(CURVE + ['_origin'], dropna=False, sort=False)
            .ape_pct.median().reset_index())


def _wilson_lower(k, n, z):
    if n == 0:
        return 0.0
    p = k / n
    z2 = z*z
    return max(0.0, ((p + z2/(2*n)) -
                     z*sqrt(p*(1-p)/n + z2/(4*n*n))) / (1 + z2/n))


def _bootstrap_lower(frame, block, cfg, seed):
    """One-sided marginal cluster bounds on risk share and median APE."""
    grouped = [v['ape_pct'].to_numpy(dtype=float) for _, v
               in frame.groupby(block, dropna=False, sort=True)]
    if len(grouped) < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    tail, med = [], []
    for _ in range(cfg.block_bootstrap_repeats):
        choice = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[i] for i in choice])
        tail.append(float(np.mean(sample > 15.0)))
        med.append(float(np.median(sample)))
    return (float(np.quantile(tail, .05)), float(np.quantile(med, .05)))


def _risk_stats(frame, cfg, *, seed, uncertainty=True):
    n = len(frame)
    bad = frame['ape_pct'].gt(15.0)
    # Distinct weekly/curve evidence must be checked separately from sample size.
    curves = frame[CURVE].drop_duplicates().shape[0]
    weeks = frame['_origin'].nunique()
    bad_rows = frame.loc[bad]
    nbad_curves = bad_rows[CURVE].drop_duplicates().shape[0]
    nbad_weeks = bad_rows['_origin'].nunique()
    metrics = dict(n_curve_weeks=int(n), weeks=int(weeks), curves=int(curves),
                   bad15=int(bad.sum()), bad15_curves=int(nbad_curves),
                   bad15_weeks=int(nbad_weeks),
                   tail15_rate=float(bad.mean()) if n else 0.,
                   mdape_pct=float(frame.ape_pct.median()) if n else np.nan,
                   tail15_lcb=0., mdape_lcb=0.)
    if n == 0:
        return metrics
    if uncertainty:
        wilson = _wilson_lower(int(bad.sum()), n, cfg.wilson_z)
        base = int.from_bytes(sha256(str(seed).encode()).digest()[:4], 'big')
        wt, wm = _bootstrap_lower(frame, '_origin', cfg, base)
        # Curves are a second source of dependence even across different weeks.
        c = frame.copy()
        c['_curve_key'] = list(map(tuple, c[CURVE].itertuples(index=False, name=None)))
        ct, cm = _bootstrap_lower(c, '_curve_key', cfg, base ^ 0x5A5A5A5A)
        metrics['tail15_lcb'] = min(wilson, wt, ct)
        metrics['mdape_lcb'] = min(wm, cm)
    return metrics


def _assess(frame, asof, cfg, key):
    # Skip expensive block bootstraps entirely for sparse models/variants.
    metrics = _risk_stats(frame, cfg, seed=key, uncertainty=False)
    n, weeks, curves = metrics['n_curve_weeks'], metrics['weeks'], metrics['curves']
    metrics.update(status='MODEL_RISK_INCONCLUSIVE', reason='INSUFFICIENT_BREADTH')
    if n < cfg.min_curve_weeks or weeks < cfg.min_weeks or curves < cfg.min_curves:
        return metrics
    metrics = _risk_stats(frame, cfg, seed=key, uncertainty=True)
    metrics.update(status='MODEL_RISK_INCONCLUSIVE', reason='NO_CONFIRMED_SYSTEMIC_RISK')
    cutoff = pd.Timestamp(asof).normalize() - pd.Timedelta(days=cfg.recent_days)
    recent = frame.loc[frame['_origin'].ge(cutoff)].copy()
    recent_by_week = recent.groupby('_origin').ape_pct.median()
    recent_tail = recent.loc[recent.ape_pct.gt(15.)]
    recent_tail_ok = (recent_tail['_origin'].nunique() >= cfg.recent_bad_weeks and
                      recent_tail[CURVE].drop_duplicates().shape[0] >= cfg.min_bad_curves)
    recent_mdape_ok = (int(recent_by_week.gt(cfg.mdape_limit_pct).sum()) >=
                       cfg.recent_bad_weeks and
                       recent.loc[recent.ape_pct.gt(cfg.mdape_limit_pct), CURVE]
                       .drop_duplicates().shape[0] >= cfg.min_bad_curves)
    broad = (metrics['bad15_curves'] >= cfg.min_bad_curves and
             metrics['bad15_weeks'] >= cfg.min_bad_weeks)
    tail_high = (broad and recent_tail_ok and
                 metrics['tail15_lcb'] > cfg.tail15_limit)
    median_high = (recent_mdape_ok and metrics['mdape_lcb'] > cfg.mdape_limit_pct)
    if not tail_high and not median_high:
        metrics.update(status='MODEL_RISK_ACCEPTABLE', reason='NO_CONFIRMED_SYSTEMIC_RISK')
        return metrics
    # A single week's evidence must never flip an entire model from hold to live.
    signal = 'TAIL15' if tail_high else 'MEDIAN_APE'
    for week in sorted(frame['_origin'].unique()):
        subset = frame.loc[frame['_origin'].ne(week)]
        if len(subset) < max(4, cfg.min_curve_weeks - 3) or subset['_origin'].nunique() < 3:
            metrics.update(status='MODEL_RISK_INCONCLUSIVE', reason='LOO_INSUFFICIENT')
            return metrics
        other = _risk_stats(subset, cfg, seed=f'{key}-{week}')
        if (signal == 'TAIL15' and other['tail15_lcb'] <= cfg.tail15_limit or
            signal == 'MEDIAN_APE' and other['mdape_lcb'] <= cfg.mdape_limit_pct):
            metrics.update(status='MODEL_RISK_INCONCLUSIVE', reason='LOO_UNSTABLE')
            return metrics
    metrics.update(status='MODEL_PROVEN_UNSAFE', reason='PERSISTENT_' + signal)
    return metrics


def apply_hierarchical_oof_guard(surface, history, asof,
                                 cfg: HierarchicalRiskConfig = HierarchicalRiskConfig()):
    """Return the grid with only proven systemic variant/model risk additionally held.

    Called *after* local OOF and observed-market guards. Previously held curves
    never contribute toward vetoing their healthy sibling curves or variants.
    """
    if surface is None or surface.empty:
        raise ValueError('Empty price surface')
    required = set(CURVE + ['publication_allowed', 'is_reliable',
                            'publication_status', 'publication_reason_code'])
    if missing := required - set(surface):
        raise ValueError(f'Hierarchical guard requires {sorted(missing)}')
    out = _normalize(surface)
    if out.duplicated(CELL).any():
        raise ValueError('Duplicate price coordinates')
    out['oof_variant_risk_state'] = 'MODEL_RISK_INCONCLUSIVE'
    out['oof_model_risk_state'] = 'MODEL_RISK_INCONCLUSIVE'
    out['oof_risk_hold_scope'] = ''
    live = out.loc[out.publication_allowed.fillna(False).astype(bool), CURVE]
    if live.empty:
        return out
    weekly = _weekly_evidence(history, asof, cfg)
    if weekly.empty:
        return out
    active = live.drop_duplicates(CURVE)
    weekly = weekly.merge(active.assign(_candidate=True), on=CURVE,
                          how='inner', validate='many_to_one')
    if weekly.empty:
        return out

    # 1) Variant risk: may suppress only its own curves.
    bad_variants = set()
    variant_states = {}
    for key, group in weekly.groupby(VARIANT, dropna=False, sort=False):
        stats = _assess(group, asof, cfg, ('variant', key))
        variant_states[key] = stats['status']
        if stats['status'] == 'MODEL_PROVEN_UNSAFE':
            bad_variants.add(key)
    vkeys = pd.Series(list(map(tuple, out[VARIANT].itertuples(index=False, name=None))),
                      index=out.index)
    out['oof_variant_risk_state'] = vkeys.map(variant_states).fillna('MODEL_RISK_INCONCLUSIVE')
    mask = vkeys.isin(bad_variants) & out.publication_allowed.astype(bool)
    out.loc[mask, ['publication_allowed', 'is_reliable']] = False
    out.loc[mask, ['publication_status', 'publication_reason_code']] = 'HOLD_OOF_VARIANT_SYSTEMIC'
    out.loc[mask, 'oof_risk_hold_scope'] = 'VARIANT'

    # 2) Model risk: at least two still-publishable variants must independently
    # contribute evidence; otherwise the variant guard is the right authority.
    alive = out.loc[out.publication_allowed.astype(bool), CURVE].drop_duplicates(CURVE)
    surviving = weekly.merge(alive.assign(_remaining=True),
                             on=CURVE, how='inner', validate='many_to_one')
    bad_models = set()
    model_states = {}
    for key, group in surviving.groupby(IDENTITY, dropna=False, sort=False):
        if group['model_variant'].nunique() < 2:
            model_states[key] = 'MODEL_RISK_INCONCLUSIVE'
            continue
        stats = _assess(group, asof, cfg, ('model', key))
        model_states[key] = stats['status']
        if stats['status'] == 'MODEL_PROVEN_UNSAFE':
            # Model-wide evidence must come from >=2 at-risk variants.
            flagged = group.loc[group.ape_pct.gt(cfg.mdape_limit_pct), 'model_variant'].nunique()
            if flagged >= 2:
                bad_models.add(key)
    mkeys = pd.Series(list(map(tuple, out[IDENTITY].itertuples(index=False, name=None))),
                      index=out.index)
    out['oof_model_risk_state'] = mkeys.map(model_states).fillna('MODEL_RISK_INCONCLUSIVE')
    mask = mkeys.isin(bad_models) & out.publication_allowed.astype(bool)
    out.loc[mask, ['publication_allowed', 'is_reliable']] = False
    out.loc[mask, ['publication_status', 'publication_reason_code']] = 'HOLD_OOF_MODEL_SYSTEMIC'
    out.loc[mask, 'oof_risk_hold_scope'] = 'MODEL'
    if out.groupby(CURVE, dropna=False).publication_allowed.nunique().gt(1).any():
        raise AssertionError('Partial mileage curve in hierarchical model guard')
    if not (out.publication_allowed.astype(bool) == out.is_reliable.astype(bool)).all():
        raise AssertionError('Reliability flag mismatch in hierarchical model guard')
    return out
