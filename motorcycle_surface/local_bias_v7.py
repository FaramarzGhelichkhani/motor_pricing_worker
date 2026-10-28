"""V7 selective local overestimation correction.

Signed log residual = log(unshifted OOF prediction / future-week median).
Only *completed* historical temporal folds are eligible: a fold's evaluation
week must already be over relative to `cutoff`. The nearest motorcycle
identity is an exact key (GROUP): there is no cross-model price pooling.
Zero shift is the only fallback when evidence is insufficient or the signal
points the wrong way (we only ever discount, never mark something up).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

GROUP = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy', 'model_variant']
COORDS = ['age', 'mileage_bucket', 'condition_tier']


def downward_monotone_envelope(baseline_prices, desired_prices, coordinates):
    """Greatest componentwise nonincreasing surface bounded above by desired.

    Given an already-isotonic `baseline_prices` and a `desired_prices` that is
    elementwise <= baseline (e.g. after a selective discount), returns the
    largest surface that is still monotone nonincreasing on every axis and
    does not exceed `desired` anywhere. This is the minimal feasible
    downward-only correction; no price can accidentally increase.
    """
    base = np.asarray(baseline_prices, float)
    target = np.asarray(desired_prices, float)
    xyz = np.asarray(coordinates, int)
    if base.ndim != 1 or target.shape != base.shape or xyz.shape != (len(base), 3):
        raise ValueError('Incompatible monotone envelope input')
    if not np.isfinite(base).all() or not np.isfinite(target).all() or np.any(base <= 0) or np.any(target <= 0):
        raise ValueError('Non-finite or nonpositive prices')
    if np.any(target > base * (1 + 1e-10)):
        raise ValueError('Selective shift may not increase baseline prices')
    if len(base) == 0:
        return target.copy()
    ordering = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    result = target.copy()
    for step, idx in enumerate(ordering):
        prev = ordering[:step]
        prev = prev[np.all(xyz[prev] <= xyz[idx], axis=1)]
        if len(prev):
            result[idx] = min(result[idx], float(result[prev].min()))
    return result


@dataclass(frozen=True)
class LocalRiskConfig:
    max_down_pct: float = 3.0
    lookback_days: int = 70
    age_sigma: float = 1.2
    mileage_sigma: float = 2.5
    condition_sigma: float = 1.4
    max_age_distance: float = 2.0
    half_life_days: float = 28.0
    min_effective_n: float = 3.0
    min_weeks: int = 2
    min_positive_share: float = 0.56
    min_signed_log_bias: float = 0.0
    uncertainty_z: float = 0.0
    strength: float = 1.0
    shrinkage_prior: float = 6.0
    recent_guard: bool = True
    recent_max_negative_log: float = 0.01

    def __post_init__(self):
        if not (0 < self.max_down_pct <= 10 and self.age_sigma > 0 and
                self.mileage_sigma > 0 and self.condition_sigma > 0 and
                self.half_life_days > 0 and self.max_age_distance >= 0 and
                self.min_effective_n >= 1 and self.min_weeks >= 1 and
                0.5 <= self.min_positive_share <= 1 and
                0 <= self.uncertainty_z <= 3 and 0 < self.strength <= 2 and
                self.shrinkage_prior >= 0 and self.lookback_days > 7):
            raise ValueError('Invalid V7 local risk parameters')


class LocalSelectiveBiasCalibrator:
    """Fit/predict a localized shrinkage estimator without global price shifts.

    The sign of an unobserved cell's error is unknowable in advance. A direct
    discount is a decision based on completed OOF evidence, never proof that
    a specific未来 price would have been wrong.
    """
    def __init__(self, config: LocalRiskConfig | None = None, **kwargs):
        self.config = config if config is not None else LocalRiskConfig(**kwargs)
        self.bank = pd.DataFrame()

    def fit(self, cells: pd.DataFrame | None, cutoff):
        self.bank = pd.DataFrame()
        if cells is None or len(cells) == 0:
            return self
        cutoff = pd.Timestamp(cutoff).normalize()
        x = cells.copy()
        required = GROUP + COORDS + ['asof', 'actual', 'price_unshifted_monotonic']
        missing = [k for k in required if k not in x]
        if missing:
            raise ValueError(f'V7 OOF cells missing {missing}')
        x['asof'] = pd.to_datetime(x['asof'], errors='coerce').dt.normalize()
        x = x.loc[x['asof'].notna() & (x['asof'] + pd.Timedelta(days=7) <= cutoff)].copy()
        if x.empty:
            return self
        p = pd.to_numeric(x['price_unshifted_monotonic'], errors='coerce')
        a = pd.to_numeric(x['actual'], errors='coerce')
        ok = p.gt(0) & a.gt(0) & np.isfinite(p) & np.isfinite(a)
        x = x.loc[ok, GROUP + COORDS + ['asof']].copy()
        x['signed_log_error'] = np.log(p.loc[ok].to_numpy() / a.loc[ok].to_numpy()).clip(-0.18, 0.18)
        for k in COORDS:
            x[k] = pd.to_numeric(x[k], errors='coerce')
        x = x.dropna(subset=COORDS + ['signed_log_error'])
        self.bank = x
        return self

    def predict(self, grid: pd.DataFrame, asof):
        z = grid.copy(); c = self.config
        n = len(z)
        z['center_adjustment_pct'] = 0.0
        z['center_calibration_source'] = 'no_positive_local_evidence'
        z['center_calibration_n_eff'] = 0.0
        z['center_calibration_weeks'] = 0
        z['center_overestimate_share'] = np.nan
        z['center_signed_median_over_pct'] = np.nan
        z['center_shift_confidence_lower'] = np.nan
        if not n or self.bank.empty:
            return z
        cutoff = pd.Timestamp(asof).normalize()
        bank = self.bank.loc[(self.bank['asof'] + pd.Timedelta(days=7) <= cutoff) &
                              (self.bank['asof'] >= cutoff - pd.Timedelta(days=c.lookback_days))]
        if bank.empty:
            return z
        for k in COORDS:
            z[k] = pd.to_numeric(z[k], errors='coerce')
        d = np.zeros(n, float); eff = np.zeros(n, float); count = np.zeros(n, int)
        shares = np.full(n, np.nan); signed = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        src = np.full(n, 'no_positive_local_evidence', dtype=object)
        train_idx = bank.groupby(GROUP, dropna=False, sort=False).indices
        for key, out_ids in z.groupby(GROUP, dropna=False, sort=False).indices.items():
            inp_ids = train_idx.get(key)
            if inp_ids is None:
                continue
            ids = np.asarray(out_ids, int)
            hist = bank.iloc[inp_ids]
            out = z.iloc[ids]
            da = np.abs(out['age'].to_numpy(float)[:, None] - hist['age'].to_numpy(float)[None, :])
            dm = np.abs(out['mileage_bucket'].to_numpy(float)[:, None] - hist['mileage_bucket'].to_numpy(float)[None, :])
            dc = np.abs(out['condition_tier'].to_numpy(float)[:, None] - hist['condition_tier'].to_numpy(float)[None, :])
            days = (np.datetime64(cutoff) - hist['asof'].to_numpy()).astype('timedelta64[D]').astype(float)
            w = np.exp(-0.5 * (da / c.age_sigma) ** 2
                       - 0.5 * (dm / c.mileage_sigma) ** 2
                       - 0.5 * (dc / c.condition_sigma) ** 2
                       - np.log(2) * days[None, :] / c.half_life_days)
            w *= (da <= c.max_age_distance)
            denom = w.sum(axis=1); w2 = (w * w).sum(axis=1)
            ne = np.divide(denom ** 2, w2, out=np.zeros(len(ids)), where=w2 > 1e-14)
            err = hist['signed_log_error'].to_numpy(float)
            avg = np.divide(w @ err, denom, out=np.zeros(len(ids)), where=denom > 1e-14)
            prob = np.divide(w @ (err > 0).astype(float), denom, out=np.zeros(len(ids)), where=denom > 1e-14)
            second = np.divide(w @ (err * err), denom, out=np.zeros(len(ids)), where=denom > 1e-14)
            stderr = np.sqrt(np.maximum(second - avg * avg, 0) / np.maximum(ne, 1.0))
            dates = hist['asof'].to_numpy()
            weeks = np.zeros(len(ids), int)
            for date in np.unique(dates):
                weeks += (w[:, dates == date].sum(axis=1) > 0.1)
            risk = avg - c.uncertainty_z * stderr
            eligible = (ne >= c.min_effective_n) & (weeks >= c.min_weeks) & (prob >= c.min_positive_share) & (risk > c.min_signed_log_bias)
            if c.recent_guard:
                latest = dates.max()
                newest = (dates == latest)
                rw = w[:, newest]; rv = err[newest]
                recent = np.divide(rw @ rv, rw.sum(axis=1), out=np.zeros(len(ids)), where=rw.sum(axis=1) > 0)
                eligible &= recent >= -c.recent_max_negative_log
            drop = np.clip((risk - c.min_signed_log_bias) * c.strength * ne / (ne + c.shrinkage_prior), 0, -np.log1p(-c.max_down_pct / 100))
            use = eligible & (drop > 1e-10)
            d[ids] = np.where(use, drop, 0)
            eff[ids] = np.where(use, ne, 0)
            count[ids] = np.where(use, weeks, 0)
            shares[ids] = np.where(use, prob, np.nan)
            signed[ids] = np.where(use, 100 * np.expm1(avg), np.nan)
            lower[ids] = np.where(use, 100 * np.expm1(risk), np.nan)
            src[ids] = np.where(use, 'group_local_kernel', 'no_positive_local_evidence')
        if 'is_factory' in z:
            fac = z['is_factory'].fillna(False).astype(bool).to_numpy()
            d[fac] = 0; src[fac] = 'factory_fixed'; eff[fac] = 0; count[fac] = 0
        z['center_adjustment_pct'] = 100 * np.expm1(-d)
        z['center_calibration_source'] = src
        z['center_calibration_n_eff'] = eff
        z['center_calibration_weeks'] = count
        z['center_overestimate_share'] = shares
        z['center_signed_median_over_pct'] = signed
        z['center_shift_confidence_lower'] = lower
        return z
    