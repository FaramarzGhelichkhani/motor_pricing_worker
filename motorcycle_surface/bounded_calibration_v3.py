"""Hard-capped narrow interval positioning.

NOTE ON HISTORY: this module used to also contain `SignedCenterCalibrator`,
a signed log-center calibrator. It was dead code — core.py never imported or
called it; `local_bias_v7.LocalSelectiveBiasCalibrator` was the calibrator
actually wired into the pipeline. It has been removed rather than kept
unused, to avoid a second, contradictory calibration path in production.

The interval positioner below is unrelated to centering: it only decides
where inside a fixed-width band the center price sits.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

G = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy', 'model_variant']


class NarrowIntervalPositioner:
    """Keeps price_high / price_low <= 1+max_width_pct/100 *exactly*.

    By default t=.5: the narrow band is symmetric in log prices. Setting
    adaptive=True permits changing t after enough completed folds, but
    historical tests must justify enabling it in production.
    """
    def __init__(self, adaptive=False, min_folds=3, min_cells=100, position_min=.35, position_max=.65):
        self.adaptive = adaptive
        self.min_folds = min_folds
        self.min_cells = min_cells
        self.pmin = position_min
        self.pmax = position_max
        self.bank = pd.DataFrame()

    def fit(self, cells, cutoff):
        if cells is None or cells.empty:
            self.bank = pd.DataFrame()
            return self
        x = cells.copy()
        x['asof'] = pd.to_datetime(x['asof']).dt.normalize()
        x = x[x['asof'] + pd.Timedelta(days=7) <= pd.Timestamp(cutoff)].copy()
        predicted = pd.to_numeric(x.get('price_center', x.get('price_monotonic')), errors='coerce')
        actual = pd.to_numeric(x['actual'], errors='coerce')
        valid = predicted.gt(0) & actual.gt(0)
        x = x.loc[valid].copy()
        x['_resid'] = np.log(actual.loc[valid] / predicted.loc[valid]).clip(-.5, .5)
        self.bank = x[['asof', '_resid']].copy()
        return self

    def position(self, asof, width):
        if not self.adaptive or self.bank.empty:
            return .5
        date = pd.Timestamp(asof)
        b = self.bank.loc[self.bank['asof'] + pd.Timedelta(days=7) <= date]
        if len(b) < self.min_cells or b['asof'].nunique() < self.min_folds:
            return .5
        W = np.log1p(width / 100)
        w = np.exp(-np.log(2) * (date - b['asof']).dt.days / 35)
        r = b['_resid'].to_numpy(float)
        positions = np.linspace(self.pmin, self.pmax, 7)
        scores = [np.average((r >= -t * W) & (r <= (1 - t) * W), weights=w) - .01 * (t - .5) ** 2 for t in positions]
        return float(positions[int(np.argmax(scores))])

    def apply(self, g, asof):
        x = g.copy()
        width = pd.to_numeric(x['band_width_pct'], errors='coerce').fillna(10.).clip(1, 10.)
        center = pd.to_numeric(x['price_center'], errors='coerce')
        t = np.array([self.position(asof, float(v)) for v in width.to_numpy(float)]) if self.adaptive else np.full(len(x), .5)
        full = np.log1p(width.to_numpy(float) / 100)
        x['price_low'] = center.to_numpy(float) * np.exp(-t * full)
        x['price_high'] = center.to_numpy(float) * np.exp((1 - t) * full)
        x['band_width_pct'] = width.to_numpy(float)
        x['interval_position'] = t
        x['interval_source'] = 'bounded_adaptive' if self.adaptive else 'bounded_symmetric'
        if np.any(x['price_high'].to_numpy(float) / x['price_low'].to_numpy(float) > 1.100000001):
            raise AssertionError('Band cap violated')
        return x