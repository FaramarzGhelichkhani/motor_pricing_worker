"""Fail-closed whole-release check on completed, out-of-time, observable price targets.

This limits observed tail RISK, never guarantees an individual future quote's error.
Cluster bootstrap respects correlation of mileage cells belonging to the same motorcycle.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_recent_oof_release(builder, *, max_tail15_upper_pct=5.0,
                                max_tail20_upper_pct=2.0, min_targets=100,
                                min_motorcycles=20, repeats=2000):
    b = getattr(builder, 'backtest_cells', None)
    needed = {'asof', 'motorcycle_model_id', 'is_copy', 'fold_publication_allowed',
              'price_center', 'actual_independent', 'future_unique_ads',
              'future_observed_days', 'future_iqr_pct'}
    if b is None or b.empty or (needed - set(b)):
        raise RuntimeError('Missing chronological OOF evidence needed for release')
    d = b.copy()
    d['asof'] = pd.to_datetime(d['asof'], errors='coerce').dt.normalize()
    last = sorted(d['asof'].dropna().unique())[-2:]
    d = d[d['asof'].isin(last)].copy()
    for col in ['price_center','actual_independent','future_unique_ads',
                'future_observed_days','future_iqr_pct']:
        d[col] = pd.to_numeric(d[col],errors='coerce')
    quality = (d['fold_publication_allowed'].fillna(False).astype(bool) &
               d.future_unique_ads.ge(3) & d.future_observed_days.ge(2) &
               d.future_iqr_pct.between(0,20) &
               np.isfinite(d.price_center) & d.price_center.gt(0) &
               np.isfinite(d.actual_independent) & d.actual_independent.gt(0))
    d = d.loc[quality].copy()
    model_keys = ['motorcycle_model_id','is_copy']
    nmodels = d[model_keys].drop_duplicates().shape[0]
    if len(d) < min_targets or nmodels < min_motorcycles or d['asof'].nunique() < 2:
        raise RuntimeError(f'Insufficient release validation: {len(d)} targets, {nmodels} models, {d["asof"].nunique()} weeks')
    err = 100 * (d.price_center/d.actual_independent-1)
    d['bad15'] = err.abs().gt(15).astype(int)
    d['bad20'] = err.abs().gt(20).astype(int)
    # Aggregate first, then bootstrap models (not individual correlated grid cells).
    cluster = d.groupby(model_keys,dropna=False).agg(n=('bad15','size'), k15=('bad15','sum'),k20=('bad20','sum'))
    a = cluster[['n','k15','k20']].to_numpy(dtype=float)
    rng = np.random.default_rng(1241)
    draws = rng.integers(0,len(a),size=(repeats,len(a)))
    totals = a[draws].sum(axis=1)
    upper15 = 100*float(np.quantile(totals[:,1]/totals[:,0],.95))
    upper20 = 100*float(np.quantile(totals[:,2]/totals[:,0],.95))
    report = dict(n=len(d), models=nmodels, weeks=d['asof'].nunique(),
                  over15=int((err>15).sum()), under15=int((err<-15).sum()),
                  bad15=int(d.bad15.sum()),bad20=int(d.bad20.sum()),
                  tail15_pct=round(float(d.bad15.mean()*100),3),
                  tail20_pct=round(float(d.bad20.mean()*100),3),
                  upper95_model_bootstrap_tail15_pct=round(upper15,3),
                  upper95_model_bootstrap_tail20_pct=round(upper20,3))
    logger.info('OOF release guard: %s',report)
    if upper15 > max_tail15_upper_pct or upper20 > max_tail20_upper_pct:
        raise RuntimeError(f'Release blocked: clustered high-error risk too high: {report}')
    return report
