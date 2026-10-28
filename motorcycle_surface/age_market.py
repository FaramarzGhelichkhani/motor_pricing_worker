"""Strictly past-only hierarchical age-market curve.

Unchanged from the previous version: this module builds the age/market
reference curve from historical ads only, with optional one-year parent
boundary extension. It is independent of the reliability/publication layer,
so no V10-related changes were needed here.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

GROUP = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy', 'model_variant']


class AgeMarketBuilder:
    def __init__(self, ads, extend_parent_supported_min_age=True, min_parent_support=2, include_age_zero=True):
        self.ads = ads.copy()
        self.include_age_zero = bool(include_age_zero)
        self.ads['gid'] = self.ads.groupby(GROUP, sort=False, dropna=False).ngroup()
        self.gids = self.ads[['gid'] + GROUP + ['segment']].drop_duplicates('gid')
        self.extend_parent_supported_min_age = bool(extend_parent_supported_min_age)
        self.min_parent_support = int(min_parent_support)

    def build_snapshot(self, cutoff):
        cutoff = pd.Timestamp(cutoff)
        h = self.ads[(self.ads.date < cutoff) & (self.ads.date >= cutoff - pd.Timedelta(days=120))]
        if h.empty:
            return pd.DataFrame()

        span = h.groupby('gid').age.agg(['min', 'max', 'size']).reset_index()
        span['grid_min_age'] = span['min'].astype(int)
        if self.include_age_zero:
            span['grid_min_age'] = 0
        span['parent_age_support'] = 0

        if self.extend_parent_supported_min_age:
            parent = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy']
            candidates = span[['gid', 'min']].merge(
                self.gids[['gid'] + parent], on='gid', how='left', validate='one_to_one'
            )
            candidates['candidate_age'] = candidates['min'].astype(int) - 1
            counts = (h.groupby(parent + ['age'], dropna=False)
                      .size().rename('parent_age_support').reset_index()
                      .rename(columns={'age': 'candidate_age'}))
            candidates = candidates.merge(
                counts, on=parent + ['candidate_age'], how='left', validate='many_to_one'
            )
            candidates['parent_age_support'] = candidates['parent_age_support'].fillna(0).astype(int)
            candidates = candidates.set_index('gid')
            span['parent_age_support'] = span['gid'].map(candidates['parent_age_support']).fillna(0).astype(int)
            eligible = (span['min'].gt(1) & span['parent_age_support'].ge(self.min_parent_support))
            if not self.include_age_zero:
                span.loc[eligible, 'grid_min_age'] -= 1

        span['age'] = span.apply(
            lambda r: list(range(int(r['grid_min_age']), int(r['max']) + 1)), axis=1
        )
        z = span.explode('age', ignore_index=True).rename(
            columns={'min': 'min_age', 'max': 'max_age', 'size': 'n_group_all'}
        )
        z['age'] = z.age.astype(int)
        z['parent_boundary_extrapolated'] = z['age'].lt(z['min_age'])
        z = z.merge(self.gids, on='gid', how='left', validate='many_to_one')
        for days in [14, 30, 60, 120]:
            sub = h[h.date >= cutoff - pd.Timedelta(days=days)]
            t = sub.groupby(['gid', 'age']).price.agg(['median', 'size']).rename(
                columns={'median': f'med{days}', 'size': f'n{days}'}).reset_index()
            z = z.merge(t, on=['gid', 'age'], how='left', validate='one_to_one')
            z[f'n{days}'] = z[f'n{days}'].fillna(0).astype('int32')
            if days in [14, 30, 120]:
                p = sub.groupby('gid').price.agg(['median', 'size']).rename(
                    columns={'median': f'group_med{days}', 'size': f'group_n{days}'}).reset_index()
                z = z.merge(p, on='gid', how='left', validate='many_to_one')
            if days == 120:
                b = sub.groupby(['real_model', 'is_copy', 'age']).price.agg(['median', 'size']).rename(
                    columns={'median': 'pool_age_median', 'size': 'pool_age_n'}).reset_index()
                z = z.merge(b, on=['real_model', 'is_copy', 'age'], how='left', validate='many_to_one')
        group_slopes = []; segment_slopes = []; model_slopes = []
        observed = z[z.n120.gt(0)].copy()
        for gid, t in observed.groupby('gid', sort=False):
            t = t.sort_values('age')
            a = t.age.to_numpy(float); y = np.log(t.med120.to_numpy(float))
            if len(a) > 1:
                slopes = (np.diff(y) / np.diff(a))
                slopes = slopes[np.isfinite(slopes) & (slopes < .03) & (slopes > -.35)]
                if len(slopes):
                    s = float(np.median(slopes))
                    group_slopes.append((gid, s))
                    info = t.iloc[0]
                    segment_slopes.append((info['segment'], info['is_copy'], s))
                    model_slopes.append((info['real_model'], info['is_copy'], s))
        g_slope = dict(group_slopes)
        mdl = pd.DataFrame(model_slopes, columns=['model', 'is_copy', 'slope']).groupby(
            ['model', 'is_copy']).slope.median().to_dict() if model_slopes else {}
        seg = pd.DataFrame(segment_slopes, columns=['seg', 'is_copy', 'slope']).groupby(
            ['seg', 'is_copy']).slope.median().to_dict() if segment_slopes else {}
        z['curve_hist'] = np.nan; z['curve_offset'] = 0.; z['curve_anchor_n'] = 0; z['curve_slope'] = np.nan
        z['age_source'] = 'none'
        for gid, idx in z.groupby('gid', sort=False).groups.items():
            t = z.loc[idx].sort_values('age'); a = t.age.to_numpy(dtype=float)
            v = t.med120.to_numpy(float); nv = t.n120.to_numpy(int)
            obs = np.isfinite(v) & (v > 0)
            if not obs.any():
                continue
            ax = a[obs]; py = np.log(v[obs]); w = np.maximum(nv[obs].astype(float), 1.)
            if len(ax) > 1:
                py = IsotonicRegression(increasing=False).fit_transform(ax, py, sample_weight=np.minimum(w, 35))
                slopes = np.diff(py) / np.diff(ax)
                middle = slopes[np.isfinite(slopes)]
                s = float(np.clip(np.median(middle) if len(middle) else -.06, -.20, -.01))
                lo = float(np.clip(slopes[0], -.20, -.005)); hi = float(np.clip(slopes[-1], -.20, -.005))
            else:
                info = t.iloc[0]
                s = float(np.clip(mdl.get((info.real_model, info.is_copy), seg.get((info.segment, info.is_copy), -.065)), -.18, -.02))
                lo = hi = s
            logs = np.interp(a, ax, py)
            logs = np.where(a < ax[0], py[0] + lo * (a - ax[0]), logs)
            logs = np.where(a > ax[-1], py[-1] + hi * (a - ax[-1]), logs)
            recent = t.med30.to_numpy(float); rn = t.n30.to_numpy(int)
            hit = np.isfinite(recent) & (recent > 0) & np.isfinite(logs)
            if hit.any():
                dif = np.log(recent[hit]) - logs[hit]
                rw = np.minimum(rn[hit], 20)
                order = np.argsort(dif)
                ds = dif[order]; ws = rw[order]
                diff = ds[np.searchsorted(np.cumsum(ws), ws.sum() * .5, side='left')]
                shift = float(np.clip(diff, -.45, .45))
                anchor_n = int(np.sum(rn[hit]))
            else:
                shift = 0.; anchor_n = 0
            z.loc[t.index, 'curve_hist'] = np.exp(logs + shift)
            z.loc[t.index, 'curve_offset'] = shift
            z.loc[t.index, 'curve_anchor_n'] = anchor_n
            z.loc[t.index, 'curve_slope'] = s
            z.loc[t.index, 'age_source'] = np.where(nv > 0, 'observed120', 'interpolated_or_extrapolated')
        z.loc[z['parent_boundary_extrapolated'], 'age_source'] = 'parent_supported_boundary_extrapolation'
        z.loc[z.age.eq(0) & z.n120.eq(0), 'age_source'] = 'age0_age_curve_extrapolation'
        for strength in [0, 1, 3, 7]:
            recent = z['med30'].to_numpy(float); curve = z['curve_hist'].to_numpy(float)
            n = z.n30.to_numpy(float)
            okay = np.isfinite(recent) & (recent > 0) & np.isfinite(curve) & (curve > 0)
            wt = n / (n + strength) if strength else np.where(n > 0, 1., 0.)
            pred = np.where(okay, np.exp(wt * np.log(np.where(okay, recent, 1)) +
                            (1 - wt) * np.log(np.where(np.isfinite(curve) & (curve > 0), curve, 1))), curve)
            z[f'blend{strength}'] = pred
            z[f'iso{strength}'] = np.nan
        for gid, idx in z.groupby('gid', sort=False).groups.items():
            t = z.loc[idx].sort_values('age'); a = t.age.to_numpy(float)
            for s in [0, 1, 3, 7]:
                p = t[f'blend{s}'].to_numpy(float); valid = np.isfinite(p) & (p > 0)
                if not valid.any():
                    continue
                w = np.minimum(t.n30.to_numpy(float), 25) + 2
                yy = IsotonicRegression(increasing=False).fit_transform(a[valid], np.log(p[valid]), sample_weight=w[valid]) if valid.sum() > 1 else np.log(p[valid])
                z.loc[t.index[valid], f'iso{s}'] = np.exp(yy)
        z['raw14'] = z.med14
        z['raw30'] = z.med30
        z['raw60'] = z.med60
        z['raw120'] = z.med120
        z['asof'] = cutoff
        return z
    