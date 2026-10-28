"""Mileage observations + hierarchical age/bucket curves, strictly as-of fit.

Never infer true odometer accuracy from unverified listing claims.
Fit on historical listings only; for test exact age-bucket cells, mask those
cells entirely. Use one generated mileage value consistently for all
deterministic mileage features.
"""
from __future__ import annotations
import logging
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

log = logging.getLogger(__name__)

G = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy', 'model_variant']
B = G + ['age', 'mileage_bucket']
# Historic bucket conventions exactly as per existing surface code. Bucket 8 has a wide, open-ended tail.
LOW = np.array([0, 0, 1001, 5001, 10001, 20001, 30001, 50001, 100001], dtype=float)
HIGH = np.array([0, 1000, 5000, 10000, 20000, 30000, 50000, 100000, 1000000], dtype=float)
PLACEHOLDER = {888888, 999999, 1000000, 111111, 1111, 123456}


def clean_observations(df):
    """Conservative syntactic clean; soft flags for plausible but low-trust readings.

    Avoid falsely converting a rare low-mileage bike to a group average.
    """
    x = df.copy()
    km = pd.to_numeric(x['mileage'], errors='coerce')
    age = pd.to_numeric(x['age'], errors='coerce')
    b = pd.to_numeric(x['mileage_bucket'], errors='coerce')
    hard_bad = km.isna() | km.lt(0) | km.gt(1000000) | km.isin(PLACEHOLDER)
    x['mileage_observed_valid'] = ~hard_bad
    x['mileage_clean_observed'] = km.mask(hard_bad)
    weight = pd.Series(1., index=x.index)
    weight = weight.mask(km.mod(1000).eq(0), .75)
    weight = weight.mask(km.mod(10000).eq(0), .55)
    weight = weight.mask(km.eq(0), .45)
    weight = weight.mask(age.ge(5) & km.le(1000), np.minimum(weight, .45))
    weight = weight.mask(km.div(age.clip(lower=1)).gt(45000), np.minimum(weight, .40))
    weight = weight.mask(hard_bad, 0.)
    bounds_ok = (km.ge(LOW[b.fillna(0).astype(int).clip(0, 8)]) & km.le(HIGH[b.fillna(0).astype(int).clip(0, 8)]))
    x['mileage_bucket_inconsistent'] = (~bounds_ok & ~hard_bad).astype('int8')
    x['mileage_observation_weight'] = weight
    return x


def prep_keys(data):
    """Normalize IDs consistently with core._keys, including missing IDs.

    IDs are CatBoost/group *categoricals*, not quantities. Never cast
    'UNKNOWN' or pandas.NA directly to int. Preserve UNKNOWN as a distinct
    key rather than assigning a fabricated numeric motorcycle identifier.
    """
    x = data.copy()
    id_numeric = pd.to_numeric(x['motorcycle_model_id'], errors='coerce')
    ids = id_numeric.astype('Int64').astype('string')
    x['motorcycle_model_id'] = ids.fillna('UNKNOWN').astype(str)
    x['is_copy'] = pd.to_numeric(x['is_copy'], errors='coerce').fillna(-1).astype(int)
    for c in ['real_brand', 'real_model', 'model_variant']:
        x[c] = x[c].fillna('UNKNOWN').astype(str).str.strip().replace('', 'UNKNOWN')
    x['age'] = pd.to_numeric(x['age'], errors='coerce').fillna(-1).astype(int)
    x['mileage_bucket'] = pd.to_numeric(x['mileage_bucket'], errors='coerce').fillna(-1).astype(int)
    return x


def pava_model(z):
    x = z.age.to_numpy(float); y = z.f.to_numpy(float); w = z.n.to_numpy(float)
    if len(x) == 1:
        return float(np.clip(y[0], .01, .99))
    return IsotonicRegression(increasing=True, y_min=.01, y_max=.99, out_of_bounds='clip').fit(x, y, sample_weight=w)


def value(v, age):
    if v is None:
        return .5
    return float(v.predict([age])[0] if hasattr(v, 'predict') else v)


def adapt_delta(points, age, span, shrink, max_delta):
    if points is None or len(points) == 0:
        return 0., 0
    ages, r, n = points
    w = np.sqrt(np.minimum(n, 15)) * np.exp(-np.abs(ages - age) / span)
    sw = float(w.sum())
    if sw < 1e-5:
        return 0., 0
    d = np.average(r, weights=w) * sw / (sw + shrink)
    return float(np.clip(d, -max_delta, max_delta)), int(n.sum())


class MileageHierarchy:
    """Global copy/age/bucket PAVA + brand/model residual + exact-variant residual.

    All statistics are computed at fit(), from TRAIN ONLY.
    """
    def __init__(self, parent_shrink=9., exact_shrink=6., kernel_age=2.75):
        self.parent_shrink = parent_shrink
        self.exact_shrink = exact_shrink
        self.kernel_age = kernel_age

    def fit(self, hist):
        started = time.perf_counter()
        log.info('[MILEAGE] Fitting hierarchical curves from %s historical rows', f'{len(hist):,}')
        t = prep_keys(clean_observations(hist))
        t = t[t.mileage_bucket.between(1, 8) & t.age.gt(0) & t.mileage_clean_observed.notna()
              & t.mileage_bucket_inconsistent.eq(0)].copy()
        b = t.mileage_bucket.to_numpy(int)
        t['f'] = np.clip((t.mileage_clean_observed.to_numpy(float) - LOW[b]) / (HIGH[b] - LOW[b]), 0, 1)
        t['n'] = t.mileage_observation_weight
        keys = ['is_copy', 'mileage_bucket']
        agg = t.groupby(keys + ['age'], sort=False).agg(f=('f', 'median'), n=('n', 'sum')).reset_index()
        self.global_fit = {k: pava_model(z) for k, z in agg.groupby(keys, sort=False)}
        self.global_bucket = t.groupby(keys, sort=False).f.median().to_dict()
        self.global_all = t.groupby('mileage_bucket').f.median().to_dict()
        self.global_support = t.groupby(keys).size().to_dict()
        self.pool_by_age = t.groupby(['age', 'mileage_bucket']).mileage_clean_observed.median().to_dict()
        self.pool_by_bucket = t.groupby('mileage_bucket').mileage_clean_observed.median().to_dict()
        self.parent = {}
        self.exact = {}
        par_keys = ['real_brand', 'real_model', 'is_copy', 'mileage_bucket']
        grouped = t.groupby(par_keys + ['age'], sort=False).agg(f=('f', 'median'), n=('n', 'sum')).reset_index()
        for key, z in grouped.groupby(par_keys, sort=False):
            prior = np.array([self.global_fraction(key[2], key[3], a) for a in z.age], float)
            residual = (z.f.to_numpy(float) - prior).clip(-.4, .4)
            self.parent[key] = (z.age.to_numpy(float), residual, z.n.to_numpy(float))
        ex_keys = G + ['mileage_bucket']
        grouped = t.groupby(ex_keys + ['age'], sort=False).agg(f=('f', 'median'), n=('n', 'sum')).reset_index()
        for key, z in grouped.groupby(ex_keys, sort=False):
            prior = np.array([self.parent_fraction(key[1], key[2], key[3], key[-1], a) for a in z.age], float)
            residual = (z.f.to_numpy(float) - prior).clip(-.45, .45)
            self.exact[key] = (z.age.to_numpy(float), residual, z.n.to_numpy(float))
        self.train_size = len(t)
        log.info('[MILEAGE] Fitted: valid=%s global=%s parent=%s exact=%s in %.1fs',
                  f'{len(t):,}', len(self.global_fit), len(self.parent), len(self.exact),
                  time.perf_counter() - started)
        return self

    def global_fraction(self, copy, b, age):
        q = self.global_fit.get((int(copy), int(b)))
        ans = value(q, age) if q is not None else float(self.global_all.get(int(b), .5))
        n = self.global_support.get((int(copy), int(b)), 0)
        return float((n * ans + 15 * self.global_bucket.get((int(copy), int(b)), ans)) / (n + 15))

    def parent_fraction(self, brand, model, copy, b, age):
        glob = self.global_fraction(copy, b, age)
        d, _ = adapt_delta(self.parent.get((brand, model, int(copy), int(b))), age, self.kernel_age, self.parent_shrink, .28)
        return float(np.clip(glob + d, .01, .99))

    def predict(self, data):
        started = time.perf_counter()
        log.info('[MILEAGE] Predicting %s age/bucket cells', f'{len(data):,}')
        x = prep_keys(data); out = []
        for r in x.itertuples(index=False):
            b = int(r.mileage_bucket); age = int(r.age); c = int(r.is_copy)
            if b == 0:
                out.append((0., 0., 0., 0., 0., 0))
                continue
            if not 1 <= b <= 8 or age < 0:
                out.append((np.nan,) * 6)
                continue
            glob = self.global_fraction(c, b, age)
            par = self.parent_fraction(str(r.real_brand), str(r.real_model), c, b, age)
            key = (str(r.motorcycle_model_id), str(r.real_brand), str(r.real_model), c, str(r.model_variant), b)
            corr, neigh = adapt_delta(self.exact.get(key), age, self.kernel_age, self.exact_shrink, .30)
            exact = np.clip(par + corr, .01, .99)
            old = self.pool_by_age.get((age, b), self.pool_by_bucket.get(b, LOW[b] + .5 * (HIGH[b] - LOW[b])))
            base = np.clip(old, LOW[b], HIGH[b])
            out.append((float(base), float(LOW[b] + glob * (HIGH[b] - LOW[b])),
                        float(LOW[b] + par * (HIGH[b] - LOW[b])), float(LOW[b] + exact * (HIGH[b] - LOW[b])),
                        neigh, float(abs(exact - par))))
        result = pd.DataFrame(out, columns=['mileage_pool_age', 'mileage_global_age', 'mileage_brand_model_age',
                                            'mileage_exact_curve_age', 'neighbor_support', 'exact_adjustment'],
                              index=data.index)
        log.info('[MILEAGE] Finished %s cells in %.1fs', f'{len(result):,}', time.perf_counter() - started)
        return result


def monotone_curve_on_age_grid(unique, km_column='km_curve'):
    """Isotonic projection of PREDICTED mileage for each exact motorcycle+bucket.

    Does not touch true historical cells; uses no heldout target values.
    """
    u = unique.copy()
    u['km_curve_before_monotone'] = u[km_column].copy()
    keys = G + ['mileage_bucket']
    for key, inds in u.groupby(keys, sort=False, dropna=False).groups.items():
        idx = np.asarray(list(inds), dtype=int)
        if len(idx) < 2:
            continue
        ordered = idx[np.argsort(u.loc[idx, 'age'].to_numpy(float))]
        x = u.loc[ordered, 'age'].to_numpy(float)
        y = u.loc[ordered, km_column].to_numpy(float)
        if np.isfinite(y).all() and np.any(np.diff(y) < -1e-5):
            model = IsotonicRegression(increasing=True, out_of_bounds='clip')
            u.loc[ordered, km_column] = model.fit_transform(x, y)
    return u


def mileage_features(age, km):
    a = pd.to_numeric(age, errors='coerce').clip(lower=1)
    m = pd.to_numeric(km, errors='coerce').clip(lower=0)
    return pd.DataFrame({'mileage_clean': m, 'mileage_log': np.log1p(m),
                         'mileage_per_year': m / a, 'relative_mileage_to_standard_age': np.log1p(m) - np.log1p(a * 7000),
                         'price_per_km_proxy': np.log1p(m) / (a + 1)}, index=age.index)
