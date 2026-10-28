"""V10: the single, final, evidence-qualified publication reliability gate.

This is the ONLY reliability/publication layer in the package. There is no
separate V8/V9 hierarchical pass anymore: in the previous codebase that layer
computed `is_reliable` / `reliability_tier`, but V10 ran afterwards and
unconditionally overwrote `is_reliable`, `publication_allowed`, and
`publish_state` from scratch. That made the entire V8/V9 computation dead
weight on the production decision (it was still exposed as diagnostic
columns, but never influenced what got published). Removing it entirely and
deriving display tiers (A/B/C/D) directly from V10 evidence keeps one
source of truth and halves the amount of calibration logic to maintain.

The final PRICE is never changed by this module. Full-grid generation and
publication are separate. Historical OOF labels become eligible only after
the whole following 7-day window has closed. Sparse/malformed future targets
never silently become trustworthy evaluation examples.

MARKET_PROVISIONAL means in-sample agreement, NOT predictive validation, and
is excluded from publication by default (`publish_market_provisional=False`).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

MODEL = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy']
CELL = MODEL + ['model_variant', 'age', 'mileage_bucket', 'condition_tier']


@dataclass
class ReliabilityV10Config:
    min_monthly_unique_ads: int = 7
    lookback_days: int = 30
    max_mdape_pct: float = 8.0
    max_mape_pct: float = 12.0
    oof_min_cells: int = 2
    oof_repeated_min_cells: int = 5
    oof_repeated_min_weeks: int = 2
    max_oof_cells_for_market_fallback: int = 2
    # Quality of a SINGLE next-week target, not support of the motorcycle.
    min_oof_target_unique_ads: int = 3
    min_oof_target_days: int = 2
    max_oof_target_iqr_pct: float = 25.0
    max_oof_target_loo_shift_pct: float = 12.0
    min_market_exact_ads_for_fallback: int = 5
    min_market_days_for_fallback: int = 3
    cell_overpricing_threshold_pct: float = 12.0
    cell_underpricing_threshold_pct: float = 15.0
    cell_min_market_ads: int = 4
    cell_min_market_days: int = 2
    cell_max_iqr_pct: float = 25.0
    oof_cell_min_weeks: int = 2
    max_price_interval_ratio: float = 1.10
    # Stage/monitor by default. Enable only with explicit acceptance of
    # training-data-fit (not independently predictive) risk.
    publish_market_provisional: bool = False
    # --- Display-tier thresholds (diagnostics only; never gate publication) ---
    tier_a_min_oof_cells: int = 5
    tier_a_min_oof_weeks: int = 2
    tier_a_max_mdape_pct: float = 8.0
    tier_b_min_oof_cells: int = 2
    tier_b_max_mdape_pct: float = 10.0


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    for c in CELL:
        if c not in d.columns:
            raise KeyError(f'Required publication identity column missing: {c}')
    d['motorcycle_model_id'] = (pd.to_numeric(d.motorcycle_model_id, errors='coerce')
                                 .astype('Int64').astype(str).replace('<NA>', 'UNKNOWN'))
    d['is_copy'] = (pd.to_numeric(d.is_copy, errors='coerce')
                     .astype('Int64').astype(str).replace('<NA>', 'UNKNOWN'))
    for c in ['real_brand', 'real_model', 'model_variant']:
        d[c] = d[c].fillna('UNKNOWN').astype(str).str.strip()
    for c in ['age', 'mileage_bucket', 'condition_tier']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d


def convert_dates(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        v = pd.to_numeric(col, errors='coerce').dropna()
        med = float(v.abs().median()) if len(v) else 0.
        unit = 'ns' if med > 1e17 else ('us' if med > 1e14 else 's')
        return pd.to_datetime(col, unit=unit, errors='coerce').dt.normalize()
    return pd.to_datetime(col, errors='coerce').dt.normalize()


def _unique_ads(ads: pd.DataFrame) -> pd.DataFrame:
    """Only identical persistent identifiers represent the same ad.

    Empty token/url never means that unrelated listings are the same ad.
    If neither is known, no deduplication is performed; this is a
    conservative choice (possible double counting of reposts), never a
    price-based outlier deletion.
    """
    a = ads.copy().sort_values('date')
    if a.empty:
        return a
    for identifier in ['token', 'url']:
        if identifier not in a:
            continue
        val = a[identifier].astype('string').str.strip()
        valid = val.notna() & val.ne('') & val.str.lower().ne('nan')
        if valid.any():
            a_known = a.loc[valid].copy()
            a_known['_ad_identifier'] = val[valid]
            a_known = a_known.drop_duplicates(['_ad_identifier'], keep='last').drop(columns='_ad_identifier')
            a = pd.concat([a_known, a.loc[~valid]], ignore_index=True)
    return a.reset_index(drop=True)


def _empty_oof_models() -> pd.DataFrame:
    return pd.DataFrame(columns=MODEL + [
        'v10_oof_cells', 'v10_oof_weeks', 'v10_oof_mdape', 'v10_oof_mape',
        'v10_oof_o10_share', 'v10_oof_u10_share', 'v10_oof_interval_coverage'])


class EvidenceFirstReliabilityV10:
    """The sole reliability/publication decision engine.

    `fit()` is implicit: all evidence banks (OOF, recent market) are built
    fresh inside `apply()` from `backtest_cells` and `ads`, strictly as-of
    `asof`. There is no separate persisted state between calls other than
    the diagnostic reports attached to `self` after the last `apply()` call.
    """

    def __init__(self, config: ReliabilityV10Config | None = None):
        self.config = config or ReliabilityV10Config()
        self.model_report = pd.DataFrame()
        self.cell_market_evidence = pd.DataFrame()
        self.oof_cell_evidence = pd.DataFrame()
        self.oof_quality_report = {}

    def _oof_bank(self, backtest_cells, cutoff):
        cfg = self.config
        if backtest_cells is None or backtest_cells.empty:
            self.oof_quality_report = {'completed_cells': 0, 'qualified_cells': 0,
                                        'unqualified_cells': 0}
            return _empty_oof_models(), pd.DataFrame(columns=CELL + [
                'v10_oof_cell_over_weeks', 'v10_oof_cell_under_weeks'])
        o = normalize_keys(backtest_cells)
        o['asof'] = convert_dates(o['asof'])
        o = o[o['asof'].notna() & (o['asof'] + pd.Timedelta(days=7) <= cutoff)].copy()
        o = o.drop_duplicates(CELL + ['asof'])
        for col in ['future_unique_ads', 'future_observed_days', 'future_iqr_pct',
                    'actual_independent', 'actual', 'price_center',
                    'future_loo_median_shift_pct']:
            if col not in o:
                o[col] = np.nan
            o[col] = pd.to_numeric(o[col], errors='coerce')
        good = (o.future_unique_ads.ge(cfg.min_oof_target_unique_ads)
                & o.future_observed_days.ge(cfg.min_oof_target_days)
                & o.future_iqr_pct.between(0, cfg.max_oof_target_iqr_pct)
                & o.future_loo_median_shift_pct.le(cfg.max_oof_target_loo_shift_pct)
                & o.actual_independent.gt(0) & o.price_center.gt(0))
        self.oof_quality_report = {
            'completed_cells': int(len(o)), 'qualified_cells': int(good.sum()),
            'unqualified_cells': int((~good).sum()),
            'with_at_least_two_unique': int(o.future_unique_ads.ge(2).sum())}
        o = o.loc[good].copy()
        if o.empty:
            return _empty_oof_models(), pd.DataFrame(columns=CELL + [
                'v10_oof_cell_over_weeks', 'v10_oof_cell_under_weeks'])
        o['v10_error_pct'] = 100 * (o.price_center / o.actual_independent - 1)
        o['v10_ape_pct'] = o.v10_error_pct.abs()
        o['v10_covered'] = (o.actual_independent.between(o.price_low, o.price_high)
                             if {'price_low', 'price_high'}.issubset(o.columns)
                             else pd.Series(False, index=o.index))
        models = o.groupby(MODEL, as_index=False, dropna=False).agg(
            v10_oof_cells=('v10_ape_pct', 'size'),
            v10_oof_weeks=('asof', 'nunique'),
            v10_oof_mdape=('v10_ape_pct', 'median'),
            v10_oof_mape=('v10_ape_pct', 'mean'),
            v10_oof_o10_share=('v10_error_pct', lambda x: float((x > 10).mean())),
            v10_oof_u10_share=('v10_error_pct', lambda x: float((x < -10).mean())),
            v10_oof_interval_coverage=('v10_covered', 'mean'))
        over = o[o.v10_error_pct.gt(cfg.cell_overpricing_threshold_pct)]
        under = o[o.v10_error_pct.lt(-cfg.cell_underpricing_threshold_pct)]
        ov = over.groupby(CELL, dropna=False).asof.nunique().rename('v10_oof_cell_over_weeks')
        uv = under.groupby(CELL, dropna=False).asof.nunique().rename('v10_oof_cell_under_weeks')
        cell = pd.concat([ov, uv], axis=1).reset_index()
        return models, cell

    def _market_bank(self, used, ads, asof):
        cfg = self.config
        if ads is None or ads.empty:
            a = pd.DataFrame(columns=CELL + ['date', 'price'])
        else:
            a = normalize_keys(ads)
            a['date'] = convert_dates(a.date)
            a['price'] = pd.to_numeric(a.price, errors='coerce')
            a = a[a.date.lt(asof) & a.date.ge(asof - pd.Timedelta(days=cfg.lookback_days))
                  & a.price.gt(0)].copy()
            if 'source' in a:
                a = a[a.source.eq('ads')].copy()
            if 'copy_label_suspicious' in a:
                a = a[pd.to_numeric(a.copy_label_suspicious, errors='coerce').fillna(0).ne(1)]
            a = _unique_ads(a)
        n30 = a.groupby(MODEL, as_index=False, dropna=False).size().rename(columns={'size': 'v10_market_month_ads'})
        matched = a.merge(used[CELL + ['price_center']], on=CELL, how='inner', validate='many_to_one')
        matched['v10_err'] = 100 * (matched.price_center / matched.price - 1)
        matched['v10_ape'] = matched.v10_err.abs()
        mr = matched.groupby(MODEL, as_index=False, dropna=False).agg(
            v10_market_matched_ads=('v10_ape', 'size'),
            v10_market_mdape=('v10_ape', 'median'),
            v10_market_mape=('v10_ape', 'mean'),
            v10_market_days=('date', 'nunique'),
            v10_market_over10_share=('v10_err', lambda x: float((x > 10).mean())))
        if matched.empty:
            cell = pd.DataFrame(columns=CELL + ['v10_cell_market_ads', 'v10_cell_market_days',
                                                 'v10_cell_market_median', 'v10_cell_iqr_pct', 'v10_cell_market_over_pct'])
        else:
            cell = matched.groupby(CELL, as_index=False, dropna=False).agg(
                v10_cell_market_ads=('price', 'size'),
                v10_cell_market_days=('date', 'nunique'),
                v10_cell_market_median=('price', 'median'),
                _q1=('price', lambda p: float(p.quantile(.25))),
                _q3=('price', lambda p: float(p.quantile(.75))))
            cell['v10_cell_iqr_pct'] = (100 * (cell._q3 - cell._q1)
                                         / cell.v10_cell_market_median.clip(lower=1))
            cell = cell.merge(used[CELL + ['price_center']], on=CELL,
                               how='left', validate='one_to_one')
            cell['v10_cell_market_over_pct'] = 100 * (cell.price_center / cell.v10_cell_market_median - 1)
            cell = cell.drop(columns=['price_center', '_q1', '_q3'])
        self.cell_market_evidence = cell.copy()
        return n30, mr, cell

    def apply(self, full_surface, ads, backtest_cells, asof):
        cfg = self.config
        cutoff = pd.Timestamp(asof).normalize()
        sf = normalize_keys(full_surface)
        # The gate must be idempotent when replaying a previously gated grid;
        # stale decision fields must not create merge suffixes or leak decisions.
        decision_cols = {'is_reliable', 'publication_allowed', 'publication_evidence_tier',
                          'production_quality_state', 'prediction_validated',
                          'prediction_quality_status', 'quality_model_status',
                          'reliability_reason', 'publish_state', 'reliability_tier'}
        stale = [c for c in sf if c.startswith('v10_') or c in decision_cols]
        if stale:
            sf = sf.drop(columns=stale)
        sf['is_factory'] = sf.get('is_factory', pd.Series(False, index=sf.index)).fillna(False).astype(bool)
        if sf.duplicated(CELL).any():
            raise ValueError('Duplicate surface coordinates: unsafe for publication')
        sf['price_center'] = pd.to_numeric(sf.price_center, errors='coerce')
        used = sf.loc[~sf.is_factory].copy()
        ov, oc = self._oof_bank(backtest_cells, cutoff)
        self.oof_cell_evidence = oc.copy()
        n30, mr, cell = self._market_bank(used, ads, cutoff)

        model = (used[MODEL].drop_duplicates()
                 .merge(n30, on=MODEL, how='left', validate='one_to_one')
                 .merge(mr, on=MODEL, how='left', validate='one_to_one')
                 .merge(ov, on=MODEL, how='left', validate='one_to_one'))
        model['v10_market_month_ads'] = model.v10_market_month_ads.fillna(0).astype(int)
        model['v10_oof_cells'] = pd.to_numeric(model.v10_oof_cells, errors='coerce').fillna(0).astype(int)
        model['v10_oof_weeks'] = pd.to_numeric(model.v10_oof_weeks, errors='coerce').fillna(0).astype(int)
        eligible = model.v10_market_month_ads.ge(cfg.min_monthly_unique_ads)
        oof_pass = (model.v10_oof_cells.ge(cfg.oof_min_cells)
                    & model.v10_oof_mdape.le(cfg.max_mdape_pct)
                    & model.v10_oof_mape.le(cfg.max_mape_pct))
        recent_pass = (model.v10_market_matched_ads.ge(cfg.min_market_exact_ads_for_fallback)
                       & model.v10_market_days.ge(cfg.min_market_days_for_fallback)
                       & model.v10_market_mdape.le(cfg.max_mdape_pct)
                       & model.v10_market_mape.le(cfg.max_mape_pct))
        market_lane = ((~oof_pass) & model.v10_oof_cells.le(cfg.max_oof_cells_for_market_fallback)
                       & recent_pass)
        repeated = (oof_pass & model.v10_oof_cells.ge(cfg.oof_repeated_min_cells)
                    & model.v10_oof_weeks.ge(cfg.oof_repeated_min_weeks))
        model['v10_sparse_oof_disagreement'] = (market_lane & model.v10_oof_cells.gt(0) & ~oof_pass)
        model['v10_evidence_lane'] = np.select(
            [~eligible, repeated, oof_pass, market_lane],
            ['LOW_MONTHLY_SUPPORT', 'OOF_REPEATED', 'OOF_LIMITED', 'MARKET_PROVISIONAL'],
            default='NOT_APPROVED')
        model['v10_model_accepted'] = eligible & (oof_pass | market_lane)
        model['v10_oof_confirmed'] = eligible & oof_pass
        model['v10_prediction_validated'] = eligible & repeated
        model['v10_note'] = np.where(market_lane,
                                      'In-sample market agreement, NOT independently validated', '')
        # --- Display tiers (diagnostics only; do not gate publication) ---
        # A: repeated, robust OOF evidence over multiple weeks.
        # B: some genuine OOF evidence, below the "repeated" bar.
        # C: no usable OOF, but in-sample recent market corroboration (MARKET_PROVISIONAL).
        # D: nothing usable yet (new / too sparse / failed quality bars).
        tier_a = (eligible & oof_pass
                  & model.v10_oof_cells.ge(cfg.tier_a_min_oof_cells)
                  & model.v10_oof_weeks.ge(cfg.tier_a_min_oof_weeks)
                  & model.v10_oof_mdape.le(cfg.tier_a_max_mdape_pct))
        tier_b = (eligible & oof_pass & ~tier_a
                  & model.v10_oof_cells.ge(cfg.tier_b_min_oof_cells)
                  & model.v10_oof_mdape.le(cfg.tier_b_max_mdape_pct))
        tier_c = (eligible & ~tier_a & ~tier_b & market_lane)
        model['reliability_tier'] = np.select([tier_a, tier_b, tier_c], ['A', 'B', 'C'], default='D')
        self.model_report = model.copy()

        sf = sf.merge(model, on=MODEL, how='left', validate='many_to_one')
        sf = sf.merge(cell, on=CELL, how='left', validate='one_to_one')
        sf = sf.merge(oc, on=CELL, how='left', validate='one_to_one')
        for numeric_evidence in ['v10_cell_market_ads', 'v10_cell_market_days',
                                  'v10_cell_iqr_pct', 'v10_cell_market_over_pct',
                                  'v10_oof_cell_over_weeks', 'v10_oof_cell_under_weeks']:
            sf[numeric_evidence] = pd.to_numeric(sf[numeric_evidence], errors='coerce')
        isfac = sf.is_factory
        credible_market = (sf.v10_cell_market_ads.fillna(0).ge(cfg.cell_min_market_ads)
                            & sf.v10_cell_market_days.fillna(0).ge(cfg.cell_min_market_days)
                            & sf.v10_cell_iqr_pct.fillna(np.inf).le(cfg.cell_max_iqr_pct))
        recent_over = credible_market & sf.v10_cell_market_over_pct.gt(cfg.cell_overpricing_threshold_pct)
        recent_under = credible_market & sf.v10_cell_market_over_pct.lt(-cfg.cell_underpricing_threshold_pct)
        historic_over = sf.v10_oof_cell_over_weeks.fillna(0).ge(cfg.oof_cell_min_weeks)
        historic_under = sf.v10_oof_cell_under_weeks.fillna(0).ge(cfg.oof_cell_min_weeks)
        sf['v10_overpricing_local_block'] = ~isfac & (recent_over | historic_over)
        sf['v10_underpricing_local_block'] = ~isfac & (recent_under | historic_under)
        sf['v10_directional_risk_source'] = np.select(
            [recent_over | recent_under, historic_over | historic_under],
            ['CURRENT_MARKET_EXACT', 'REPEATED_INDEPENDENT_OOF'], default='NONE')
        p = sf.price_center
        lo = pd.to_numeric(sf.price_low, errors='coerce')
        hi = pd.to_numeric(sf.price_high, errors='coerce')
        valid_identity = ~sf.motorcycle_model_id.isin(['UNKNOWN', 'nan', '<NA>'])
        valid_copy = sf.is_copy.isin(['0', '1'])
        sf['v10_invalid'] = ~(valid_identity & valid_copy & p.gt(0) & lo.gt(0)
                               & hi.gt(0) & hi.ge(p) & p.ge(lo)
                               & (hi / lo).le(cfg.max_price_interval_ratio + 1e-8))
        active = sf.v10_model_accepted.eq(True)
        provisional = sf.v10_evidence_lane.eq('MARKET_PROVISIONAL')
        approved = (active & ~sf.v10_overpricing_local_block
                    & ~sf.v10_underpricing_local_block & ~sf.v10_invalid)
        if not cfg.publish_market_provisional:
            approved &= ~provisional
        sf['publication_allowed'] = np.where(isfac, ~sf.v10_invalid, approved).astype(bool)
        sf['is_reliable'] = sf.publication_allowed  # DB compatibility; never a predictive claim
        sf['prediction_validated'] = np.where(isfac, False, sf.v10_prediction_validated.eq(True))
        sf['publication_evidence_tier'] = np.where(isfac, 'FACTORY', sf.v10_evidence_lane.fillna('NOT_APPROVED'))
        sf['reliability_tier'] = np.where(isfac, 'FACTORY', sf.reliability_tier.fillna('D'))
        sf['production_quality_state'] = np.select([
            sf.v10_invalid, isfac, sf.v10_overpricing_local_block,
            sf.v10_underpricing_local_block, ~active,
            provisional & (not cfg.publish_market_provisional), provisional],
            ['BLOCK_INVALID', 'FACTORY_CONFIRMED', 'BLOCK_HIGH_LOCAL_OVERPRICE',
             'BLOCK_HIGH_LOCAL_UNDERPRICE', 'HOLD_MODEL_METRICS',
             'HOLD_MARKET_PROVISIONAL', 'APPROVED_PROVISIONAL'], default='APPROVED_OOF')
        sf['publish_state'] = sf.production_quality_state
        sf['quality_model_status'] = sf.v10_evidence_lane.fillna('NOT_APPROVED')
        sf['prediction_quality_status'] = np.where(isfac, 'FACTORY', np.where(
            sf.v10_oof_confirmed.eq(True), 'OOF_PASS', 'MARKET_FIT_ONLY_OR_FAIL'))
        sf['reliability_reason'] = sf.production_quality_state.str.lower()
        # Invariants required before a caller may send rows to a DB.
        assert not (sf.is_reliable & (sf.v10_invalid | sf.v10_overpricing_local_block
                                      | sf.v10_underpricing_local_block)).any()
        return sf
    