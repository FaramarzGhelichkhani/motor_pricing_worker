"""Full two-stage motorcycle price surface pipeline, post-preprocessing.

Data causal contract:
- Each snapshot uses only dated rows strictly before its asof.
- Stage-1's training labels are completed future-week medians from older snapshots.
- Stage-2 training ads see the age reference that was available at their weekly origin.
- All reported evaluation cells use future ads only after the forecast origin.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional
import logging
import sys
import time
import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from .age_market import AgeMarketBuilder, AgeRatePredictor
from .mileage import MileageHierarchy, LOW, HIGH, monotone_curve_on_age_grid
from .database import SurfaceDatabaseMixin
from .reliability import HierarchicalSurfaceReliability
from .bounded_calibration_v3 import NarrowIntervalPositioner
from .local_bias_v7 import LocalSelectiveBiasCalibrator, LocalRiskConfig, downward_monotone_envelope
from .publication_policy import apply_publication, PublicationConfig, POLICY_VERSION
from .recent_market_guard import apply_recent_market_guard
from .weighted_isotonic_v3 import weighted_ragged_isotonic
from .coverage_policy_v4 import apply_age_window_certificate

log = logging.getLogger(__name__)
GROUP = ['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']
PARENT = GROUP[:-1]  # one motorcycle identity; ignore variant, age, mileage, condition
CELL = GROUP + ['age','mileage_bucket','condition_tier']
CATS = ['real_brand','real_model','motorcycle_model_id','model_variant','is_copy']
AGE_CATS = CATS + ['segment']
AGE_NUM = ['age','min_age','max_age','n_group_all','med14','med30','med60','med120',
           'n14','n30','n60','n120','group_med14','group_med30','group_med120',
           'group_n14','group_n30','group_n120','curve_hist','curve_offset',
           'curve_anchor_n','curve_slope','pool_age_median','pool_age_n']
AGE_LOG = ['med14','med30','med60','med120','group_med14','group_med30',
           'group_med120','curve_hist','pool_age_median']
AGE_FEATURES = AGE_CATS + AGE_NUM
FEATURES_24 = CATS + ['age','age_ref_log','mileage_bucket','mileage_clean',
    'mileage_log','mileage_per_year','relative_mileage_to_standard_age',
    'price_per_km_proxy','condition_tier','n14','n30','n120','group_n30',
    'curve_anchor_n','age_ref_fallback','age_model_correction',
    'local_14_rel','local_30_rel','local_120_rel']
assert len(FEATURES_24) == 24

@dataclass
class SurfaceConfig:
    history_days: int = 120
    age_iterations: int = 2000
    # New experiment: supervised adjacent-year log price ratios.
    # Set to legacy_price to reproduce the previous independent-price target.
    age_target_mode: str = "direct_rate"
    residual_iterations: int = 3000
    age_depth: int = 4
    residual_depth: int = 5
    age_learning_rate: float = .035
    residual_learning_rate: float = 0.033
    random_seed: int = 1241
    threads: int = 6
    condition_levels: tuple = (1, 2, 3)
    bucket_levels: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    isotonic_warn_pct: float = 3.0
    isotonic_block_pct: float = 5.0
    compact_output_default: bool = True
    min_age_training_cells: int = 200
    # Calibration uses completed earlier-fold OOF evidence.
    center_calibration_enabled: bool = True
    # Local selective calibration.
    local_age_sigma: float = 1.2
    local_mileage_sigma: float = 2.5
    local_condition_sigma: float = 1.4
    local_max_age_distance: float = 2.0
    local_half_life_days: float = 28.0
    local_min_effective_n: float = 3.0
    local_min_weeks: int = 2
    local_positive_share: float = 0.56
    local_min_signed_log_bias: float = 0.0
    local_uncertainty_z: float = 0.0
    local_strength: float = 1.0
    local_shrinkage_prior: float = 6.0
    local_recent_guard: bool = True
    local_recent_max_negative_log: float = 0.01
    local_lookback_days: int = 70
    risk_max_down_pct: float = 3.0
    center_max_adjust_pct: float = 3.0
    weighted_isotonic_enabled: bool = True
    # Adaptive position may increase or decrease coverage; default OFF until validated.
    bounded_interval_adaptive_position: bool = False
    max_width_pct: float = 10.
    min_width_pct: float = 1.
    price_width_per_mape: float = 1.3
    show_progress: bool = True
    extend_parent_supported_min_age: bool = True
    min_parent_age_support: int = 2
    residual_loss_function: str = "Quantile:alpha=0.48"
    log_every_n_groups: int = 200
    grid_age_bucket_quantile: float = 0.99
    grid_min_age_observations: int = 100
    grid_age_cap_defaults: tuple = (3, 5, 6, 7)
    grid_history_days: int = 120
    # Extra safety for interval calibration after unexpected market drift.
    # Estimated using only completed previous folds; no same-fold labels.
    interval_empirical_guard: bool = True
    interval_guard_quantile: float = 0.65
    interval_guard_prior_strength: int = 12
    # Tier C is an estimated/review price and may be exposed to users as such.
    reliability_publish_review_tier_c: bool = True
    # Fill economically valid interior coordinates only when a GROUP is predominantly
    # A/B/C and has independent, completed-backtest evidence of usable pricing.
    complete_dominantly_reliable_groups: bool = False  # Never promote unsupported D to C by default
    # V4: prioritize independently validated cells through age six, rather
    # than letting sparse historical ages 7..14 disqualify young-age coverage.
    completion_priority_max_age: int = 6
    # Default audit to avoid silently removing >60% of reliable prices.
    # Set 'strict' for a public catalog ONLY after inspecting completeness impact.
    coverage_publication_mode: str = 'audit'  # 'audit' | 'strict'
    coverage_required_max_age: int = 6
    # Fixed 10%-wide consumer intervals are a PRODUCT constraint.  Low
    # empirical interval coverage does not, by itself, make the point price bad.
    reliability_c_min_interval_coverage_pct: float = 0.0
    completion_min_reliable_share: float = 0.55
    completion_min_reliable_cells: int = 12
    completion_min_reliable_ages: int = 2
    completion_min_group_backtest_cells: int = 5
    completion_min_group_backtest_weeks: int = 2
    completion_max_group_backtest_mape_pct: float = 12.0
    completion_min_group_backtest_coverage_pct: float = 0.0
    completion_max_cell_estimated_mape_pct: float = 12.0
    completion_min_cell_estimated_coverage_pct: float = 0.0
    # Publication support is counted for one motorcycle identity (ID+brand+model+copy),
    # regardless of variant, age, mileage, or condition. [asof-30d, asof).
    min_motorcycle_monthly_ads: int = 7
    # Generate every snapshot-supported model for analysis; 7 ads is a PUBLISH gate.
    generate_all_models_for_review: bool = True
    # V9 empirical OOF model-wide qualification; tune on longer chronological folds.
    quality_min_cells: int = 8
    quality_min_weeks: int = 3
    quality_min_ages: int = 2
    quality_max_mdape_pct: float = 8.0
    quality_max_mape_pct: float = 10.0
    quality_max_abs_bias_pct: float = 7.0
    quality_max_o10_share: float = .25
    quality_max_u10_share: float = .25
    quality_min_coverage_pct: float = 35.0
    quality_min_good_week_share: float = .75
    # Expanded OOF publication lanes require independent repeated predictions
    # AND corroborating recent matching; do not equate with strict OOF validation.
    quality_enable_expanded_approval: bool = True
    quality_enable_two_ad_lane: bool = False  # Empirically high O10; stay disabled
    quality_expanded_market_min_cells: int = 3
    quality_expanded_market_min_matched_ads: int = 15
    quality_expanded_market_max_median_ape_pct: float = 8.0
    quality_expanded_market_max_abs_signed_bias_pct: float = 8.0
    quality_market_group_max_abs_bias_pct: float = 8.0
    quality_max_cell_isotonic_change_pct: float = 5.0
    min_model_risk_target_ads: int = 5
    min_model_risk_target_days: int = 2
    max_model_risk_target_iqr_pct: float = 25.0
    model_risk_min_cells: int = 8
    model_risk_min_weeks: int = 3
    model_risk_min_mdape_pct: float = 15.0
    model_risk_min_o10_share: float = 0.50
    reliability_backtest_folds: int = 4
    reliability_backtest_lookback_days: int = 42
    residual_early_stopping_rounds: int = 150
    residual_validation_days: int = 14
    residual_monotone_constraints: bool = True


def _keys(df):
    z = df.copy()
    for c in ['real_brand','real_model','model_variant','segment']:
        if c not in z: z[c] = 'UNKNOWN'
        z[c] = z[c].fillna('UNKNOWN').astype(str).str.strip().replace('', 'UNKNOWN')
    z['motorcycle_model_id'] = pd.to_numeric(z['motorcycle_model_id'],errors='coerce').astype('Int64').astype(str)
    z['motorcycle_model_id'] = z['motorcycle_model_id'].replace('<NA>','UNKNOWN')
    z['is_copy'] = pd.to_numeric(z['is_copy'],errors='coerce').fillna(-1).astype(int).astype(str)
    return z


def _parse_source_is_original(values: pd.Series) -> pd.Series:
    """Parse DB is_original without guessing from is_copy or brand.

    Interpret known 0/1 and boolean strings from advertisement rows only.
    """
    s = pd.Series(values, index=values.index).astype('string').str.strip().str.lower()
    return pd.to_numeric(s.replace({
        'true': '1', 'false': '0', '1.0': '1', '0.0': '0',
    }), errors='coerce').where(lambda x: x.isin([0, 1]))


def _source_original_by_model_id(data: pd.DataFrame) -> pd.Series:
    """Map model_id -> source/DB is_original, without consulting or rewriting is_copy.

    A model ID with conflicting DB original labels is deliberately omitted
    rather than guessed from a majority. Such an ID cannot publish a surface.
    """
    needed = {'motorcycle_model_id', 'is_original'}
    if not needed.issubset(data.columns):
        raise ValueError(f'Missing surface identity columns: {sorted(needed - set(data.columns))}')
    mid = pd.to_numeric(data['motorcycle_model_id'], errors='coerce')
    org = _parse_source_is_original(data['is_original'])
    usable = mid.notna() & org.isin([0, 1])
    x = pd.DataFrame({
        'motorcycle_model_id': mid.loc[usable].astype('Int64').astype(str),
        'is_original': org.loc[usable].astype('int8'),
    })
    if x.empty:
        return pd.Series(dtype='int8', name='is_original')
    by_id = x.groupby('motorcycle_model_id')['is_original'].agg(['nunique', 'first'])
    ambiguous = int(by_id['nunique'].ne(1).sum())
    if ambiguous:
        log.warning('[SURFACE IDENTITY] %s model IDs have contradictory source is_original; no surfaces for those IDs', ambiguous)
    labels = by_id.loc[by_id['nunique'].eq(1), 'first'].astype('int8')
    labels.name = 'is_original'
    return labels


def _keep_surface_identities(grid: pd.DataFrame, original_by_id: pd.Series) -> pd.DataFrame:
    """SURFACE ONLY: require known model ID and is_copy == 1 - DB is_original.

    Never infer source is_original from is_copy and never filter training ads.
    Only ad-derived used-price candidates pass through this gate.
    """
    if grid.empty:
        return grid.copy()
    x = grid.copy()
    mid = pd.to_numeric(x['motorcycle_model_id'], errors='coerce')
    model_key = mid.astype('Int64').astype(str)
    db_original = model_key.map(original_by_id)
    copy = pd.to_numeric(x['is_copy'], errors='coerce')
    keep = mid.notna() & db_original.isin([0, 1]) & copy.isin([0, 1]) & copy.eq(1 - db_original)
    # Preserve authoritative source identity consistency.
    if 'is_original' in x.columns:
        source = _parse_source_is_original(x['is_original'])
        keep &= source.isna() | source.eq(db_original)
    x = x.loc[keep.fillna(False)].copy()
    x['is_original'] = db_original.loc[x.index].astype('int8')
    return x


def _previous_saturday(date_series):
    d = pd.to_datetime(date_series).dt.normalize()
    return d - pd.to_timedelta((d.dt.weekday - 5) % 7, unit='D')


def _pav_down(x):
    """Weighted (equal weight here) PAVA in log-price, nonincreasing."""
    avg, counts = [], []
    for v in np.asarray(x, dtype=float):
        avg.append(float(v)); counts.append(1)
        while len(avg) > 1 and avg[-2] < avg[-1]:
            n = counts[-2] + counts[-1]
            avg[-2] = (avg[-2]*counts[-2] + avg[-1]*counts[-1]) / n
            counts[-2] = n; counts.pop(); avg.pop()
    return np.repeat(avg, counts)


def _metrics(y, prediction):
    v = np.asarray(y,float); p = np.asarray(prediction,float)
    m = np.isfinite(v) & np.isfinite(p) & (v>0) & (p>0)
    a = 100*np.abs(p[m]/v[m]-1)
    return {'n':int(len(a)), 'mape_pct':float(a.mean()) if len(a) else None,
            'mdape_pct':float(np.median(a)) if len(a) else None,
            'p90ape_pct':float(np.quantile(a,.9)) if len(a) else None}


class MotorcyclePriceSurface(SurfaceDatabaseMixin):
    """In-memory facade: fit(DataFrame) -> create_surface() -> DataFrame -> update_database().

    No preprocessing changes are applied to input labels or Condition.
    Suspicious copy-label ads are excluded from used prices and mileage.
    """
    def __init__(
        self,
        crawler_db=None,
        production_db=None,
        db_target='crawler',
        batch_size=500,
        config: Optional[SurfaceConfig] = None,
        history_model=None,
        surface_model=None,
        publish_only_reliable_db=True,
    ):
        # Preserve convenience of the previous package: MotorcyclePriceSurface(SurfaceConfig(...)).
        if isinstance(crawler_db, SurfaceConfig) and config is None:
            config, crawler_db = crawler_db, None
        self.config = config or SurfaceConfig()
        selected_loss = str(self.config.residual_loss_function)
        if selected_loss != 'MAE':
            m = re.fullmatch(r'Quantile:alpha=(0(?:\.\d+)?|1(?:\.0+)?)', selected_loss)
            if not m or not 0.0 < float(m.group(1)) < 1.0:
                raise ValueError('residual_loss_function must be MAE or Quantile:alpha=<0..1>')
        self._enable_progress_logging()
        self.crawler_db = crawler_db
        self.production_db = production_db
        self.db_target = db_target
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.history_model = history_model
        self.surface_model = surface_model
        self.publish_only_reliable_db = bool(publish_only_reliable_db)
        self.algo = f'TwoStage24:{self.config.residual_loss_function}' 
        self.today = None
        self._surface_df = None
        self._model_index_history_data = None
        self._input_year_data = None
        self._surface_original_by_id = pd.Series(dtype='int8', name='is_original')
        self.ads = None
        self.market_builder = None
        self.snapshots = None
        self.train_ads = None
        self.age_model = None
        self.residual_model = None
        self.mileage_model = None
        self.calibration = pd.DataFrame()
        self.calibration_global = {}
        self.center_calibrator = LocalSelectiveBiasCalibrator(LocalRiskConfig(
            max_down_pct=self.config.risk_max_down_pct,
            lookback_days=self.config.local_lookback_days,
            age_sigma=self.config.local_age_sigma,
            mileage_sigma=self.config.local_mileage_sigma,
            condition_sigma=self.config.local_condition_sigma,
            max_age_distance=self.config.local_max_age_distance,
            half_life_days=self.config.local_half_life_days,
            min_effective_n=self.config.local_min_effective_n,
            min_weeks=self.config.local_min_weeks,
            min_positive_share=self.config.local_positive_share,
            min_signed_log_bias=self.config.local_min_signed_log_bias,
            uncertainty_z=self.config.local_uncertainty_z,
            strength=self.config.local_strength,
            shrinkage_prior=self.config.local_shrinkage_prior,
            recent_guard=self.config.local_recent_guard,
            recent_max_negative_log=self.config.local_recent_max_negative_log,
        ))
        self.interval_positioner = NarrowIntervalPositioner(
            adaptive=self.config.bounded_interval_adaptive_position)
        self.asof = None
        self.backtest_results = pd.DataFrame()
        self.backtest_cells = pd.DataFrame()
        self.backtest_group_metrics = pd.DataFrame()
        self.grid_bucket_limits = pd.DataFrame()
        self.reliability_policy = None
        self.backtest_reliability_metrics = pd.DataFrame()
        self.backtest_reliability_group_metrics = pd.DataFrame()

    def _enable_progress_logging(self):
        """Print progress to console/Jupyter without writing any files."""
        package_logger = logging.getLogger('motorcycle_surface')
        if self.config.show_progress:
            package_logger.setLevel(logging.INFO)
            if not any(getattr(h, '_motorcycle_surface_progress', False)
                       for h in package_logger.handlers):
                handler = logging.StreamHandler(sys.stdout)
                handler._motorcycle_surface_progress = True
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S'))
                package_logger.addHandler(handler)
            package_logger.propagate = False
        else:
            for handler in list(package_logger.handlers):
                if getattr(handler, '_motorcycle_surface_progress', False):
                    package_logger.removeHandler(handler)
            package_logger.propagate = False

    def load_clean_data(self, data: pd.DataFrame):
        """Only a pandas DataFrame is accepted; never reads a path or file."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Expected pandas.DataFrame, not a filename/path')
        log.info('[DATA] Loading cleaned DataFrame (%s rows, %s columns)', f'{len(data):,}', len(data.columns))
        df = data.copy()
        self._surface_df = None
        self._model_index_history_data = None
        self.snapshots = None
        self.train_ads = None
        self.age_model = None
        self.residual_model = None
        self.mileage_model = None
        self.backtest_results = pd.DataFrame()
        self.backtest_cells = pd.DataFrame()
        self.backtest_group_metrics = pd.DataFrame()
        self.grid_bucket_limits = pd.DataFrame()
        self.reliability_policy = None
        self.backtest_reliability_metrics = pd.DataFrame()
        self.backtest_reliability_group_metrics = pd.DataFrame()
        self.calibration = pd.DataFrame()
        self.calibration_global = {}
        self.center_calibrator.fit(pd.DataFrame(), cutoff=pd.Timestamp.today())
        self.interval_positioner.fit(pd.DataFrame(), cutoff=pd.Timestamp.today())
        if 'source' not in df:
            raise ValueError("Missing 'source'; cannot verify ads-only input")
        ads_mask = df['source'].astype('string').str.strip().str.lower().eq('ads').fillna(False)
        excluded = int((~ads_mask).sum())
        df = df.loc[ads_mask].copy()
        df['source'] = 'ads'
        log.info('[DATA] excluded %s non-ads rows BEFORE identity/year/feature work', excluded)
        if df.empty:
            raise ValueError('No ad observations: factory data cannot train used-price models')
        self._input_year_data = df[[c for c in (GROUP + ['age', 'date', 'production_year']) if c in df.columns]].copy()
        necessary = GROUP + ['date','source','price','age','mileage','mileage_bucket','condition_tier']
        missing = [c for c in necessary if c not in df]
        if missing: raise ValueError(f'Missing preprocessing columns: {missing}')
        df = _keys(df)
        # Keep authoritative per-ID original labels for SURFACE GATING ONLY.
        # Training observations and their is_copy labels are left untouched.
        self._surface_original_by_id = _source_original_by_model_id(df)
        df['date'] = pd.to_datetime(df['date'],errors='coerce').dt.normalize()
        for c in ['price','age','mileage','mileage_bucket','condition_tier']:
            df[c] = pd.to_numeric(df[c],errors='coerce')
        if 'mileage_clean' not in df: df['mileage_clean'] = df['mileage']
        df['mileage_clean'] = pd.to_numeric(df['mileage_clean'],errors='coerce')
        # Only the specialized Preprocessing label weight (or a neutral 1.0)
        # is consumed. There is no dependency on previous Audit-only fields.
        if 'label_weight' not in df:
            df['label_weight'] = 1.0
        df['label_weight'] = pd.to_numeric(df['label_weight'],errors='coerce').fillna(1.).clip(lower=.01)
        suspicious = pd.to_numeric(df.get('copy_label_suspicious',pd.Series(0,index=df.index)),errors='coerce').fillna(0)
        used = df['source'].eq('ads') & df['price'].gt(0) & df['age'].ge(0) & df['motorcycle_model_id'].ne('UNKNOWN') & suspicious.eq(0)
        # Existing price benchmark included age>0 ads whether or not flag_new==1; preserve parity.
        self.ads = df.loc[used].copy()
        self.ads['age'] = self.ads['age'].astype(int)
        self.ads = self.ads[self.ads.mileage_bucket.between(0,8) & self.ads.condition_tier.isin(self.config.condition_levels)].copy()
        # Bucket 0 represents reported zero odometer; do not impute a nonzero value.
        self.ads.loc[self.ads.mileage_bucket.eq(0), 'mileage_clean'] = 0.0
        self.ads['segment'] = self.ads['segment'].fillna('UNKNOWN').astype(str)
        self.ads = self.ads[self.ads['date'].notna()]
        if self.ads.empty: raise ValueError('No eligible used ads')
        self.market_builder = AgeMarketBuilder(
            self.ads,
            extend_parent_supported_min_age=self.config.extend_parent_supported_min_age,
            min_parent_support=self.config.min_parent_age_support,
            include_age_zero=True,
        )
        self.ads = self.market_builder.ads.copy()
        log.info('[DATA] Valid used ads=%s | dates=%s -> %s', f'{len(self.ads):,}', self.ads.date.min().date(), self.ads.date.max().date())
        return self

    def _origins(self, asof):
        first = self.ads.date.min()+pd.Timedelta(days=28)
        days = pd.date_range(first, pd.Timestamp(asof)-pd.Timedelta(days=1),freq='D')
        saturdays=[d for d in days if d.weekday()==5]
        return saturdays + ([pd.Timestamp(asof)] if pd.Timestamp(asof) not in saturdays else [])

    def _age_frame(self,z):
        x=z.copy()
        for c in AGE_CATS: x[c] = x[c].fillna('UNKNOWN').astype(str)
        for c in AGE_LOG:
            x[c]=np.log(pd.to_numeric(x[c],errors='coerce').where(pd.to_numeric(x[c],errors='coerce')>0))
        for c in AGE_NUM: x[c] = pd.to_numeric(x[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
        return x

    def _fit_age_from_past(self, history, cutoff):
        """Train age stage using only completed one-week future labels.

        direct_rate learns log(P_age / P_age+1) on observed adjacent ages
        of the SAME variant/motorcycle identity and completed forecast week.
        Inference integrates those learned year-to-year differences and
        anchors the curve to historical price levels (never future labels).

        legacy_price is retained for local A/B comparisons.
        """
        cutoff = pd.Timestamp(cutoff)
        past = history[
            (history['asof'] + pd.Timedelta(days=7) <= cutoff)
            & (history.future_n >= 3)
            & history.curve_hist.notna()
            & history.future_med.gt(0)
        ].copy()
        if len(past) < self.config.min_age_training_cells or past['asof'].nunique() < 3:
            return None

        mode = self.config.age_target_mode
        if mode not in ('direct_rate', 'legacy_price'):
            raise ValueError("age_target_mode must be 'direct_rate' or 'legacy_price'")

        started = time.perf_counter()
        if mode == 'legacy_price':
            model = CatBoostRegressor(
                iterations=self.config.age_iterations,
                depth=self.config.age_depth,
                learning_rate=self.config.age_learning_rate,
                l2_leaf_reg=6,
                loss_function='MAE',
                random_seed=self.config.random_seed,
                thread_count=self.config.threads,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(
                past[AGE_FEATURES],
                np.log(past.future_med) - past.curve_hist,
                cat_features=AGE_CATS,
            )
            log.info('[AGE] Legacy independent-price target: %s cells, %.1fs',
                     len(past), time.perf_counter() - started)
            return model

        # Forward adjacent-age join: every pair has the SAME asof and GROUP.
        # The shifted 'next' row has age+1; this never mixes companies, variants,
        # weeks, or real/copy identities.
        older = past[['asof'] + GROUP + [
            'age', 'future_med', 'future_n', 'curve_hist'
        ]].copy()
        older['age'] = older['age'] - 1
        older = older.rename(columns={
            'future_med': 'next_future_med',
            'future_n': 'next_future_n',
            'curve_hist': 'next_curve_hist',
        })
        pairs = past.merge(
            older, on=['asof'] + GROUP + ['age'],
            how='inner', validate='one_to_one',
        )
        minimum = max(45, self.config.min_age_training_cells // 2)
        if len(pairs) < minimum or pairs['asof'].nunique() < 3:
            log.warning('[AGE RATE] Not enough completed adjacent-age pairs: %s',
                        len(pairs))
            return None

        target = np.log(
            pairs.future_med.to_numpy(float)
            / pairs.next_future_med.to_numpy(float)
        )
        target = np.clip(target, -0.35, 0.35)
        valid = np.isfinite(target)
        if valid.sum() < 40:
            return None

        model = CatBoostRegressor(
            iterations=self.config.age_iterations,
            depth=self.config.age_depth,
            learning_rate=self.config.age_learning_rate,
            l2_leaf_reg=8,
            loss_function='MAE',
            random_seed=self.config.random_seed,
            thread_count=self.config.threads,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            pairs.loc[valid, AGE_FEATURES], target[valid],
            cat_features=AGE_CATS,
        )
        log.info('[AGE RATE] Supervised adjacent pairs=%s origins=%s elapsed=%.1fs',
                 int(valid.sum()), pairs['asof'].nunique(), time.perf_counter() - started)
        return AgeRatePredictor(model, feature_columns=AGE_FEATURES)

    def build_market_snapshots(self, asof):
        self.asof = pd.Timestamp(asof).normalize()
        self.today = self.asof.date()
        if self.asof <= self.ads.date.min()+pd.Timedelta(days=28):
            raise ValueError('Need > 4 weeks of history for the age model')
        all_frames=[]
        origins = self._origins(self.asof)
        log.info('[AGE] Building %s causal snapshots up to %s', len(origins), self.asof.date())
        started = time.perf_counter()
        for index, dt in enumerate(origins, 1):
            log.info('[AGE] Snapshot %s/%s | asof=%s | historical days=%s',
                     index, len(origins), dt.date(), (dt-self.ads.date.min()).days)
            z=self.market_builder.build_snapshot(dt)
            if z.empty: continue
            z=self._age_frame(z)
            z['asof']=dt
            f=self.ads[self.ads.date.ge(dt)&self.ads.date.lt(dt+pd.Timedelta(days=7))].groupby(['gid','age']).price.agg(future_med='median',future_n='size').reset_index()
            z=z.merge(f,on=['gid','age'],how='left',validate='one_to_one')
            # Future fields for current asof are NEVER used by training or feature generation.
            known=pd.concat(all_frames,ignore_index=True) if all_frames else pd.DataFrame()
            model=self._fit_age_from_past(known,dt) if len(known) else None
            z['age_model_logprice']=z['curve_hist']
            z['age_model_is_fitted']=int(model is not None)
            if model is not None:
                z['age_model_logprice']=z['curve_hist'] + model.predict(z[AGE_FEATURES])
            if dt==self.asof: self.age_model=model
            all_frames.append(z)
            log.info('[AGE] Snapshot %s/%s done: cells=%s model=%s', index, len(origins), f'{len(z):,}', bool(model))
        self.snapshots=pd.concat(all_frames,ignore_index=True)
        # Absolute last asof exists even if not Saturday.
        if not self.snapshots['asof'].eq(self.asof).any(): raise RuntimeError('Latest age snapshot is missing')
        latest = self.snapshots[self.snapshots['asof'].eq(self.asof)]
        if 'parent_boundary_extrapolated' in latest:
            log.info('[AGE] Latest parent-supported younger-age cells: %s', int(latest['parent_boundary_extrapolated'].sum()))
        log.info('[AGE] All snapshots complete: %s rows in %.1fs', f'{len(self.snapshots):,}', time.perf_counter()-started)
        return self.snapshots

    @staticmethod
    def add_features(x):
        z=x.copy()
        z['mileage_clean']=pd.to_numeric(z.mileage_clean,errors='coerce').clip(lower=0)
        z['age']=pd.to_numeric(z.age,errors='coerce')
        z['mileage_log']=np.log1p(z.mileage_clean)
        z['mileage_per_year']=z.mileage_clean/z.age.clip(lower=1)
        z['relative_mileage_to_standard_age']=z.mileage_log-np.log1p(z.age*7000)
        z['price_per_km_proxy']=z.mileage_log/(z.age+1.)
        z['age_ref_log']=z.age_model_logprice
        z['age_model_correction']=z.age_model_logprice-z.curve_hist
        for d in (14,30,120): z[f'local_{d}_rel']=z[f'med{d}']-z.age_ref_log
        for c in CATS: z[c]=z[c].fillna('UNKNOWN').astype(str)
        for c in FEATURES_24:
            if c not in z:raise ValueError(f'Missing price model feature: {c}')
        return z

    def build_residual_training_data(self):
        if self.snapshots is None: raise RuntimeError('build_market_snapshots first')
        z=self.ads.copy()
        z['asof']=_previous_saturday(z.date)
        # Only snapshots from those weekly origins can be used.
        s=self.snapshots[self.snapshots['asof'].ne(self.asof)].copy()
        market_cols=['asof']+GROUP+['age','curve_hist','age_model_logprice',
            'n14','n30','n120','group_n30','curve_anchor_n','med14','med30','med120','curve_slope','min_age','max_age']
        z=z.merge(s[market_cols],on=['asof']+GROUP+['age'],how='left',validate='many_to_one')
        group_curve=s.groupby(['asof']+GROUP,as_index=False).agg(group_ref_log=('curve_hist','median'),
            group_min_age=('min_age','min'),group_max_age=('max_age','max'),group_slope=('curve_slope','median'))
        z=z.merge(group_curve,on=['asof']+GROUP,how='left',validate='many_to_one')
        missing=z.curve_hist.isna()
        # Match inference: no same-age historical support is a fallback even if the curve exists.
        z['age_ref_fallback']=(missing | z.n120.fillna(0).eq(0)).astype(int)
        center=(z.group_min_age+z.group_max_age)*.5
        projection=z.group_ref_log+z.group_slope.fillna(-.06).clip(-.18,-.01)*(z.age-center)
        z.loc[missing,'curve_hist']=projection.loc[missing]
        z.loc[missing,'age_model_logprice']=projection.loc[missing]
        z=z[z.age_model_logprice.between(10,26)&z.mileage_clean.notna()&z.mileage_bucket.between(0,8)&z.condition_tier.isin(self.config.condition_levels)].copy()
        z['mileage_bucket']=z.mileage_bucket.astype(int)
        z.loc[z.mileage_bucket.eq(0),'mileage_clean']=0.0
        z['condition_tier']=z.condition_tier.astype(int)
        z=self.add_features(z)
        z['residual_target']=np.log(z.price)-z.age_ref_log
        self.train_ads=z
        log.info('[RESIDUAL] Causal features prepared: %s training ads (%.1f%% retained)',
                 f'{len(z):,}', 100*len(z)/max(len(self.ads),1))
        return z

    def _new_residual_model(self, iterations=None):
        constraints = (
            {"mileage_clean": -1, "condition_tier": -1}
            if self.config.residual_monotone_constraints
            else None
        )

        return CatBoostRegressor(
            iterations=iterations or self.config.residual_iterations,
            depth=self.config.residual_depth,
            learning_rate=self.config.residual_learning_rate,
            l2_leaf_reg=5,
            loss_function=self.config.residual_loss_function,
            monotone_constraints=constraints,
            random_seed=self.config.random_seed,
            thread_count=self.config.threads,
            verbose=1000,
            allow_writing_files=False,
        )

    def _fit_residual_with_early_stopping(self, data):
        cutoff = data["date"].max() - pd.Timedelta(
            days=self.config.residual_validation_days
        )

        train = data[data.date < cutoff]
        valid = data[data.date >= cutoff]

        if len(train) < 1000 or len(valid) < 200:
            model = self._new_residual_model()
            model.fit(
                data[FEATURES_24], data.residual_target,
                cat_features=CATS,
                sample_weight=data.label_weight.clip(lower=.01)
            )
            return model

        model = self._new_residual_model()

        model.fit(
            train[FEATURES_24],
            train.residual_target,
            cat_features=CATS,
            sample_weight=train.label_weight.clip(lower=.01),
            eval_set=(
                valid[FEATURES_24],
                valid.residual_target
            ),
            early_stopping_rounds=(
                self.config.residual_early_stopping_rounds
            ),
            use_best_model=True
        )

        best_trees = model.best_iteration_ + 1

        final_model = self._new_residual_model(
            iterations=best_trees
        )

        final_model.fit(
            data[FEATURES_24],
            data.residual_target,
            cat_features=CATS,
            sample_weight=data.label_weight.clip(lower=.01)
        )

        return final_model

    def fit(self, data=None, asof=None, do_backtest=True, backtest_dates=None):
        started = time.perf_counter()
        log.info('[FIT] START | do_backtest=%s | age_trees=%s | residual_trees=%s | residual_loss=%s | threads=%s',
                 do_backtest, self.config.age_iterations, self.config.residual_iterations,
                 self.config.residual_loss_function, self.config.threads)
        if data is not None: self.load_clean_data(data)
        if self.ads is None: raise RuntimeError('Provide a cleaned pandas DataFrame')
        if asof is None: asof=self.ads.date.max()+pd.Timedelta(days=1)
        log.info('[FIT] Stage 1/5: causal Age/Market snapshots')
        self.build_market_snapshots(asof)
        log.info('[FIT] Stage 2/5: assemble causal Residual features')
        z=self.build_residual_training_data()
        train=z[z.date.lt(self.asof)]
        if len(train)<1000:raise ValueError('Insufficient rows with causal training features')
        log.info('[FIT] Stage 3/5: time backtest (enabled=%s)', do_backtest)
        if do_backtest:self.backtest(backtest_dates)
        else:
            self.calibration = pd.DataFrame()
            self.calibration_global = {}
            self.center_calibrator.fit(pd.DataFrame(), cutoff=self.asof)
            self.interval_positioner.fit(pd.DataFrame(), cutoff=self.asof)
            log.warning('[CALIBRATION] No backtest: group-specific shifts are disabled and interval widths are prior-based')
        log.info('[FIT] Stage 4/5: train final Residual CatBoost (%s rows, %s features)', f'{len(train):,}',len(FEATURES_24))
        model_started = time.perf_counter()
        # This helper already selects the best iteration and refits on all admissible data.
        # A second .fit() here would defeat the intended training workflow.
        self.residual_model = self._fit_residual_with_early_stopping(train)
        log.info('[FIT] Final Residual fitted in %.1fs',time.perf_counter()-model_started)
        log.info('[FIT] Stage 5/5: train historical mileage hierarchy')
        self.mileage_model=MileageHierarchy().fit(self.ads[self.ads.date.lt(self.asof)])
        log.info('[FIT] DONE | samples=%s | age_model=%s | total=%.1fs',
                 f'{len(train):,}', self.age_model is not None,time.perf_counter()-started)
        return self

    def market_at(self, asof):
        asof=pd.Timestamp(asof).normalize()
        z=self.snapshots[self.snapshots['asof'].eq(asof)].copy()
        if z.empty:
            if asof>self.asof:
                raise ValueError('New asof after training: rerun fit on newly cleaned data first')
            raise ValueError('Requested asof absent from trained snapshots')
        z['age_ref_fallback']=z.n120.fillna(0).eq(0).astype(int)
        return z

    def _bucket_limits_from_past(self, history, asof):
        """Causal empirical maximum bucket by age (99th percentile, default).

        Ages 0..3 use history >=min_count; rare reported high mileage will not
        force extreme hypothetical surface rows. Age >=4 keeps bucket 8.
        This only constrains published grid coordinates, never relabels ads.
        """
        defaults = tuple(self.config.grid_age_cap_defaults)
        if len(defaults) != 4:
            raise ValueError('grid_age_cap_defaults must give age 0,1,2,3 caps')
        h = history[(history.date < pd.Timestamp(asof)) &
                    (history.date >= pd.Timestamp(asof) - pd.Timedelta(days=self.config.grid_history_days))]
        h = h[h.mileage_bucket.between(0, 8) & h.age.ge(0)].copy()
        report = []
        for age in range(4):
            a = pd.to_numeric(h.loc[h.age.eq(age),'mileage_bucket'],errors='coerce').dropna()
            if len(a) >= self.config.grid_min_age_observations:
                cap = int(np.ceil(np.quantile(a, self.config.grid_age_bucket_quantile)))
                source = 'historical_q99'
            else:
                cap = int(defaults[age]);source = 'default_low_support'
            report.append({'age':age,'max_bucket':int(np.clip(cap,0,8)),
                           'n_age':int(len(a)),'source':source})
        # Age caps must be non-decreasing: wider used mileage as age increases.
        for j in range(1,len(report)):
            report[j]['max_bucket'] = max(report[j]['max_bucket'],report[j-1]['max_bucket'])
        report.append({'age':4,'max_bucket':8,'n_age':int(h.age.ge(4).sum()),
                       'source':'default_age_4_plus'})
        self.grid_bucket_limits=pd.DataFrame(report)
        return {r['age']:r['max_bucket'] for r in report}

    def _recent_motorcycle_counts(self, history, asof):
        """Count distinct eligible ads, not reposts, within the preceding 30d.

        The identity is PARENT, intentionally ignoring variant, age, mileage,
        and condition. Never include future advertisements.
        """
        recent=history.loc[history.date.ge(pd.Timestamp(asof)-pd.Timedelta(days=30))
                           & history.date.lt(pd.Timestamp(asof))].copy()
        if recent.empty:
            return pd.DataFrame(columns=PARENT+['motorcycle_n30'])
        if 'token' in recent and recent['token'].notna().any():
            known=recent[recent.token.notna()].drop_duplicates(PARENT+['token'])
            missing=recent[recent.token.isna()]
            recent=pd.concat([known,missing],ignore_index=True)
        elif 'url' in recent and recent['url'].notna().any():
            known=recent[recent.url.notna()].drop_duplicates(PARENT+['url'])
            missing=recent[recent.url.isna()]
            recent=pd.concat([known,missing],ignore_index=True)
        return (recent.groupby(PARENT,dropna=False).size()
                .rename('motorcycle_n30').reset_index())

    def _complete_observed_age_span(self, market, history):
        """Make every integer age from the historically observed min..max.

        No ages outside the observed range are invented. A snapshot-gap age
        receives an interpolation of snapshot age references (in log space),
        is tagged unverified and never promoted to a directly observed age.
        """
        if market.empty:return market
        bounds=history.groupby(GROUP,dropna=False).age.agg(
            observed_min_age='min',observed_max_age='max').reset_index()
        market=market.merge(bounds,on=GROUP,how='left',validate='many_to_one')
        out=[];added=0
        for _,q in market.groupby(GROUP,dropna=False,sort=False):
            q=q.sort_values('age').copy()
            low=q.observed_min_age.iloc[0]
            high=q.observed_max_age.iloc[0]
            if pd.isna(low) or pd.isna(high):
                out.append(q);continue
            low,high=int(low),int(high)
            anchors=q.copy()
            q=q[q.age.between(low,high)].copy()
            if q.empty:
                # Snapshot provided only a parent-supported boundary anchor.
                # Project it onto the nearest observed age, never outside the
                # actual dataset-supported age span; keep the row unverified.
                distances=np.minimum(abs(anchors.age-low),abs(anchors.age-high))
                q=anchors.loc[[distances.idxmin()]].copy()
                old_age=float(q.age.iloc[0])
                new_age=float(np.clip(old_age,low,high))
                slope=float(pd.to_numeric(q.get('curve_slope',pd.Series([-.06])),errors='coerce').median())
                if not np.isfinite(slope):slope=-.06
                slope=float(np.clip(slope,-.18,-.01))
                for col in ('curve_hist','age_model_logprice'):
                    if col in q:
                        q[col]=pd.to_numeric(q[col],errors='coerce')+(new_age-old_age)*slope
                q['age']=int(new_age)
                q['age_source']='interpolated_snapshot_gap'
                for col in ('n14','n30','n60','n120','curve_anchor_n'):
                    if col in q:q[col]=0
                for col in ('med14','med30','med60','med120'):
                    if col in q:q[col]=np.nan
            q['min_age']=low
            q['max_age']=high
            # Keep exactly the ages actually observed in the historical ads.
            desired=np.arange(low,high+1,dtype=int)
            present=set(pd.to_numeric(q.age,errors='coerce').dropna().astype(int))
            missing=[a for a in desired if a not in present]
            if not missing:
                out.append(q);continue
            xx=q.age.to_numpy(dtype=float)
            # Nearest snapshot row provides ONLY group-static/categorical data.
            extra=[]
            for a in missing:
                near=q.iloc[int(np.argmin(np.abs(xx-a)))].copy()
                near['age']=a
                for col in ('curve_hist','age_model_logprice'):
                    if col not in q:continue
                    yy=pd.to_numeric(q[col],errors='coerce').to_numpy(float)
                    good=np.isfinite(xx)&np.isfinite(yy)
                    if good.sum()>=2:
                        xs=xx[good];vs=yy[good]
                        interp=float(np.interp(a,xs,vs))
                        if a<xs.min():
                            slope=float(np.clip(np.nanmedian(np.diff(vs)/np.diff(xs)),-.18,-.01))
                            interp=float(vs[0]+(a-xs[0])*slope)
                        elif a>xs.max():
                            slope=float(np.clip(np.nanmedian(np.diff(vs)/np.diff(xs)),-.18,-.01))
                            interp=float(vs[-1]+(a-xs[-1])*slope)
                        near[col]=interp
                    elif good.sum()==1:
                        slope=float(pd.to_numeric(q.get('curve_slope',pd.Series([-.06])),errors='coerce').median())
                        if not np.isfinite(slope):slope=-.06
                        near[col]=float(yy[good][0]+(a-xx[good][0])*np.clip(slope,-.18,-.01))
                    else:
                        near[col]=np.nan
                # Never reuse an observed age's local reference/count as proof.
                for col in ('n14','n30','n60','n120','curve_anchor_n'):
                    if col in q:near[col]=0
                for col in ('med14','med30','med60','med120'):
                    if col in q:near[col]=np.nan
                near['age_source']='interpolated_snapshot_gap'
                near['age_ref_fallback']=1
                extra.append(near)
            out.append(pd.concat([q,pd.DataFrame(extra)],ignore_index=True))
            added+=len(extra)
        market=pd.concat(out,ignore_index=True)
        market=market.drop(columns=['observed_min_age','observed_max_age'])
        if market.duplicated(GROUP+['age']).any():
            raise AssertionError('Duplicate age anchors after age-span completion')
        log.info('[AGE SPAN] Added %s snapshot-gap age anchors within observed per-group ranges',added)
        return market

    def _build_used_grid(self, asof, history):
        log.info('[GRID] Building economically plausible ragged grid at %s',pd.Timestamp(asof).date())
        m=self.market_at(asof).copy()
        support=self._recent_motorcycle_counts(history,asof)
        m=m.merge(support,on=PARENT,how='left',validate='many_to_one')
        m['motorcycle_n30']=m.motorcycle_n30.fillna(0).astype(int)
        old=len(m)
        if not self.config.generate_all_models_for_review:
            m=m[m.motorcycle_n30.ge(self.config.min_motorcycle_monthly_ads)].copy()
        log.info('[MONTHLY SUPPORT] generated %s/%s age anchors; %s below %s ads (publish gate)',
                 len(m),old,int(m.motorcycle_n30.lt(self.config.min_motorcycle_monthly_ads).sum()),
                 self.config.min_motorcycle_monthly_ads)
        if m.empty:
            raise ValueError('No eligible snapshot age anchors')
        m=self._complete_observed_age_span(m,history)
        caps=self._bucket_limits_from_past(history, asof)
        m['max_grid_bucket']=m.age.map(lambda a: caps.get(int(a),8)).astype(int)
        # Bucket 0 can only represent Condition 1; nonzero buckets allow 1..3.
        b=pd.DataFrame({'mileage_bucket':list(self.config.bucket_levels)})
        b=b.merge(pd.DataFrame({'condition_tier':list(self.config.condition_levels)}),how='cross')
        b=b.loc[b.mileage_bucket.ne(0)|b.condition_tier.eq(1)]
        g=m.merge(b,how='cross')
        g=g[g.mileage_bucket.le(g.max_grid_bucket)].copy()
        log.info('[GRID] %s group-age anchors -> %s valid price cells; bounds=%s',
                 f'{len(m):,}',f'{len(g):,}',caps)
        k=GROUP+['age','mileage_bucket']
        u=g[k].drop_duplicates().reset_index(drop=True)
        log.info('[GRID] %s age cells | %s full cells | %s unique mileage cells', f'{len(m):,}', f'{len(g):,}', f'{len(u):,}')
        mh=self.mileage_model if (pd.Timestamp(asof)==self.asof and self.mileage_model is not None) else MileageHierarchy().fit(history)
        km=mh.predict(u)
        b=u.mileage_bucket
        u['km_curve']=np.select([b.eq(0),b.between(1,4),b.eq(5),b.isin([6,7]),b.eq(8)],
            [0.0,km.mileage_pool_age,km.mileage_global_age,km.mileage_exact_curve_age,km.mileage_brand_model_age],default=np.nan)
        u=monotone_curve_on_age_grid(u,km_column='km_curve')
        h=history.copy()
        h['mileage_bucket']=pd.to_numeric(h.mileage_bucket,errors='coerce')
        h['mileage_clean']=pd.to_numeric(h.mileage_clean,errors='coerce')
        h=h[h.mileage_bucket.between(0,8)&h.mileage_clean.notna()].copy()
        bounds=h.mileage_bucket.to_numpy(int)
        h=h[h.mileage_clean.between(0,1000000)&h.mileage_clean.ge(LOW[bounds])&h.mileage_clean.le(HIGH[bounds])]
        local=h.groupby(k,dropna=False).mileage_clean.agg(local_n='size',local_km='median').reset_index()
        u=u.merge(local,on=k,how='left',validate='one_to_one')
        u['local_n']=u.local_n.fillna(0).astype(int)
        use_local = u.local_n.ge(3) | (u.mileage_bucket.le(5)&u.local_n.ge(2))
        u['mileage_clean']=np.where(b.eq(0),0.0,np.where(use_local,u.local_km,u.km_curve))
        if u.mileage_clean.isna().any():
            raise RuntimeError(f'Unfilled mileage in {int(u.mileage_clean.isna().sum())} grid cells')
        bucket=u.mileage_bucket.to_numpy(int)
        u['mileage_clean']=np.clip(u.mileage_clean.to_numpy(float),LOW[bucket],HIGH[bucket])
        u['km_origin']=np.where(b.eq(0),'reported_zero',np.where(use_local,'observed_age_bucket','hierarchical_age_curve'))
        g=g.merge(u[k+['mileage_clean','km_origin','local_n']],on=k,how='left',validate='many_to_one')
        ch=(history[history.condition_tier.isin(self.config.condition_levels)]
            .groupby(CELL).size().rename('local_cell_n').reset_index())
        g=g.merge(ch,on=CELL,how='left',validate='many_to_one')
        g['local_cell_n']=g.local_cell_n.fillna(0).astype(int)
        g=self.add_features(g)
        g['is_factory']=False
        g['market_asof']=pd.Timestamp(asof)
        log.info('[GRID] Used grid: %s cells | %s with no exact historical support', f'{len(g):,}',f'{int(g.local_cell_n.eq(0).sum()):,}')
        return g


    @staticmethod
    def _ragged_monotone_projection(prices, coordinates, fixed_anchor=None):
        """PAVA alternating slices + product-order closure on valid cells only.

        Every cell is an existing (age, bucket, condition) combination. No
        absent grid cell participates in fitting or pushes down valid prices.
        """
        y=np.log(np.asarray(prices,float))
        coords=np.asarray(coordinates,int)
        for _ in range(9):
            prev=y.copy()
            for axis in range(3):
                other=[j for j in range(3) if j!=axis]
                groups={}
                for ii,v in enumerate(coords):
                    key=tuple(int(v[j]) for j in other)
                    groups.setdefault(key,[]).append(ii)
                for ids in groups.values():
                    ids=np.asarray(ids,int)
                    ids=ids[np.argsort(coords[ids,axis])]
                    if len(ids)>1:y[ids]=_pav_down(y[ids])
            if np.max(np.abs(prev-y))<1e-10:break
        if fixed_anchor is not None:
            anchor_log=np.log(float(fixed_anchor))
            y=np.minimum(y,anchor_log)
            mask=np.all(coords==[0,0,1],axis=1)
            if mask.sum()!=1:raise ValueError('Factory anchor coordinate missing or duplicated')
            y[mask]=anchor_log
        # Exact monotone closure for all comparable valid cells. Sparse, irregular
        # grids cannot use reshape or rectangular minimum.accumulate.
        order=np.lexsort((coords[:,2],coords[:,1],coords[:,0]))
        for pos,t in enumerate(order):
            predecessors=order[:pos]
            predecessors=predecessors[np.all(coords[predecessors] <= coords[t],axis=1)]
            if len(predecessors):y[t]=min(y[t],np.min(y[predecessors]))
        return np.exp(y)

    def _project_monotone(self, grid):
        """Enforce monotonicity on the ad-derived used-price grid."""
        g = grid.sort_values(CELL).reset_index(drop=True)
        g['price_monotonic'] = np.nan
        g['price_unshifted_monotonic'] = np.nan
        g['center_adjustment_effective_pct'] = 0.0
        g['shift_pre_isotonic_pct'] = (-g['center_adjustment_pct']).clip(lower=0.)
        g['center_shift_isotonic_spillover'] = False
        violations = {'age_before': 0, 'mileage_before': 0, 'condition_before': 0}
        gi = g.groupby(GROUP, sort=False).indices
        for ix, (_, indices) in enumerate(gi.items(), 1):
            ids = np.asarray(indices, int)
            coords = g.loc[ids,['age','mileage_bucket','condition_tier']].to_numpy(int)
            raw = g.loc[ids,'price_raw'].to_numpy(float)
            c30 = pd.to_numeric(g.loc[ids,'n30'],errors='coerce').fillna(0).clip(lower=0).to_numpy(float)
            cc = pd.to_numeric(g.loc[ids,'local_cell_n'],errors='coerce').fillna(0).clip(lower=0).to_numpy(float)
            weights = 1+np.minimum(3,np.log1p(c30))+1.5*np.minimum(3,np.log1p(cc))
            if self.config.weighted_isotonic_enabled:
                base = weighted_ragged_isotonic(raw,coords,weights=weights,fixed_anchor=None)
            else:
                base = self._ragged_monotone_projection(raw,coords,fixed_anchor=None)
            pct = g.loc[ids,'center_adjustment_pct'].to_numpy(float)
            desired = base * (1+pct/100)
            final = downward_monotone_envelope(base,desired,coords)
            g.loc[ids,'price_unshifted_monotonic'] = base
            g.loc[ids,'price_monotonic'] = final
            g.loc[ids,'price_shifted_pre_iso'] = desired
            spilled = (pct >= -1e-10) & (final < base*(1-1e-9))
            g.loc[ids,'center_shift_isotonic_spillover'] = spilled
            for axis,name in enumerate(violations):
                by={}
                for j,v in enumerate(coords):
                    by.setdefault(tuple(v[k] for k in range(3) if k!=axis),[]).append(j)
                for positions in by.values():
                    positions=np.asarray(positions,int)
                    positions=positions[np.argsort(coords[positions,axis])]
                    violations[name]+=int((np.diff(raw[positions])>1e-9).sum())
            if self.config.log_every_n_groups and ix%self.config.log_every_n_groups==0:
                log.info('[ISO] %s/%s used-price groups projected',ix,len(gi))
        g['isotonic_change_pct'] = 100*(g.price_unshifted_monotonic/g.price_raw-1)
        g['center_adjustment_effective_pct'] = 100*(g.price_monotonic/g.price_unshifted_monotonic-1)
        g['shift_down_pct'] = (-g['center_adjustment_effective_pct']).clip(lower=0.)
        g['isotonic_warning'] = g.isotonic_change_pct.abs().gt(self.config.isotonic_warn_pct)
        g['isotonic_block'] = g.isotonic_change_pct.abs().gt(self.config.isotonic_block_pct)
        if g['center_adjustment_effective_pct'].gt(1e-7).any():
            raise AssertionError('Selective calibration increased an isotonic center')
        return g,violations

    def _global_calibration(self):
        return self.calibration_global or {'mape_pct':10.,'signed_bias_pct':0.}

    def _intervals(self,g):
        cal=self.calibration
        if len(cal):g=g.merge(cal,on=GROUP,how='left',validate='many_to_one')
        else:
            g['group_eval_n']=0;g['group_mape_pct']=np.nan;g['group_overpricing_pct']=np.nan;g['group_guard_width_pct']=np.nan
        prior=float(self._global_calibration().get('mape_pct',10))
        n=g['group_eval_n'].fillna(0).clip(lower=0)
        local=g['group_mape_pct'].fillna(prior)
        smoothed=(n*local+12*prior)/(n+12)
        width=np.clip(self.config.price_width_per_mape*smoothed,
                      self.config.min_width_pct,self.config.max_width_pct)
        if self.config.interval_empirical_guard:
            prior_guard=float(self._global_calibration().get('coverage_guard_width_pct',0.))
            guard_col=g.get('group_guard_width_pct',pd.Series(np.nan,index=g.index))
            guard_local=guard_col.fillna(prior_guard)
            strength=max(1,int(self.config.interval_guard_prior_strength))
            empirical_guard=(n*guard_local+strength*prior_guard)/(n+strength)
            width=np.clip(np.maximum(width,empirical_guard),
                          self.config.min_width_pct,self.config.max_width_pct)
        # Downside-risk bias has been estimated from COMPLETED past folds
        # and subtracted BEFORE the final monotone projection. No second shift
        # is applied here; that would reintroduce age/mileage inversions.
        g['group_mape_shrunk_pct']=smoothed
        g['band_width_pct']=width
        center=g.price_monotonic.to_numpy(float)
        fac=np.sqrt(1+width/100)
        g['price_center']=center
        g['price_low']=center/fac
        g['price_high']=center*fac
        # USER PRODUCT CONTRACT: never widen the full interval beyond 10%.
        g=self.interval_positioner.apply(g,pd.Timestamp(g['market_asof'].iloc[0]))
        month_support=pd.to_numeric(g.get('motorcycle_n30',pd.Series(0,index=g.index)),errors='coerce').fillna(0)
        g['publish_state']=np.where(month_support.ge(self.config.min_motorcycle_monthly_ads),
                                     'MODEL_SUPPORT_REVIEW','BLOCK_LOW_MONTHLY_SUPPORT')
        # Isotonic severity is diagnostic, not proof of bad predictions.  The
        # reliability policy sets the final publication state after OOF review.
        return g

    def build_surface(self, asof=None, residual_model=None):
        if self.snapshots is None:raise RuntimeError('fit() before build_surface()')
        started=time.perf_counter()
        asof=pd.Timestamp(asof if asof is not None else self.asof).normalize()
        model=residual_model if residual_model is not None else self.residual_model
        if model is None:raise RuntimeError('No fitted Residual model')
        log.info('[SURFACE] START used-only %s',asof.date())
        history=self.ads[self.ads.date.lt(asof)].copy()
        used=self._build_used_grid(asof,history)
        used['reference_price']=np.exp(used.age_ref_log)
        used['price_raw']=np.exp(used.age_ref_log+model.predict(used[FEATURES_24]))
        if not np.isfinite(used.price_raw).all():raise ValueError('Non-finite Residual prices')
        # V7: localized OOF-positive-only adjustment. Raw prices are first
        # weighted-isotonic, then discounted with minimal monotone closure.
        used=self.center_calibrator.predict(used,asof)
        if not self.config.center_calibration_enabled:
            used['center_adjustment_pct']=0.0
            used['center_calibration_source']='disabled'
        used['price_shifted_pre_iso']=(used['price_raw']*(1+used['center_adjustment_pct']/100.))
        # This is evidence-based proposed discount, not the unknowable true error.
        used['estimated_overpricing_pct']=used.get('center_signed_median_over_pct',np.nan)
        used['shift_calibration_n_eff']=used['center_calibration_n_eff']
        used['shift_calibration_weeks']=used['center_calibration_weeks']
        used['shift_calibration_source']=used['center_calibration_source']
        if not used.price_shifted_pre_iso.gt(0).all():
            raise ValueError('Invalid signed center calibration price')
        used,viol=self._project_monotone(used)
        used=self._intervals(used)
        used['surface_date']=asof
        # Single source of truth for ad-derived used-price surfaces.
        # This does NOT remove mismatched is_copy ads from model training.
        candidates_n = len(used)
        used = _keep_surface_identities(used, self._surface_original_by_id)
        log.info('[SURFACE IDENTITY] kept=%s/%s (known ID and complementary DB is_original)',
                 f'{len(used):,}', f'{candidates_n:,}')
        if used.empty:
            raise ValueError('No valid surface cells: check motorcycle_model_id and source is_original')
        if used.duplicated(CELL).any():raise AssertionError('Duplicate surface cells')
        r=(used.price_high/used.price_low).to_numpy(float)
        if not np.isfinite(r).all() or np.any(r < 1) or np.any(r>1.10000001):
            raise AssertionError('Interval bounds invalid or exceed 10pct HARD CAP')
        if not (used.price_low.gt(0) & used.price_high.ge(used.price_center) &
                used.price_center.ge(used.price_low)).all():
            raise AssertionError('All intervals must contain the center price')
        if used['center_adjustment_effective_pct'].abs().gt(self.config.center_max_adjust_pct+1e-3).any():
            raise AssertionError('Effective monotonic center correction exceeds 3pct')
        log.info('[SURFACE] DONE used=%s | warnings=%s | blocked=%s | violations_before=%s | elapsed=%.1fs',
                 f'{len(used):,}',int(used.isotonic_warning.sum()),int(used.isotonic_block.sum()),viol,time.perf_counter()-started)
        return used


    def _future_cells(self,dt):
        t=self.ads[self.ads.date.ge(dt)&self.ads.date.lt(dt+pd.Timedelta(days=7))].copy()
        fields=CELL+['actual','future_n','actual_independent','future_unique_ads',
                     'future_observed_days','future_iqr_pct',
                     'future_loo_median_shift_pct']
        if t.empty:return pd.DataFrame(columns=fields)
        # Preserve the LEGACY all-ad weekly median for the general Backtest.
        # A distinct, deduplicated median is used by the publication risk bank.
        raw=t.groupby(CELL,dropna=False).price.agg(actual='median',future_n='size').reset_index()
        if 'token' in t and t.token.notna().any():
            known=t[t.token.notna()].sort_values('date').drop_duplicates(CELL+['token'],keep='last')
            missing=t[t.token.isna()]
            unique=pd.concat([known,missing],ignore_index=True)
        elif 'url' in t and t.url.notna().any():
            known=t[t.url.notna()].sort_values('date').drop_duplicates(CELL+['url'],keep='last')
            missing=t[t.url.isna()]
            unique=pd.concat([known,missing],ignore_index=True)
        else:
            unique=t
        g=unique.groupby(CELL,dropna=False)
        evidence=g.price.agg(actual_independent='median',future_unique_ads='size',
                             _q25=lambda v:v.quantile(.25),
                             _q75=lambda v:v.quantile(.75)).reset_index()
        evidence['future_iqr_pct']=100*(evidence._q75-evidence._q25)/evidence.actual_independent.clip(lower=1)
        evidence=evidence.drop(columns=['_q25','_q75'])
        days=g.date.nunique().rename('future_observed_days').reset_index()
        def loo_stability(prices):
            values=np.asarray(prices,dtype=float)
            values=values[np.isfinite(values)&(values>0)]
            if len(values)<3:
                return np.nan
            med=float(np.median(values))
            return (100*max(abs(float(np.median(np.delete(values,i)))/med-1)
                            for i in range(len(values))) if med>0 else np.nan)
        stability=(g.price.agg(loo_stability)
                   .rename('future_loo_median_shift_pct').reset_index())
        return (raw.merge(evidence,on=CELL,validate='one_to_one')
                   .merge(days,on=CELL,validate='one_to_one')
                   .merge(stability,on=CELL,validate='one_to_one'))

    def _backtest_single(self,dt,shift_calibration=None,return_grid=False):
        z=self.train_ads
        tr=z[z.date.lt(dt)]
        if len(tr)<1000:raise ValueError('Not enough training ads for fold')
        log.info('[BACKTEST] Origin=%s | residual train rows=%s',dt.date(),f'{len(tr):,}')
        started=time.perf_counter()
        model = self._fit_residual_with_early_stopping(tr)
        log.info('[BACKTEST] Residual trained in %.1fs, creating historical surface...',time.perf_counter()-started)
        test=z[z['asof'].eq(dt)]
        pred=np.exp(test.age_ref_log+model.predict(test[FEATURES_24]))
        ad_metrics=_metrics(test.price,pred)
        grid=self.build_surface(asof=dt,residual_model=model)
        future=self._future_cells(dt).query('future_n>=2').copy()
        columns=CELL+['price_raw','price_unshifted_monotonic','price_monotonic',
                      'price_center','price_low','price_high','band_width_pct',
                      'shift_down_pct','shift_pre_isotonic_pct',
                      'center_adjustment_pct','center_adjustment_effective_pct',
                      'publish_state','isotonic_change_pct']
        merged=future.merge(grid[columns],on=CELL,how='left',validate='one_to_one')
        missing_grid_n=int(merged.price_monotonic.isna().sum())
        cells=merged[merged.price_monotonic.notna()].copy()
        cells['asof']=dt
        cellhist=self.ads[self.ads.date.lt(dt)].groupby(CELL).size().rename('history_cell_n').reset_index()
        cells=cells.merge(cellhist,on=CELL,how='left',validate='many_to_one')
        cells['history_cell_n']=cells.history_cell_n.fillna(0).astype(int)
        cells['abs_error_pct']=100*np.abs(cells.price_monotonic/cells.actual-1)
        cells['inside_interval']=cells.actual.between(cells.price_low,cells.price_high)
        cells['above_floor']=cells.actual>=cells.price_low
        cells['display_candidate']=cells.publish_state.eq('MODEL_SUPPORT_REVIEW')
        # Actual DB publishing also requires separately backtested group reliability.
        # Do not call candidates 'published': history group reliability may veto them.
        raw_m=_metrics(cells.actual,cells.price_raw)
        mono_m=_metrics(cells.actual,cells.price_monotonic)
        # Same OOF cells and origin: raw vs shift-then-ISO vs ISO-then-shift.
        # The last alternative is a diagnostic only: it can break monotonicity.
        post_iso_price=(cells.price_unshifted_monotonic *
                        (1+cells.center_adjustment_pct/100))
        baseline_m=_metrics(cells.actual,cells.price_unshifted_monotonic)
        post_iso_m=_metrics(cells.actual,post_iso_price)
        base_signed=float((100*(cells.price_unshifted_monotonic/cells.actual-1)).mean()) if len(cells) else np.nan
        pre_signed=float((100*(cells.price_monotonic/cells.actual-1)).mean()) if len(cells) else np.nan
        post_signed=float((100*(post_iso_price/cells.actual-1)).mean()) if len(cells) else np.nan
        unseen_m=_metrics(cells.loc[cells.history_cell_n.eq(0),'actual'],cells.loc[cells.history_cell_n.eq(0),'price_monotonic'])
        displayed=cells[cells.display_candidate]
        disp_m=_metrics(displayed.actual,displayed.price_monotonic)
        def pct(x):return 100*float(x.mean()) if len(x) else np.nan
        report={'asof':str(dt.date()),'train_n':len(tr),'test_ads_n':len(test),
                'ad_mape_pct':ad_metrics['mape_pct'],
                'future_cells_ge2_universe':len(future),'future_cells_ge2':len(cells),
                'future_cells_excluded_by_grid':missing_grid_n,
                'surface_cell_recall_pct':100*len(cells)/len(future) if len(future) else np.nan,
                'raw_cell_mape_pct':raw_m['mape_pct'],'isotonic_cell_mape_pct':mono_m['mape_pct'],
                'unshifted_isotonic_mape_pct':baseline_m['mape_pct'],
                'shift_before_isotonic_mape_pct':mono_m['mape_pct'],
                'shift_after_isotonic_mape_pct':post_iso_m['mape_pct'],
                'unshifted_isotonic_signed_error_pct':base_signed,
                'shift_before_isotonic_signed_error_pct':pre_signed,
                'shift_after_isotonic_signed_error_pct':post_signed,
                'shift_effective_cells_ge2':int(cells.shift_down_pct.gt(1e-9).sum()),
                'baseline_overestimate_share_pct':100*float((cells.price_unshifted_monotonic>cells.actual).mean()) if len(cells) else np.nan,
                'shift_before_iso_overestimate_share_pct':100*float((cells.price_monotonic>cells.actual).mean()) if len(cells) else np.nan,
                'shift_after_iso_overestimate_share_pct':100*float((post_iso_price>cells.actual).mean()) if len(cells) else np.nan,
                'unseen_full_cells_ge2':unseen_m['n'],'unseen_full_cells_mape_pct':unseen_m['mape_pct'],
                'interval_coverage_pct':pct(cells.inside_interval),
                'floor_coverage_pct':pct(cells.above_floor),
                'display_candidate_cells_ge2':len(displayed),
                'display_candidate_share_pct':100*len(displayed)/len(cells) if len(cells) else np.nan,
                'display_candidate_cell_mape_pct':disp_m['mape_pct'],
                'display_candidate_interval_coverage_pct':pct(displayed.inside_interval),
                'display_candidate_floor_coverage_pct':pct(displayed.above_floor)}
        return (report,cells,grid) if return_grid else (report,cells)

    def _calibrate(self, cells):
        # Called only with completed (previous) fold cells. The OOF bank is
        # evaluated against the NEXT forecast origin at predict time.
        cutoff = self.asof if cells is None or len(cells)==0 else (
            pd.Timestamp(cells['asof'].max()) + pd.Timedelta(days=7))
        self.center_calibrator.fit(cells,cutoff=cutoff)
        self.interval_positioner.fit(cells,cutoff=cutoff)
        if len(cells)==0:
            self.calibration=pd.DataFrame()
            self.calibration_global={'mape_pct':10.,'signed_bias_pct':0.,'coverage_guard_width_pct':0.}
            return
        x=cells.copy()
        x['ape_pct']=100*np.abs(x.price_monotonic/x.actual-1)
        x['signed_pct']=100*(x.price_monotonic/x.actual-1)
        # Required high/low ratio width of a symmetric geometric interval.
        # This is computed from previous completed folds; never future target of same fold.
        ratio=np.maximum(x.actual/x.price_monotonic,x.price_monotonic/x.actual)
        x['required_width_pct']=100*(np.square(ratio)-1)
        q=float(self.config.interval_guard_quantile)
        if not 0.5<=q<=0.95:raise ValueError('interval_guard_quantile must be in [0.5,0.95]')
        self.calibration_global={'mape_pct':float(x.ape_pct.mean()),
                                 'signed_bias_pct':float(x.signed_pct.mean()),
                                 'coverage_guard_width_pct':float(x.required_width_pct.quantile(q))}
        self.calibration=x.groupby(GROUP,as_index=False,dropna=False).agg(
            group_eval_n=('ape_pct','size'),group_mape_pct=('ape_pct','mean'),
            group_overpricing_pct=('signed_pct','mean'),
            group_guard_width_pct=('required_width_pct',lambda v:float(v.quantile(q))))

    def _new_reliability_policy(self):
        return HierarchicalSurfaceReliability(
            min_c_interval_coverage_pct=self.config.reliability_c_min_interval_coverage_pct,
            min_monthly_samples=self.config.min_motorcycle_monthly_ads,
            min_target_ads=self.config.min_model_risk_target_ads,
            min_target_days=self.config.min_model_risk_target_days,
            max_target_iqr_pct=self.config.max_model_risk_target_iqr_pct,
            model_risk_min_cells=self.config.model_risk_min_cells,
            model_risk_min_weeks=self.config.model_risk_min_weeks,
            model_risk_min_mdape_pct=self.config.model_risk_min_mdape_pct,
            model_risk_min_o10_share=self.config.model_risk_min_o10_share,
            quality_min_cells=self.config.quality_min_cells,
            quality_min_weeks=self.config.quality_min_weeks,
            quality_min_ages=self.config.quality_min_ages,
            quality_max_mdape_pct=self.config.quality_max_mdape_pct,
            quality_max_mape_pct=self.config.quality_max_mape_pct,
            quality_max_abs_bias_pct=self.config.quality_max_abs_bias_pct,
            quality_max_o10_share=self.config.quality_max_o10_share,
            quality_max_u10_share=self.config.quality_max_u10_share,
            quality_min_coverage_pct=self.config.quality_min_coverage_pct,
            quality_min_good_week_share=self.config.quality_min_good_week_share,
            enable_expanded_approval=self.config.quality_enable_expanded_approval,
            enable_two_ad_lane=self.config.quality_enable_two_ad_lane,
            expanded_market_min_cells=self.config.quality_expanded_market_min_cells,
            expanded_market_min_matched_ads=self.config.quality_expanded_market_min_matched_ads,
            expanded_market_max_median_ape_pct=self.config.quality_expanded_market_max_median_ape_pct,
            expanded_market_max_abs_signed_bias_pct=self.config.quality_expanded_market_max_abs_signed_bias_pct,
            market_group_max_abs_bias_pct=self.config.quality_market_group_max_abs_bias_pct,
            max_cell_isotonic_change_pct=self.config.quality_max_cell_isotonic_change_pct)

    def _attach_fold_reliability(self, cells, past_cells, dt):
        """Predict reliability from completed EARLIER folds, before seeing this fold."""
        h=self.ads[self.ads.date.lt(dt)]
        h30=h[h.date.ge(dt-pd.Timedelta(days=30))]
        t=cells.copy()
        age30=h30.groupby(GROUP+['age']).size().rename('n30').reset_index()
        grp30=h30.groupby(GROUP).size().rename('group_n30').reset_index()
        ga=h.groupby(GROUP+['age']).size().rename('_hist_age_n').reset_index()
        t=t.merge(age30,on=GROUP+['age'],how='left',validate='many_to_one')
        t=t.merge(grp30,on=GROUP,how='left',validate='many_to_one')
        t=t.merge(self._recent_motorcycle_counts(h,dt),on=PARENT,how='left',validate='many_to_one')
        t['motorcycle_n30']=t.motorcycle_n30.fillna(0).astype(int)
        t=t.merge(ga,on=GROUP+['age'],how='left',validate='many_to_one')
        t['local_cell_n']=t.history_cell_n
        t['age_source']=np.where(t._hist_age_n.fillna(0).gt(0),'observed120','interpolated_or_extrapolated')
        policy=self._new_reliability_policy().fit(past_cells,asof=dt)
        return policy.apply(t,asof=dt,publish_review=self.config.reliability_publish_review_tier_c).drop(columns=['_hist_age_n'])

    @staticmethod
    def _summarize_reliability(cells, keys):
        if cells is None or cells.empty:return pd.DataFrame()
        x=cells.copy()
        x['ape']=100*(x.price_center/x.actual-1).abs()
        x['covered']=x.actual.between(x.price_low,x.price_high).astype(float)
        x['floor_covered']=(x.actual>=x.price_low).astype(float)
        return (x.groupby(keys,as_index=False,dropna=False).agg(
            evaluated_cells=('ape','size'),mape_post_surface_pct=('ape','mean'),
            interval_coverage_pct=('covered',lambda v:100*float(v.mean())),
            floor_coverage_pct=('floor_covered',lambda v:100*float(v.mean())),
            publishable_cells=('is_reliable','sum')))

    def backtest(self, dates=None):
        if dates is None:
            valid=sorted(t for t in self.snapshots['asof'].unique() if
                pd.Timestamp(t)+pd.Timedelta(days=7)<=self.asof and pd.Timestamp(t)>=self.asof-pd.Timedelta(days=self.config.reliability_backtest_lookback_days))
            dates=valid[-self.config.reliability_backtest_folds:]
        dates=[pd.Timestamp(d).normalize() for d in dates]
        if not dates:raise ValueError('No completed backtest origins available')
        reports=[]; cell_results=[]
        log.info('[BACKTEST] Starting %s historical folds: %s',len(dates),[str(x.date()) for x in dates])
        self._calibrate(pd.DataFrame())
        for i,dt in enumerate(dates):
            log.info('[BACKTEST] Fold %s/%s at %s',i+1,len(dates),dt.date())
            r,c,fold_grid=self._backtest_single(dt,return_grid=True)
            earlier=pd.concat(cell_results,ignore_index=True) if cell_results else pd.DataFrame()
            c=self._attach_fold_reliability(c,earlier,dt)
            # OOF gate is decided BEFORE joining any targets from this fold.
            # Folds without two completed independent histories remain unapproved.
            # This is audit-only and never sees the current fold's targets.
            if earlier.empty:
                decisions = fold_grid[CELL].copy()
                decisions['fold_publication_allowed'] = False
            else:
                historic_gate = apply_publication(fold_grid, dt, earlier, PublicationConfig())
                decisions = (historic_gate[CELL + ['publication_allowed']]
                    .rename(columns={'publication_allowed':'fold_publication_allowed'}))
            # Fold grid normalizes is_copy into int8 inside the strict policy;
            # the original cell frame uses a string CatBoost category.
            # Only normalize the decision keys (leave core calibration keys intact).
            for name in ('motorcycle_model_id','real_brand','real_model','model_variant','is_copy'):
                decisions[name]=decisions[name].astype(str)
            c=c.merge(decisions,on=CELL,how='left',validate='many_to_one')
            if c.fold_publication_allowed.isna().any():
                raise AssertionError('OOF cell missing historical publication decision')
            chosen=c.loc[c.fold_publication_allowed]
            r['reliability_auto_cells']=len(chosen)
            r['reliability_auto_coverage_pct']=100*float(chosen.actual.between(chosen.price_low,chosen.price_high).mean()) if len(chosen) else np.nan
            r['reliability_auto_mape_pct']=float((100*(chosen.price_center/chosen.actual-1).abs()).mean()) if len(chosen) else np.nan
            r['reliability_auto_model_ids']=int(chosen.motorcycle_model_id.nunique())
            r['calibration_using_only_earlier_folds']=i>0
            reports.append(r);cell_results.append(c)
            # Calibrate only using completed earlier folds; fold itself never calibrates itself.
            self._calibrate(pd.concat(cell_results,ignore_index=True))
            log.info('Backtest %s %s',str(dt.date()),r)
        self.backtest_results=pd.DataFrame(reports)
        self.backtest_cells=pd.concat(cell_results,ignore_index=True)
        self.backtest_group_metrics=self.evaluate_groups(self.backtest_cells)
        self.backtest_reliability_metrics=self._summarize_reliability(
            self.backtest_cells,['asof','reliability_tier'])
        self.backtest_reliability_group_metrics=self._summarize_reliability(
            self.backtest_cells,['asof']+GROUP+['reliability_tier'])
        return self.backtest_results

    @staticmethod
    def evaluate_groups(cells: pd.DataFrame) -> pd.DataFrame:
        """Real post-Surface future-median metrics, per group and forecast week.

        Groups with zero eligible cells have NaN coverage, not misleading zero.
        Never treat these metrics as observed future accuracy of today's grid.
        """
        if cells is None or cells.empty:return pd.DataFrame()
        x=cells.copy()
        x['ape_pct']=100*np.abs(x.price_monotonic/x.actual-1)
        x['inside_interval']=x.actual.between(x.price_low,x.price_high)
        x['above_floor']=x.actual>=x.price_low
        x['display_candidate']=x.publish_state.eq('MODEL_SUPPORT_REVIEW')
        def group_summary(chunk):
            eligible=chunk.loc[chunk.display_candidate]
            return pd.Series({
                'future_cells_ge2':len(chunk),
                'mape_post_surface_pct':float(chunk.ape_pct.mean()),
                'median_ape_post_surface_pct':float(chunk.ape_pct.median()),
                'interval_coverage_all_pct':100*float(chunk.inside_interval.mean()),
                'floor_coverage_all_pct':100*float(chunk.above_floor.mean()),
                'display_candidate_cells':len(eligible),
                'display_candidate_mape_pct':float(eligible.ape_pct.mean()) if len(eligible) else np.nan,
                'display_candidate_interval_coverage_pct':100*float(eligible.inside_interval.mean()) if len(eligible) else np.nan,
                'display_candidate_floor_coverage_pct':100*float(eligible.above_floor.mean()) if len(eligible) else np.nan,
            })
        output=[]
        for key,z in x.groupby(['asof']+GROUP,dropna=False,sort=False):
            output.append(dict(zip(['asof']+GROUP,key),**group_summary(z).to_dict()))
        return pd.DataFrame(output)

    def _resolve_production_year(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Recover Iranian production_year from actual data (not 2026-age Gregorian math).

        Prefer observed same group+age production_year, then inferred dataset's
        reference Persian year minus age.  Never silently calculate using Gregorian year.
        """
        grid = surface.copy()
        hist = self.ads[self.ads.date.lt(self.asof)].copy()
        source = hist
        if 'production_year' not in source:
            grid['production_year'] = np.nan
            grid['production_year_source'] = 'missing'
            return grid
        source['production_year'] = pd.to_numeric(source['production_year'], errors='coerce')
        source = source[source.production_year.between(1300, 1600)].copy()
        if source.empty:
            grid['production_year'] = np.nan
            grid['production_year_source'] = 'missing'
            return grid
        source['age'] = pd.to_numeric(source['age'], errors='coerce')
        exact = (source.groupby(GROUP + ['age'], as_index=False, dropna=False)
                 .production_year.agg(lambda x: float(x.mode().iloc[0]) if not x.mode().empty else np.nan))
        exact = exact.rename(columns={'production_year': '_actual_production_year'})
        grid = grid.merge(exact, on=GROUP + ['age'], how='left', validate='many_to_one')
        ref = source.loc[source.age.between(0, 60)].copy()
        ref['_ref_year'] = (ref.production_year + ref.age).round()
        # Published model years are Persian; derive the current reference year from the dataset.
        plausible = ref._ref_year[ref._ref_year.between(1350, 1600)]
        reference_year = int(plausible.mode().iloc[0]) if len(plausible) else None
        if reference_year is not None:
            fallback = reference_year - pd.to_numeric(grid.age, errors='coerce')
        else:
            fallback = pd.Series(np.nan, index=grid.index)
        grid['production_year_source'] = np.where(grid._actual_production_year.notna(), 'observed', 'derived_from_dataset')
        grid['production_year'] = grid._actual_production_year.fillna(fallback)
        grid.loc[grid.production_year.isna(), 'production_year_source'] = 'missing'
        grid = grid.drop(columns=['_actual_production_year'])
        return grid

    def _build_model_index_history(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Legacy history view, populated without in-sample price evaluation.

        Normalize IDs before grouping or joining history.
        """
        surface = surface.copy()
        surface['motorcycle_model_id'] = (
            pd.to_numeric(surface['motorcycle_model_id'], errors='coerce')
            .astype('Int64').astype(str).replace('<NA>', 'UNKNOWN')
        )
        asof = self.asof
        used = self.ads[(self.ads.date.lt(asof)) &
                        (self.ads.date.ge(asof - pd.Timedelta(days=self.config.history_days)))].copy()
        used['recent30'] = used.date.ge(asof - pd.Timedelta(days=30)).astype(int)
        stats = used.groupby(GROUP, as_index=False, dropna=False).agg(
            total_samples=('price', 'size'), samples_last_month=('recent30', 'sum'),
            price_p10=('price', lambda v: v.quantile(.10)),
            price_p50=('price', 'median'),
            price_p90=('price', lambda v: v.quantile(.90)),
        )
        # ModelIndexHistory includes only surfaced ad-derived groups.
        stats = stats.merge(surface[GROUP].drop_duplicates(), on=GROUP,
                            how='inner', validate='one_to_one')
        if stats.empty:
            return stats
        cells = self.backtest_cells.copy()
        if len(cells):
            # DB group reliability is calibrated on candidate-display cells only.
            # Full matched-cell metrics remain in backtest_group_metrics.
            cells = cells[cells.publish_state.eq('MODEL_SUPPORT_REVIEW')].copy()
            if len(cells):
                cells['ape'] = 100 * np.abs(cells.price_monotonic / cells.actual - 1)
                cells['inside'] = cells.actual.between(cells.price_low, cells.price_high)
                metrics = cells.groupby(GROUP, as_index=False, dropna=False).agg(
                    tested_samples=('ape', 'size'), mape=('ape', 'mean'), mdape=('ape', 'median'),
                    coverage_percent=('inside', lambda x: 100 * float(x.mean())),
                )
                stats = stats.merge(metrics, on=GROUP, how='left', validate='one_to_one')
            else:
                stats['tested_samples'] = 0
                stats['mape'] = np.nan
                stats['mdape'] = np.nan
                stats['coverage_percent'] = np.nan
        else:
            stats['tested_samples'] = 0
            stats['mape'] = np.nan
            stats['mdape'] = np.nan
            stats['coverage_percent'] = np.nan
        stats['tested_samples'] = stats.tested_samples.fillna(0).astype(int)
        # No candidate samples = unknown accuracy, not zero error/zero coverage.

        shifts = surface.groupby(GROUP, as_index=False, dropna=False).agg(
            optimal_shift=('shift_down_pct', lambda x: -float(x.mean()) if len(x) else 0.))
        stats = stats.merge(shifts, on=GROUP, how='left', validate='one_to_one')
        stats['optimal_shift'] = stats.optimal_shift.fillna(0.)
        stats['is_reliable'] = (
            stats.coverage_percent.ge(30) & stats.mdape.le(10) &
            stats.mape.le(15) & stats.tested_samples.ge(3) &
            stats.samples_last_month.ge(7)
        )
        return stats

    def _complete_dominantly_reliable_groups(self, surface: pd.DataFrame) -> pd.DataFrame:
        """Fill interior gaps in a predominantly A/B/C group, without inventing prices.

        The *full* predicted grid is already present when this method is called.
        Only a D tier may be upgraded to C. The model prediction and isotonic
        prices remain unchanged. The source of the upgraded reliability claim is
        completed, earlier-fold group evaluation, never today's observed prices.

        We deliberately do NOT promote age 0, the maximum age, unsupported
        younger-age extrapolation, a hard isotonic block, or any D-heavy group.
        This is a tier-C reviewed estimate, not an observed local cell.
        """
        g = surface.copy()
        g['completed_by_group_policy'] = False
        g['completion_group_reliable_share'] = np.nan
        g['completion_group_backtest_n'] = 0
        g['completion_group_backtest_weeks'] = 0
        g['completion_group_backtest_mape_pct'] = np.nan
        if not self.config.complete_dominantly_reliable_groups or g.empty:
            return g

        # The policy is applied after the authoritative ID/copy surface gate.
        # At this point, each GROUP contains all economically permitted coordinates.
        age = pd.to_numeric(g['age'], errors='coerce')
        maxage = pd.to_numeric(g['max_age'], errors='coerce')
        minage = pd.to_numeric(g['min_age'], errors='coerce')
        iso = pd.to_numeric(g['isotonic_change_pct'], errors='coerce').abs()
        hard_block = (g['isotonic_block'].fillna(False).astype(bool) | iso.gt(5.0)
                      | ~np.isfinite(iso))
        # Age 0 and the right-hand boundary are deliberately exempt. A young
        # age below the youngest observed age is extrapolation, not interpolation.
        interior = (age.gt(0) & age.ge(minage)
                    & age.lt(maxage)
                    & age.le(self.config.completion_priority_max_age))
        prevalidated = (g['reliability_tier'].isin(['A', 'B', 'C'])
                        & ~hard_block & interior)

        x = g[GROUP].copy()
        x['_interior'] = interior.to_numpy(bool)
        x['_prevalidated'] = prevalidated.to_numpy(bool)
        x['_valid_age'] = age.where(prevalidated).to_numpy()
        by_group = x.groupby(GROUP, dropna=False, as_index=False, sort=False).agg(
            interior_n=('_interior', 'sum'),
            validated_n=('_prevalidated', 'sum'),
            validated_ages=('_valid_age', 'nunique'),
        )
        by_group['_share'] = (by_group['validated_n'] /
                              by_group['interior_n'].replace(0, np.nan))

        # Extract the *group-specific* independent history bank. Using a cohort
        # prior here would incorrectly grant trust to a model never backtested.
        bank = (getattr(self.reliability_policy, 'tables', {}) or {}).get('group')
        required = GROUP + ['_ev_group_n', '_ev_group_weeks',
                            '_ev_group_mape', '_ev_group_coverage']
        if bank is None or not set(required).issubset(bank.columns):
            log.warning('[GRID COMPLETION] No completed group backtest bank; no D upgrades')
            return g
        bank = bank[required].drop_duplicates(GROUP)
        by_group = by_group.merge(bank, on=GROUP, how='left', validate='one_to_one')
        qualified = (
            by_group['_share'].ge(self.config.completion_min_reliable_share)
            & by_group['validated_n'].ge(self.config.completion_min_reliable_cells)
            & by_group['validated_ages'].ge(self.config.completion_min_reliable_ages)
            & by_group['_ev_group_n'].ge(self.config.completion_min_group_backtest_cells)
            & by_group['_ev_group_weeks'].ge(self.config.completion_min_group_backtest_weeks)
            & by_group['_ev_group_mape'].le(self.config.completion_max_group_backtest_mape_pct)
            & by_group['_ev_group_coverage'].ge(self.config.completion_min_group_backtest_coverage_pct)
        )
        by_group['_qualified'] = qualified
        keep_cols = GROUP + ['_share', '_qualified', '_ev_group_n', '_ev_group_weeks',
                             '_ev_group_mape']
        # Avoid adding a duplicate many-to-many relationship to the grid.
        g = g.merge(by_group[keep_cols], on=GROUP, how='left', validate='many_to_one')
        g['completion_group_reliable_share'] = g['_share']
        g['completion_group_backtest_n'] = g['_ev_group_n'].fillna(0).astype(int)
        g['completion_group_backtest_weeks'] = g['_ev_group_weeks'].fillna(0).astype(int)
        g['completion_group_backtest_mape_pct'] = g['_ev_group_mape']

        can_complete = (
            g['_qualified'].fillna(False)
            & g['reliability_tier'].eq('D')
            & interior
            & age.ge(minage)
            & ~hard_block
            & pd.to_numeric(g['price_center'], errors='coerce').gt(0)
            & pd.to_numeric(g['estimated_mape_pct'], errors='coerce').le(
                self.config.completion_max_cell_estimated_mape_pct)
            & pd.to_numeric(g['estimated_interval_coverage_pct'], errors='coerce').ge(
                self.config.completion_min_cell_estimated_coverage_pct)
        )
        # Only interpolate across ages actually observed for the same exact
        # group in the 120-day history. This excludes extrapolation beyond the
        # oldest and youngest observed samples even if age=0 was generated.
        supported_age = (g.loc[pd.to_numeric(g.get('n120', 0), errors='coerce').gt(0),
                                 GROUP + ['age']]
                         .drop_duplicates())
        if not supported_age.empty:
            observed = supported_age.groupby(GROUP, dropna=False, as_index=False).agg(
                _first_observed_age=('age', 'min'), _last_observed_age=('age', 'max'))
            g = g.merge(observed, on=GROUP, how='left', validate='many_to_one')
            can_complete = can_complete & age.ge(g['_first_observed_age']) & age.le(g['_last_observed_age'])
            g = g.drop(columns=['_first_observed_age', '_last_observed_age'])
        else:
            can_complete = can_complete & False

        g.loc[can_complete, 'reliability_tier'] = 'C'
        g.loc[can_complete, 'is_reliable'] = bool(self.config.reliability_publish_review_tier_c)
        g.loc[can_complete, 'reliability_reason'] = 'completed_interior_from_validated_group'
        g.loc[can_complete, 'completed_by_group_policy'] = True
        g = g.drop(columns=['_share', '_qualified', '_ev_group_n',
                            '_ev_group_weeks', '_ev_group_mape'])
        log.info('[GRID COMPLETION] reliable-majority GROUPs=%s/%s; D->C interior=%s; remaining interior D=%s',
                 int(qualified.sum()), len(by_group), int(can_complete.sum()),
                 int((interior & g['reliability_tier'].eq('D')).sum()))
        return g

    def create_surface(self, segment_metrics=None, asof=None, debug=False) -> pd.DataFrame:
        """Return compact DataFrame by default; debug=True provides all features.

        Database persistence uses the full internal frame; compact results never
        remove information required by update_database().
        """
        if segment_metrics is not None:
            log.warning('segment_metrics ignored: calibration is derived from causal backtest')
        if asof is not None and pd.Timestamp(asof).normalize()!=self.asof:
            raise ValueError('Only fitted asof supported; rerun fit for another origin')
        result=self.build_surface(asof=self.asof)
        # Normalize IDs before downstream GROUP merges and grouping.
        result['motorcycle_model_id'] = (
            pd.to_numeric(result['motorcycle_model_id'], errors='coerce')
            .astype('Int64').astype(str).replace('<NA>', 'UNKNOWN')
        )
        if result.duplicated(CELL).any():
            raise ValueError('Duplicate surface cells after motorcycle_model_id normalization')
        result['adjusted_price']=result['price_center']
        result['suggested_lower']=result['price_low']
        result['suggested_upper']=result['price_high']
        result['predicted_price']=result['price_raw']
        result['optimal_shift']=-result['shift_down_pct'].fillna(0)
        result['MAPE(%)']=result.get('group_mape_shrunk_pct')
        result=self._resolve_production_year(result)
        # Only previously completed evaluation folds are used to estimate cell quality.
        self.reliability_policy=self._new_reliability_policy().fit(self.backtest_cells,asof=self.asof)
        # Recent in-sample price agreement is an independent safety monitor,
        # not an OOF prediction validation or a substitute for it.
        self.reliability_policy.fit_market_alignment(result,self.ads,asof=self.asof)
        result=self.reliability_policy.apply(
            result,asof=self.asof,publish_review=self.config.reliability_publish_review_tier_c)
        # Fill *already predicted* valid interior cells only for groups whose
        # majority is A/B/C and whose independent earlier-fold results pass.
        result = self._complete_dominantly_reliable_groups(result)
        # DIAGNOSTICS ONLY: age-six completeness must NEVER veto A/B/C publication.
        # The untouched V3 database.py publishes all reliable candidate rows.
        caps = (self.grid_bucket_limits.set_index('age')['max_bucket'].to_dict()
                if self.grid_bucket_limits is not None and not self.grid_bucket_limits.empty
                else {})
        result = apply_age_window_certificate(
            result,
            max_required_age=self.config.coverage_required_max_age,
            mileage_caps=caps,
            bucket_levels=self.config.bucket_levels,
            condition_levels=self.config.condition_levels,
            publication_mode='audit',  # full-group publication veto is forbidden
        )
        log.info('[AGE6 COVERAGE] eligible groups=%s | completely certified=%s | mode=%s',
                 int(result.loc[result.coverage_window_applicable, GROUP].drop_duplicates().shape[0]),
                 int(result.loc[result.coverage_full_to_age6, GROUP].drop_duplicates().shape[0]),
                 self.config.coverage_publication_mode)

        # Sole, fail-closed publication authority; no V10/V13/V14/V15 fallback.
        result=apply_publication(result, self.asof, self.backtest_cells,
                                 PublicationConfig())
        # Post-prediction check against USED ads available before the run date.
        # Stage 1 age reference is already included in price_center for cells
        # without exact listings; no factory data or future ads are consulted.
        result=apply_recent_market_guard(result, self.ads, self.asof)
        result['is_reliable']=result['publication_allowed'].astype(bool)
        if result['is_factory'].any():
            raise AssertionError('Factory price leaked into used-price surface')
        if not result['policy_version'].eq(POLICY_VERSION).all():
            raise AssertionError('Publication policy version mismatch')
        log.info('[PUBLICATION RISK_GUARD] approved=%s/%s | motor ids=%s',
                 int(result.publication_allowed.sum()), len(result),
                 int(result.loc[result.publication_allowed,'motorcycle_model_id'].nunique()))

        # Preserve ALL constructed prices, including unobserved D cells.
        # Publication approval is separate from tier; D is not promoted to C.
        # Sparse models stay in the full grid for review, but are not published.
        blocked = int((~result['publication_allowed']).sum())
        log.info('[PUBLICATION] retained full %s-cell grid; approved=%s blocked=%s',
                 f'{len(result):,}',int(result.publication_allowed.sum()),blocked)
        self._surface_df=result.copy()
        self._model_index_history_data=self._build_model_index_history(result)
        # Group-level DB flag is an aggregate; the source of truth is per-cell is_reliable.
        group_valid=(result.groupby(GROUP,dropna=False,as_index=False).is_reliable.any()
                     .rename(columns={'is_reliable':'_any_cell_reliable'}))
        self._model_index_history_data=self._model_index_history_data.merge(
            group_valid,on=GROUP,how='left',validate='one_to_one')
        self._model_index_history_data['is_reliable']=self._model_index_history_data['_any_cell_reliable'].fillna(False).astype(bool)
        self._model_index_history_data=self._model_index_history_data.drop(columns=['_any_cell_reliable'])
        self._surface_debug_df=result.copy() if debug else None
        if debug or not self.config.compact_output_default:
            return result
        cols=GROUP+['is_original','age','production_year','production_year_source','mileage_bucket','mileage_clean',
              'condition_tier','adjusted_price','suggested_lower','suggested_upper','predicted_price',
              'band_width_pct','shift_down_pct','estimated_overpricing_pct',
              'center_adjustment_pct','center_adjustment_effective_pct',
              'center_calibration_source','center_overestimate_share','center_signed_median_over_pct',
              'center_shift_confidence_lower','center_shift_isotonic_spillover','center_calibration_n_eff',
              'center_calibration_weeks','interval_position','interval_source',
              'shift_calibration_n_eff','shift_calibration_weeks','shift_calibration_source',
              'price_shifted_pre_iso','shift_pre_isotonic_pct',
              'price_unshifted_monotonic','group_mape_shrunk_pct','group_eval_n',
              'n30','local_cell_n','age_source','km_origin','publish_state',
              'is_factory',
              'isotonic_change_pct','isotonic_warning','isotonic_block','surface_date',
              'is_reliable','publication_allowed','publication_status','publication_reason_code',
              'policy_version','oof_n','oof_weeks','oof_median_abs_pct','oof_over15','oof_under15','publication_evidence_source',
              'prediction_validated','prediction_quality_status',
              'quality_expansion_lane','publication_evidence_tier','quality_market_matched_ads',
              'quality_model_status','production_quality_state','quality_oof_n',
              'quality_oof_weeks','quality_oof_ages','quality_oof_mdape_pct',
              'quality_oof_mape_pct','quality_oof_robust_mape_pct','quality_oof_signed_median_pct',
              'quality_oof_o10_share','quality_oof_u10_share',
              'quality_oof_coverage_pct','quality_oof_floor_pct',
              'quality_oof_shift_mape_delta_pct','quality_oof_iso_mape_delta_pct',
              'market_n_stable_cells','market_median_signed_pct','market_median_ape_pct',
              'market_model_drift_review','market_cell_n','market_cell_signed_pct',
              'market_cell_mismatch_review',
              'motorcycle_n30','motorcycle_support_eligible',
              'model_oof_n','model_oof_weeks','model_oof_mdape_pct','model_oof_mape_pct',
              'model_oof_o10_share','model_oof_u10_share','model_oof_high_risk',
              'reliability_tier','reliability_evidence_level','reliability_eval_n',
              'reliability_eval_weeks','estimated_mape_pct','estimated_interval_coverage_pct',
              'estimated_floor_coverage_pct','reliability_reason',
              'completed_by_group_policy','completion_group_reliable_share',
              'completion_group_backtest_n','completion_group_backtest_weeks',
              'completion_group_backtest_mape_pct',
              'coverage_expected_required','coverage_predicted_required',
              'coverage_reliable_required','coverage_missing_grid',
              'coverage_unreliable_required','coverage_full_to_age6',
              'coverage_window_applicable','coverage_reliable_share',
              'coverage_publish_ready','coverage_publication_mode']
        return result[[c for c in cols if c in result.columns]].copy()

    @property
    def surface_df(self):
        return None if self._surface_df is None else self._surface_df.copy()

    @property
    def production_surface(self):
        """Only approved price rows, without modifying the full surface grid."""
        if self._surface_df is None:return None
        return self._surface_df.loc[self._surface_df.publication_allowed].copy()

    @property
    def model_quality_report(self):
        """One row per PARENT with OOF and recent-market diagnostics."""
        if self._surface_df is None:return pd.DataFrame()
        if self.reliability_policy is None:return pd.DataFrame()
        base=self._surface_df.groupby(PARENT,dropna=False,as_index=False).agg(
            total_surface_cells=('publication_allowed','size'),
            approved_surface_cells=('publication_allowed','sum'),
            monthly_ads=('motorcycle_n30','max'),
            model_quality_status=('quality_model_status','first'))
        q=self.reliability_policy.model_quality.copy()
        if not q.empty:base=base.merge(q,on=PARENT,how='left',validate='one_to_one')
        m=self.reliability_policy.recent_market_group
        if m is not None and not m.empty:base=base.merge(m,on=PARENT,how='left',validate='one_to_one')
        return base

    @property
    def model_index_history_data(self):
        return None if self._model_index_history_data is None else self._model_index_history_data.copy()

    @property
    def debug_surface_df(self):
        """Full feature values of most recently produced surface, if available."""
        return None if self._surface_df is None else self._surface_df.copy()
