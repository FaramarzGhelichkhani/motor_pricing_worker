"""Hierarchical price evidence and model-level publication safeguards.

Publication contract:
  * >= min_monthly_samples DISTINCT eligible ads in the preceding 30 days for
    the model identity PARENT (variant/age/mileage/condition ignored).
  * No false negative caused by missing cell-level backtest observations.
  * Block a model only for repeated, robust, independent historical OOF failures.
  * Leave reliability_tier='D' for unverified estimates, even when published.
  * Valid factory quotations bypass the used-ad sample minimum.

Completed OOF weeks only. No file I/O or model retraining.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GROUP = ['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']
CELL = GROUP + ['age','mileage_bucket','condition_tier']
PARENT = ['motorcycle_model_id','real_brand','real_model','is_copy']


def age_band(values):
    s=pd.to_numeric(values,errors='coerce').fillna(-1)
    return np.select([s.eq(0),s.eq(1),s.eq(2),s.le(4),s.le(7)],
                     ['0','1','2','3-4','5-7'],default='8+')


def km_band(values):
    s=pd.to_numeric(values,errors='coerce').fillna(-1)
    return np.select([s.eq(0),s.le(2),s.le(5)],['0','1-2','3-5'],default='6-8')


class _V8ReliabilityBase:
    """Historical evidence tier plus independent model-wide publication decision.

    `fit(cells, asof)` consumes COMPLETED future-week medians from earlier
    forecast origins. Sparse/noisy cells remain in the general Core backtest,
    but cannot themselves cause a model-level publication veto.

    `apply(grid)` requires `motorcycle_n30`, counted by Core on PARENT from
    ads dated [asof-30 days, asof) and excluding factory rows.
    """
    LEVELS = [
        ('exact',CELL,3,2,5.),
        ('group_age',GROUP+['age'],4,2,8.),
        ('group_bucket_condition',GROUP+['mileage_bucket','condition_tier'],4,2,8.),
        ('group',GROUP,5,2,10.),
        ('parent',PARENT,8,2,14.),
        ('cohort',['_age_band','_km_band','condition_tier'],15,2,18.),
    ]

    def __init__(self,min_week_diversity=2,min_c_interval_coverage_pct=28.0,
                 min_monthly_samples=7,min_target_ads=5,min_target_days=2,
                 max_target_iqr_pct=25.,model_risk_min_cells=8,
                 model_risk_min_weeks=3,model_risk_min_mdape_pct=15.,
                 model_risk_min_o10_share=0.50):
        self.min_week_diversity=int(min_week_diversity)
        self.min_c_interval_coverage_pct=float(min_c_interval_coverage_pct)
        self.min_monthly_samples=int(min_monthly_samples)
        self.min_target_ads=int(min_target_ads)
        self.min_target_days=int(min_target_days)
        self.max_target_iqr_pct=float(max_target_iqr_pct)
        self.model_risk_min_cells=int(model_risk_min_cells)
        self.model_risk_min_weeks=int(model_risk_min_weeks)
        self.model_risk_min_mdape_pct=float(model_risk_min_mdape_pct)
        self.model_risk_min_o10_share=float(model_risk_min_o10_share)
        if self.min_monthly_samples < 1: raise ValueError('min_monthly_samples >= 1 required')
        self.tables={}
        self.model_risk=pd.DataFrame()
        self.overall={'n':0,'weeks':0,'mape':10.,'coverage':45.,'floor':80.}

    @staticmethod
    def _normalize(frame):
        x=frame.copy()
        # Collapse integer, float, and nullable ID representations safely.
        mid=pd.to_numeric(x['motorcycle_model_id'],errors='coerce')
        x['motorcycle_model_id']=mid.astype('Int64').astype(str).replace('<NA>','UNKNOWN')
        ic=pd.to_numeric(x['is_copy'],errors='coerce')
        x['is_copy']=ic.astype('Int64').astype(str).replace('<NA>','UNKNOWN')
        for col in ['real_brand','real_model','model_variant']:
            x[col]=x[col].fillna('UNKNOWN').astype(str).str.strip().replace('', 'UNKNOWN')
        x['_age_band']=age_band(x['age'])
        x['_km_band']=km_band(x['mileage_bucket'])
        return x

    def fit(self,cells:pd.DataFrame,asof=None):
        self.tables={}
        self.model_risk=pd.DataFrame()
        self.overall={'n':0,'weeks':0,'mape':10.,'coverage':45.,'floor':80.}
        if cells is None or cells.empty:return self
        x=self._normalize(cells)
        x['asof']=pd.to_datetime(x['asof'],errors='coerce').dt.normalize()
        if asof is not None:
            x=x[x['asof']+pd.Timedelta(days=7)<=pd.Timestamp(asof).normalize()]
        for k in ('actual','price_center','price_low','price_high'):
            x[k]=pd.to_numeric(x[k],errors='coerce')
        # Only the risk/reliability bank uses the deduplicated future target;
        # the legacy public Backtest continues to expose its original median.
        if 'actual_independent' in x:
            independent=pd.to_numeric(x['actual_independent'],errors='coerce')
            x['_target']=independent.where(independent.gt(0),x['actual'])
        else:
            x['_target']=x['actual']
        x=x[x._target.gt(0)&x.price_center.gt(0)&x['asof'].notna()].copy()
        if x.empty:return self
        # Statistical evidence bank: require enough independent ad observations.
        support=pd.to_numeric(x.get('future_unique_ads',x.get('future_n',pd.Series(0,index=x.index))),errors='coerce').fillna(0)
        days=pd.to_numeric(x.get('future_observed_days',pd.Series(0,index=x.index)),errors='coerce').fillna(0)
        dispersion=pd.to_numeric(x.get('future_iqr_pct',pd.Series(np.nan,index=x.index)),errors='coerce')
        # Missing independence/day/dispersion evidence must NOT certify badness.
        credible=support.ge(self.min_target_ads)&days.ge(self.min_target_days)&dispersion.le(self.max_target_iqr_pct)
        x=x.loc[credible].copy()
        if x.empty:return self
        x['_signed']=100*(x.price_center/x._target-1)
        x['_ape']=x._signed.abs()
        x['_o10']=x._signed.gt(10.).astype(float)
        x['_u10']=x._signed.lt(-10.).astype(float)
        x['_covered']=x._target.between(x.price_low,x.price_high).astype(float)*100
        x['_floor']=x._target.ge(x.price_low).astype(float)*100
        self.overall={'n':len(x),'weeks':x['asof'].nunique(),
                      'mape':float(x._ape.mean()),'coverage':float(x._covered.mean()),
                      'floor':float(x._floor.mean())}
        for name,keys,_,_,_ in self.LEVELS:
            t=x.groupby(keys,dropna=False,as_index=False).agg(
                n=('_ape','size'),weeks=('asof','nunique'),
                mape=('_ape','mean'),coverage=('_covered','mean'),floor=('_floor','mean'))
            self.tables[name]=t.rename(columns={p:f'_ev_{name}_{p}' for p in ('n','weeks','mape','coverage','floor')})
        mr=x.groupby(PARENT,dropna=False,as_index=False).agg(
            model_oof_n=('_ape','size'),model_oof_weeks=('asof','nunique'),
            model_oof_mdape_pct=('_ape','median'),model_oof_mape_pct=('_ape','mean'),
            model_oof_o10_share=('_o10','mean'),model_oof_u10_share=('_u10','mean'))
        # A *repeated* high median error AND high O10/U10 rate is required.
        # One extreme wrong label cannot veto a motorcycle's entire catalogue.
        mr['model_oof_high_risk']=(
            mr.model_oof_n.ge(self.model_risk_min_cells)
            & mr.model_oof_weeks.ge(self.model_risk_min_weeks)
            & mr.model_oof_mdape_pct.ge(self.model_risk_min_mdape_pct)
            & (mr.model_oof_o10_share+mr.model_oof_u10_share).ge(self.model_risk_min_o10_share))
        self.model_risk=mr
        return self

    def apply(self,grid:pd.DataFrame,asof=None,publish_review=False):
        g=self._normalize(grid)
        if g.empty:return g.drop(columns=['_age_band','_km_band'])
        g['estimated_mape_pct']=float(self.overall['mape'])
        g['estimated_interval_coverage_pct']=float(self.overall['coverage'])
        g['estimated_floor_coverage_pct']=float(self.overall['floor'])
        g['reliability_evidence_level']='global_prior'
        g['reliability_eval_n']=0
        g['reliability_eval_weeks']=0
        for name,keys,nmin,wmin,tau in reversed(self.LEVELS):
            t=self.tables.get(name)
            if t is None or t.empty:continue
            g=g.merge(t,on=keys,how='left',validate='many_to_one')
            n=g[f'_ev_{name}_n'].fillna(0)
            weeks=g[f'_ev_{name}_weeks'].fillna(0)
            usable=n.ge(nmin)&weeks.ge(max(wmin,self.min_week_diversity))
            weight=n/(n+tau)
            for target,prop in [('estimated_mape_pct','mape'),
                                ('estimated_interval_coverage_pct','coverage'),
                                ('estimated_floor_coverage_pct','floor')]:
                direct=g[f'_ev_{name}_{prop}'].fillna(g[target])
                g[target]=np.where(usable,weight*direct+(1-weight)*g[target],g[target])
            g['reliability_evidence_level']=np.where(usable,name,g.reliability_evidence_level)
            g['reliability_eval_n']=np.where(usable,n,g.reliability_eval_n)
            g['reliability_eval_weeks']=np.where(usable,weeks,g.reliability_eval_weeks)
            g=g.drop(columns=[f'_ev_{name}_{p}' for p in ('n','weeks','mape','coverage','floor')])
        if not self.model_risk.empty:
            g=g.merge(self.model_risk,on=PARENT,how='left',validate='many_to_one')
        for col in ('model_oof_n','model_oof_weeks','model_oof_mdape_pct','model_oof_mape_pct',
                    'model_oof_o10_share','model_oof_u10_share'):
            if col not in g:g[col]=np.nan
        if 'model_oof_high_risk' not in g:g['model_oof_high_risk']=False
        g['model_oof_high_risk']=g.model_oof_high_risk.eq(True).astype(bool)
        month=pd.to_numeric(g.get('motorcycle_n30',pd.Series(0,index=g.index)),errors='coerce').fillna(0)
        g['motorcycle_n30']=month.astype(int)
        g['motorcycle_support_eligible']=month.ge(self.min_monthly_samples)
        age_recent=pd.to_numeric(g.get('n30',pd.Series(0,index=g.index)),errors='coerce').fillna(0)
        group_recent=pd.to_numeric(g.get('group_n30',pd.Series(0,index=g.index)),errors='coerce').fillna(0)
        cell_samples=pd.to_numeric(g.get('local_cell_n',pd.Series(0,index=g.index)),errors='coerce').fillna(0)
        iso=pd.to_numeric(g.get('isotonic_change_pct',pd.Series(0,index=g.index)),errors='coerce').abs().fillna(1e3)
        age0=pd.to_numeric(g.age,errors='coerce').eq(0)
        extrap=g.get('age_source',pd.Series('',index=g.index)).fillna('').astype(str).str.contains('extrapolat',case=False)
        factory=g.get('is_factory',pd.Series(False,index=g.index)).fillna(False).astype(bool)
        id_known=~g.motorcycle_model_id.isin(['UNKNOWN','nan','<NA>'])
        direct=g.reliability_evidence_level.isin(['exact','group_age','group_bucket_condition','group'])
        parent_known=g.reliability_evidence_level.eq('parent')
        history_enough=(direct&g.reliability_eval_n.ge(4))|(parent_known&g.reliability_eval_n.ge(8))
        grade_a=(~age0&~extrap&id_known&cell_samples.ge(2)&age_recent.ge(7)&group_recent.ge(12)
                 &iso.le(3)&history_enough&g.reliability_eval_n.ge(6)
                 &g.reliability_eval_weeks.ge(2)&g.estimated_mape_pct.le(8)
                 &g.estimated_interval_coverage_pct.ge(42))
        grade_b=(~age0&~extrap&id_known&age_recent.ge(3)&group_recent.ge(7)
                 &iso.le(3)&history_enough&g.estimated_mape_pct.le(10)
                 &g.estimated_interval_coverage_pct.ge(35))
        grade_c=(~age0&id_known&age_recent.ge(1)&group_recent.ge(5)
                 &iso.le(5)&g.reliability_eval_n.ge(3)
                 &g.estimated_mape_pct.le(12)
                 &g.estimated_interval_coverage_pct.ge(self.min_c_interval_coverage_pct))
        g['reliability_tier']=np.select([factory,grade_a,grade_b,grade_c],
                                        ['FACTORY','A','B','C'],default='D')
        g['prediction_validated']=g.reliability_tier.isin(['A','B'])
        prices_ok=(pd.to_numeric(g.price_center,errors='coerce').gt(0)
                   &np.isfinite(pd.to_numeric(g.price_center,errors='coerce'))
                   &pd.to_numeric(g.price_low,errors='coerce').gt(0)
                   &pd.to_numeric(g.price_high,errors='coerce').ge(pd.to_numeric(g.price_center,errors='coerce'))
                   &pd.to_numeric(g.price_center,errors='coerce').ge(pd.to_numeric(g.price_low,errors='coerce')))
        invalid=~prices_ok|~id_known
        block_model=~factory&g.model_oof_high_risk
        low_support=~factory&~g.motorcycle_support_eligible
        allowed=(~invalid)&(~block_model)&(~low_support)
        g['publication_allowed']=allowed.astype(bool)
        # database.py's existing `publish_only_reliable_db` reads is_reliable.
        # Keep it as a COMPATIBILITY publication gate; statistical validation
        # is separately exposed as prediction_validated / reliability_tier.
        g['is_reliable']=g.publication_allowed
        g['publish_state']=np.select([
            invalid,factory,low_support,block_model,g.prediction_validated,iso.gt(5)],
            ['BLOCK_INVALID_ID_OR_PRICE','FACTORY_ANCHORED','BLOCK_LOW_MONTHLY_SUPPORT',
             'BLOCK_REPEATED_OOF_MODEL_ERROR','PUBLISH_VALIDATED','PUBLISH_ESTIMATE_ISOTONIC_REVIEW'],
            default='PUBLISH_MODEL_SUPPORTED_ESTIMATE')
        g['prediction_quality_status']=np.select([
            invalid,block_model,g.prediction_validated],
            ['INVALID','REPEATED_HIGH_ERROR','VALIDATED'],default='ESTIMATED_UNVERIFIED')
        g['reliability_reason']=np.select([
            invalid,factory,low_support,block_model,iso.gt(5),g.prediction_validated],
            ['invalid_identity_or_price','factory_anchored','monthly_model_support_below_minimum',
             'repeated_independent_model_oof_error','large_isotonic_review_not_proven_wrong',
             'historically_validated'],default='model_supported_no_local_test_required')
        for metric in ('estimated_mape_pct','estimated_interval_coverage_pct','estimated_floor_coverage_pct'):
            g.loc[factory,metric]=np.nan
        return g.drop(columns=['_age_band','_km_band'])


# ===========================================================================
# V9.1: quality evidence and production approval, never a generation filter.
# ===========================================================================
class _V9ReliabilityBase(_V8ReliabilityBase):
    """Two-evidence reliability engine: causal OOF + recent market sanity.

    - OOF (completed, independent targets): may VALIDATE prediction quality.
    - Recent in-sample market matching: may FLAG drift/disagreement, NEVER validate.
    - The group identity for model-wide validation is PARENT; no age or
      variant split in the model-level quality gate.
    - Exact cell does NOT require OOF observations when the PARENT passes.
    - Every generated surface row is retained; publication_allowed is distinct.
    """

    def __init__(self, *args,
                 quality_min_cells=8, quality_min_weeks=3, quality_min_ages=2,
                 quality_max_mdape_pct=8.0, quality_max_mape_pct=10.0,
                 quality_max_abs_bias_pct=7.0, quality_max_o10_share=.25,
                 quality_max_u10_share=.25, quality_min_coverage_pct=50.0,
                 quality_min_good_week_share=.75,
                 market_days=30, market_cell_min_ads=3, market_cell_min_days=2,
                 market_cell_max_iqr_pct=25.0,
                 market_group_min_stable_cells=5,
                 market_group_max_abs_bias_pct=8.0,
                 market_local_min_ads=5, market_local_min_days=3,
                 market_local_max_iqr_pct=20., market_local_max_abs_bias_pct=15.,
                 max_cell_isotonic_change_pct=5.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.quality_min_cells=int(quality_min_cells)
        self.quality_min_weeks=int(quality_min_weeks)
        self.quality_min_ages=int(quality_min_ages)
        self.quality_max_mdape_pct=float(quality_max_mdape_pct)
        self.quality_max_mape_pct=float(quality_max_mape_pct)
        self.quality_max_abs_bias_pct=float(quality_max_abs_bias_pct)
        self.quality_max_o10_share=float(quality_max_o10_share)
        self.quality_max_u10_share=float(quality_max_u10_share)
        self.quality_min_coverage_pct=float(quality_min_coverage_pct)
        self.quality_min_good_week_share=float(quality_min_good_week_share)
        self.market_days=int(market_days)
        self.market_cell_min_ads=int(market_cell_min_ads)
        self.market_cell_min_days=int(market_cell_min_days)
        self.market_cell_max_iqr_pct=float(market_cell_max_iqr_pct)
        self.market_group_min_stable_cells=int(market_group_min_stable_cells)
        self.market_group_max_abs_bias_pct=float(market_group_max_abs_bias_pct)
        self.market_local_min_ads=int(market_local_min_ads)
        self.market_local_min_days=int(market_local_min_days)
        self.market_local_max_iqr_pct=float(market_local_max_iqr_pct)
        self.market_local_max_abs_bias_pct=float(market_local_max_abs_bias_pct)
        self.max_cell_isotonic_change_pct=float(max_cell_isotonic_change_pct)
        self.model_quality=pd.DataFrame()
        self.recent_market_group=pd.DataFrame()
        self.recent_market_cell=pd.DataFrame()

    @staticmethod
    def _finite_metric(frame, field):
        return pd.to_numeric(frame[field],errors='coerce').replace([np.inf,-np.inf],np.nan)

    def fit(self, cells: pd.DataFrame, asof=None):
        super().fit(cells,asof=asof)
        self.model_quality=pd.DataFrame()
        self.recent_market_group=pd.DataFrame()
        self.recent_market_cell=pd.DataFrame()
        if cells is None or cells.empty:return self
        x=self._normalize(cells)
        x['asof']=pd.to_datetime(x['asof'],errors='coerce').dt.normalize()
        if asof is not None:
            x=x.loc[x['asof'].add(pd.Timedelta(days=7)).le(pd.Timestamp(asof).normalize())].copy()
        for col in ('actual','price_center','price_raw','price_unshifted_monotonic','price_low','price_high',
                    'future_unique_ads','future_n','future_observed_days','future_iqr_pct','isotonic_change_pct'):
            if col not in x:x[col]=np.nan
            x[col]=self._finite_metric(x,col)
        if 'actual_independent' in x:
            x['target']=self._finite_metric(x,'actual_independent')
            # Missing independent median does not become accepted by falling back.
        else:
            x['target']=x['actual']
        credible=(x['asof'].notna()&x.target.gt(0)&x.price_center.gt(0)&
                  x.future_unique_ads.ge(self.min_target_ads)&
                  x.future_observed_days.ge(self.min_target_days)&
                  x.future_iqr_pct.le(self.max_target_iqr_pct))
        z=x.loc[credible].copy()
        if z.empty:return self
        z['signed_pct']=100*(z.price_center/z.target-1)
        z['ape_pct']=z.signed_pct.abs()
        # Gate on capped error to prevent ONE wrongly-labelled future cell
        # from turning a generally accurate model into a false rejection.
        # Retain uncapped MAPE for transparent diagnostics.
        z['robust_ape_pct']=z.ape_pct.clip(upper=25.0)
        z['over10']=z.signed_pct.gt(10).astype(float)
        z['under10']=z.signed_pct.lt(-10).astype(float)
        z['covered_pct']=z.target.between(z.price_low,z.price_high).astype(float)*100
        z['floor_pct']=z.target.ge(z.price_low).astype(float)*100
        z['raw_ape_pct']=100*(z.price_raw/z.target-1).abs()
        z['before_shift_ape_pct']=100*(z.price_unshifted_monotonic/z.target-1).abs()
        z['shift_damage_pct']=z.ape_pct-z.before_shift_ape_pct
        z['isotonic_damage_pct']=z.before_shift_ape_pct-z.raw_ape_pct
        z['abs_iso_pct']=z.isotonic_change_pct.abs()
        # Per-week medians avoid a dominant, high-volume week masquerading as
        # repeated independent evidence.
        weekly=(z.groupby(PARENT+['asof'],dropna=False,as_index=False).agg(
            wk_signed_median_pct=('signed_pct','median'),wk_n=('ape_pct','size')))
        weekly['wk_good']=weekly.wk_signed_median_pct.abs().le(8.)
        weekly['wk_bad']=weekly.wk_signed_median_pct.abs().ge(10.)
        w=weekly.groupby(PARENT,dropna=False,as_index=False).agg(
            model_week_good_share=('wk_good','mean'),
            model_week_bad_count=('wk_bad','sum'),
            model_week_median_abs_bias_pct=('wk_signed_median_pct',lambda v:float(v.abs().median())),
            model_week_max_abs_bias_pct=('wk_signed_median_pct',lambda v:float(v.abs().max())))
        m=z.groupby(PARENT,dropna=False,as_index=False).agg(
            quality_oof_n=('ape_pct','size'),quality_oof_weeks=('asof','nunique'),
            quality_oof_ages=('age','nunique'),
            quality_oof_mdape_pct=('ape_pct','median'),quality_oof_mape_pct=('ape_pct','mean'),
            quality_oof_robust_mape_pct=('robust_ape_pct','mean'),
            quality_oof_signed_median_pct=('signed_pct','median'),
            quality_oof_o10_share=('over10','mean'),quality_oof_u10_share=('under10','mean'),
            quality_oof_coverage_pct=('covered_pct','mean'),
            quality_oof_floor_pct=('floor_pct','mean'),
            quality_oof_shift_mape_delta_pct=('shift_damage_pct','mean'),
            quality_oof_iso_mape_delta_pct=('isotonic_damage_pct','mean'),
            quality_oof_iso_p90_pct=('abs_iso_pct',lambda v:float(v.quantile(.90))))
        m=m.merge(w,on=PARENT,how='left',validate='one_to_one')
        enough=(m.quality_oof_n.ge(self.quality_min_cells)
                &m.quality_oof_weeks.ge(self.quality_min_weeks)
                &m.quality_oof_ages.ge(self.quality_min_ages))
        good=(m.quality_oof_mdape_pct.le(self.quality_max_mdape_pct)
              &m.quality_oof_robust_mape_pct.le(self.quality_max_mape_pct)
              &m.quality_oof_signed_median_pct.abs().le(self.quality_max_abs_bias_pct)
              &m.quality_oof_o10_share.le(self.quality_max_o10_share)
              &m.quality_oof_u10_share.le(self.quality_max_u10_share)
              &m.quality_oof_coverage_pct.ge(self.quality_min_coverage_pct)
              &m.model_week_good_share.ge(self.quality_min_good_week_share))
        # Fail requires both repeated week-level errors and substantial breadth
        # of large errors.  A single anomalous future target never suffices.
        fail=(enough&m.quality_oof_mdape_pct.ge(12.)
              &(m.quality_oof_o10_share+m.quality_oof_u10_share).ge(.45)
              &m.model_week_bad_count.ge(2))
        m['quality_model_status']=np.select([fail,enough&good,enough],
             ['FAIL_REPEATED','PASS','REVIEW_MIXED'],default='INSUFFICIENT_OOF')
        m['quality_model_pass']=m.quality_model_status.eq('PASS')
        self.model_quality=m
        return self

    def fit_market_alignment(self, grid:pd.DataFrame, ads:pd.DataFrame, asof):
        """Use recent training ads as a SAFETY MONITOR, never as predictive validation.

        All matching respects exact variant, age, mileage, and condition.  This
        check is performed after all used prices (including isotonic and shift)
        have been generated and before the DB publication flag is assigned.
        """
        self.recent_market_group=pd.DataFrame()
        self.recent_market_cell=pd.DataFrame()
        if grid is None or grid.empty or ads is None or ads.empty:return self
        d=pd.Timestamp(asof).normalize()
        s=self._normalize(grid)
        s=s[~s.get('is_factory',pd.Series(False,index=s.index)).fillna(False).astype(bool)]
        s=s.drop_duplicates(CELL)
        a=ads.copy()
        a['date']=pd.to_datetime(a['date'],errors='coerce').dt.normalize()
        a=a[a.date.ge(d-pd.Timedelta(days=self.market_days))&a.date.lt(d)].copy()
        if a.empty:return self
        if 'copy_label_suspicious' in a:
            a=a[pd.to_numeric(a.copy_label_suspicious,errors='coerce').fillna(0).eq(0)]
        if 'token' in a: a=a.drop_duplicates('token')
        elif 'url' in a:a=a.drop_duplicates('url')
        a=self._normalize(a)
        a['price']=pd.to_numeric(a['price'],errors='coerce')
        a=a[a.price.gt(0)].copy()
        if a.empty:return self
        a=a.merge(s[CELL+['price_center']],on=CELL,how='inner',validate='many_to_one')
        if a.empty:return self
        t=a.groupby(CELL,dropna=False,as_index=False).agg(
            market_cell_n=('price','size'),
            market_cell_days=('date','nunique'),
            market_cell_median=('price','median'),
            market_cell_q25=('price',lambda v:float(v.quantile(.25))),
            market_cell_q75=('price',lambda v:float(v.quantile(.75))),
            surface_cell_price=('price_center','first'))
        t['market_cell_iqr_pct']=100*(t.market_cell_q75-t.market_cell_q25)/t.market_cell_median
        t['market_cell_signed_pct']=100*(t.surface_cell_price/t.market_cell_median-1)
        t['market_cell_stable']=(t.market_cell_n.ge(self.market_cell_min_ads)
                                 &t.market_cell_days.ge(self.market_cell_min_days)
                                 &t.market_cell_iqr_pct.le(self.market_cell_max_iqr_pct))
        t['market_cell_mismatch_review']=(t.market_cell_n.ge(self.market_local_min_ads)
                                 &t.market_cell_days.ge(self.market_local_min_days)
                                 &t.market_cell_iqr_pct.le(self.market_local_max_iqr_pct)
                                 &t.market_cell_signed_pct.abs().gt(self.market_local_max_abs_bias_pct))
        self.recent_market_cell=t[CELL+['market_cell_n','market_cell_days','market_cell_iqr_pct',
                             'market_cell_signed_pct','market_cell_stable','market_cell_mismatch_review']]
        good=t.loc[t.market_cell_stable].copy()
        if not good.empty:
            h=good.groupby(PARENT,dropna=False,as_index=False).agg(
                market_n_stable_cells=('market_cell_signed_pct','size'),
                market_median_signed_pct=('market_cell_signed_pct','median'),
                market_median_ape_pct=('market_cell_signed_pct',lambda v:float(v.abs().median())),
                market_o10_share=('market_cell_signed_pct',lambda v:float(v.gt(10).mean())),
                market_u10_share=('market_cell_signed_pct',lambda v:float(v.lt(-10).mean())))
            h['market_model_drift_review']=(h.market_n_stable_cells.ge(self.market_group_min_stable_cells)
                  &h.market_median_signed_pct.abs().gt(self.market_group_max_abs_bias_pct))
            self.recent_market_group=h
        return self

    def apply(self,grid:pd.DataFrame,asof=None,publish_review=False):
        g=super().apply(grid,asof=asof,publish_review=publish_review)
        if g.empty:return g
        if not self.model_quality.empty:
            g=g.merge(self.model_quality,on=PARENT,how='left',validate='many_to_one')
        if not self.recent_market_group.empty:
            g=g.merge(self.recent_market_group,on=PARENT,how='left',validate='many_to_one')
        if not self.recent_market_cell.empty:
            g=g.merge(self.recent_market_cell,on=CELL,how='left',validate='many_to_one')
        for col in ('quality_oof_n','quality_oof_weeks','quality_oof_ages','quality_oof_mdape_pct',
                    'quality_oof_mape_pct','quality_oof_robust_mape_pct','quality_oof_signed_median_pct',
                    'quality_oof_o10_share','quality_oof_u10_share',
                    'quality_oof_coverage_pct','quality_oof_floor_pct',
                    'quality_oof_shift_mape_delta_pct','quality_oof_iso_mape_delta_pct',
                    'quality_oof_iso_p90_pct','model_week_good_share','model_week_bad_count',
                    'model_week_median_abs_bias_pct','model_week_max_abs_bias_pct',
                    'market_n_stable_cells','market_median_signed_pct','market_median_ape_pct',
                    'market_o10_share','market_u10_share',
                    'market_cell_n','market_cell_days','market_cell_iqr_pct','market_cell_signed_pct'):
            if col not in g:g[col]=np.nan
        if 'quality_model_status' not in g:g['quality_model_status']='INSUFFICIENT_OOF'
        g['quality_model_status']=g.quality_model_status.fillna('INSUFFICIENT_OOF')
        if 'quality_model_pass' not in g:g['quality_model_pass']=False
        g['quality_model_pass']=g.quality_model_pass.eq(True).astype(bool)
        for c in ('market_model_drift_review','market_cell_mismatch_review','market_cell_stable'):
            if c not in g:g[c]=False
            g[c]=g[c].eq(True).astype(bool)
        factory=g.get('is_factory',pd.Series(False,index=g.index)).fillna(False).astype(bool)
        approved_support=g.motorcycle_support_eligible.fillna(False).astype(bool)
        id_good=~g.motorcycle_model_id.isin(['UNKNOWN','<NA>','nan'])
        price=pd.to_numeric(g.price_center,errors='coerce')
        lo=pd.to_numeric(g.price_low,errors='coerce')
        hi=pd.to_numeric(g.price_high,errors='coerce')
        invalid=(~id_good|~np.isfinite(price)|price.le(0)|~np.isfinite(lo)|lo.le(0)
                 |~np.isfinite(hi)|hi.lt(price)|price.lt(lo)|hi.div(lo).gt(1.10+1e-8))
        iso=pd.to_numeric(g.get('isotonic_change_pct',pd.Series(np.nan,index=g.index)),errors='coerce')
        extreme_iso=iso.abs().gt(self.max_cell_isotonic_change_pct)|(~np.isfinite(iso))
        # Repeated model OOF qualification is the prerequisite for publishing
        # used prices. Recent exact-cell samples can veto an implausible number,
        # but cannot certify it (they may have been used for training).
        valid=(~invalid)&(approved_support)&g.quality_model_pass&(~g.market_model_drift_review)
        valid=valid&(~g.market_cell_mismatch_review)&(~extreme_iso)
        g['publication_allowed']=np.where(factory,~invalid,valid).astype(bool)
        g['is_reliable']=g.publication_allowed.copy()  # DB integration contract
        g['prediction_validated']=np.where(factory,False,g.quality_model_pass)
        g['production_quality_state']=np.select([
            invalid,factory,~approved_support,g.quality_model_status.eq('FAIL_REPEATED'),
            ~g.quality_model_pass,g.market_model_drift_review,g.market_cell_mismatch_review,
            extreme_iso],
            ['BLOCK_INVALID','FACTORY_CONFIRMED','HOLD_MONTHLY_SUPPORT','BLOCK_REPEATED_OOF_ERROR',
             'HOLD_NOT_OOF_VALIDATED','HOLD_MARKET_DRIFT','HOLD_CELL_MARKET_DISAGREEMENT',
             'HOLD_ISOTONIC_REVIEW'], default='APPROVED')
        g['publish_state']=g.production_quality_state
        g['prediction_quality_status']=np.select([
            invalid,g.quality_model_status.eq('FAIL_REPEATED'),g.quality_model_pass],
            ['INVALID','REPEATED_HIGH_ERROR','OOF_VALIDATED'],default='NOT_VALIDATED')
        g['reliability_reason']=g.production_quality_state.str.lower()
        return g

# =============================================================================
# V9.2: multi-week graded OOF evidence rather than all-or-nothing 5-ad cells.
# =============================================================================
class HierarchicalSurfaceReliability(_V9ReliabilityBase):
    """Evidence-calibrated, two-lane model publication gate.

    STRICT: historical OOF using >=5 independent target ads, broad week/age
        diversity and the standard predictive-quality thresholds.
    EXPANDED_3: at least 5 >=3-ad future cells over 2 completed weeks,
        plus recent exact-cell market corroboration and tighter median bias.
    EXPANDED_2: at least 8 independent >=2-ad future cells over 3 completed
        weeks, including >=3 >=3-ad cells, plus the same market corroboration.

    Expanded approval is explicitly *not* labeled prediction_validated. The
    market check is in-sample: it can corroborate, but never alone approve.
    Existing repeated-error, malformed identity, interval, local disagreement,
    and isotonic blocking rules are preserved. All full-grid rows remain.
    """

    def __init__(self, *args, enable_expanded_approval=True,
                 enable_two_ad_lane=False, expanded_market_min_cells=3,
                 expanded_market_min_matched_ads=15,
                 expanded_market_max_median_ape_pct=8.0,
                 expanded_market_max_abs_signed_bias_pct=8.0,
                 expanded3_min_cells=6, expanded3_min_weeks=3,
                 expanded2_min_cells=8, expanded2_min_weeks=3,
                 expanded2_min_three_ad_cells=3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_expanded_approval=bool(enable_expanded_approval)
        self.enable_two_ad_lane=bool(enable_two_ad_lane)
        self.expanded_market_min_cells=int(expanded_market_min_cells)
        self.expanded_market_min_matched_ads=int(expanded_market_min_matched_ads)
        self.expanded_market_max_median_ape_pct=float(expanded_market_max_median_ape_pct)
        self.expanded_market_max_abs_signed_bias_pct=float(expanded_market_max_abs_signed_bias_pct)
        self.expanded3_min_cells=int(expanded3_min_cells)
        self.expanded3_min_weeks=int(expanded3_min_weeks)
        self.expanded2_min_cells=int(expanded2_min_cells)
        self.expanded2_min_weeks=int(expanded2_min_weeks)
        self.expanded2_min_three_ad_cells=int(expanded2_min_three_ad_cells)
        self.expanded_metrics=pd.DataFrame()

    def _expanded_oof_summary(self, cells, asof, min_ads):
        """Build model-level evidence from distinct completed OOF *cells*.

        No OOF target based on fewer than two independently observed ads may
        participate. Even the lower-support path requires repeated weeks and
        independently stronger (>=3-ad) target cells.
        """
        if cells is None or cells.empty:return pd.DataFrame(columns=PARENT)
        x=self._normalize(cells)
        x['asof']=pd.to_datetime(x['asof'],errors='coerce').dt.normalize()
        if asof is not None:
            cutoff=pd.Timestamp(asof).normalize()
            x=x.loc[x['asof'].add(pd.Timedelta(days=7)).le(cutoff)].copy()
        for col in ('actual_independent','actual','price_center','price_low','price_high',
                    'future_unique_ads','future_observed_days','future_iqr_pct'):
            if col not in x:x[col]=np.nan
            x[col]=pd.to_numeric(x[col],errors='coerce')
        # Never silently substitute the old target when the deduped target exists.
        x['target']=x.actual_independent if 'actual_independent' in cells else x.actual
        x=x.loc[x.target.gt(0)&x.price_center.gt(0)
                &x.future_unique_ads.ge(min_ads)
                &x.future_observed_days.ge(2)
                &x.future_iqr_pct.le(25.)
                &x['asof'].notna()].copy()
        if x.empty:return pd.DataFrame(columns=PARENT)
        x['signed']=100*(x.price_center/x.target-1)
        x['ape']=x.signed.abs()
        x['robust_ape']=x.ape.clip(upper=25.)
        x['o10']=x.signed.gt(10)
        x['u10']=x.signed.lt(-10)
        x['covered']=x.target.between(x.price_low,x.price_high)
        weekly=x.groupby(PARENT+['asof'],dropna=False,as_index=False).signed.median()
        weekly['good']=weekly.signed.abs().le(8.)
        w=weekly.groupby(PARENT,dropna=False,as_index=False).agg(
            weeks=('asof','size'), good_week_share=('good','mean'))
        m=x.groupby(PARENT,dropna=False,as_index=False).agg(
            n=('ape','size'), ages=('age','nunique'),
            mdape=('ape','median'), robust_mape=('robust_ape','mean'),
            median_bias=('signed','median'), o10=('o10','mean'), u10=('u10','mean'),
            coverage=('covered','mean'),
            n_three_ad_cells=('future_unique_ads',lambda v:int(v.ge(3).sum())))
        return m.merge(w,on=PARENT,how='left',validate='one_to_one')

    def fit(self, cells: pd.DataFrame, asof=None):
        super().fit(cells,asof=asof)
        self.expanded_metrics=pd.DataFrame()
        if not self.enable_expanded_approval or cells is None or cells.empty:
            return self
        three=self._expanded_oof_summary(cells,asof,3)
        two=self._expanded_oof_summary(cells,asof,2)
        if self.model_quality.empty and three.empty and two.empty:return self
        # The conservative >=5-ad baseline is retained as highest rank.
        q=self.model_quality.copy() if not self.model_quality.empty else pd.DataFrame(columns=PARENT)
        for label,bank in [('three',three),('two',two)]:
            if not bank.empty:
                bank=bank.rename(columns={c:f'exp_{label}_{c}' for c in bank.columns if c not in PARENT})
                q=q.merge(bank,on=PARENT,how='outer',validate='one_to_one')
        if 'quality_model_status' not in q:q['quality_model_status']='INSUFFICIENT_OOF'
        q['quality_model_status']=q['quality_model_status'].astype('object').fillna('INSUFFICIENT_OOF')
        if 'quality_model_pass' not in q:q['quality_model_pass']=False
        q['quality_model_pass']=q['quality_model_pass'].eq(True).astype(bool)
        strong=q.quality_model_status.eq('PASS')
        failed=q.quality_model_status.eq('FAIL_REPEATED')
        for name in ['three','two']:
            for col in ('n','weeks','ages','mdape','robust_mape','median_bias','o10',
                        'u10','coverage','good_week_share','n_three_ad_cells'):
                k=f'exp_{name}_{col}'
                if k not in q:q[k]=np.nan
        # Requirements selected from the empirical V8 sensitivity sweep;
        # thresholds should be forward-validated with more than four folds.
        moderate=(q.exp_three_n.ge(self.expanded3_min_cells)
                  &q.exp_three_weeks.ge(self.expanded3_min_weeks)
                  &q.exp_three_ages.ge(2)
                  &q.exp_three_mdape.le(8.)
                  &q.exp_three_robust_mape.le(10.)
                  &q.exp_three_median_bias.abs().le(7.)
                  &q.exp_three_o10.le(.25)&q.exp_three_u10.le(.25)
                  &q.exp_three_coverage.ge(.35)
                  &q.exp_three_good_week_share.ge(2/3))
        multi=(self.enable_two_ad_lane &q.exp_two_n.ge(self.expanded2_min_cells)
               &q.exp_two_weeks.ge(self.expanded2_min_weeks)
               &q.exp_two_ages.ge(2)
               &q.exp_two_n_three_ad_cells.ge(self.expanded2_min_three_ad_cells)
               &q.exp_two_mdape.le(7.)
               &q.exp_two_robust_mape.le(9.)
               &q.exp_two_median_bias.abs().le(5.)
               &q.exp_two_o10.le(.20)&q.exp_two_u10.le(.20)
               &q.exp_two_coverage.ge(.35)
               &q.exp_two_good_week_share.ge(2/3))
        q['quality_expansion_lane']=np.select(
            [strong,~failed&moderate,~failed&multi],
            ['STRICT_OOF','EXPANDED_3_AD','EXPANDED_MULTI_WEEK'],default='NONE')
        upgrade=~strong&~failed&q.quality_expansion_lane.isin(['EXPANDED_3_AD','EXPANDED_MULTI_WEEK'])
        q.loc[upgrade,'quality_model_status']='PASS_CONDITIONAL_OOF'
        # Important: the strict OOF validation marker is NOT changed. An
        # expanded candidate still needs independent market corroboration.
        q['quality_model_pass']=strong
        self.model_quality=q
        self.expanded_metrics=q[PARENT+['quality_expansion_lane']].copy()
        return self

    def apply(self, grid:pd.DataFrame, asof=None, publish_review=False):
        g=super().apply(grid,asof=asof,publish_review=publish_review)
        if g.empty:return g
        if 'quality_expansion_lane' not in g:g['quality_expansion_lane']='NONE'
        g['quality_expansion_lane']=g.quality_expansion_lane.fillna('NONE')
        if not self.enable_expanded_approval:
            g['publication_evidence_tier']=np.where(g.production_quality_state.eq('APPROVED'),
                'STRICT_OOF',np.where(g.get('is_factory',False),'FACTORY','NONE'))
            return g
        # Recent sample compatibility: matched ads cover actual CELLs and are
        # deduplicated by token/url in fit_market_alignment (not by price error).
        if not self.recent_market_cell.empty:
            r=self.recent_market_cell.copy()
            a=r.groupby(PARENT,dropna=False,as_index=False).agg(
                quality_market_matched_ads=('market_cell_n','sum'))
            g=g.merge(a,on=PARENT,how='left',validate='many_to_one')
        else:
            g['quality_market_matched_ads']=0
        if 'quality_market_matched_ads' not in g:g['quality_market_matched_ads']=0
        sufficient_market=(g.quality_market_matched_ads.fillna(0).ge(self.expanded_market_min_matched_ads)
            &pd.to_numeric(g.market_n_stable_cells,errors='coerce').fillna(0).ge(self.expanded_market_min_cells)
            &pd.to_numeric(g.market_median_ape_pct,errors='coerce').le(self.expanded_market_max_median_ape_pct)
            &pd.to_numeric(g.market_median_signed_pct,errors='coerce').abs().le(self.expanded_market_max_abs_signed_bias_pct))
        can_reopen=(g.production_quality_state.eq('HOLD_NOT_OOF_VALIDATED')
            &g.quality_expansion_lane.isin(['EXPANDED_3_AD','EXPANDED_MULTI_WEEK'])
            &g.motorcycle_support_eligible.fillna(False)
            &~g.model_oof_high_risk.fillna(False)
            &~g.quality_model_status.eq('FAIL_REPEATED')
            &~g.market_model_drift_review.fillna(False)
            &~g.market_cell_mismatch_review.fillna(False)
            &pd.to_numeric(g.isotonic_change_pct,errors='coerce').abs().le(self.max_cell_isotonic_change_pct)
            &sufficient_market)
        g.loc[can_reopen,'publication_allowed']=True
        g.loc[can_reopen,'is_reliable']=True
        g.loc[can_reopen,'production_quality_state']=np.where(
            g.loc[can_reopen,'quality_expansion_lane'].eq('EXPANDED_3_AD'),
            'APPROVED_OOF_MODERATE','APPROVED_OOF_MULTI_CELL')
        g['publication_evidence_tier']=np.select([
            g.get('is_factory',pd.Series(False,index=g.index)).fillna(False).astype(bool)
               &g.publication_allowed,
            g.production_quality_state.eq('APPROVED'),
            g.production_quality_state.eq('APPROVED_OOF_MODERATE'),
            g.production_quality_state.eq('APPROVED_OOF_MULTI_CELL')],
            ['FACTORY','STRICT_OOF','MODERATE_OOF','MULTI_CELL_OOF'],default='NOT_APPROVED')
        g['publish_state']=g.production_quality_state
        g['reliability_reason']=g.production_quality_state.str.lower()
        # D remains D; evidence tier is not a synthetic A/B/C upgrade.
        return g
