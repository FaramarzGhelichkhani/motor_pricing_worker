"""Fail-closed chronological OOF release audit.

Only historical fold publication decisions determine pass/fail. Auditing
today's candidates against historical labels is explicitly diagnostic only.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
CURVE = ['motorcycle_model_id','real_brand','real_model','is_copy',
         'model_variant','age','condition_tier']

class ReleaseRiskError(RuntimeError):
    def __init__(self,message,report):
        self.report=report
        super().__init__(f'{message}: {report}')

def _normalize_keys(frame):
    d=frame.copy()
    d['motorcycle_model_id']=(pd.to_numeric(d.motorcycle_model_id,errors='coerce')
                              .astype('Int64').astype(str))
    d['is_copy']=pd.to_numeric(d.is_copy,errors='coerce').fillna(0).astype(int)
    for col in ['real_brand','real_model','model_variant']:
        d[col]=d[col].fillna('').astype(str)
    for col in ['age','condition_tier']:
        d[col]=pd.to_numeric(d[col],errors='coerce')
    return d

def _risk_metrics(d, *, repeats, seed=1241):
    if d.empty:
        return {'available':False,'n':0,'models':0,'weeks':0}
    d=d.copy()
    error=100.0*(d.price_center/d.actual_independent-1.0)
    d['bad15']=error.abs().gt(15).astype(int)
    d['bad20']=error.abs().gt(20).astype(int)
    cluster=(d.groupby(['motorcycle_model_id','is_copy'],dropna=False)
             .agg(n=('bad15','size'),k15=('bad15','sum'),k20=('bad20','sum')))
    matrix=cluster[['n','k15','k20']].to_numpy(dtype=float)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(matrix),size=(repeats,len(matrix)))
    totals=matrix[draws].sum(axis=1)
    upper15=100.0*float(np.quantile(totals[:,1]/totals[:,0],.95))
    upper20=100.0*float(np.quantile(totals[:,2]/totals[:,0],.95))
    return dict(available=True,n=int(len(d)),models=int(len(cluster)),
                weeks=int(d['asof'].nunique()),
                mape_pct=round(float(error.abs().mean()),3),
                over15=int((error>15).sum()),under15=int((error<-15).sum()),
                bad15=int(d.bad15.sum()),bad20=int(d.bad20.sum()),
                tail15_pct=round(float(d.bad15.mean()*100),3),
                tail20_pct=round(float(d.bad20.mean()*100),3),
                upper95_model_bootstrap_tail15_pct=round(upper15,3),
                upper95_model_bootstrap_tail20_pct=round(upper20,3),
                upper95_model_bootstrap_tail15_pct_raw=upper15,
                upper95_model_bootstrap_tail20_pct_raw=upper20)

def validate_recent_oof_release(builder, *, max_tail15_upper_pct=5.0,
                                max_tail20_upper_pct=4.0, min_targets=100,
                                min_motorcycles=20, repeats=2000,
                                weeks=2, sensitivity_weeks=(4,6)):
    b=getattr(builder,'backtest_cells',None)
    needed={'asof','motorcycle_model_id','is_copy','fold_publication_allowed',
            'price_center','actual_independent','future_unique_ads',
            'future_observed_days','future_iqr_pct'}
    if b is None or b.empty or needed-set(b):
        raise ReleaseRiskError('Missing chronological OOF evidence',{'available':False})
    d=b.copy()
    d['asof']=pd.to_datetime(d['asof'],errors='coerce').dt.normalize()
    all_dates=sorted(d['asof'].dropna().unique())
    if not all_dates:
        raise ReleaseRiskError('No valid OOF origins',{'available':False})
    for col in ['price_center','actual_independent','future_unique_ads',
                'future_observed_days','future_iqr_pct']:
        d[col]=pd.to_numeric(d[col],errors='coerce')
    quality=(d.future_unique_ads.ge(3) & d.future_observed_days.ge(2) &
             d.future_iqr_pct.between(0,20) &
             np.isfinite(d.price_center) & d.price_center.gt(0) &
             np.isfinite(d.actual_independent) & d.actual_independent.gt(0))
    d=d.loc[quality].copy()
    d['fold_publication_allowed']=d.fold_publication_allowed.fillna(False).astype(bool)
    history=d.loc[d.fold_publication_allowed & d['asof'].isin(all_dates[-weeks:])].copy()
    primary=_risk_metrics(history,repeats=repeats)
    report={'historical_policy_risk':primary,
            'current_candidate_risk_audit':{'available':False,'reason':'no_current_surface'},
            'sensitivity_by_weeks':{},'top_historical_errors':[],
            'thresholds':{'upper15_pct':max_tail15_upper_pct,
                          'upper20_pct':max_tail20_upper_pct,'min_targets':min_targets,
                          'min_models':min_motorcycles}}
    for window in sensitivity_weeks:
        dates=all_dates[-int(window):]
        v=d.loc[d.fold_publication_allowed & d['asof'].isin(dates)]
        report['sensitivity_by_weeks'][str(window)]=_risk_metrics(v,repeats=repeats)
        report['sensitivity_by_weeks'][str(window)]['available_origins']=len(dates)

    current=getattr(builder,'_surface_df',None)
    if isinstance(current,pd.DataFrame) and not current.empty and (
        set(CURVE+['publication_allowed']).issubset(current) and
        set(CURVE).issubset(d)):
        allowed=_normalize_keys(current.loc[
            current.publication_allowed.fillna(False).astype(bool),CURVE])
        allowed=allowed.drop_duplicates(CURVE)
        historic=_normalize_keys(history)
        selected=historic.merge(allowed.assign(_current_allowed=True),
                                on=CURVE,how='inner',validate='many_to_one')
        audit=_risk_metrics(selected,repeats=repeats)
        audit['selection_bias_warning']='Diagnostic only: candidate selection may use past outcomes'
        report['current_candidate_risk_audit']=audit

    if len(history):
        error=100*(history.price_center/history.actual_independent-1)
        severe=history.loc[error.abs().gt(20)].copy()
        if len(severe):
            severe['signed_error_pct']=error.loc[severe.index].round(3)
            keep=[c for c in ['asof','motorcycle_model_id','real_brand',
                  'real_model','model_variant','age','condition_tier',
                  'mileage_bucket','signed_error_pct'] if c in severe]
            severe['asof']=severe['asof'].dt.strftime('%Y-%m-%d')
            severe=severe.reindex(severe.signed_error_pct.abs().sort_values(
                ascending=False).index)
            report['top_historical_errors']=severe[keep].head(20).to_dict('records')

    logger.info('Causal OOF release audit: %s',report)
    if (primary['n']<min_targets or primary['models']<min_motorcycles or
        primary['weeks']<weeks):
        raise ReleaseRiskError('Insufficient release validation',report)
    if (primary['upper95_model_bootstrap_tail15_pct_raw']>max_tail15_upper_pct or
        primary['upper95_model_bootstrap_tail20_pct_raw']>max_tail20_upper_pct):
        raise ReleaseRiskError('Release blocked: chronological policy risk too high',report)
    return report
