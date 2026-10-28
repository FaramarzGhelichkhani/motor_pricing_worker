"""V10: Stable family-price reference, bounded model-year extrapolation, and safe Persian text evidence.

Contract
--------
* ``is_original`` is an IMMUTABLE database field about the standardized brand.
* ``is_copy`` describes the ADVERTISED vehicle, which may disagree with the brand.
* Share price evidence among real_model family members ACROSS real_brand/model ID.
* Price bands are fit by production year and publication month; mileage/condition
  are NOT independently modeled by this clustering step. No hard-coded prices.
* Never flip a row with contradictory text, weak reference market, unreliable price,
  or conflicting strong signals. Uncertain rows retain their initial is_copy.
* Price audit columns are not price-model features. Run only after raw DB merge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

try:
    from dictionary.mappings import BRAND_NAME_MAP, ORIGINAL_BRANDS, IRANIAN_BRANDS
except ImportError:
    BRAND_NAME_MAP, ORIGINAL_BRANDS, IRANIAN_BRANDS = {}, (), ()


import numpy as np
import pandas as pd

# Names derived from original Iranian assembler list and frequently-observed raw aliases.
# They are WEAK evidence of a locally manufactured/assembled vehicle, NOT proof
# that a particular advertised vehicle is a copy.
LOCAL_BRANDS = (
    'کویر','نیرو موتور','نیرو‌موتور','همتاز','دایچی','هانی','نامی','شاهین موتور',
    'جهانرو','پیشتاز','تکتاز','متین','پرواز','ثاقب','کبیر','زیگما','جترو','احسان',
    'رهرو','تیزپر','ساوین','نیکتاز','تلاش','باهر','زمرد','سپهر','پاسارگاد','برین',
    'پردیس','کیوان','سهند','سحر','آریا','داتیس','عرفان','پویان','رخش','آوا',
    'توسن','دلتا','توسان','ایران دو چرخ','سالار گستر',
)
RAW_LOCAL_ALIASES = (
    'kavir','kavir-motor','niroo','niroomotor','hani','salar gostar','salar',
    'kabir','kabir-motor','pishro','pishro','sagheb','saghebkhodro','saqeb',
    'jahan-hamta','hamtaz','hany','irandocharkh','iran doocharkh',
    'tosan','tosanmoharekeshargh','dino','delta','kavir-motor','tizpar',
    'rahro','pishtaz','daichi','shahin motor','parvaz',
    'blenta','hi click','tekno vario','ava click','crystal','nami','sagheb','tizpar','saqeb','ravan','kavir','niroo motor','irandocharkh ava','tosanmoharekeshargh','salar gostar',
)
# Distributor metadata such as "Armanmotor Jahantond" is not a maker identity.
_MOTOR = (r'(?:موتور(?:\s*سیکلت)?|اسکوتر|هوندا|هندا|کلیک|کیلیک|کیلک|واریو|'
          r'ایروکس|آیروکس|یاماها|سی\s*جی|پالسار|بنلی|vespa|pcx|adv|'
          r'click|vario|aerox|n\s*max|nmax|honda|yamaha|cg\s*\d+|'
          r'cdi\s*\d+|scooter|motocycle|motorcycle)')
_O = r'(?:اورجینال|اصلی|اصل|original|تایلندی|اندونزیایی|ژاپنی|وارداتی)'
_C = r'(?:طرح|های\s*کپی|هایکپی|high\s*copy|کپی|clone|fake|فیک)'
_O_NEAR = re.compile(rf'(?:{_MOTOR})[^،؛.!؟\n]{{0,30}}(?:{_O})|(?:{_O})[^،؛.!؟\n]{{0,26}}(?:{_MOTOR})',re.I)
_C_NEAR = re.compile(rf'(?:{_MOTOR})[^،؛.!؟\n]{{0,30}}(?:{_C})|(?:{_C})[^،؛.!؟\n]{{0,26}}(?:{_MOTOR})',re.I)
_LEAD_C = re.compile(r'^\s*(?:فروش\s+)?(?:موتور(?:\s*سیکلت)?\s+)?(?:طرح|هایکپی|های\s*کپی)\s+',re.I)
_NEG = re.compile(r'(?:طرح|کپی|اصلی|اصل|اورجینال)[^،؛.!؟\n]{0,24}(?:نیست|نبوده|نمیباشد|نمی\s*باشد|نداره)|(?:نیست|نبوده)\s*(?:طرح|کپی|اصلی)',re.I)
_PART = re.compile(r'(?:سند|مدارک|برگه|فاکتور|کارت|لوازم|قطعات|لاستیک|باتری|باتری|لنت|روغن|تسمه|فلاپ|قفل|آچار|پلاک|رنگ)\s+(?:اصل|اصلی|اورجینال|تایلندی)',re.I)
_FINANCE = re.compile(r'(?:پیش\s*پرداخت|اقساط|قسطی|قسط\s*بندی|طرح\s*(?:فروش|تعویض|اقساط|ترافیک))',re.I)


def _normal(x: pd.Series) -> pd.Series:
    s=x.fillna('').astype('string').str.lower()
    for a,b in [('ي','ی'),('ك','ک'),('ۀ','ه'),('ـ',''),('\u200c',' '),('\u200f',' '),('أ','ا'),('إ','ا')]:
        s=s.str.replace(a,b,regex=False)
    return s.str.replace(r'\s+',' ',regex=True).str.strip()


def _numeric(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return (pd.to_numeric(df[name],errors='coerce').astype(float) if name in df
            else pd.Series(default,index=df.index,dtype=float))


def _family(df: pd.DataFrame) -> pd.Series:
    if 'real_model' not in df:
        return pd.Series('',index=df.index,dtype='string')
    # Standardized real_model is trusted; strip only spelling separators, not digits.
    return _normal(df['real_model']).str.replace(r'(?<=[a-z])(?=\d)', ' ',regex=True).str.replace(r'\s+',' ',regex=True)


def _contains(s: pd.Series, p) -> pd.Series:
    return s.str.contains(p,regex=True,na=False).fillna(False).astype(bool)


def _raw_brand_hints(raw: pd.Series) -> tuple[pd.Series,pd.Series]:
    # Combine specific aliases into one compiled regex; never use generic
    # words like 'motor', 'iran', 'company'.
    aliases=sorted({p.lower().replace('\u200c',' ') for p in (LOCAL_BRANDS+RAW_LOCAL_ALIASES+tuple(IRANIAN_BRANDS)+tuple(k for k,v in BRAND_NAME_MAP.items() if v in set(IRANIAN_BRANDS)))},key=len,reverse=True)
    local=_contains(raw,'(?:'+'|'.join(map(re.escape,aliases))+')')
    # Intentionally count only high-specificity origin brand hints.
    o = _contains(raw,r'\b(?:honda|yamaha|vespa|kawasaki|suzuki|bajaj|tvs)\b|هوندا|یاماها')
    return local.astype(bool),o.astype(bool)



# ---------------------------------------------------------------------------
# V10: FAMILY-WIDE, FROZEN, UNSUPERVISED PRICE REFERENCE
# ---------------------------------------------------------------------------
import json
from pathlib import Path
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture

@dataclass(frozen=True)
class PriceClusterConfig:
    min_year_n: int = 60
    min_family_n: int = 180
    min_component_n: int = 7
    min_component_fraction: float = .015
    min_bic_gain: float = 20.0
    min_log_separation: float = .48
    min_sigma_separation: float = 2.8
    min_anchor_years: int = 2
    max_extrapolate_years: int = 5
    min_posterior: float = .94
    min_abs_log_odds: float = 2.75
    max_scoring_months_ahead: int = 1
    min_price: float = 30_000_000
    max_price: float = 15_000_000_000
    min_ad_price_confidence: float = .0
    suspicious_multiplier: float = .30


def _price_month(df: pd.DataFrame) -> pd.Series:
    if 'publish_date' in df:
        m = df['publish_date'].astype('string').str.extract(r'^(\d{4})[-/](\d{1,2})')
        y=pd.to_numeric(m[0],errors='coerce'); mo=pd.to_numeric(m[1],errors='coerce')
        return (12*y+mo).where(y.between(1300,2200)&mo.between(1,12)).astype(float)
    if 'date' in df:
        d=pd.to_datetime(df['date'],errors='coerce')
        return (12*d.dt.year+d.dt.month).astype(float)
    return pd.Series(np.nan,index=df.index,dtype=float)


def _reference_valid_rows(df:pd.DataFrame,cfg:PriceClusterConfig, *,for_fit:bool=True)->pd.Series:
    """Source restrictions apply to FIT, not to independent ad SCORING.

    A scoring batch may carry `source=divar` or an inherited source value;
    rejecting it silently used to appear as an inexplicable `unavailable`.
    Factory quotes are never scored as used-ad listings.
    """
    v=_numeric(df,'price')
    ok=v.between(cfg.min_price,cfg.max_price)
    ok &= _numeric(df,'production_year').between(1370,1600)
    ok &= _price_month(df).notna() & _family(df).ne('')
    if 'is_real_price' in df:ok &= ~_numeric(df,'is_real_price').eq(0)
    if 'flag_installment' in df:ok &= ~_numeric(df,'flag_installment').eq(1)
    if 'source' in df:
        source = _normal(df['source'])
        if for_fit:
            ok &= source.isin(['ads', 'ad', 'advertisement', 'divar', 'شیپور', 'دیوار'])
        else:
            ok &= ~source.isin(['factory', 'کارخانه'])
    return ok.fillna(False)


def _mode_candidate(y, cfg:PriceClusterConfig):
    """1-vs-2 Gaussian mixture fitted ONLY on log prices in the model-year.

    Labels, brands, maker strings and raw fields do not participate in the fit.
    """
    x=np.asarray(y,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)<cfg.min_year_n:return None
    z=x.reshape(-1,1)
    g1=GaussianMixture(1,random_state=41,reg_covar=.0025,max_iter=150).fit(z)
    g2=GaussianMixture(2,random_state=41,n_init=3,reg_covar=.0025,max_iter=150).fit(z)
    bic=float(g1.bic(z)-g2.bic(z))
    sort=np.argsort(g2.means_.ravel())
    means=g2.means_.ravel()[sort]; sd=np.sqrt(g2.covariances_.ravel()[sort]); w=g2.weights_[sort]
    separation=float((means[1]-means[0])/max(sd))
    counts=w*len(x)
    if (bic<cfg.min_bic_gain or min(counts)<cfg.min_component_n or
        min(w)<cfg.min_component_fraction or
        means[1]-means[0]<cfg.min_log_separation or
        separation<cfg.min_sigma_separation):
        return None
    return {'low':float(means[0]),'high':float(means[1]),
            'sd_low':float(sd[0]),'sd_high':float(sd[1]),
            'weight_low':float(w[0]),'weight_high':float(w[1]),
            'bic_gain':bic,'separation':separation,'n':int(len(x))}


def _fit_trend(years,vals, cap=.25):
    yrs=np.asarray(years,dtype=float); vv=np.asarray(vals,dtype=float)
    if len(yrs)<2:return 0.
    rates=[]
    for i in range(len(yrs)):
        for j in range(i+1,len(yrs)):
            if yrs[i]!=yrs[j]:rates.append((vv[j]-vv[i])/(yrs[j]-yrs[i]))
    return float(np.clip(np.median(rates),-.06,cap)) if rates else 0.


def _price_monthly_trend(f, modes):
    """Estimate inflation from repeated monthly observations WITHIN each price mode.

    Mode membership is evaluated solely by price in each confirmed model-year.
    Avoid the mix-of-original-and-copy time bias of V7's pooled 30th percentile.
    """
    slopes=[]
    for yr,m in modes.items():
        z=f[f['_year'].eq(yr)]
        p=z['_lp'].to_numpy()
        # Use low-component prices far below midpoint for a clean trend estimate.
        reliable=z[p<(.65*m['low']+.35*m['high'])]
        med=reliable.groupby('_month')['_lp'].agg(['size','median'])
        med=med[med['size']>=5]
        if len(med)<3:continue
        a=med.index.to_numpy(dtype=float);b=med['median'].to_numpy()
        slopes.extend([(b[j]-b[i])/(a[j]-a[i]) for i in range(len(a)) for j in range(i+1,len(a)) if a[j]!=a[i]])
    return float(np.clip(np.median(slopes),-.045,.045)) if slopes else 0.


def _validate_semantics(yr_frame, mode, family_col='real_brand'):
    """Post-fit guard only. Never alters Gaussian prices, thresholds, or labels.

    Different seller brands can offer genuine vehicles, so do NOT insist every
    expensive ad has an OEM brand. However the low-priced mode should span at
    least two standardized brands to avoid trim/repair-only bimodality.
    """
    z=yr_frame
    if family_col not in z:return True
    split=(mode['low']+mode['high'])/2
    a=z.loc[z['_lp']<split,'real_brand'].dropna()
    b=z.loc[z['_lp']>=split,'real_brand'].dropna()
    if len(a)<5 or len(b)<5:return False
    if a.nunique()<2:return False
    # The OEM enrichment is a downstream validation only, never a price feature.
    oems=set(ORIGINAL_BRANDS) if ORIGINAL_BRANDS else {'Honda','Yamaha','Vespa','Suzuki','Kawasaki','Bajaj','TVS','SYM'}
    o_lo=a.isin(oems).mean();o_hi=b.isin(oems).mean()
    return bool(o_hi>=.50 and (o_hi-o_lo)>=.30)


def fit_price_reference(reference_df: pd.DataFrame,
                        config:Optional[PriceClusterConfig]=None,
                        *, required_families:Optional[list[str]]=None)->dict:
    """Train ONE stable family price model from the full, unmodified historical pool.

    Excludes brands, is_copy, is_original, title and description from the fit.
    The real_brand column is used ONLY to reject implausible OEM/copy readings
    of statistically separated price modes (post-fit check).
    Fit once BEFORE filtering to a brand, production year, or minority label.
    """
    cfg=config or PriceClusterConfig()
    valid=_reference_valid_rows(reference_df,cfg)
    w=pd.DataFrame({'_fam':_family(reference_df)[valid],
                    '_year':_numeric(reference_df,'production_year')[valid].astype(int),
                    '_month':_price_month(reference_df)[valid],
                    '_lp':np.log(_numeric(reference_df,'price')[valid]),
                    'real_brand':(reference_df['real_brand'][valid] if 'real_brand' in reference_df else '')})
    models={}
    for fam,g in w.groupby('_fam',sort=True):
        if len(g)<cfg.min_family_n:continue
        # Stage 1: strong anchors at individual model-years; price-only fit.
        anchors={}
        for yr,z in g.groupby('_year',sort=True):
            md=_mode_candidate(z['_lp'].to_numpy(),cfg)
            if md is not None and _validate_semantics(z,md):anchors[int(yr)]=md
        if len(anchors)<cfg.min_anchor_years:continue
        # Stage 2: shared year-dependent ORIGINAL/COPY bands estimated from
        # confirmed anchor modes. Makes sparse years stable rather than forcing
        # two modes from 17 ads (e.g. Click150 1405).
        yrs=sorted(anchors)
        low_slope=_fit_trend(yrs,[anchors[j]['low'] for j in yrs])
        high_slope=_fit_trend(yrs,[anchors[j]['high'] for j in yrs])
        # MONTH slope is computed once on the same full family reference.
        monthly=_price_monthly_trend(g,anchors)
        latest=float(g['_month'].max())
        # Price corrections need refit anchors after time normalization.
        g=g.copy()
        g['_lp']=g['_lp']+monthly*np.clip(latest-g['_month'],0,12)
        anchors2={}
        for yr,z in g.groupby('_year',sort=True):
            md=_mode_candidate(z['_lp'].to_numpy(),cfg)
            if md is not None and _validate_semantics(z,md):anchors2[int(yr)]=md
        if len(anchors2)>=cfg.min_anchor_years:
            anchors=anchors2
            yrs=sorted(anchors)
            low_slope=_fit_trend(yrs,[anchors[j]['low'] for j in yrs])
            high_slope=_fit_trend(yrs,[anchors[j]['high'] for j in yrs])
        else:
            # If normalizing destroys robust evidence, fail closed.
            continue
        centers={}
        for yr,z in g.groupby('_year',sort=True):
            yr=int(yr)
            if yr in anchors:
                md=anchors[yr]
                centers[str(yr)]={'low':md['low'],'high':md['high'],
                    'sd_low':md['sd_low'],'sd_high':md['sd_high'],
                    'method':'year_two_modes','n':len(z)}
                continue
            nearest=min(yrs,key=lambda j:(abs(j-yr),j))
            if abs(nearest-yr)>cfg.max_extrapolate_years:continue
            donor=anchors[nearest]
            lo=donor['low']+low_slope*(yr-nearest)
            hi=donor['high']+high_slope*(yr-nearest)
            # One-mode year: adjust its own low-band center when its bulk is
            # closer to the predicted cheap band; NEVER force a second mode.
            med=float(z['_lp'].median())
            if abs(med-lo)<abs(med-hi):
                lo=float(np.clip(med,lo-.23,lo+.23))
            else:
                hi=float(np.clip(med,hi-.23,hi+.23))
            if hi-lo<cfg.min_log_separation:continue
            centers[str(yr)]={'low':lo,'high':hi,
                'sd_low':float(np.median([anchors[j]['sd_low'] for j in yrs])),
                'sd_high':float(np.median([anchors[j]['sd_high'] for j in yrs])),
                'method':'borrowed_family_bands','n':len(z), 'donor_year':int(nearest), 'donor_distance':int(abs(nearest-yr))}
        models[fam]={'ref_month':latest,'monthly_slope':monthly,
                     'anchors':[int(x) for x in yrs],
                     'year_bands':centers,'sample_count':len(g),
                     'train_min_month':float(g['_month'].min()),
                     'train_max_month':float(g['_month'].max())}
    missing=[]
    for item in required_families or []:
        key=str(_family(pd.DataFrame({'real_model':[item]})).iloc[0])
        if key not in models:
            missing.append(key)
    if missing:
        raise ValueError(
            f'V10 reference missing required families {missing}. '
            f'Fit on ALL brands and model-years before filtering. '
            f'Valid rows: {int(valid.sum())}; available families: {list(models)}. '
            f'`segment` is NOT a dependency.'
        )
    return {'schema':'copy_price_reference_v10','config':cfg.__dict__,
            'families':models,'reference_n':int(valid.sum()),
            'reference_family_counts':w['_fam'].value_counts().astype(int).to_dict()}



def save_price_reference(ref:dict,path)->None:
    Path(path).write_text(json.dumps(ref,ensure_ascii=False,indent=2))


def load_price_reference(path)->dict:
    obj=json.loads(Path(path).read_text())
    if obj.get('schema')!='copy_price_reference_v10':raise ValueError('Incorrect V10 reference schema')
    return obj


def predict_price_evidence(target_df:pd.DataFrame,reference:dict)->pd.DataFrame:
    """Applying a frozen price reference is independent of target batch composition."""
    if reference.get('schema')!='copy_price_reference_v10':raise ValueError('Fit V10 reference first')
    cfg=PriceClusterConfig(**reference['config'])
    result=pd.DataFrame(index=target_df.index)
    result['copy_price_cluster']=np.nan
    result['copy_price_trusted']=0
    result['copy_price_only_confidence']=np.nan
    result['copy_price_cluster_boundary']=np.nan
    result['copy_price_lower_mode']=np.nan
    result['copy_price_upper_mode']=np.nan
    result['copy_price_cluster_n']=np.nan
    result['copy_price_cluster_method']='unavailable'
    result['copy_price_gap_log']=np.nan
    result['copy_price_log_odds']=np.nan
    result['copy_price_reference_scope']='frozen_full_family'
    result['copy_price_evidence_status']='unavailable'
    result['copy_price_reference_has_family']=0
    result['copy_price_reference_has_year_band']=0
    result['copy_price_band_donor_year']=np.nan
    result['copy_price_band_donor_distance']=np.nan
    result['copy_price_reference_month_in_scope']=0
    ok=_reference_valid_rows(target_df,cfg,for_fit=False)
    years=_numeric(target_df,'production_year')
    months=_price_month(target_df)
    price=_numeric(target_df,'price')
    fam=_family(target_df)
    # Explicit diagnostics: formerly *all* failure modes were indistinguishable.
    result['copy_price_evidence_status']='missing_family_reference'
    missing_data=_numeric(target_df,'price').isna() | _numeric(target_df,'production_year').isna() | _price_month(target_df).isna()
    result.loc[missing_data,'copy_price_evidence_status']='missing_price_year_or_date'
    result.loc[~missing_data & ~ok,'copy_price_evidence_status']='invalid_price_or_source'
    for name,m in reference['families'].items():
        # A frozen reference may be used for subsequent ads, but not forever:
        # refuse to extrapolate an out-of-date market beyond the configured horizon.
        time_ok = months.le(m['train_max_month'] + cfg.max_scoring_months_ahead) & months.ge(m['train_min_month'] - 1)
        group_mask = fam.eq(name).fillna(False)
        result.loc[group_mask,'copy_price_reference_has_family']=1
        result.loc[group_mask,'copy_price_evidence_status']='outside_reference_date_scope'
        result.loc[group_mask & time_ok.fillna(False),'copy_price_reference_month_in_scope']=1
        year_available=years.isin([int(v) for v in m['year_bands']])
        result.loc[group_mask & time_ok.fillna(False) & ~year_available,'copy_price_evidence_status']='missing_year_band'
        candidate=group_mask & time_ok.fillna(False) & year_available
        result.loc[candidate,'copy_price_reference_has_year_band']=1
        result.loc[candidate & ~ok,'copy_price_evidence_status']='invalid_price_or_source'
        idx=target_df.index[ok & candidate]
        if len(idx)==0:continue
        for year,b in m['year_bands'].items():
            ix=idx[years.loc[idx].eq(int(year))]
            if len(ix)==0:continue
            lp=np.log(price.loc[ix])+m['monthly_slope']*np.clip(m['ref_month']-months.loc[ix],0,12)
            lo,hi=b['low'],b['high']
            result.loc[ix,'copy_price_band_donor_year']=b.get('donor_year',int(year))
            result.loc[ix,'copy_price_band_donor_distance']=b.get('donor_distance',0)
            s0=max(b['sd_low'],.075);s1=max(b['sd_high'],.075)
            # equal prior prevents rare OEM price stratum from being missed
            ll0=-.5*((lp-lo)/s0)**2-np.log(s0)
            ll1=-.5*((lp-hi)/s1)**2-np.log(s1)
            lodds=ll1-ll0
            # if data are close to the middle, abstain even with a strong MAP
            confident=np.abs(lodds)>=max(cfg.min_abs_log_odds,
                                           np.log(cfg.min_posterior/(1-cfg.min_posterior)))
            confident &= (np.minimum(np.abs(lp-lo)/s0,np.abs(lp-hi)/s1) <= 2.7)
            confident &= (np.abs(lp-(lo+hi)/2) > .055)
            cl=(lodds>0).astype(int)
            result.loc[ix,'copy_price_log_odds']=lodds
            result.loc[ix,'copy_price_cluster']=cl.where(confident,np.nan)
            result.loc[ix,'copy_price_trusted']=confident.astype('int8')
            result.loc[ix,'copy_price_only_confidence']=np.abs(lodds)
            # log-midpoint for explainability, not a fixed threshold
            bound=np.exp((lo+hi)/2-m['monthly_slope']*np.clip(m['ref_month']-months.loc[ix],0,12))
            result.loc[ix,'copy_price_cluster_boundary']=bound
            result.loc[ix,'copy_price_lower_mode']=np.exp(lo-m['monthly_slope']*np.clip(m['ref_month']-months.loc[ix],0,12))
            result.loc[ix,'copy_price_upper_mode']=np.exp(hi-m['monthly_slope']*np.clip(m['ref_month']-months.loc[ix],0,12))
            result.loc[ix,'copy_price_cluster_n']=b['n']
            result.loc[ix,'copy_price_cluster_method']=b['method']
            result.loc[ix,'copy_price_evidence_status']='price_overlap_abstain'
            result.loc[ix[confident],'copy_price_evidence_status']='trusted_price_cluster'
            result.loc[ix,'copy_price_gap_log']=hi-lo
    return result

def audit_reference_coverage(df:pd.DataFrame,ref:dict)->pd.DataFrame:
    """Explain missing evidence without modifying label columns."""
    scored=predict_price_evidence(df,ref)
    keep=['copy_price_evidence_status','copy_price_reference_has_family',
          'copy_price_reference_has_year_band','copy_price_reference_month_in_scope',
          'copy_price_trusted','copy_price_cluster_method']
    return scored[keep].copy()


def assert_price_reference_coverage(target_df:pd.DataFrame,ref:dict,
                                    *, required_families:Optional[list[str]]=None,
                                    require_all_valid_rows:bool=False)->pd.DataFrame:
    """Fail FAST if a required family cannot be priced using the frozen ref.

    Returns the audit table on success. This does not demand two clusters for
    the entire motorcycle catalog: it checks ONLY explicitly required families.
    `require_all_valid_rows=True` also rejects date/year coverage holes.
    """
    tab = audit_reference_coverage(target_df,ref)
    fam=_family(target_df)
    for requested in required_families or []:
        key=str(_family(pd.DataFrame({'real_model':[requested]})).iloc[0])
        mask=fam.eq(key)
        if not mask.any():
            raise ValueError(f'Required family {requested!r} does not appear in target_df')
        if key not in ref['families']:
            raise ValueError(f'V10 reference has no price model for {key!r}; fit on ALL brands and years first.')
        statuses=tab.loc[mask,'copy_price_evidence_status'].value_counts().to_dict()
        if require_all_valid_rows:
            failures={k:v for k,v in statuses.items() if k in {
                'missing_family_reference','missing_year_band','outside_reference_date_scope',
                'missing_price_year_or_date','invalid_price_or_source'
            }}
            if failures:
                raise ValueError(f'V10 family {key!r} reference coverage failure: {failures}. '
                                 'Inspect source, publish_date, production_year, and fit-time filters.')
    return tab


# ---------------------------------------------------------------------------
# HIERARCHICAL SEMANTIC EVIDENCE — INDEPENDENT OF PRICE CLUSTER FITTING
# ---------------------------------------------------------------------------

# Specific maker aliases seen in the data; augmented by dictionary.mappings.
_LOCAL_MAKER_ALIASES = (
    'saghebkhodro','saqeb','sagheb','irandocharkh','iran doocharkh',
    'tosanmoharekeshargh','tosan','to san','niroomotor','niroo motor',
    'jahan-hamta','jahan hamta','zomorod-kavir','zomorod kavir',
    'salar gostar','salar','shahin motor','delta','blenta','ava',
    'kavir-motor','kavir','hani','hamtaz','dino','kabir','tizpar','pishtaz','nami',
    'نیروموتور','نیرو موتور','ثاقب خودرو','ثاقب','آوا','توسن','توسان',
    'شاهین موتور','دلتا','کبیر موتور','کبیر','سالار گستر','بلنتا',
    'ایران دوچرخ','ایران دو چرخ','زمرد کویر','کویر موتور','کویر',
    'هانی','همتاز','تیزپر','پیشتاز','جهان همتا','نامی','ساوین','سگال',
)
# Use Persian-aware token boundaries to avoid treating "اصلاً" (at all) as
# "اصل" (authentic). The old regex matched the first three letters of "اصلا".
_WORD_LEFT = r'(?<![^\W_])'
_WORD_RIGHT = r'(?![^\W_])'
_ORIGINAL_TERMS = r'(?:' + _WORD_LEFT + r'(?:اصلی|اصل|اورجینال|تایلندی|ژاپنی|اندونزیایی|وارداتی|original|genuine|thai(?:land)?)' + _WORD_RIGHT + r'|ساخت\s*تایلند)'
_COPY_TERMS = r'(?:' + _WORD_LEFT + r'(?:طرح|های\s*کپی|هایکپی|high\s*copy|کپی|clone|فیک|fake)' + _WORD_RIGHT + r')'
_VEHICLE_WORDS = (r'(?:کلیک|کیلیک|کیلک|واریو|click|vario|موتور(?:\s*سیکلت)?|اسکوتر|هوندا|هندا|'
                  r'یاماها|aerox|ایروکس|آیروکس|nmax|pcx|adv|باکسر|boxer|وسپا|vespa|'
                  r'گس[\s\-]*گس|gasgas|crf|سی\s*آر\s*اف|پالسار|pulsar|آپاچی|apache|فورزا|forza|'
                  r'دوک|duke|کاوازاکی|ktm|کی\s*تی\s*ام|گلکسی|galaxy|بنلی|benelli|هارلی|harley|کراس|'
                  r'cdi|cg[\s\-]*\d+|suzuki|سوزوکی|kymco|کیمکو|cfmoto|لیفان|lifan)')
# Parts and consumables are not evidence of authenticity of the VEHICLE.
_PART_CLAIM = re.compile(
    r'(?:انجین|موتور\s*فنی|فلاپ|فیلتر|لاستیک|کمک|ترمز|رادیاتور|قفل|لوازم|قطعات|رنگ|فرمان|کاربرات|سرسیلندر|اگزایتری|'
    r'تسمه|روغن|باتری|باطری|سند|مدارک|لنت|اگزوز|باکس|سوییچ|سوئیچ|کیلس|کی\s*لس|هواکش)'
    r'[^،؛.!؟\n]{0,35}' + _ORIGINAL_TERMS,re.I)
_NEGATED = re.compile(r'(?<![^\W_])(?:طرح|کپی|اصلی|اصل|اورجینال)(?![^\W_])'
    r'[^،؛.!؟\n]{0,16}(?:نیست|نبوده|نمی\s*باشد|نداره|ندارد)|'
    r'(?:نیست|نبوده)\s*(?:طرح|کپی|اصل|اصلی)', re.I)
_MODEL_CONTEXT=re.compile(r'کلیک|کیلیک|کیلک|واریو|click|vario|موتور|اسکوتر|150|۱۵۰|160|۱۶۰|aerox|ایروکس|آیروکس',re.I)

# A refusal of installments is not an installment listing. Remove only the
# negative *clause*, not all evidence in the entire title/description.
_FINANCE_DENIAL = re.compile(
    r'(?:بدون|فاقد|عدم)\s*(?:فروش\s*)?(?:اقساط|قسط|قسطی|پیش\s*پرداخت)|'
    r'(?:اقساط|قسطی|قسط\s*بندی|پیش\s*پرداخت)'
    r'[^.،؛!؟\n]{0,28}?(?:ندارم|نداریم|ندارد|نیست|نبوده|نمی\s*دهم|نمی\s*دهیم|'
    r'نمی\s*فروشم|نمی\s*فروشیم|قبول\s*نمی\s*کنم|نمی\s*کنم|امکان\s*ندارد)', re.I)
_FINANCE_POSITIVE = re.compile(r'(?:اقساط|قسطی|قسط\s*بندی|پیش\s*پرداخت|'
    r'فروش\s*قسطی|فروش\s*اقساطی|طرح\s*فروش\s*اقساطی)',re.I)

def _has_positive_finance(s:pd.Series)->pd.Series:
    cleaned=s.str.replace(_FINANCE_DENIAL,' ',regex=True)
    return _contains(cleaned,_FINANCE_POSITIVE)


def _maker_pattern():
    aliases=set(_LOCAL_MAKER_ALIASES)
    aliases.update(map(str,IRANIAN_BRANDS))
    aliases.update(str(k) for k,v in BRAND_NAME_MAP.items() if v in set(IRANIAN_BRANDS))
    aliases={a.lower().strip() for a in aliases if len(a.strip())>=3}
    # English word edges; Persian strings do not require \b after normalization.
    chunks=[]
    for a in sorted(aliases,key=len,reverse=True):
        esc=re.escape(a)
        if re.search('[a-z]',a):chunks.append(r'(?<![a-z])'+esc+r'(?![a-z])')
        else:chunks.append(esc)
    return re.compile('(?:'+'|'.join(chunks)+')',re.I)

_LOCAL_MAKER_RE = _maker_pattern()

# Copy-related words in barter/comparison and cosmetic-design descriptions
# describe OTHER vehicles or parts, not the offered bike's authenticity.
_SWAP_COPY_CONTEXT = re.compile(
    r'(?:معاوضه|مبادله|تعویض|عوض\s*کردن)[^\n.،؛!؟]{0,100}(?:طرح|کپی)[^\n.،؛!؟]{0,24}'
    r'(?:نمی\s*کنم|نمیکنم|نمی\s*شود|نمی\s*خوام|ندارم)?', re.I)
_DECOR_COPY_CONTEXT = re.compile(
    r'(?:بدنه|رنگ|استیکر|قاب|فلاپ|کاور|روکش|برچسب|طرح\s*رنگ|طرح\s*فروش|طرح\s*ترافیک)'
    r'[^\n.،؛!؟]{0,42}(?:طرح|کپی)[^\n.،؛!؟]{0,12}',re.I)

_SWAP_ORIGINAL_CONTEXT = re.compile(
    r'(?:معاوضه|مبادله|تعویض)\s*(?:با)?[^\n.،؛!؟]{0,100}'
    r'(?:\bاصل\b|\bاصلی\b|اورجینال|تایلندی|original)', re.I)

def _claim_spacing(s:pd.Series)->pd.Series:
    # Sellers often attach 'طرح' to motorcycle names: موتورطرح، طرحadv، 185طرح.
    return (s.str.replace(r'(?<=[a-z0-9])(?=طرح)', ' ', regex=True)
             .str.replace(r'(موتور|موتورسیکلت|هوندا|کلیک)(?=طرح)',r'\1 ',regex=True)
             .str.replace(r'(?<=طرح)(?=[a-z]|کلیک|وسپا|واریو|باکسر)', ' ', regex=True))

def _original_claim_scope(s:pd.Series)->pd.Series:
    return s.str.replace(_SWAP_ORIGINAL_CONTEXT,' ',regex=True)

def _copy_claim_scope(s:pd.Series)->pd.Series:
    return s.str.replace(_SWAP_COPY_CONTEXT,' ',regex=True).str.replace(
        _DECOR_COPY_CONTEXT,' ',regex=True)


def _vehicle_claim(s:pd.Series,kind:str)->pd.Series:
    """Find claims about the WHOLE MOTORCYCLE, not spare parts or its engine."""
    term=_ORIGINAL_TERMS if kind=='original' else _COPY_TERMS
    # Remove misleading component phrases before inspecting full-vehicle claims.
    clean=(s.str.replace(_PART_CLAIM,' ',regex=True)
              .str.replace(r'همه\s*چیزش\s*(?:اصل|اصلی)|قطعاتش\s*(?:اصل|اصلی)',' ',regex=True)) if kind=='original' else s
    vicinity=rf'(?:{_VEHICLE_WORDS})[^،؛.!؟\n]{{0,32}}(?:{term})|(?:{term})[^،؛.!؟\n]{{0,32}}(?:{_VEHICLE_WORDS})'
    return _contains(clean,vicinity)


def _correct_is_copy_labels_previous(df:pd.DataFrame)->pd.DataFrame:
    """PURE evidence extractor. No label/price/DB identity is changed."""
    out=df.copy()
    t=_claim_spacing(_normal(out.get('title',pd.Series('',index=out.index))))
    d=_claim_spacing(_normal(out.get('description',pd.Series('',index=out.index))))
    raw=_normal(out.get('brand_model_raw',pd.Series('',index=out.index)))
    # 'Honda Click'/'Vario' are platform descriptions, not OEM claims.
    t_o=_vehicle_claim(_original_claim_scope(t),'original');d_o=_vehicle_claim(_original_claim_scope(d),'original')
    t_c=_vehicle_claim(_copy_claim_scope(t),'copy');d_c=_vehicle_claim(_copy_claim_scope(d),'copy')
    neg=_contains(t+' '+d,_NEGATED)
    fin=_has_positive_finance(t+' '+d)
    # Phrases 'طرح فروش', 'طرح اقساط' are not proof of a copy.
    # Non-vehicle uses of طرح are removed contextually, not by excluding
    # the entire title just because it also mentions an installment plan.

    conflict=(t_o|d_o)&(t_c|d_c) | neg
    text_proposal=pd.Series(np.nan,index=out.index,dtype=float)
    text_proposal.loc[(t_o|d_o)&~conflict]=0
    text_proposal.loc[(t_c|d_c)&~conflict]=1
    text_strength=pd.Series('none',index=out.index,dtype='object')
    text_strength.loc[text_proposal.notna()&~(t_o|t_c)]='description'
    text_strength.loc[text_proposal.notna()&(t_o|t_c)]='title'
    text_strength.loc[conflict]='conflict'
    title_local=_contains(t,_LOCAL_MAKER_RE.pattern)&_contains(t,_MODEL_CONTEXT.pattern)
    desc_local=_contains(d,_LOCAL_MAKER_RE.pattern)&_contains(d,_MODEL_CONTEXT.pattern)
    raw_local=_contains(raw,_LOCAL_MAKER_RE.pattern)
    # A specific local factory + product name in raw: strong. Bare 'delta'
    # or other one-word catalog values are weak unless corroborated by price.
    raw_specific=raw_local&(_contains(raw,r'click|vario|کلیک|واریو|hi\s*click|blenta|ava\s*150|crt\s*160|wave|tekno|\b(?:125|150|160|170|180|200)\b') |
                              (raw.str.len()>=12))
    raw_oem=_contains(raw,r'(?<![a-z])(?:honda|yamaha|vespa|suzuki|kawasaki)(?![a-z])|هوندا|یاماها')
    # Explicitly different vehicle family: a mapping quality issue, not copy evidence.
    family=_family(out)
    family_click=family.str.contains('click',regex=False,na=False)
    different_title=family_click & _contains(t,r'اکتیوا|activa|فورزا|forza|ایروکس|aerox|nmax|انمکس') & ~_contains(t,r'کلیک|کیلیک|کیلک|click|واریو')
    different_raw=family_click & _contains(raw,r'\b(?:kps|cg|cdi|nmax|aerox|adv|pcx|activa|forza)\b') & ~_contains(raw,r'click|vario|کلیک|واریو')
    out['copy_family_key']=family
    out['copy_text_title_original']=t_o.astype('int8')
    out['copy_text_desc_original']=d_o.astype('int8')
    out['copy_text_title_copy']=t_c.astype('int8')
    out['copy_context_ignored']=( (t.str.contains(_COPY_TERMS,regex=True,na=False) & ~t_c) | (d.str.contains(_COPY_TERMS,regex=True,na=False) & ~d_c)).astype('int8')
    out['copy_text_desc_copy']=d_c.astype('int8')
    out['copy_text_proposal']=text_proposal
    out['copy_text_strength']=text_strength
    out['copy_label_text_conflict']=conflict.astype('int8')
    out['copy_text_negation']=neg.astype('int8')
    out['copy_title_local_hint']=title_local.astype('int8')
    out['copy_desc_local_hint']=desc_local.astype('int8')
    out['copy_raw_local_hint']=raw_local.astype('int8')
    out['copy_raw_local_specific']=raw_specific.astype('int8')
    out['copy_raw_oem_hint']=raw_oem.astype('int8')
    out['copy_raw_copy_hint']=_contains(raw,r'طرح|های\s*کپی|\bcopy\b|\bclone\b').astype('int8')
    out['copy_title_model_mismatch']=different_title.astype('int8')
    out['copy_raw_model_mismatch']=different_raw.astype('int8')
    out['copy_finance_ad']=fin.astype('int8')
    return out



def _correct_is_copy_labels(df:pd.DataFrame,suspicious_multiplier:float=.30,
                            *,price_reference:Optional[dict]=None,
                            use_price:bool=True,
                            config:Optional[PriceClusterConfig]=None)->pd.DataFrame:
    """V10: stable frozen price bands + independent text evidence + abstention.

    CRITICAL: fit_price_reference(FULL historical data) once and pass its result.
    No refit on the current filtered target. 'is_original' NEVER mutated.
    No brand_model_raw-only or local-brand-only fallback in the absence of price.
    """
    if not 0<suspicious_multiplier<=1:raise ValueError('invalid suspicious_multiplier')
    for col in ('is_original','is_copy','price','real_model','production_year'):
        if col not in df:raise KeyError(f'Missing required field: {col}')
    if use_price and price_reference is None:
        raise ValueError('V10 requires a FROZEN price_reference: '
                         'fit_price_reference(full_unfiltered_ads_df) once and pass it. '
                         'Silently refitting on a subset is deliberately prohibited.')
    db_original=df['is_original'].copy(deep=True)
    initial_copy=df['is_copy'].copy(deep=True)
    out=_correct_is_copy_labels_previous(df)
    out['is_copy_before_clean']=initial_copy
    if 'label_weight_pre_copy_audit' not in out:
        out['label_weight_pre_copy_audit']=_numeric(out,'label_weight',1.).fillna(1.).clip(lower=1e-6)
    if use_price:
        price_evidence=predict_price_evidence(out,price_reference)
    else:
        price_evidence=predict_price_evidence(out,{'schema':'copy_price_reference_v10',
            'config':(config or PriceClusterConfig()).__dict__,'families':{}})
        price_evidence['copy_price_cluster_method']='disabled'
    for col in price_evidence:out[col]=price_evidence[col]
    n=lambda col:_numeric(out,col)
    original=n('is_copy'); text=n('copy_text_proposal')
    trusted=n('copy_price_trusted').eq(1)&n('copy_price_cluster').isin([0,1])
    price_label=(1-n('copy_price_cluster')).where(trusted)
    explicit_orig=text.eq(0);explicit_copy=text.eq(1)
    local_title=n('copy_title_local_hint').eq(1)
    local_desc=n('copy_desc_local_hint').eq(1)
    local_raw=n('copy_raw_local_hint').eq(1)
    raw_mismatch=n('copy_raw_model_mismatch').eq(1)
    title_mismatch=n('copy_title_model_mismatch').eq(1)
    conflict_text=n('copy_label_text_conflict').eq(1)
    invalid=~original.isin([0,1])
    identity_missing=(n('motorcycle_model_id').isna()|n('is_original').isna()) if 'motorcycle_model_id' in out else n('is_original').isna()
    invalid_price=n('copy_finance_ad').eq(1)|~n('price').gt(0)
    price_text_conflict=trusted & text.isin([0,1]) & price_label.ne(text)
    # An EXPLICIT local manufacturer of the advertised bike in title is
    # a genuine conflict when high-priced, unlike raw catalog metadata.
    price_maker_conflict=trusted & price_label.eq(0)&local_title
    # Description may mention another bike/company; require support from raw.
    price_maker_conflict |= trusted & price_label.eq(0)&local_desc&local_raw
    text_maker_conflict=explicit_orig & local_title
    # Raw model mismatch is review-worthy, not automatic ban on detecting a
    # price stratum: a noisy catalog can map Vario to some other Click variant.
    # But when target TITLE says a wholly different model, do not flip.
    freeze=(identity_missing|conflict_text|invalid|invalid_price|title_mismatch|
            price_text_conflict|price_maker_conflict|text_maker_conflict)
    proposed=pd.Series(np.nan,index=out.index,dtype=float)
    proposed.loc[trusted]=price_label.loc[trusted]
    # When the complete family price reference gives no decisive verdict:
    # * strongly explicit copy/original VEHICLE claims are still usable
    # * bare manufacturer in brand_model_raw is NEVER sufficient to flip.
    no_price=~trusted
    proposed.loc[no_price & explicit_copy]=1.
    strong_original_title=n('copy_text_title_original').eq(1)
    db_orig_oem=n('is_original').eq(1)
    proposed.loc[no_price & explicit_orig & strong_original_title & db_orig_oem & ~local_title & ~local_desc]=0.
    # Even an explicit original claim of an own-brand local model can refer
    # to an original component, so require more than text-only evidence.
    local_brands=set(IRANIAN_BRANDS) if IRANIAN_BRANDS else {
        'Kavir','Niroo Motor','Hani','Kabir','Salar Gostar','Saqeb',
        'Jahan Hamta','Shahin Motor','Toosan','Iran Doocharkh'}
    only_local=out.get('real_brand',pd.Series('',index=out.index)).isin(local_brands)
    local_orig_not_confirmed=no_price & explicit_orig & only_local
    freeze|=local_orig_not_confirmed
    proposed.loc[local_orig_not_confirmed]=np.nan
    changed=proposed.isin([0.,1.]) & proposed.ne(original) & ~freeze
    if changed.any():
        if pd.api.types.is_integer_dtype(initial_copy.dtype):
            out.loc[changed,'is_copy']=proposed.loc[changed].astype(initial_copy.dtype)
        else:
            out.loc[changed,'is_copy']=proposed.loc[changed]
    reasons=pd.Series('no_decisive_evidence',index=out.index,dtype='object')
    reasons.loc[trusted]='trusted_price_consistent'
    reasons.loc[changed & trusted & price_label.eq(0)]='corrected_price_high'
    reasons.loc[changed & trusted & price_label.eq(1)]='corrected_price_low'
    reasons.loc[changed & trusted & price_label.eq(0)&local_raw]='corrected_price_high_raw_maker_ignored'
    reasons.loc[changed & no_price & explicit_orig]='corrected_explicit_vehicle_original'
    reasons.loc[changed & no_price & explicit_copy]='corrected_explicit_vehicle_copy'
    reasons.loc[price_text_conflict]='conflict_price_vs_explicit_vehicle_text'
    reasons.loc[price_maker_conflict]='conflict_price_high_vs_named_maker'
    reasons.loc[text_maker_conflict]='conflict_explicit_original_vs_named_maker'
    reasons.loc[conflict_text]='conflict_text_frozen'
    reasons.loc[title_mismatch]='title_model_mismatch_frozen'
    reasons.loc[local_orig_not_confirmed]='unverified_local_original_claim_frozen'
    reasons.loc[invalid_price]='invalid_price_or_installment_frozen'
    reasons.loc[invalid]='invalid_copy_label_frozen'
    reasons.loc[identity_missing]='missing_identity_frozen'
    out['copy_identity_missing']=identity_missing.astype('int8')
    out['copy_label_decision_reason']=reasons
    out['copy_label_changed']=changed.astype('int8')
    out['copy_label_suspicious']=freeze.astype('int8')
    out['copy_label_review_flag']=(freeze|raw_mismatch|(trusted & price_label.eq(0)&local_raw)|
        (no_price & (local_raw|local_title|local_desc))).astype('int8')
    out['copy_label_evidence_price']=price_label
    out['copy_label_evidence_text']=text
    out['copy_label_evidence_local_maker']=(local_title|local_desc|local_raw).astype('int8')
    out['copy_raw_price_disagreement']=(trusted & price_label.eq(0)&local_raw).astype('int8')
    out['copy_label_db_mismatch']=(n('is_copy').isin([0,1])&n('is_original').isin([0,1])&
                                    n('is_copy').ne(1-n('is_original'))).astype('int8')
    # Explicit, non-compounding quality weights. Existing 1.0/1.5 is retained
    # as `label_weight_pre_copy_audit` and multiplied by a quality factor.
    # An unavailable reference is NOT evidence of a bad label. No maker-only
    # correction is allowed just because the price evidence is absent.
    factor=pd.Series(1.0,index=out.index,dtype='float64')
    no_evidence=reasons.eq('no_decisive_evidence')
    # Raw local maker is *expected* for an Iranian-brand copy listing;
    # reduce trust only when this hint contradicts a claimed OEM-original
    # listing or when the raw model actually appears mismapped.
    db_oem=n('is_original').eq(1)
    labeled_original=n('is_copy').eq(0)
    weak_hint = raw_mismatch | (db_oem & labeled_original &
                               (local_raw | local_title | local_desc))
    factor.loc[no_evidence & ~weak_hint]=0.90
    factor.loc[no_evidence & weak_hint]=0.55
    factor.loc[reasons.isin(['corrected_explicit_vehicle_copy',
                              'corrected_explicit_vehicle_original'])]=0.80
    factor.loc[reasons.isin(['corrected_price_low','corrected_price_high'])]=0.95
    factor.loc[reasons.eq('corrected_price_high_raw_maker_ignored')]=0.80
    factor.loc[reasons.str.startswith('conflict_',na=False)]=min(suspicious_multiplier,0.30)
    factor.loc[reasons.eq('conflict_text_frozen')]=min(suspicious_multiplier,0.30)
    factor.loc[reasons.eq('unverified_local_original_claim_frozen')]=0.35
    factor.loc[reasons.eq('invalid_copy_label_frozen')]=0.05
    factor.loc[reasons.eq('title_model_mismatch_frozen')]=0.05
    factor.loc[reasons.eq('invalid_price_or_installment_frozen')]=0.05
    factor.loc[reasons.eq('trusted_price_consistent')]=1.0
    factor.loc[identity_missing]=0.30
    out['copy_label_audit_multiplier']=factor.astype('float32')
    out['copy_label_weight_factor']=out['copy_label_audit_multiplier']
    out['label_weight']=n('label_weight_pre_copy_audit').fillna(1).clip(lower=1e-6)*out['copy_label_audit_multiplier']
    pd.testing.assert_series_equal(out['is_original'],db_original,check_dtype=True)
    pd.testing.assert_series_equal(out.loc[freeze,'is_copy'],initial_copy.loc[freeze],check_dtype=True)
    return out

__all__=['PriceClusterConfig','fit_price_reference','save_price_reference',
         'load_price_reference','predict_price_evidence','audit_reference_coverage','assert_price_reference_coverage',
         '_correct_is_copy_labels_previous','_correct_is_copy_labels']
