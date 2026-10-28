"""Motorcycle text variant labelling, v2 (2026-10).

Compatibility entry points:
    _add_model_variant(df, min_variant_support=7, min_overlap_cells=3,
                       min_price_gap=0.04, *, fit_df=None, registry=None,
                       return_registry=False)
    _set_model_variant(df, text_col='model_variant', *, registry=None)

No changes to motorcycle_model_id, is_original, is_copy or ad price. Price
is used *only* to decide whether OPTIONAL features may split a market model;
explicit engine architecture uses textual evidence, not a price threshold.
For temporal evaluation, fit registry on past ads only, then apply it to held-out
ads/factory records; never fit a price-derived registry on the same test fold.
"""
from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd

BASE_FEATURES = (
    'AIR_COOLING','LIQUID_COOLING','KEYLESS','TURBO','NEW_FACE','NEO',
    'CBS','RIDE_MODES','TWIN_CYLINDER','BOX','SPORT_RIM','ABS','TCS',
    'THAILAND','INDONESIA'
)
STRUCTURAL = ('SINGLE_CYLINDER','TWIN_CYLINDER','FOUR_CYLINDER')
# Accessories are not a factory model: retain features but never split pricing.
ACCESSORIES = {'BOX','SPORT_RIM'}
OPTIONAL = tuple(x for x in BASE_FEATURES if x not in ACCESSORIES and x != 'TWIN_CYLINDER')
ALL_FEATURES = tuple(dict.fromkeys(BASE_FEATURES + STRUCTURAL))
# Preserve legacy order of multi-feature variant tokens for database compatibility.
MODEL_VARIANT_ORDER = tuple(dict.fromkeys(
    tuple(k for x in BASE_FEATURES for k in (("SINGLE_CYLINDER","FOUR_CYLINDER",x) if x == "TWIN_CYLINDER" else (x,)))
))

# Explicit cylinder text: a naked 'تک'/'جفت' only within BN250 TITLE,
# not in descriptions or other models (e.g. 'تک ریموت', 'تک سند').
CYL_PATS = {
  'SINGLE_CYLINDER': re.compile(r'(?<!\w)(?:تک[\s‌\-]*سیلندر|تکسیلندر|یک[\s‌\-]*سیلندر|[۱1][\s‌\-]*سیلندر|single[\s-]*(?:cylinder|cyl)|one[\s-]*cylinder)(?!\w)', re.I),
  'TWIN_CYLINDER': re.compile(r'(?<!\w)(?:دو[\s‌\-]*سیلندر|دوسیلندر|جفت[\s‌\-]*سیلندر|[۲2][\s‌\-]*سیلندر|twin[\s-]*(?:cylinder|cyl)|2[\s-]*cyl)(?!\w)', re.I),
  'FOUR_CYLINDER': re.compile(r'(?<!\w)(?:چهار[\s‌\-]*سیلندر|چهارسیلندر|[۴4][\s‌\-]*سیلندر|four[\s-]*(?:cylinder|cyl)|4[\s-]*cyl|zx[\s‌\-]*25r?)(?!\w)', re.I),
}
BN_SINGLE = re.compile(r'(?<![\w])تک(?![\w])')
BN_TWIN = re.compile(r'(?<![\w])جفت(?![\w])')
NINJA_SL = re.compile(r'(?<![a-z0-9])(?:sl[\s-]*250|250[\s-]*sl|sl)(?![a-z0-9])', re.I)
# Explicit negation of optional features is not positive evidence.
NEG_PREFIX = re.compile(r'(?:بدون|فاقد|ندارد|نداشته|no\s+)\s*$',re.I)
FEATURE_PAT = {
 'AIR_COOLING': r'هوا[\s‌\-]*خنک|خنک[\s‌\-]*کننده[\s‌\-]*هوا|air[\s-]*cool(?:ed|ing)',
 'LIQUID_COOLING': r'آب[\s‌\-]*خنک|مایع[\s‌\-]*خنک|خنک[\s‌\-]*کننده[\s‌\-]*مایع|liquid[\s-]*cool(?:ed|ing)|water[\s-]*cool(?:ed|ing)|رادیاتور|رادیاتوری',
 'KEYLESS': r'کیلس|کی[\s‌\-]*لس|کلید[\s‌\-]*لس|(?<!\w)keyless(?!\w)|(?<!\w)kls(?!\w)|smart[\s-]*key|اسمارت[\s‌\-]*کی',
 'TURBO': r'توربو(?:[\s‌\-]*شارژ)?|(?<!\w)turbo(?:charged|[\s-]*charged)?(?!\w)',
 'NEW_FACE': r'نیو[\s‌\-]*فیس|فیس[\s‌\-]*جدید|new[\s-]*face',
 'NEO': r'(?<!\w)neo(?!\w)|نئو',
 'CBS': r'(?<!\w)cbs(?!\w)|ترمز[\s‌\-]*ترکیبی',
 'RIDE_MODES': r'rid(?:e|ing)[\s-]*modes?|حالت[\s‌\-]*رانندگی|مود[\s‌\-]*(?:sport|city|rain)',
 'BOX': r'باکس[\s‌\-]*(?:دار|پشت)|box[\s-]*دار|with[\s-]*box',
 'SPORT_RIM': r'رینگ[\s‌\-]*اسپرت|اسپرت[\s‌\-]*رینگ|sport[\s-]*(?:rim|wheel)',
 'ABS': r'ای[\s‌\-]*بی[\s‌\-]*اس|ایبیاس|(?<!\w)abs(?!\w)',
 'TCS': r'ترکشن[\s‌\-]*کنترل|traction[\s-]*control|(?<!\w)tcs(?!\w)',
 'THAILAND': r'تایلند|(?<!\w)thailand(?!\w)|تایلندی',
 'INDONESIA': r'اندونزی|(?<!\w)indonesia(?:n)?(?!\w)',
}
COMPILED = {k:re.compile(v,re.I) for k,v in FEATURE_PAT.items()}


def _norm(t):
    if pd.isna(t): return ''
    s=unicodedata.normalize('NFKC', str(t)).lower().replace('ي','ی').replace('ك','ک')
    return s.replace('ـ','').replace('\u200c',' ')


def _no_negations(s, pat):
    for m in pat.finditer(s):
        if not NEG_PREFIX.search(s[max(0,m.start()-22):m.start()]):
            return True
    return False


def _group_key(row):
    mid = row.get('motorcycle_model_id',None)
    cp = row.get('is_copy',None)
    try: copy = str(int(float(cp)))
    except (ValueError,TypeError): copy='?'
    try:
        if mid is not None and pd.notna(mid):
            return f'id:{int(float(mid))}|copy:{copy}'
    except (ValueError,TypeError): pass
    return f'brand:{_norm(row.get("real_brand", ""))}|model:{_norm(row.get("real_model", ""))}|copy:{copy}'


def _extract_flags(df, *, factory_text_col=None):
    flags = np.zeros((len(df),len(ALL_FEATURES)), dtype=np.int8)
    status=[]; evidence=[]
    keys = list(ALL_FEATURES)
    k_to_i = {k:i for i,k in enumerate(keys)}
    use_titles = 'title' in df and factory_text_col is None
    for j, row in enumerate(df[[c for c in ('title','description','real_model','motorcycle_model_id',factory_text_col) if c is not None and c in df.columns]].to_dict('records')):
        title = _norm(row.get('title','')) if use_titles else ''
        desc = _norm(row.get('description','')) if use_titles else ''
        factory = _norm(row.get(factory_text_col,'')) if factory_text_col is not None else ''
        # Parse user-supplied canonical tokens: factory model_variant may itself
        # be SINGLE_CYLINDER+ABS, not plain natural language.
        canon = set(re.findall(r'(?<!\w)(?:'+'|'.join(ALL_FEATURES)+r')(?!\w)',factory.upper())) if factory else set()
        model = _norm(row.get('real_model',''))
        mid = row.get('motorcycle_model_id')
        benelli250 = (model in ('bn 250','bn250'))
        ninja250 = (model in ('ninja 250','ninja250'))
        if factory_text_col is None:
            searches=[('title',title),('description',desc)]
        else:
            searches=[('factory',factory)]
        cyl_status='none'; cyl_source='none'
        for source, piece in searches:
            hits=set()
            for k,p in CYL_PATS.items():
                if _no_negations(piece,p): hits.add(k)
            if benelli250 and source=='title':
                if BN_SINGLE.search(piece) and not re.search(r'تک\s*(?:برگ|ریموت|سویچ|سوییچ|دست|مالک|رنگ|سند|کلید)',piece): hits.add('SINGLE_CYLINDER')
                if BN_TWIN.search(piece): hits.add('TWIN_CYLINDER')
            if ninja250 and NINJA_SL.search(piece): hits.add('SINGLE_CYLINDER')
            # Canonical input tokens apply only when parsing factory.
            if source=='factory': hits |= set(canon).intersection(STRUCTURAL)
            if hits:
                if len(hits)==1:
                    flags[j,k_to_i[next(iter(hits))]]=1
                    cyl_status='explicit'; cyl_source=source
                else: cyl_status='conflict'; cyl_source=source
                break  # title evidence outranks description
        # other feature flags from text; never set cooling if both variants present
        whole = title+' '+desc if use_titles else factory
        for k,p in COMPILED.items():
            if k in ('AIR_COOLING','LIQUID_COOLING'):
                pass
            if k in canon or _no_negations(whole,p): flags[j,k_to_i[k]]=1
        if flags[j,k_to_i['AIR_COOLING']] and flags[j,k_to_i['LIQUID_COOLING']]:
            flags[j,k_to_i['AIR_COOLING']]=flags[j,k_to_i['LIQUID_COOLING']]=0
        # For explicit factory 'TWIN_CYLINDER' token it's already set by structural.
        status.append(cyl_status); evidence.append(cyl_source)
    return pd.DataFrame(flags,index=df.index,columns=[f'variant_{x.lower()}' for x in ALL_FEATURES]), pd.Series(status,index=df.index), pd.Series(evidence,index=df.index)


def _fit_variant_registry(train, raw, min_variant_support=7, min_overlap_cells=3, min_price_gap=.04):
    # Keys always scoped to real DB model identity and ad originality.
    keys=train.apply(_group_key,axis=1)
    ads = train['source'].astype(str).eq('ads') if 'source' in train else pd.Series(True,index=train.index)
    valid_price=pd.to_numeric(train['price'],errors='coerce').gt(0) if 'price' in train else pd.Series(False,index=train.index)
    valid = ads & valid_price
    enabled={}
    grp_keys=keys[valid]; train_idx=train.index[valid]
    # Positional arrays via reset_index to remain correct if source has duplicate index.
    tt=pd.DataFrame({
      'g':keys.to_numpy(),
      'age':pd.to_numeric(train['age'],errors='coerce').to_numpy() if 'age' in train else np.nan,
      'mileage_bucket':pd.to_numeric(train['mileage_bucket'],errors='coerce').to_numpy() if 'mileage_bucket' in train else np.nan,
      'price':pd.to_numeric(train['price'],errors='coerce').to_numpy() if 'price' in train else np.nan,
      'ads':valid.to_numpy(),
    })
    for col in raw.columns:
      feature=col.removeprefix('variant_').upper()
      if feature in ACCESSORIES: continue
      tt['detected']=raw[col].to_numpy()
      # Market support; count positives by exact model not other brands.
      c=tt.loc[tt.ads].groupby(['g','detected']).size().unstack('detected',fill_value=0)
      positives=c.get(1,pd.Series(dtype=float))
      if feature in STRUCTURAL:
        # No PRICE gate. For broad catalog models require market support;
        # BN250/Ninja250 have explicit known heterogeneous engine identity.
        for g,n1 in positives.items():
          if n1>=min_variant_support: enabled.setdefault(g,set()).add(feature)
        continue
      candidates=[g for g,n in positives.items() if n>=min_variant_support and c.loc[g].get(0,0)>=min_variant_support]
      if not candidates: continue
      pairs=tt.loc[tt.ads & tt.g.isin(candidates)].dropna(subset=['age','mileage_bucket']).groupby(['g','age','mileage_bucket','detected']).price.median().unstack('detected')
      if not {0,1}.issubset(pairs.columns):continue
      pairs=pairs.dropna(subset=[0,1]);pairs['gap']=np.log(pairs[1]/pairs[0]);
      for g,t in pairs.groupby(level=0):
        if len(t)<min_overlap_cells: continue
        gap=float(t['gap'].median()); consistency=max(float((t.gap>0).mean()),float((t.gap<0).mean()))
        if abs(np.expm1(gap))>=min_price_gap and consistency>=.67:
          enabled.setdefault(g,set()).add(feature)
    return {'version':'variant_v2','enabled':{k:sorted(v) for k,v in enabled.items()},
            'min_variant_support':min_variant_support,'min_overlap_cells':min_overlap_cells,
            'min_price_gap':min_price_gap}


def _apply(df, raw, status, sources, registry, *, factory=False, keep_factory_features=True):
    out=df.copy()
    keys=out.apply(_group_key,axis=1)
    enabled={k:set(v) for k,v in registry['enabled'].items()}
    flags=raw.copy()
    recognized_multi=df['real_model'].fillna('').astype(str).str.lower().str.replace(' ','',regex=False).isin(['bn250','ninja250']).to_numpy() if 'real_model' in df else np.zeros(len(df),dtype=bool)
    for col in flags:
        f=col.removeprefix('variant_').upper()
        if f in ACCESSORIES: # raw features preserved for analysis; no pricing split
            flags[col]=raw[col]
        else:
            supported = keys.map(lambda k: f in enabled.get(k,set())).to_numpy()
            allowed = (supported | recognized_multi) if f in STRUCTURAL else supported
            flags[col]=(raw[col].to_numpy() & allowed).astype('int8')
    # at most 1 structural flag; title explicit outweighs description.
    structural_cols=[f'variant_{k.lower()}' for k in STRUCTURAL]
    if (flags[structural_cols].sum(axis=1)>1).any(): raise ValueError('Invalid multiple cylinder labels')
    out=out.drop(columns=[c for c in ALL_FEATURES if c in out],errors='ignore')
    out=out.drop(columns=list(flags.columns),errors='ignore')
    out=pd.concat([out,flags],axis=1)
    # Price-defining variants exclude accessories. Restrict variants to 1 structural
    # + optional model features (a combination may still be informative).
    act_features=[f for f in MODEL_VARIANT_ORDER if f in set(STRUCTURAL).union(OPTIONAL) or (factory and keep_factory_features and f in ACCESSORIES)]
    act_cols=[f'variant_{k.lower()}' for k in act_features]
    names=np.array([c.replace('variant_','').upper() for c in act_cols],dtype=object)
    av=flags[act_cols].to_numpy();labels=['+'.join(names[row.astype(bool)]) or 'DEFAULT' for row in av]
    out['model_variant']=labels
    out['variant_cylinder_evidence']=status.astype(str).to_numpy()
    out['variant_evidence_source']=sources.astype(str).to_numpy()
    active=(av.sum(axis=1)>0)
    structural_active=flags[structural_cols].sum(axis=1).to_numpy()>0
    conf=np.where(structural_active & sources.eq('title').to_numpy(),.98,
         np.where(structural_active,.85,np.where(active,.75,0))).astype('float32')
    conf=np.where(status.eq('conflict').to_numpy(),0.,conf).astype('float32')
    out['variant_confidence']=conf
    out['variant_source']=np.where(structural_active,'explicit_identity',np.where(active,'text+market','default'))
    out['variant_conflict']=status.eq('conflict').astype('int8').to_numpy()
    out['variant_registry_version']='variant_v2'
    return out


def _add_model_variant(df, min_variant_support=7, min_overlap_cells=3, min_price_gap=0.04, *,
                       fit_df=None, registry=None, return_registry=False):
    """Detect/fit variant labels on ads. Supports train-only registry for leakage-safe holdout.

    If registry is supplied: apply it without touching prices.
    If fit_df is supplied: fit on fit_df then apply to df (temporal evaluation).
    Else: fit on df (in-sample descriptive cleaning, NOT leakage-safe scoring).
    """
    if registry is None:
        train=df if fit_df is None else fit_df
        train_raw,_,_=_extract_flags(train)
        registry=_fit_variant_registry(train,train_raw,min_variant_support,min_overlap_cells,min_price_gap)
    raw,status,sources=_extract_flags(df)
    result=_apply(df,raw,status,sources,registry)
    assert len(result)==len(df)
    return (result,registry) if return_registry else result


def _set_model_variant(df, text_col='model_variant', *, registry=None, alignment='preserve'):
    """Factory variant detection; never fit on factory prices.

    Default ``alignment='preserve'`` keeps explicit catalog features (including
    BOX/SPORT_RIM) even if they are not activated in sparse ad pricing evidence.
    With ``registry`` supplied, ``variant_factory_unaligned`` flags features that
    have no corresponding activated ads variant. They are *not silently lost*.
    Opt into ``alignment='ads'`` only when intentionally harmonizing both domains;
    it can collapse optional factory variants to DEFAULT.
    """
    if alignment not in ('preserve','ads'):
        raise ValueError("alignment must be 'preserve' or 'ads'")
    raw,status,sources=_extract_flags(df,factory_text_col=text_col)
    original_registry=registry
    if alignment=='preserve' or registry is None:
       k=df.apply(_group_key,axis=1)
       en={}
       for key,fl in zip(k,raw.to_numpy()):
           active=[f for f,v in zip(ALL_FEATURES,fl) if v]
           if active: en.setdefault(key,set()).update(active)
       registry={'version':'variant_v2','enabled':{g:sorted(v) for g,v in en.items()}}
    out=_apply(df,raw,status,sources,registry,factory=True,keep_factory_features=(alignment=='preserve'))
    if original_registry is not None:
       enabled={k:set(v) for k,v in original_registry['enabled'].items()}
       kk=df.apply(_group_key,axis=1)
       unmatched=[]
       for key,fl in zip(kk,raw.to_numpy()):
           labels=[f for f,v in zip(ALL_FEATURES,fl) if v]
           unmatched.append(int(any(f not in enabled.get(key,set()) for f in labels)))
       out['variant_factory_unaligned']=np.array(unmatched,dtype=np.int8)
    else:
       out['variant_factory_unaligned']=0
    return out
