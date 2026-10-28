import datetime as dt
import logging
import itertools
import numpy as np
import pandas as pd

from core.models import ModelIndexHistory, PriceSurface
from ._04_trainer_2 import ProductionCatBoostModel
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)


class MotorcyclePriceSurface:
    
    def __init__(
        self,
        crawler_db=None,
        production_db=None,
        db_target='crawler',
        batch_size=500
    ):
        self.today=dt.date.today()

        self.crawler_db=crawler_db
        self.production_db=production_db

        self.db_target=db_target
        self.batch_size = batch_size
        self.algo =  'Quantile:alpha=0.48'

        self._surface_df=None
        self._model_index_history_data=None

    def fit(self, df):
        price_baseline_features=[c for c in df.columns if c.startswith('price_')]
        parent_features=[c for c in df.columns if c.startswith('parent_')]
        baseline_features=[c for c in df.columns if c.startswith('baseline_')]+['adaptive_baseline','profile_support']
        momentum_features=[c for c in df.columns if c.startswith('momentum_')]
        variant_features = [c for c in df.columns if c.startswith('variant_')]

        num_features=['age','mileage','mileage_bucket','is_copy','is_real_price','condition_tier','is_original',
                      'technical_score','mileage_gap','mileage_gap_ratio','mileage_clean','annual_mileage_rate',
                      'mileage_age_consistency','mileage_confidence','mileage_is_suspicious','mileage_reference_median',
                      'mileage_reference_mad','mileage_reference_count','mileage_robust_z', 'flag_accessories']

        categorical_features=['color','real_brand','real_model','source','segment','model_variant','parent_id',
                              'baseline_monthly_level','baseline_weekly_level',]

        forbidden_features=['price','adaptive_residual','adaptive_residual_log','price_l1_monthly_median_residual_log',
                            'city','district','date','label_weight','variant_source']

        features=list(set(price_baseline_features+parent_features+baseline_features+momentum_features+
                          num_features+variant_features+categorical_features)-set(forbidden_features))


        self._features=features
        self._cat_features=categorical_features

        predictor=ProductionCatBoostModel(
            features=features,
            categorical_features=categorical_features,
            test_days=5,
            monotone_config={'positive':[],'negative':['age','mileage_bucket','condition_tier','is_copy']},
            cb_params={  'loss_function': self.algo, 'iterations': 3000, 'learning_rate':0.0018}  
        )

        predictor.evaluate(df)
        print("Global Metrics:\n",predictor.get_global_metrics())

        group_cols=['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']
        segment_metrics=predictor.get_segment_metrics(group_cols=group_cols)
        print("\nSegment Metrics:\n",segment_metrics.head())

        predictor.train_production(df)
        self._df=df
        self._predictor=predictor

        logger.info("Model Predictor FITTED Successfully.")
        return segment_metrics

    def _repair_monotonic(self,grid,group_cols,df=None,min_samples=3,max_adjust_pct=3.0,gap_pct=0.05,tol_pct=0.25,max_passes=3):
        grid=grid.copy()

        if 'model_price' not in grid:
            grid['model_price']=grid['predicted_price'].astype(float)
        grid['adjusted_price']=grid['model_price'].copy()

        level_keys=group_cols+['age','mileage_bucket','condition_tier']

        if df is not None:
            support=df.groupby(level_keys,dropna=False).size()
            idx=pd.MultiIndex.from_frame(grid[level_keys])
            grid['level_samples']=support.reindex(idx).fillna(0).to_numpy()
        elif 'level_samples' not in grid:
            grid['level_samples']=0

        fixed_factory=(
            grid['age'].eq(0) &
            grid['mileage_bucket'].eq(0) &
            grid['factory_anchor_price'].notna()
        )

        def _factory_price(i):
            v=pd.to_numeric(grid.loc[i,'factory_anchor_price'],errors='coerce')
            return float(v) if np.isfinite(v) and v>0 else np.nan

        def _target(ids,i,prev_v,next_v):
            cur=float(grid.loc[ids[i],'model_price'])

            if not np.isfinite(cur) or cur<=0:
                return np.nan

            # anomaly داخلی: interpolation بین دو همسایه
            if prev_v is not None and next_v is not None and prev_v>next_v>0:
                target=np.sqrt(prev_v*next_v)
                low=next_v*(1+gap_pct/100)
                high=prev_v*(1-gap_pct/100)
                return float(np.clip(target,low,high))

            # anomaly در لبه
            if prev_v is None and next_v is not None and next_v>0:
                return float(next_v*(1+gap_pct/100))

            if next_v is None and prev_v is not None and prev_v>0:
                return float(prev_v*(1-gap_pct/100))

            return np.nan

        axes=[
            ('age',['mileage_bucket','condition_tier']),
            ('mileage_bucket',['age','condition_tier'])
        ]

        for _ in range(max_passes):
            changed=False
            drop_ids=set()

            for order_col,other_cols in axes:
                keys=group_cols+other_cols

                for _,grp in grid.groupby(keys,dropna=False):
                    ids=grp.sort_values(order_col).index.tolist()

                    if len(ids)<2:
                        continue

                    for i in range(len(ids)):
                        idx=ids[i]
                        cur=float(grid.loc[idx,'model_price'])

                        if not np.isfinite(cur) or cur<=0 or fixed_factory.loc[idx]:
                            continue

                        prev_id=ids[i-1] if i>0 else None
                        next_id=ids[i+1] if i<len(ids)-1 else None

                        prev_v=(
                            _factory_price(prev_id)
                            if prev_id is not None and fixed_factory.loc[prev_id]
                            else float(grid.loc[prev_id,'model_price'])
                            if prev_id is not None else None
                        )

                        next_v=(
                            float(grid.loc[next_id,'model_price'])
                            if next_id is not None else None
                        )

                        # monotonic violation
                        violation=False

                        if prev_v is not None and cur>=prev_v*(1+tol_pct/100):
                            violation=True

                        if next_v is not None and cur<=next_v*(1-tol_pct/100):
                            violation=True

                        if not violation:
                            continue

                        target=_target(ids,i,prev_v,next_v)

                        if not np.isfinite(target) or target<=0:
                            continue

                        adjust_pct=abs(target-cur)/cur*100
                        samples=float(grid.loc[idx,'level_samples'])

                        # anomaly شدید یا کم‌نمونه → حذف
                        if adjust_pct>max_adjust_pct:
                            drop_ids.add(idx)
                            continue

                        # فقط خود anomaly تغییر کند
                        grid.loc[idx,'adjusted_price']=target
                        changed=True

            if drop_ids:
                grid=grid.drop(index=list(drop_ids))
                changed=True

            if not changed:
                break

        grid['adjustment_pct']=(
            (grid['adjusted_price']-grid['model_price'])/
            grid['model_price']*100
        )

        return grid

    def _apply_factory_and_monotonic(self,grid,df,max_age_days=30,min_zero_samples=7,max_zero_mape=5):
        group_cols=['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']

        grid=grid.copy()
        df=df.copy()

        grid['model_price']=grid['predicted_price'].astype(float)
        grid['adjusted_price']=grid['model_price'].copy()
        grid['factory_anchor_price']=np.nan
        grid['zero_km_mape']=np.nan
        grid['zero_km_samples']=0
        grid['cap_limited']=False

        # 1) آخرین Factory معتبر
        factory=df[
            df['source'].eq('factory') &
            df['age'].eq(0) &
            df['mileage'].eq(0) &
            df['price'].notna() &
            df['date'].notna()
        ].copy()

        if not factory.empty:
            cutoff=df['date'].max()-pd.Timedelta(days=max_age_days)
            factory=factory[factory['date'].ge(cutoff)].copy()

            if not factory.empty:
                factory=(
                    factory.sort_values('date')
                    .drop_duplicates(group_cols,keep='last')
                )
                factory_prices=factory.set_index(group_cols)['price']
                idx=pd.MultiIndex.from_frame(grid[group_cols])
                grid['factory_anchor_price']=factory_prices.reindex(idx).to_numpy()

        # 2) Zero-km دارای Factory
        zero=grid['age'].eq(0) & grid['mileage_bucket'].eq(0)
        zero_factory=zero & grid['factory_anchor_price'].notna()

        grid.loc[zero_factory,'adjusted_price']=grid.loc[
            zero_factory,'factory_anchor_price'
        ]

        # 3) Zero-km بدون Factory
        zero_no_factory=zero & grid['factory_anchor_price'].isna()
        zero_groups=grid.loc[zero_no_factory,group_cols].drop_duplicates()

        observed_zero=df[
            df['age'].eq(0) &
            df['mileage'].eq(0) &
            df['price'].notna() &
            df['price'].gt(0) &
            df['source'].ne('factory')
        ].copy()

        quality=pd.DataFrame(columns=group_cols+['zero_km_samples','zero_km_mape'])

        if not zero_groups.empty and not observed_zero.empty:
            zero_pred=grid.loc[
                zero_no_factory,group_cols+['model_price']
            ].drop_duplicates(group_cols)

            observed_zero=observed_zero.merge(
                zero_pred,on=group_cols,how='left'
            )
            observed_zero=observed_zero[
                observed_zero['model_price'].notna()
            ].copy()

            if not observed_zero.empty:
                observed_zero['APE']=(
                    np.abs(observed_zero['price']-observed_zero['model_price'])/
                    observed_zero['price']*100
                )
                quality=observed_zero.groupby(
                    group_cols,dropna=False
                ).agg(
                    zero_km_samples=('price','count'),
                    zero_km_mape=('APE','mean')
                ).reset_index()

        if not quality.empty:
            q=quality.set_index(group_cols)
            gidx=pd.MultiIndex.from_frame(grid[group_cols])

            grid['zero_km_samples']=q['zero_km_samples'].reindex(gidx).fillna(0).to_numpy()
            grid['zero_km_mape']=q['zero_km_mape'].reindex(gidx).to_numpy()

        # Zero-km بدون Factory و بدون عملکرد مناسب → حذف
        if not zero_groups.empty:
            quality_check=zero_groups.merge(
                quality,on=group_cols,how='left'
            )

            bad_zero_keys=quality_check[
                quality_check['zero_km_samples'].isna() |
                quality_check['zero_km_samples'].lt(min_zero_samples) |
                quality_check['zero_km_mape'].isna() |
                quality_check['zero_km_mape'].gt(max_zero_mape)
            ][group_cols]

            if not bad_zero_keys.empty:
                bad_idx=pd.MultiIndex.from_frame(bad_zero_keys)
                grid_idx=pd.MultiIndex.from_frame(grid[group_cols])

                grid=grid.loc[
                    ~(zero_no_factory & grid_idx.isin(bad_idx))
                ].copy()

        # 4) Factory فقط سقف بخش بالاتر از Factory
        #    سطوح زیر Factory کاملاً دست‌نخورده می‌مانند.
        gap=0.05/100

        def _apply_factory_chain():
            keys=group_cols+['mileage_bucket','condition_tier']

            groups=grid.loc[
                grid['age'].gt(0) &
                grid['factory_anchor_price'].notna()
            ].groupby(keys,dropna=False).groups.values()

            for idx_group in groups:
                ids=grid.loc[
                    list(idx_group)
                ].sort_values('age').index.tolist()

                if not ids:
                    continue

                anchor=float(grid.loc[
                    ids[0],'factory_anchor_price'
                ])

                if not np.isfinite(anchor) or anchor<=0:
                    continue

                ceiling=anchor*(1-gap)

                model=np.asarray(
                    grid.loc[ids,'model_price'],
                    dtype=float
                ).copy()

                # فقط levelهایی که خروجی خام مدل از Factory بالاتر است.
                above=model>ceiling

                if not above.any():
                    continue

                i=0

                while i<len(ids):
                    if not above[i]:
                        i+=1
                        continue

                    start=i

                    while i<len(ids) and above[i]:
                        i+=1

                    end=i-1
                    next_idx=i if i<len(ids) else None

                    raw=model[start:end+1]

                    if next_idx is not None:
                        # اولین level زیر Factory؛ این level و بعدی‌ها untouched
                        right=float(model[next_idx])

                        if not np.isfinite(right) or right<=0:
                            continue

                        k=len(raw)

                        # فاصله قابل استفاده بین Factory و level ثابت بعدی
                        available=max(ceiling/right-1,0.)

                        step_gap=min(
                            gap,
                            available/(k+1) if available>0 else 0.
                        )

                        right_target=right*(1+step_gap)

                        # اگر فضای کافی نبود، کوچک‌ترین فاصله ممکن
                        if right_target>=ceiling:
                            right_target=right+(ceiling-right)/(k+1)

                        if right_target<=right:
                            right_target=right*(1+1e-6)

                        # interpolation در فضای log بر اساس shape مدل
                        raw_start=raw[0]

                        if (
                            not np.isfinite(raw_start) or
                            raw_start<=0
                        ):
                            continue

                        raw_end=right

                        if raw_end<=0:
                            continue

                        denom=np.log(raw_end/raw_start)

                        if not np.isfinite(denom) or denom==0:
                            t=np.linspace(
                                0,1,k+1
                            )[1:]
                        else:
                            t=np.log(raw/raw_start)/denom

                        t=np.clip(t,0,1)
                        t=np.maximum.accumulate(t)

                        adjusted=np.exp(
                            np.log(ceiling)+
                            t*(
                                np.log(right_target)-
                                np.log(ceiling)
                            )
                        )

                        for idx,value in zip(
                            ids[start:end+1],
                            adjusted
                        ):
                            if np.isfinite(value) and value>0:
                                grid.loc[
                                    idx,'adjusted_price'
                                ]=value
                                grid.loc[
                                    idx,'cap_limited'
                                ]=True

                    else:
                        # اگر تا انتهای path بالاتر از Factory هست،
                        # از نسبت خروجی مدل به صورت level-by-level استفاده کن.
                        prev_model=raw[0]
                        prev_adjusted=ceiling

                        grid.loc[
                            ids[start],'adjusted_price'
                        ]=ceiling
                        grid.loc[
                            ids[start],'cap_limited'
                        ]=True

                        for j in range(1,len(raw)):
                            current_model=raw[j]

                            if (
                                not np.isfinite(current_model) or
                                current_model<=0 or
                                not np.isfinite(prev_model) or
                                prev_model<=0
                            ):
                                continue

                            ratio=current_model/prev_model
                            ratio=np.clip(
                                ratio,
                                1e-6,
                                1-gap
                            )

                            current_adjusted=prev_adjusted*ratio

                            grid.loc[
                                ids[start+j],
                                'adjusted_price'
                            ]=current_adjusted
                            grid.loc[
                                ids[start+j],
                                'cap_limited'
                            ]=True

                            prev_model=current_model
                            prev_adjusted=current_adjusted

        _apply_factory_chain()

        # grid=self._repair_monotonic(grid,group_cols,df=self._df,min_samples=3,max_adjust_pct=4.0)

        # 5) Factory zero-km immutable
        zero_factory=(
            grid['age'].eq(0) &
            grid['mileage_bucket'].eq(0) &
            grid['factory_anchor_price'].notna()
        )

        grid.loc[
            zero_factory,'adjusted_price'
        ]=grid.loc[
            zero_factory,'factory_anchor_price'
        ]

        grid['adjustment_pct']=(
            (grid['adjusted_price']-grid['model_price'])/
            grid['model_price']*100
        )

        return grid

    def _generate_and_predict_grid(self,model_index_history_data):

            group_cols=['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']
            grid_keys=['age','mileage_bucket','condition_tier']
            pool_cols=['motorcycle_model_id','real_model','is_copy','model_variant']

            df=self._df.merge(
                model_index_history_data[group_cols].drop_duplicates(),
                on=group_cols,how='inner'
            ).copy()

            feature_cols=[c for c in self._features if c not in group_cols+grid_keys]
            missing=[c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Grid source is missing required features: {missing}")

            cat_cols=[c for c in feature_cols if c in self._cat_features]
            num_cols=[c for c in feature_cols if c not in cat_cols]
            variant_cols=[c for c in feature_cols if c.startswith('variant_')]

            mileage_core=[c for c in [
                'mileage',
                'mileage_clean',
                'mileage_gap',
                'mileage_gap_ratio',
                'mileage_robust_z',
                'mileage_age_consistency',
                'annual_mileage_rate',
                'mileage_confidence',
                'price_per_km_proxy'
            ] if c in feature_cols]

            mileage_ref=[c for c in [
                'mileage_reference_median',
                'mileage_reference_mad',
                'mileage_reference_count'
            ] if c in feature_cols]

            mileage_flag=[c for c in [
                'mileage_is_suspicious'
            ] if c in feature_cols]

            mileage_cols=mileage_core+mileage_ref+mileage_flag

            condition_cols=[
                c for c in ['technical_score']
                if c in feature_cols
            ]

            context_cols=[
                c for c in feature_cols
                if c not in mileage_cols+condition_cols+variant_cols
            ]

            # ---------------------------------------------------------
            # helpers
            # ---------------------------------------------------------
            def _mode(x):
                x=x.dropna()
                return x.mode().iloc[0] if not x.empty else np.nan

            def _safe_num(x):
                return pd.to_numeric(x,errors='coerce')

            # ---------------------------------------------------------
            # 1) ساخت grid
            # ---------------------------------------------------------
            age_support=(
                df.groupby(
                    group_cols+['age'],
                    dropna=False
                )
                .size()
                .rename('n')
                .reset_index()
            )

            valid_ages=age_support[
                age_support['n'].ge(2)
            ]

            rows=[]

            for _,r in valid_ages.iterrows():

                g=[r[c] for c in group_cols]
                age=int(r['age'])

                for km,cond in itertools.product(
                    range(9),
                    [1,2,3]
                ):

                    if age==0:
                        if km>0 or cond!=1:
                            continue
                    elif km==0:
                        continue

                    rows.append(
                        g+[age,km,cond]
                    )

            grid=pd.DataFrame(
                rows,
                columns=group_cols+grid_keys
            )

            

            if grid.empty:
                return grid

            year_map=df.groupby(
                            group_cols+['age'],
                            dropna=False
                        )['production_year'].agg(_mode)
            
            year_idx=pd.MultiIndex.from_frame(
                grid[group_cols+['age']]
            )

            grid['production_year']=pd.Series(
                year_map.reindex(year_idx).to_numpy(),
                index=grid.index
            )

            # ---------------------------------------------------------
            # مقدار اولیه featureها
            # ---------------------------------------------------------
            for c in feature_cols:
                grid[c]=(
                    pd.Series(
                        pd.NA,
                        index=grid.index,
                        dtype='object'
                    )
                    if c in cat_cols else np.nan
                )

            # ---------------------------------------------------------
            # variant flags
            # ---------------------------------------------------------
            for c in variant_cols:
                token=c.replace('variant_','').upper()

                grid[c]=(
                    grid['model_variant']
                    .fillna('')
                    .astype(str)
                    .str.upper()
                    .str.contains(token,regex=False)
                    .astype('int8')
                )

            exact=group_cols+grid_keys
            age_cond=group_cols+['age','condition_tier']
            age_only=group_cols+['age']
            group_only=group_cols

            # =========================================================
            # 2) Mileage reference در سطح group + age
            # =========================================================
            if mileage_ref:

                ref=df.groupby(
                    age_only,
                    dropna=False
                )[mileage_ref].median()

                idx=pd.MultiIndex.from_frame(
                    grid[age_only]
                )

                for c in mileage_ref:
                    grid[c]=pd.Series(
                        ref[c].reindex(idx).to_numpy(),
                        index=grid.index
                    )

            # =========================================================
            # 3) Mileage profile اصلی
            #
            # local:
            #   group + age + bucket
            #
            # pool:
            #   model + real_model + copy + variant + age + bucket
            #
            # global:
            #   copy + age + bucket
            #
            # برای bucket missing:
            #   internal -> interpolation
            #   edge -> age-aware annual shape + local scale
            # =========================================================

            valid=df[
                df['age'].gt(0) &
                df['mileage_bucket'].between(1,8) &
                df['mileage'].notna()
            ].copy()

            if not valid.empty:

                valid['age']=valid['age'].astype(int)
                valid['mileage_bucket']=valid['mileage_bucket'].astype(int)

                valid['_annual_mileage']=(
                    valid['mileage']/
                    valid['age'].clip(lower=1)
                )

                # -----------------------------------------------------
                # global annual shape
                # -----------------------------------------------------
                annual_by_copy=(
                    valid.groupby(
                        ['is_copy','mileage_bucket'],
                        dropna=False
                    )['_annual_mileage']
                    .median()
                    .unstack('mileage_bucket')
                    .reindex(columns=range(1,9))
                )

                annual_by_copy=(
                    annual_by_copy
                    .interpolate(axis=1)
                    .ffill(axis=1)
                    .bfill(axis=1)
                )

                annual_global=annual_by_copy.median(axis=0)

                # -----------------------------------------------------
                # helper: complete یک profile از bucketهای مشاهده شده
                # -----------------------------------------------------
                def _complete_mileage_profile(series,age,is_copy):

                    p=pd.Series(
                        np.nan,
                        index=range(1,9),
                        dtype=float
                    )

                    if series is not None and not series.empty:

                        q=series.copy()

                        q.index=q.index.astype(int)

                        p.loc[
                            q.index.intersection(p.index)
                        ]=_safe_num(
                            q.loc[
                                q.index.intersection(p.index)
                            ]
                        )

                    observed=p.notna()

                    # هیچ sample محلی نداریم
                    if not observed.any():

                        try:
                            rates=annual_by_copy.loc[is_copy].to_numpy(
                                dtype=float
                            )
                        except Exception:
                            rates=annual_global.to_numpy(
                                dtype=float
                            )

                        return pd.Series(
                            rates*max(int(age),1),
                            index=range(1,9)
                        )

                    # -------------------------------------------------
                    # اول gap داخلی با interpolation
                    # -------------------------------------------------
                    p=p.interpolate(
                        method='index',
                        limit_area='inside'
                    )

                    # -------------------------------------------------
                    # edgeها با global bucket shape
                    # ولی scale شده با mileage خود گروه
                    # -------------------------------------------------
                    try:
                        rates=annual_by_copy.loc[is_copy].to_numpy(
                            dtype=float
                        )
                    except Exception:
                        rates=annual_global.to_numpy(
                            dtype=float
                        )

                    default=pd.Series(
                        rates*max(int(age),1),
                        index=range(1,9)
                    )

                    ratios=p[observed]/default[observed].replace(
                        0,np.nan
                    )

                    ratios=ratios.replace(
                        [np.inf,-np.inf],
                        np.nan
                    ).dropna()

                    scale=float(
                        ratios.median()
                    ) if not ratios.empty else 1.0

                    edge_default=default*scale

                    p=p.combine_first(edge_default)

                    return p

                # -----------------------------------------------------
                # local mileage matrix
                # -----------------------------------------------------
                local=(
                    valid.groupby(
                        group_cols+[
                            'age',
                            'mileage_bucket'
                        ],
                        dropna=False
                    )['mileage']
                    .median()
                    .unstack('mileage_bucket')
                    .reindex(columns=range(1,9))
                )

                # -----------------------------------------------------
                # pool mileage matrix
                # -----------------------------------------------------
                pool=(
                    valid.groupby(
                        pool_cols+[
                            'age',
                            'mileage_bucket'
                        ],
                        dropna=False
                    )['mileage']
                    .median()
                    .unstack('mileage_bucket')
                    .reindex(columns=range(1,9))
                )

                # -----------------------------------------------------
                # global age + copy profile
                # -----------------------------------------------------
                global_age=(
                    valid.groupby(
                        [
                            'is_copy',
                            'age',
                            'mileage_bucket'
                        ],
                        dropna=False
                    )['mileage']
                    .median()
                    .unstack('mileage_bucket')
                    .reindex(columns=range(1,9))
                )

                # -----------------------------------------------------
                # ساخت profile نهایی برای group + age
                # فقط روی تعداد group-age واقعی iteration می‌کنیم،
                # نه روی کل grid
                # -----------------------------------------------------
                local_complete=[]

                for key,series in local.iterrows():

                    is_copy=key[group_cols.index('is_copy')]
                    age=key[-1]

                    p=_complete_mileage_profile(
                        series,
                        age,
                        is_copy
                    )

                    rec=dict(
                        zip(
                            group_cols+['age'],
                            key
                        )
                    )

                    for b in range(1,9):
                        rec[f'km_{b}']=p.loc[b]

                    local_complete.append(rec)

                local_complete=pd.DataFrame(local_complete)

                # -----------------------------------------------------
                # pool complete profiles
                # -----------------------------------------------------
                pool_complete=[]

                for key,series in pool.iterrows():

                    is_copy=key[pool_cols.index('is_copy')]
                    age=key[-1]

                    p=_complete_mileage_profile(
                        series,
                        age,
                        is_copy
                    )

                    rec=dict(
                        zip(
                            pool_cols+['age'],
                            key
                        )
                    )

                    for b in range(1,9):
                        rec[f'km_{b}']=p.loc[b]

                    pool_complete.append(rec)

                pool_complete=pd.DataFrame(pool_complete)

                # -----------------------------------------------------
                # merge local
                # -----------------------------------------------------
                grid=grid.merge(
                    local_complete,
                    on=group_cols+['age'],
                    how='left'
                )

                # -----------------------------------------------------
                # merge pool
                # -----------------------------------------------------
                grid=grid.merge(
                    pool_complete,
                    on=pool_cols+['age'],
                    how='left',
                    suffixes=('','_pool')
                )

                # -----------------------------------------------------
                # برای هر row فقط bucket خودش انتخاب شود
                # -----------------------------------------------------
                bucket_num=grid['mileage_bucket'].astype(int)

                local_cols=[
                    f'km_{i}' for i in range(1,9)
                ]

                pool_cols_km=[
                    f'km_{i}_pool' for i in range(1,9)
                ]

                local_arr=grid[local_cols].to_numpy(dtype=float)
                pool_arr=grid[pool_cols_km].to_numpy(dtype=float)

                col_idx=bucket_num.to_numpy()-1
                row_idx=np.arange(len(grid))

                local_mileage=local_arr[
                    row_idx,
                    col_idx
                ]

                pool_mileage=pool_arr[
                    row_idx,
                    col_idx
                ]

                # -----------------------------------------------------
                # global age + copy
                # -----------------------------------------------------
                global_idx=pd.MultiIndex.from_frame(
                    grid[['is_copy','age']]
                )

                global_profile=(
                    global_age
                    .reindex(global_idx)
                )

                global_profile.index=grid.index

                global_arr=global_profile.to_numpy(
                    dtype=float
                )

                global_mileage=global_arr[
                    row_idx,
                    col_idx
                ]

                # -----------------------------------------------------
                # fallback annual default
                # -----------------------------------------------------
                copy_rates=annual_by_copy.reindex(
                    grid['is_copy']
                )

                copy_rates.index=grid.index

                copy_rates=copy_rates.fillna(
                    annual_global
                )

                default_matrix=(
                    copy_rates.to_numpy(dtype=float)*
                    grid['age'].to_numpy(dtype=float)[:,None]
                )

                default_mileage=default_matrix[
                    row_idx,
                    col_idx
                ]

                synthetic=local_mileage.copy()

                miss=np.isnan(synthetic)

                synthetic[miss]=pool_mileage[miss]

                miss=np.isnan(synthetic)

                synthetic[miss]=global_mileage[miss]

                miss=np.isnan(synthetic)

                synthetic[miss]=default_mileage[miss]

                grid['_synthetic_mileage']=synthetic

                # =====================================================
                # mileage core از یک مقدار واحد
                # =====================================================
                used=grid['age'].gt(0)

                m=pd.to_numeric(
                    grid['_synthetic_mileage'],
                    errors='coerce'
                )

                age_safe=grid['age'].clip(lower=1).astype(float)
                expected=grid['age'].astype(float)*7000.0

                if 'mileage' in feature_cols:
                    grid.loc[used,'mileage']=m.loc[used]

                if 'mileage_clean' in feature_cols:
                    grid.loc[used,'mileage_clean']=m.loc[used]

                if 'annual_mileage_rate' in feature_cols:
                    grid.loc[used,'annual_mileage_rate']=(
                        m.loc[used]/
                        age_safe.loc[used]
                    )

                if 'mileage_gap' in feature_cols:
                    grid.loc[used,'mileage_gap']=(
                        m.loc[used]-
                        expected.loc[used]
                    )

                if 'mileage_gap_ratio' in feature_cols:
                    grid.loc[used,'mileage_gap_ratio']=(
                        m.loc[used]/
                        expected.loc[used].replace(
                            0,np.nan
                        )
                    )

                # =====================================================
                # سایر mileage featureها
                #
                # exact bucket
                # -> pool bucket
                # -> global age+bucket
                # -> age
                # -> group
                #
                # اینها را بازسازی نمی‌کنیم چون فرمول اصلی‌شان
                # در preprocessing مشخص شده.
                # =====================================================
                other_cols=[
                    c for c in mileage_core
                    if c not in [
                        'mileage',
                        'mileage_clean',
                        'annual_mileage_rate',
                        'mileage_gap',
                        'mileage_gap_ratio'
                    ]
                ]

                other_cols=[
                    c for c in other_cols
                    if c in df.columns
                ]

                if other_cols:

                    exact_stats=df.groupby(
                        group_cols+grid_keys,
                        dropna=False
                    )[other_cols].median()

                    pool_stats=df.groupby(
                        pool_cols+grid_keys,
                        dropna=False
                    )[other_cols].median()

                    global_stats=df.groupby(
                        ['is_copy','age','mileage_bucket'],
                        dropna=False
                    )[other_cols].median()

                    age_stats=df.groupby(
                        group_cols+['age'],
                        dropna=False
                    )[other_cols].median()

                    group_stats=df.groupby(
                        group_cols,
                        dropna=False
                    )[other_cols].median()

                    exact_idx=pd.MultiIndex.from_frame(
                        grid[group_cols+grid_keys]
                    )

                    pool_idx=pd.MultiIndex.from_frame(
                        grid[pool_cols+grid_keys]
                    )

                    global_idx=pd.MultiIndex.from_frame(
                        grid[['is_copy','age','mileage_bucket']]
                    )

                    age_idx=pd.MultiIndex.from_frame(
                        grid[group_cols+['age']]
                    )

                    group_idx=pd.MultiIndex.from_frame(
                        grid[group_cols]
                    )

                    e=exact_stats.reindex(exact_idx)
                    e.index=grid.index

                    p=pool_stats.reindex(pool_idx)
                    p.index=grid.index

                    g=global_stats.reindex(global_idx)
                    g.index=grid.index

                    a=age_stats.reindex(age_idx)
                    a.index=grid.index

                    gr=group_stats.reindex(group_idx)
                    gr.index=grid.index

                    grid[other_cols]=(
                        e
                        .combine_first(p)
                        .combine_first(g)
                        .combine_first(a)
                        .combine_first(gr)
                    )

                # =====================================================
                # robust z
                # =====================================================
                if (
                    'mileage_robust_z' in feature_cols and
                    'mileage_reference_median' in grid.columns and
                    'mileage_reference_mad' in grid.columns
                ):

                    den=1.4826*pd.to_numeric(
                        grid['mileage_reference_mad'],
                        errors='coerce'
                    )

                    robust_z=(
                        m-
                        pd.to_numeric(
                            grid['mileage_reference_median'],
                            errors='coerce'
                        )
                    )/den.replace(0,np.nan)

                    grid.loc[
                        used,
                        'mileage_robust_z'
                    ]=robust_z.loc[used]

                # =====================================================
                # flag
                # =====================================================
                if mileage_flag:

                    flag_stats=df.groupby(
                        group_cols+grid_keys,
                        dropna=False
                    )[mileage_flag].median()

                    flag_idx=pd.MultiIndex.from_frame(
                        grid[group_cols+grid_keys]
                    )

                    fv=flag_stats.reindex(flag_idx)
                    fv.index=grid.index

                    for c in mileage_flag:
                        if c in fv.columns:
                            grid[c]=fv[c]

                    # fallback group+age
                    if grid[mileage_flag].isna().any().any():

                        fa=df.groupby(
                            group_cols+['age'],
                            dropna=False
                        )[mileage_flag].median()

                        ai=pd.MultiIndex.from_frame(
                            grid[group_cols+['age']]
                        )

                        av=fa.reindex(ai)
                        av.index=grid.index

                        for c in mileage_flag:
                            grid[c]=grid[c].fillna(av[c])

                # =====================================================
                # cleanup
                # =====================================================
                drop_cols=[
                    c for c in
                    local_cols+
                    pool_cols_km+
                    ['_synthetic_mileage']
                    if c in grid.columns
                ]

                grid.drop(
                    columns=drop_cols,
                    inplace=True,
                    errors='ignore'
                )

            # =========================================================
            # 4) fallback بسیار محدود mileage core
            # =========================================================
            if mileage_core:

                age_mileage=df.groupby(
                    age_only,
                    dropna=False
                )[mileage_core].median()

                age_idx=pd.MultiIndex.from_frame(
                    grid[age_only]
                )

                for c in mileage_core:
                    if c in grid.columns:
                        grid[c]=grid[c].fillna(
                            pd.Series(
                                age_mileage[c].reindex(age_idx).to_numpy(),
                                index=grid.index
                            )
                        )

                group_mileage=df.groupby(
                    group_only,
                    dropna=False
                )[mileage_core].median()

                group_idx=pd.MultiIndex.from_frame(
                    grid[group_only]
                )

                for c in mileage_core:
                    if c in grid.columns:
                        grid[c]=grid[c].fillna(
                            pd.Series(
                                group_mileage[c].reindex(group_idx).to_numpy(),
                                index=grid.index
                            )
                        )

            # =========================================================
            # 5) Condition / Context مثل نسخه working
            # =========================================================
            def _fill(cols,levels):

                cols=[
                    c for c in cols
                    if c in df.columns
                ]

                if not cols:
                    return

                agg={
                    c:(
                        _mode
                        if c in cat_cols
                        else 'median'
                    )
                    for c in cols
                }

                for keys in levels:

                    stats=df.groupby(
                        keys,
                        dropna=False
                    )[cols].agg(
                        {
                            c:agg[c]
                            for c in cols
                        }
                    )

                    vals=stats.reindex(
                        pd.MultiIndex.from_frame(
                            grid[keys]
                        )
                    )

                    vals.index=grid.index

                    for c in cols:
                        grid[c]=grid[c].fillna(
                            vals[c]
                        )

            _fill(
                condition_cols,
                [exact,age_cond,age_only,group_only]
            )

            _fill(
                context_cols,
                [age_only,group_only]
            )

            # =========================================================
            # 6) Color مثل نسخه working
            # =========================================================
            if 'color' in cat_cols:

                color_age=df.groupby(
                    group_cols+['age'],
                    dropna=False
                )['color'].agg(_mode)

                idx=pd.MultiIndex.from_frame(
                    grid[group_cols+['age']]
                )

                grid['color']=grid['color'].fillna(
                    pd.Series(
                        color_age.reindex(idx).to_numpy(),
                        index=grid.index
                    )
                )

                grid['color']=grid['color'].fillna(
                    _mode(df['color'])
                )

            # =========================================================
            # 7) fixed numeric flags
            # =========================================================
            for c in num_cols:

                grid[c]=pd.to_numeric(
                    grid[c],
                    errors='coerce'
                )

                if c=='is_original':
                    grid[c]=1-grid['is_copy']

                elif c=='is_real_price':
                    grid[c]=1

                elif c=='flag_accessories':
                    grid[c]=0

            # =========================================================
            # 8) categorical
            # =========================================================
            for c in cat_cols:
                grid[c]=(
                    grid[c]
                    .astype('string')
                    .fillna('unknown')
                    .astype(str)
                )

            # =========================================================
            # 9) prediction
            # =========================================================
            X_predict=grid[self._features].copy()

            for c in self._cat_features:
                if c in X_predict.columns:
                    X_predict[c]=X_predict[c].astype(str)

            grid['predicted_price']=self._predictor.get_predict(
                X_predict
            )

            # =========================================================
            # 10) factory + monotonic
            # =========================================================
            grid=self._apply_factory_and_monotonic(
                grid,
                df,
                max_age_days=30
            )

            # =========================================================
            # 11) metrics
            # =========================================================
            metrics_cols=[
                'MAPE(%)',
                'total_samples',
                'price_p10',
                'price_p50',
                'price_p90',
                'optimal_shift'
            ]

            metrics=model_index_history_data[
                group_cols+metrics_cols
            ].drop_duplicates(group_cols)

            return grid[
                group_cols+grid_keys+
                [
                    'production_year',
                    'model_price',
                    'adjusted_price',
                    'factory_anchor_price',
                    'zero_km_mape',
                    'zero_km_samples',
                    'cap_limited'
                ]
            ].merge(
                metrics,
                on=group_cols,
                how='left'
            )

    def _fit_optimal_shifts(self,eval_df,min_samples=7,max_shift=3,max_coverage_drop=2.5):
        group_cols=['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']

        x=eval_df.copy()
        x=x[x['price'].notna() & x['prediction'].notna() & (x['price']>0)].copy()

        # قیمت کارخانه صفرکیلومتر نباید در تعیین Shift مؤثر باشد.
        if 'source' in x.columns:
            factory_zero=(
                x['source'].eq('factory') &
                x['age'].eq(0) &
                x['mileage'].eq(0)
            )
            x=x.loc[~factory_zero].copy()

        x['ape']=np.abs(x['price']-x['prediction'])/x['price']*100

        def get_width(d):
            mape=d['ape'].mean()
            return max(1.,2.*mape) if mape<5 else 10.

        def find_shift(d):
            if d.empty:
                return 0.

            width=get_width(d)
            rows=[]

            for s in np.arange(0,-max_shift-.1,-0.5):
                center=d['prediction']*(1+s/100)
                lo=center*(1-width/200)
                hi=center*(1+width/200)

                total_cov=((d['price']>=lo)&(d['price']<=hi)).mean()*100
                floor_cov=(d['price']>=lo).mean()*100
                rows.append((s,total_cov,floor_cov))

            z=pd.DataFrame(
                rows,
                columns=['shift','coverage','floor_coverage']
            )

            base_cov=z.loc[z['shift'].eq(0),'coverage'].iloc[0]
            z=z[z['coverage']>=base_cov-max_coverage_drop]

            return float(
                z.sort_values(
                    ['floor_coverage','coverage','shift'],
                    ascending=[False,False,False]
                ).iloc[0]['shift']
            )

        global_shift=find_shift(x)
        rows=[]

        for keys,d in x.groupby(group_cols,dropna=False):
            if len(d)<min_samples:
                continue
            rows.append(list(keys)+[find_shift(d)])

        return pd.DataFrame(
            rows,
            columns=group_cols+['optimal_shift']
        ),global_shift

    def _generate_dynamic_bounds(self,row):
        p=pd.to_numeric(row.get('adjusted_price'),errors='coerce')
        mape=pd.to_numeric(row.get('MAPE(%)'),errors='coerce')
        zero_mape=pd.to_numeric(row.get('zero_km_mape'),errors='coerce')
        shift=pd.to_numeric(row.get('optimal_shift'),errors='coerce')

        if not np.isfinite(p) or p<=0:
            return pd.Series([np.nan,np.nan])

        is_zero=(
            row.get('age')==0 and
            row.get('mileage_bucket')==0
        )

        # Factory zero-km: بدون Shift و فقط ±0.5%
        if is_zero and pd.notna(row.get('factory_anchor_price')):
            return pd.Series([p*.995,p*1.005])

        # Non-factory zero-km: فقط با MAPE اختصاصی خودش
        if is_zero:
            if not np.isfinite(zero_mape):
                return pd.Series([np.nan,np.nan])

            width=min(10.,max(1.,2.*float(zero_mape)))
            center=p
            half=width/200.

            return pd.Series([
                center*(1-half),
                center*(1+half)
            ])

        # Used
        mape=float(mape) if np.isfinite(mape) else 13.
        shift=float(shift) if np.isfinite(shift) else 0.

        width=max(1.,2.*mape) if mape<5 else 10.
        center=p*(1+shift/100.)
        half=width/200.

        lower=center*(1-half)
        upper=center*(1+half)

        return pd.Series([
            max(0.,lower),
            max(lower,upper)
        ])

    def _evaluate_price_surface(self):
        join_keys=['motorcycle_model_id','real_brand','real_model','is_copy','model_variant',
                   'age','mileage_bucket','condition_tier']

        merged=pd.merge(
            self._df,
            self._surface_df[join_keys+['adjusted_price','suggested_lower','suggested_upper']],
            on=join_keys,how='inner'
        )

        merged['is_covered']=merged['price'].between(merged['suggested_lower'],merged['suggested_upper'])
        merged['APE']=np.abs(merged['price']-merged['adjusted_price'])/merged['price']*100

        merged['width_percent'] = (
            (merged['suggested_upper'] - merged['suggested_lower'])
            / merged['adjusted_price'] * 100
        )

        report=merged.groupby(
            ['motorcycle_model_id','real_brand','real_model','is_copy','model_variant'],
            dropna=False
        ).agg(
            tested_samples=('price','count'),
            coverage_percent=('is_covered',lambda x:x.mean()*100),
            mean_width_percent=('width_percent','mean'),
            mape=('APE','mean'),
            mdape=('APE','median')
        ).reset_index()

        return report

    def create_surface(self, segment_metrics):
        df=self._df[
            self._df.motorcycle_model_id.notna() &
            (self._df.is_copy==1-self._df.is_original)
        ].copy()

        last_month_threshold=df['date'].max()-pd.Timedelta(days=30)
        df['is_last_month']=df['date']>=last_month_threshold

        group_cols=['motorcycle_model_id','real_brand','real_model','is_copy','model_variant']

        price_metrics=df.groupby(group_cols, dropna=False).agg(
            total_samples=('price','count'),
            samples_last_month=('is_last_month','sum'),
            price_p10=('price',lambda x:x.quantile(.10)),
            price_p50=('price',lambda x:x.quantile(.50)),
            price_p90=('price',lambda x:x.quantile(.90))
        ).reset_index()

        model_index_history_data=pd.merge(price_metrics,segment_metrics,on=group_cols,how='left')

        shift_df,global_shift=self._fit_optimal_shifts(self._predictor.test_df)
        model_index_history_data=model_index_history_data.merge(shift_df,on=group_cols,how='left')
        model_index_history_data['optimal_shift']=model_index_history_data['optimal_shift'].fillna(global_shift)


        # ابتدا همه گروه‌ها Surface می‌گیرند؛ هنوز is_reliable اعمال نمی‌شود.
        surface_grid=self._generate_and_predict_grid(model_index_history_data)
        surface_grid[['suggested_lower','suggested_upper']]=surface_grid.apply(
            self._generate_dynamic_bounds,axis=1
        )
        self._surface_df=surface_grid.copy()

        surface_report=self._evaluate_price_surface()

        model_index_history_data=model_index_history_data.merge(
            surface_report[group_cols+['tested_samples','coverage_percent','mape','mdape']],
            on=group_cols,how='left'
        )

        model_index_history_data['is_reliable']=(
            model_index_history_data['coverage_percent'].ge(30) &
            model_index_history_data['mdape'].le(10) &
            model_index_history_data['mape'].le(15) &
            model_index_history_data['tested_samples'].ge(3) &
            model_index_history_data['samples_last_month'].ge(7)
        )

        self._model_index_history_data=model_index_history_data

        snapshot_keys=model_index_history_data[
            group_cols+['is_reliable']
        ].drop_duplicates(group_cols)

        self._surface_df=self._surface_df.merge(
            snapshot_keys,
            on=group_cols,
            how='inner'
        )

        self._surface_df=self._surface_df[
            self._surface_df['is_reliable']
        ].drop(columns='is_reliable')

        self._surface_df=self._surface_df.drop_duplicates(
            group_cols+['age','mileage_bucket','condition_tier']
        )

        logger.info("PRICE SURFACE FITTED Successfully.")
        return self._surface_df

    def _build_history_rows(self,target):
        df=self._model_index_history_data

        rows=[]

        for _,row in df.iterrows():
            item={
                'motorcycle_model_id':int(row['motorcycle_model_id']),
                'is_copy':bool(row['is_copy']),
                'model_variant':row['model_variant'],
                'price_low':row['price_p10'],
                'price_mid':row['price_p50'],
                'price_high':row['price_p90'],
                'mape':row['mape'],
                'sample_count':row['total_samples'],
                'is_reliable':bool(row['is_reliable']),
                'date':self.today,
                'color_coefficients':{},
                'algorithm':self.algo,
                'coverage_percent':row['coverage_percent'],
                'shift':row['optimal_shift']
            }

            if target=='crawler':
                item['real_brand']=row['real_brand']
                item['real_model']=row['real_model']

            rows.append(item)

        return rows

    def _save_history(self,db,target):

        history_model=ModelIndexHistory

        rows=self._build_history_rows(target)

        if not rows:
            return pd.DataFrame()

        conflict_cols=[
            'date',
            'motorcycle_model_id',
            'is_copy',
            'model_variant'
        ]

        if target=='crawler':
            conflict_cols=[
                'real_brand',
                'real_model',
                'date',
                'motorcycle_model_id',
                'is_copy',
                'model_variant'
            ]

        all_ids=[]

        for start in range(0,len(rows),self.batch_size):

            batch=rows[start:start+self.batch_size]

            stmt=pg_insert(history_model).values(batch)

            update_cols={
                'price_low':stmt.excluded.price_low,
                'price_mid':stmt.excluded.price_mid,
                'price_high':stmt.excluded.price_high,
                'mape':stmt.excluded.mape,
                'coverage_percent':stmt.excluded.coverage_percent,
                'shift':stmt.excluded.shift,
                'sample_count':stmt.excluded.sample_count,
                'algorithm':stmt.excluded.algorithm,
                'is_reliable':stmt.excluded.is_reliable,
                'color_coefficients':stmt.excluded.color_coefficients
            }

            stmt=stmt.on_conflict_do_update(
                index_elements=conflict_cols,
                set_=update_cols
            )

            result=db.execute(
                stmt.returning(
                    history_model.id,
                    history_model.motorcycle_model_id,
                    history_model.is_copy,
                    history_model.model_variant,
                    *(
                        [
                            history_model.real_brand,
                            history_model.real_model
                        ]
                        if target=='crawler'
                        else []
                    )
                )
            )

            all_ids.extend(
                result.mappings().all()
            )

        return pd.DataFrame(all_ids).rename(
            columns={'id':'snapshot_id'}
        )

    def _build_surface_rows(self,snapshot_ids,target):

        df=self._surface_df.copy()

        join_cols=[
            'motorcycle_model_id',
            'is_copy',
            'model_variant'
        ]

        if target=='crawler':
            join_cols += [
                'real_brand',
                'real_model'
            ]

        df=df.merge(
            snapshot_ids,
            on=join_cols,
            how='left'
        )

        if df['snapshot_id'].isna().any():
            raise ValueError(
                'Some Surface rows have no snapshot_id.'
            )

        rows=[]

        for _,row in df.iterrows():

            item={
                'snapshot_id':int(row['snapshot_id']),
                'motorcycle_model_id':int(row['motorcycle_model_id']),
                'year':int(row['production_year']),
                'mileage_bucket':int(row['mileage_bucket']),
                'condition_tier':int(row['condition_tier']),
                'model_variant':row['model_variant'],
                'is_copy':bool(row['is_copy']),
                'color':'مشکی',
                'price_low':int(round(row['suggested_lower'])),
                'price_mid':int(round(row['adjusted_price'])),
                'price_high':int(round(row['suggested_upper']))
            }

            if target=='crawler':
                item['real_brand']=row['real_brand']
                item['real_model']=row['real_model']

            rows.append(item)

        return rows
    
    def _save_surface(self,db,target,snapshot_ids):

        surface_model=PriceSurface
            
        rows=self._build_surface_rows(
            snapshot_ids,
            target
        )

        if not rows:
            return


        motorcycle_ids=[
            int(x)
            for x in self._surface_df[
                'motorcycle_model_id'
            ]
            .dropna()
            .unique()
        ]

        db.execute(
            delete(surface_model).where(
                surface_model.motorcycle_model_id.in_(
                    motorcycle_ids
                )
            )
        )

        for start in range(0,len(rows),self.batch_size):

            batch=rows[
                start:start+self.batch_size
            ]

            db.execute(
                pg_insert(surface_model),
                batch
            )
    
    def _save_target(self,db,target):

        try:

            snapshot_ids=self._save_history(
                db,
                target
            )

            self._save_surface(
                db,
                target,
                snapshot_ids
            )

            db.commit()

            logger.info(
                f"{target.upper()} DATABASE UPDATED SUCCESSFULLY."
            )

        except Exception:

            db.rollback()

            logger.exception(
                f"{target.upper()} DATABASE UPDATE FAILED. ROLLED BACK."
            )

            raise
    
    def update_database(self):

        if self._surface_df is None:
            raise RuntimeError(
                "Call fit() and create_surface() first."
            )

        if self.db_target=='crawler':

            self._save_target(
                self.crawler_db,
                'crawler'
            )

        elif self.db_target=='production':

            self._save_target(
                self.production_db,
                'production'
            )

        elif self.db_target=='both':

            self._save_target(
                self.crawler_db,
                'crawler'
            )

            self._save_target(
                self.production_db,
                'production'
            )

        else:
            raise ValueError(
                "db_target must be "
                "'crawler', 'production', or 'both'."
            )
