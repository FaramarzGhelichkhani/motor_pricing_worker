import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

class ProductionCatBoostModel:
    def __init__(
        self, 
        features, 
        categorical_features, 
        target='price_l1_monthly_median_residual_log', 
        test_days=10, 
        cb_params=None, 
        monotone_config=None
    ):
        """
        کلاس مدیریت مدل CatBoost برای محیط پروداکشن.
        
        monotone_config: دیکشنری برای تعیین جهت فیچرها. مثال:
            {'positive': ['exact_momentum_7d_30d'], 'negative': ['mileage', 'age']}
        """
        self.features = features
        self.categorical_features = categorical_features
        self.target = target
        self.test_days = test_days
        self.cb_params = cb_params or {}
        
        # تنظیمات پیش‌فرض Constraints
        self.monotone_config = monotone_config or {
            'positive': [],
            'negative': ['mileage', 'mileage_bucket','age','condition_tier', 'is_copy']
        }
        
        self.eval_model = None
        self.prod_model = None
        self.test_df = None
        self.best_iteration = None

    def _prepare_data(self, df):
        """اعمال پیش‌پردازش‌های اولیه روی داده‌ها (تبدیل تایپ و مدیریت مقادیر نال)"""
        df = df.sort_values("date").reset_index(drop=True).copy()
        df = df.dropna(subset=self.target)
        
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("__MISSING__").astype(str)
                
        for col in self.features:
            if col not in self.categorical_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df

    def _calculate_weights(self, df):
        """محاسبه وزن ترکیبی (زمان + لیبل) برای آموزش"""
        # وزن لیبل
        label_weight = (
            pd.to_numeric(df["label_weight"], errors="coerce")
            .fillna(1.0)
            .clip(lower=0.0)
            .values
        ) if "label_weight" in df.columns else np.ones(len(df))
        
        # وزن زمان
        max_date = df["date"].max()
        age_days = (max_date - df["date"]).dt.days.clip(lower=0)
        max_age = max(1, age_days.max())
        time_weight = ((max_age + 1 - age_days) / (max_age + 1)).astype(float).values
        
        return time_weight * label_weight

    def _get_monotone_constraints(self):
        """ساخت دیکشنری Monotone Constraints فقط برای فیچرهای موجود"""
        constraints = {}
        for feature in self.features:
            if feature in self.monotone_config.get('positive', []):
                constraints[feature] = 1
            elif feature in self.monotone_config.get('negative', []):
                constraints[feature] = -1
        return constraints

    def _get_default_params(self):
        """پارامترهای پایه مدل"""
        params = {
            'loss_function': 'Quantile:alpha=0.50',  
            'eval_metric': 'MAE',
            'iterations': 6000, 
            'learning_rate': 0.0021, 
            'depth': 5,
            'l2_leaf_reg': 4, 
            'random_strength': 0.4079, 
            'bagging_temperature': 0.518,
            'border_count': 128, 
            'random_seed': 42,
            'early_stopping_rounds':200,
            'monotone_constraints': self._get_monotone_constraints()
        }
        params.update(self.cb_params)
        print(params['loss_function'])
        print(params['iterations'])
        return params

    def evaluate(self, df):
        """
        تقسیم داده‌ها به ترین، ولیدیشن و تست، آموزش مدل با early_stopping 
        و ذخیره بهترین نتایج برای تست عملکرد.
        """
        df = self._prepare_data(df)
        
        max_date = df["date"].max()
        test_cutoff = max_date - pd.Timedelta(days=self.test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=self.test_days)
        
        train_df = df[df["date"] <= val_cutoff].copy()
        val_df = df[(df["date"] > val_cutoff) & (df["date"] <= test_cutoff)].copy()
        self.test_df = df[df["date"] > test_cutoff].copy()
        
        train_pool = Pool(
            data=train_df[self.features], 
            label=train_df[self.target], 
            weight=self._calculate_weights(train_df), 
            cat_features=self.categorical_features
        )
        
        val_pool = Pool(
            data=val_df[self.features], 
            label=val_df[self.target], 
            weight=self._calculate_weights(val_df),
            cat_features=self.categorical_features
        )
        
        params = self._get_default_params()
        params['early_stopping_rounds'] = 200
        params['verbose'] = 500

        self.eval_model = CatBoostRegressor(**params)
        self.eval_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # ذخیره بهترین تعداد دور برای آموزش نهایی پروداکشن
        self.best_iteration = self.eval_model.get_best_iteration()
        
        # پیش‌بینی روی تست برای محاسبه متریک‌ها
        self.test_df['prediction'] =  np.exp( self.eval_model.predict(self.test_df[self.features]) ) * self.test_df["price_l1_monthly_median"]
        
        print(f"\n[INFO] Evaluation completed. Best iteration found: {self.best_iteration}")

    def train_production(self, df):
        """
        آموزش مدل نهایی روی *کل* داده‌ها.
        در اینجا از best_iteration به‌دست‌آمده از مرحله ارزیابی استفاده می‌شود.
        """
        df = self._prepare_data(df)
        
        train_pool = Pool(
            data=df[self.features], 
            label=df[self.target], 
            weight=self._calculate_weights(df), 
            cat_features=self.categorical_features
        )
        
        params = self._get_default_params()
        
        # چون ولیدیشن نداریم، early_stopping را حذف می‌کنیم و 
        # ایتریشن را روی بهترین ایتریشن مرحله ارزیابی تنظیم می‌کنیم
        if self.best_iteration:
            params['iterations'] = self.best_iteration
            
        # حذف کلیدهای مربوط به early stopping
        params.pop('early_stopping_rounds', None)
        params['verbose'] = 500
        
        self.prod_model = CatBoostRegressor(**params)
        self.prod_model.fit(train_pool)
        
        print("\n[INFO] Production model trained successfully on ALL data.")
        return self.prod_model

    @staticmethod
    def _calculate_metrics_logic(df, target_col, pred_col):
        df = df[[target_col, pred_col]].dropna()
        if len(df) == 0:
            return pd.Series(dtype=float)

        y_true = df[target_col].to_numpy(dtype=float)
        y_pred = df[pred_col].to_numpy(dtype=float)

        error = y_pred - y_true
        abs_error = np.abs(error)
        ape = abs_error / np.maximum(np.abs(y_true), 1e-8) * 100

        downside = np.maximum(error / np.maximum(np.abs(y_pred), 1e-8), 0) * 100

        return pd.Series({
            "n_samples": len(df),
            "MdAPE(%)": np.median(ape),
            "MAPE(%)": np.mean(ape),
            "MedAE": np.median(abs_error),
            "MAE": np.mean(abs_error),
            "P90_AE": np.percentile(abs_error, 90),
            "P95_AE": np.percentile(abs_error, 95),
            "P90_APE(%)": np.percentile(ape, 90),
            "P95_APE(%)": np.percentile(ape, 95),
            "Downside_Rate(%)": np.mean(downside > 0) * 100,
            "Downside_P50(%)": np.percentile(downside, 50),
            "Downside_P75(%)": np.percentile(downside, 75),
            "Downside_P90(%)": np.percentile(downside, 90),
        })

    def get_global_metrics(self):
        """دریافت متریک‌های عملکرد روی کل داده‌های تست"""
        if self.test_df is None or 'prediction' not in self.test_df:
            raise ValueError("You must call `evaluate()` before getting metrics.")
            
        return self._calculate_metrics_logic(self.test_df, 'price', 'prediction').to_frame().T

    def get_segment_metrics(self, group_cols):
        """
        دریافت متریک‌های عملکرد به تفکیک سگمنت‌های دلخواه.
        مثال: group_cols = ['real_brand', 'real_model']
        """
        if self.test_df is None or 'prediction' not in self.test_df:
            raise ValueError("You must call `evaluate()` before getting metrics.")
            
        # گروه‌بندی و محاسبه متریک‌ها برای هر گروه
        metrics_df = self.test_df[self.test_df.is_copy == 1 - self.test_df.is_original].groupby(group_cols).apply(
            lambda g: self._calculate_metrics_logic(g, 'price', 'prediction')
        ).reset_index()
        
        return metrics_df.sort_values("n_samples", ascending=False).reset_index(drop=True)

    def get_predict(self, X):
        pred =  self.prod_model.predict(X)
        price_pred = np.exp(pred) * X["price_l1_monthly_median"]
        return price_pred
        