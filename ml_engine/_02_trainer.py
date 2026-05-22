import datetime
import jdatetime
from dictionary.mappings import IRANIAN_BRANDS
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import QuantileRegressor
from sqlalchemy.orm import Session
from core.models import ModelIndexHistory, PriceSurface

# --- Constants ---
LOOKBACK_DAYS = 25
MIN_MODEL_ROWS = 15
CURRENT_MARKET_YEAR = jdatetime.date.today().year
EXPECTED_KM_PER_YEAR = 7000
VALIDATION_RATIO = 0.2
MIN_VALID_ROWS = 3
MIN_FIT_ROWS = 5
RELIABLE_MAPE = 12.0

BUCKET_TYPICAL_MILEAGE = {
    0: 0, 1: 500, 2: 3000, 3: 7500, 4: 15000, 
    5: 23500, 6: 40000, 7: 80000, 8: 105000,
}

def _apply_dynamic_margin(mid_price: float, bucket: int):
    """اعمال حاشیه تجاری داینامیک بر اساس کارکرد برای تولید بازه قیمت"""
    if bucket == 0: margin = 0.02
    elif bucket <= 3: margin = 0.04
    else: margin = 0.05

    low = int(mid_price * (1.0 - margin))
    high = int(mid_price * (1.0 + margin))
    return low, int(mid_price), high

class PriceModelTrainer:
    def __init__(self, df: pd.DataFrame, db_session: Session, verbose=True):
        self.df = df.copy()
        self.db = db_session
        self.verbose = verbose
        # self.today = jdatetime.date.today() 
        self.today = datetime.date.today()
        self.color_coefficients = {}

    def log(self, msg):
        if self.verbose:
            print(msg)

    def _row_weights(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty: return np.array([], dtype=float)
        
        max_date = frame["publish_date"].max()
        age_days = frame["publish_date"].apply(lambda d: (max_date - d).days)
        
        age_days = age_days.clip(lower=0, upper=LOOKBACK_DAYS)
        
        date_weights = 1.0 + (LOOKBACK_DAYS - age_days) * 0.08
        
        guess_correct_weight = np.where(frame["is_system_guess_correct"].fillna(0).astype(int) == 1, 1.5, 1.0)
        tech_scores = frame["technical_score"].fillna(5).astype(float)
        quality_weights = 1.0 + (tech_scores - 5) * 0.1 
        
        return (date_weights.to_numpy() * guess_correct_weight * quality_weights).astype(float)

    def _extract_color_coefficients(self) -> dict:
        """استخراج ضریب قیمتی هر رنگ نسبت به مشکی/سفید برای کل بازار"""
        work_df = self.df.copy()
        work_df["model_median_price"] = work_df.groupby(["real_brand", "real_model"])["price"].transform("median")
        work_df["price_ratio"] = work_df["price"] / (work_df["model_median_price"] + 1)
        
        color_stats = work_df.groupby("color")["price_ratio"].median().to_dict()
        base_ratio = color_stats.get("مشکی", color_stats.get("سفید", 1.0))
        if base_ratio == 0: base_ratio = 1.0
        
        color_coeffs = {}
        for c, r in color_stats.items():
            coeff = max(0.95, min(1.05, r / base_ratio))
            color_coeffs[c] = round(coeff, 3)
            
        return color_coeffs

    def _split_data(self, group_df: pd.DataFrame, test_ratio=0.1):
        """تقسیم داده‌ها بر اساس زمان برای تست مدل"""
        group_df = group_df.sort_values("publish_date").copy()
        n_test = max(3, int(round(len(group_df) * test_ratio)))
        if len(group_df) - n_test < 5: n_test = max(1, len(group_df) - 5)
        if n_test <= 0 or len(group_df) - n_test < 5: return None, None
        return group_df.iloc[:-n_test].copy(), group_df.tail(n_test).copy()

    def _train_global_models(self):
        """آموزش مدل جامع CatBoost روی کل دیتای بازار"""
        self.log("[TRAIN] Training Global CatBoost Model...")
        
        global_features = [
            "real_brand", "real_model", "color", "age", "mileage", 
            "clutch_type", "brake_type", "start_type", "engine_condition", 
            "body_condition", "document_status", "is_copy", "seller_type", 
            "technical_score",'engine_volume',
            "m_cat_zero_dry", "m_cat_near_zero", "m_cat_very_low", "m_cat_low", 
            "m_cat_low_mid", "m_cat_mid_low", "m_cat_mid", "m_cat_high", "m_cat_very_high", 
            "expected_mileage", "mileage_gap", "mileage_gap_ratio", "price_per_km_proxy",
            "flag_clean", "flag_accessories", "flag_new_consumables", "flag_first_owner",
            "flag_new", "flag_white_doc", "flag_full_docs", "flag_incomplete_docs",
            "flag_insurance", "flag_accident", "flag_engine_issue", "flag_installment",
            "flag_swap", "flag_urgent", "flag_service", "title_len", "desc_len", "segment"
        ]
        cat_cols = ["real_brand", "real_model", "color", "segment", "clutch_type", "brake_type", "start_type",\
                     "engine_condition", "body_condition", "document_status", "seller_type"]

        X = self.df[global_features].copy()
        for c in cat_cols: X[c] = X[c].astype(str)
        y = np.log1p(self.df["price"].to_numpy(dtype=float))
        w = self._row_weights(self.df)

        cat_idx = [X.columns.get_loc(c) for c in cat_cols]
        pool = Pool(X, y, cat_features=cat_idx, weight=w)

        # قید اکید یکنوایی (Monotonic Constraints) برای جلوگیری از باگ‌های قیمتی
        model = CatBoostRegressor(
            loss_function="MAE", 
            iterations=500, depth=6, learning_rate=0.05, random_seed=42, verbose=False,
            monotone_constraints={"age": -1, "mileage": -1, "expected_mileage": -1, "mileage_gap": -1}
        )
        model.fit(pool)

        return {"features": global_features, "cat_cols": cat_cols, "mid": model}

    def _train_local_models(self, train_df: pd.DataFrame):
        """آموزش مدل خطی محلی برای هر مدل خاص"""
        local_features = [
            "age", "m_cat_zero_dry", "m_cat_near_zero", "m_cat_very_low", "m_cat_low", 
            "m_cat_low_mid", "m_cat_mid_low", "m_cat_mid", "m_cat_high", "m_cat_very_high"
        ]
        X = train_df[local_features].copy()
        y = np.log1p(train_df["price"].to_numpy(dtype=float))
        w = self._row_weights(train_df)

        model = QuantileRegressor(quantile=0.50, alpha=1e-4, solver="highs")
        try:
            model.fit(X, y, sample_weight=w)
        except TypeError:
            model.fit(X, y)

        return {"features": local_features, "mid": model}

    @staticmethod
    def _metrics(y_true, mid):
        mape = float(np.mean(np.abs(y_true - mid) / np.maximum(y_true, 1)) * 100)
        return {"mape": mape, "median_pred": float(np.median(mid))}

    def _predict_mid(self, algorithm, global_models, local_models, df, local_weight=None):
        if algorithm in ["global", "hybrid"]:
            X_g = df[global_models["features"]].copy()
            for c in global_models["cat_cols"]: X_g[c] = X_g[c].astype(str)
            g_log = global_models["mid"].predict(X_g)
        
        if algorithm in ["local", "hybrid"]:
            X_l = df[local_models["features"]].copy()
            l_log = local_models["mid"].predict(X_l)

        if algorithm == "global": final_log = g_log
        elif algorithm == "local": final_log = l_log
        else: final_log = (1.0 - local_weight) * g_log + local_weight * l_log

        return np.expm1(final_log)

    def _select_best_algorithm(self, global_models, fit_df, val_df):
        candidates = []
        y_val_true = val_df["price"].to_numpy(dtype=float)

        try:
            g_mid = self._predict_mid("global", global_models, None, val_df)
            candidates.append({"algorithm": "global", "local_weight": None, "mape": self._metrics(y_val_true, g_mid)["mape"], "local_models": None})
        except Exception as e:
            self.log(f"[WARN] Global validation failed: {e}")

        local_models_fit = None
        try:
            local_models_fit = self._train_local_models(fit_df)
            l_mid = self._predict_mid("local", None, local_models_fit, val_df)
            candidates.append({"algorithm": "local", "local_weight": None, "mape": self._metrics(y_val_true, l_mid)["mape"], "local_models": local_models_fit})
        except Exception as e:
            self.log(f"[WARN] Local validation failed: {e}")

        if local_models_fit is not None:
            for lw in [0.25, 0.50, 0.75]:
                try:
                    hy_mid = self._predict_mid("hybrid", global_models, local_models_fit, val_df, lw)
                    candidates.append({"algorithm": "hybrid", "local_weight": lw, "mape": self._metrics(y_val_true, hy_mid)["mape"], "local_models": local_models_fit})
                except Exception:
                    pass

        if not candidates: return None
        candidates.sort(key=lambda x: x["mape"])
        return candidates[0]

    def _update_db_surface(self, brand, model_name, test_mape, sample_count, y_preds, algo, model_df, global_models, local_models, local_weight):
        """ذخیره دیتای ارزیابی و تولید دیتای مصنوعی (Surface) در دیتابیس"""
        overall_mid = int(np.median(y_preds)) if len(y_preds) else 0
        
        history_obj = self.db.query(ModelIndexHistory).filter_by(
            real_brand=brand, real_model=model_name, date=self.today
        ).first()

        if not history_obj:
            history_obj = ModelIndexHistory(
                real_brand=brand, real_model=model_name, date=self.today
            )
            self.db.add(history_obj)

        history_obj.price_mid = overall_mid
        history_obj.price_low = int(overall_mid * 0.95)
        history_obj.price_high = int(overall_mid * 1.05)
        history_obj.mape = test_mape
        history_obj.is_reliable = (test_mape <= RELIABLE_MAPE)
        history_obj.sample_count = sample_count
        history_obj.color_coefficients = self.color_coefficients
        history_obj.algorithm = algo
        
        self.db.commit()

        if not history_obj.is_reliable:
            return {"created": 0, "status": "unreliable"}

        self.db.query(PriceSurface).filter_by(
            real_brand=brand, real_model=model_name
        ).delete()

        # 3. تولید دیتای مصنوعی
        target_years = sorted(model_df["production_year"].dropna().astype(int).unique().tolist())
        if not target_years: return {"created": 0, "status": "no_years"}
        
        # استخراج مقادیر متداول این مدل (Mode)
        typical = {
            "title_len": model_df["title_len"].median(),
            "desc_len": model_df["desc_len"].median(),
            "real_brand": brand,
            "real_model": model_name,
            "segment": model_df["segment"].iloc[0],
            # مقداردهی دیفالت به فیچرهای جدید
            "clutch_type": model_df["clutch_type"].mode()[0] if not model_df["clutch_type"].mode().empty else "نامشخص",
            "brake_type": model_df["brake_type"].mode()[0] if not model_df["brake_type"].mode().empty else "نامشخص",
            "start_type": model_df["start_type"].mode()[0] if not model_df["start_type"].mode().empty else "نامشخص",
            "engine_condition": model_df["engine_condition"].mode()[0] if not model_df["engine_condition"].mode().empty else "سالم",
            "body_condition": model_df["body_condition"].mode()[0] if not model_df["body_condition"].mode().empty else "سالم",
            "document_status":  model_df["document_status"].mode()[0] if not model_df["document_status"].mode().empty else "مدارک کامل",
            "is_copy": 1 if brand in IRANIAN_BRANDS else 0,
            "seller_type": "private",
            "technical_score": 5,
            'engine_volume': model_df["engine_volume"].mode()[0] if not model_df["engine_volume"].mode().empty else 0
        }
        
        flags = [col for col in model_df.columns if col.startswith("flag_")]
        for f in flags:
            typical[f] = int(model_df[f].mode()[0]) if not model_df.empty else 0
        typical["flag_clean"] = 1 
        synthetic_rows = []
        for y in target_years:
            for b in range(9):
                age = max(0, CURRENT_MARKET_YEAR - y)
                mileage = BUCKET_TYPICAL_MILEAGE.get(b, 20000)
                exp_mileage = age * EXPECTED_KM_PER_YEAR
                
                row = {
                    "production_year": y, "mileage_bucket": b,
                    "age": age, "mileage": mileage,
                    "expected_mileage": exp_mileage,
                    "mileage_gap": mileage - exp_mileage,
                    "mileage_gap_ratio": mileage / (exp_mileage + 1.0),
                    "price_per_km_proxy": np.log1p(mileage) / (age + 1.0),
                    "color": "مشکی", 
                }
                
                for i, name in enumerate(["zero_dry", "near_zero", "very_low", "low", "low_mid", "mid_low", "mid", "high", "very_high"]):
                    row[f"m_cat_{name}"] = 1 if b == i else 0
                
                row.update(typical)
                if b == 0: row["flag_new"] = 1
                synthetic_rows.append(row)

        syn_df = pd.DataFrame(synthetic_rows)
        preds_mid = np.array(self._predict_mid(algo, global_models, local_models, syn_df, local_weight))
        
        # ==============================================================
        # اعمال الگوریتم PAVA (ایزوتونیک) برای حفظ منطق قیمت-کارکرد
        # ==============================================================
        for y in target_years:
            mask = syn_df["production_year"] == y
            year_indices = syn_df[mask].sort_values("mileage_bucket").index.tolist()
            
            for i in range(len(year_indices) - 2, -1, -1):
                curr_idx = year_indices[i]     
                next_idx = year_indices[i+1]   
                if preds_mid[next_idx] > preds_mid[curr_idx]:
                    preds_mid[curr_idx] = preds_mid[next_idx]
        # ==============================================================

        # ساخت آبجکت‌های دیتابیس برای Insert گروهی (Bulk)
        surface_objs = []
        for idx, row in syn_df.iterrows():
            low_p, mid_p, high_p = _apply_dynamic_margin(preds_mid[idx], row["mileage_bucket"])
            
            surface_objs.append(PriceSurface(
                snapshot_id=history_obj.id,
                real_brand=brand,
                real_model=model_name,
                year=int(row["production_year"]),
                mileage_bucket=int(row["mileage_bucket"]),
                color="مشکی", 
                price_low=low_p,
                price_mid=mid_p,
                price_high=high_p,
            ))

        # اینزرت سریع و گروهی در دیتابیس
        self.db.bulk_save_objects(surface_objs)
        self.db.commit()
        
        return {"created": len(surface_objs), "status": "ok"}

    def execute_pipeline(self):
        if self.df.empty: return {"status": "no_data"}
        self.log(f"[INFO] Initial rows for training: {len(self.df)}")

        self.color_coefficients = self._extract_color_coefficients()
        global_models = self._train_global_models()

        results = []
        
        # پردازش مدل به مدل
        for (brand, model_name), model_df in self.df.groupby(["real_brand", "real_model"]):
            if len(model_df) < MIN_MODEL_ROWS: continue

            train_df, test_df = self._split_data(model_df, test_ratio=0.2)
            if train_df is None: continue

            fit_df, val_df = self._split_data(train_df, test_ratio=VALIDATION_RATIO)
            if fit_df is None: continue

            selection = self._select_best_algorithm(global_models, fit_df, val_df)
            if not selection: continue

            algo = selection["algorithm"]
            lw = selection["local_weight"]
            final_local = self._train_local_models(train_df) if algo != "global" else None

            # تست نهایی
            y_test_true = test_df["price"].to_numpy(dtype=float)
            y_test_pred = self._predict_mid(algo, global_models, final_local, test_df, lw)
            test_mape = self._metrics(y_test_true, y_test_pred)["mape"]

            # ذخیره در دیتابیس Worker
            surf_res = self._update_db_surface(
                brand, model_name, test_mape, len(model_df), y_test_pred, 
                algo, model_df, global_models, final_local, lw
            )
            
            self.log(f"[DONE] {brand} {model_name} | MAPE: {test_mape:.1f}% | Surfaces: {surf_res['created']}")
            results.append({"brand": brand, "model": model_name, "algo": algo, "mape": test_mape})

        return {"status": "success", "processed_models": len(results), "details": results}
    