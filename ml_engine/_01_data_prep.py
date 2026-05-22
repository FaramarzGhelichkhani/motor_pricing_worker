import jdatetime
import pandas as pd
import numpy as np
import datetime as dt
from dictionary.mappings import GLOBAL_PLATFORM_MAP, IRANIAN_BRANDS
from core.config import IQR_K
# --- Constants ---
LOOKBACK_DAYS = 25
CURRENT_MARKET_YEAR = jdatetime.date.today().year
EXPECTED_KM_PER_YEAR = 7000
CONFIRMED_WEIGHT_MULTIPLIER = 1.8

def _to_jalali_date(date_str):
        parts = list(map(int, str(date_str).split('-')))
        return jdatetime.date(parts[0], parts[1], parts[2])

def _surface_color_bucket(color):
    c = str(color or "").strip().lower()
    if c in {"سفید", "مشکی", "قرمز", "آبی"}:
        return c
    return "سایر"

def _robust_mask_log_price(prices, iqr_k=IQR_K):
    x = pd.to_numeric(pd.Series(prices), errors="coerce").astype(float).to_numpy()
    valid = np.isfinite(x) & (x > 0)
    if valid.sum() < 4: return np.ones(len(x), dtype=bool)

    lx = np.log(x[valid])
    q1, q3 = np.percentile(lx, 25), np.percentile(lx, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0: return np.ones(len(x), dtype=bool)

    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    mask_valid = (lx >= lo) & (lx <= hi)
    out = np.ones(len(x), dtype=bool)
    out[np.where(valid)[0]] = mask_valid
    return out

def _sequential_outlier_removal(df: pd.DataFrame, min_group_size: int = 6):
    work = df.copy()
    def apply_iqr(frame, group_cols):
        keep_mask = pd.Series(True, index=frame.index)
        for _, g in frame.groupby(group_cols, dropna=False):
            if len(g) >= min_group_size:
                m = _robust_mask_log_price(g["price"].values)
                keep_mask.loc[g.index] = keep_mask.loc[g.index] & pd.Series(m, index=g.index)
        return frame.loc[keep_mask].copy()

    work = apply_iqr(work, ["real_brand", "real_model"])
    work = apply_iqr(work, ["real_brand", "real_model", "production_year"])
    return apply_iqr(work, ["real_brand", "real_model", "production_year", "mileage_bucket"])

def _segment_motorcycles(df):
    economic_models = [
        "CG 125", "CG 150", "CG 200", "CDI 125", "Boxer 150", 
        "HLX 150", "Rockz 125", "Wego 110", "Vino 50", "Pulsar 135", 
        "Pulsar 180", "Pulsar 200", "S2", "Daichi 150", "KPS 200",
    ]
    high_class_models = [
        "CB 1300", "CRF 250", "ADV 160", "ADV 350", "PCX 160", "Forza 350", 
        "NMAX 155", "XMAX 250", "R25", "MT 25", "WR 155", "Ninja 250", 
        "Z 250", "Galaxy JR 300", "Primavera 150", "Sprint 150", "GTS 250", 
        "GTS 300", "GTV 300"
    ]

    def apply_segment(row):
        model = str(row.get("real_model", ""))
        brand = str(row.get("real_brand", ""))

        if any(hc.lower() in model.lower() for hc in high_class_models): return "High-Class"
        if any(eco.lower() in model.lower() for eco in economic_models): return "Economic"

        copy_prone_models = ["Click", "Aerox", "ADV", "NVX"]
        is_copy_prone = any(c.lower() in model.lower() for c in copy_prone_models)

        if is_copy_prone:
            if brand in IRANIAN_BRANDS:
                if "click" in model.lower(): return "Economic"
                return "Medium"
            else:
                return "Medium"  
        return "Medium"

    df["segment"] = df.apply(apply_segment, axis=1)
    return df

class DataPreprocessor:
    def __init__(self, raw_df: pd.DataFrame):
        self.df = raw_df.copy()
        self.today = dt.date.today()

    def _impute_missing_values(self):
        """پر کردن مقادیر Null بر اساس گروه بندی برند و مدل موتور"""
        
        # 1. متغیرهای طبقه‌بندی (Categorical) -> پر کردن با نما (Mode) گروه
        cat_cols = [
            'clutch_type', 'brake_type', 'start_type', 
            'engine_condition', 'body_condition', 'document_status', 'seller_type'
        ]
        for col in cat_cols:
            if col in self.df.columns:
                # محاسبه نما (بیشترین تکرار) در هر گروه
                self.df[col] = self.df.groupby(['real_brand', 'real_model'])[col].transform(
                    lambda x: x.mode()[0] if not x.mode().empty else np.nan
                )
                self.df[col] = self.df[col].fillna('نامشخص')

        num_cols = ['engine_volume', 'technical_score']
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df.groupby(['real_brand', 'real_model'])[col].transform(
                    lambda x: x.median()
                )
                self.df[col] = self.df[col].fillna(0)
                
        return self

    def _filter_by_mapping(self):
        """
        فقط سطرهایی را نگه می‌دارد که (real_brand, real_model)
        در مپینگ تعریف شده باشند.
        """
        allowed_pairs = set()

        for model_name, info in GLOBAL_PLATFORM_MAP.items():
            parent_brand = info.get("parent_brand")
            if parent_brand:
                allowed_pairs.add((parent_brand, model_name))

            for brand in info.get("copy_brands", []):
                allowed_pairs.add((brand, model_name))

        self.df = self.df[
            self.df.apply(
                lambda row: (row["real_brand"], row["real_model"]) in allowed_pairs,
                axis=1
            )
        ].copy()

        return self

    def clean_and_prepare(self) -> pd.DataFrame:
        if self.df.empty: return self.df

        # --- 1. فیلترهای پایه ---
        # فقط آگهی‌های معتبر و پردازش شده را نگه می‌داریم
        self.df = self.df[(self.df['is_valid_ad'] == 1) & (self.df['status'] == 'PROCESSED_OK')].copy()
    
        self.df = self.df.dropna(subset=["price", "real_brand", "real_model", "production_year", "mileage"]).copy()
        self.df = self.df[(self.df["price"] > 0) & (self.df["mileage"] >= 0) & (self.df["production_year"] >= 1390)].copy()

        self._filter_by_mapping()

        if self.df.empty:
            return self.df

        self._impute_missing_values()
        
        self.df["publish_date"] = self.df["publish_date"].apply(_to_jalali_date)
        self.df["color"] = self.df["color"].fillna("unknown").apply(_surface_color_bucket)
        
        self.df["age"] = (CURRENT_MARKET_YEAR - self.df["production_year"]).clip(lower=0)
        self.df["expected_mileage"] = self.df["age"] * EXPECTED_KM_PER_YEAR
        self.df["mileage_gap"] = self.df["mileage"] - self.df["expected_mileage"]
        self.df["mileage_gap_ratio"] = self.df["mileage"] / (self.df["expected_mileage"] + 1.0)
        self.df["price_per_km_proxy"] = np.log1p(self.df["mileage"]) / (self.df["age"] + 1.0)

        # One-Hot Encoding برای باکت‌ها
        for i, name in enumerate(["zero_dry", "near_zero", "very_low", "low", "low_mid", "mid_low", "mid", "high", "very_high"]):
            self.df[f"m_cat_{name}"] = (self.df["mileage_bucket"] == i).astype(int)

        self.df["title_len"] = self.df["title"].astype(str).str.len()
        self.df["desc_len"] = self.df["description"].astype(str).str.len()

        self.df = _segment_motorcycles(self.df)
        self.df = _sequential_outlier_removal(self.df)

        return self.df
    