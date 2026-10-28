import re
import jdatetime
import pandas as pd
import numpy as np
import datetime as dt
from typing import Optional
from dictionary.mappings import GLOBAL_PLATFORM_MAP, IRANIAN_BRANDS
from core.config import IQR_K
from .baseline import build_profiles_and_adaptive_price_baseline
from .condition_dropin import (
    _build_condition_features,
)
from ml_engine.copy_label_cleaning import (
    fit_price_reference,
    _correct_is_copy_labels,
)

from ml_engine.variant_labeling_v2 import (
    _add_model_variant,
    _set_model_variant,
)
# --- Constants ---

CURRENT_MARKET_YEAR = jdatetime.date.today().year
EXPECTED_KM_PER_YEAR = 7000
CONFIRMED_WEIGHT_MULTIPLIER = 1.8

    
def _convert_jalali_to_greg(jalali_str):
    try:
        # فرض بر این است که رشته ورودی به فرمت '1405-05-16' است
        year, month, day = map(int, str(jalali_str).split('-'))
        # تبدیل به آبجکت datetime استاندارد پانداس
        return pd.to_datetime(jdatetime.date(year, month, day).togregorian())
    except:
        return pd.NaT

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

def _sequential_outlier_removal(df: pd.DataFrame, min_group_size: int = 5):
    work = df.copy()
    def apply_iqr(frame, group_cols):
        keep_mask = pd.Series(True, index=frame.index)
        for _, g in frame.groupby(group_cols, dropna=False):
            if len(g) >= min_group_size:
                m = _robust_mask_log_price(g["price"].values)
                keep_mask.loc[g.index] = keep_mask.loc[g.index] & pd.Series(m, index=g.index)
        return frame.loc[keep_mask].copy()

    work = apply_iqr(work, ["real_brand", "real_model", "is_copy", "model_variant"])
    work = apply_iqr(work, ["real_brand", "real_model", "is_copy", "model_variant","production_year"])
    return apply_iqr(work, ["real_brand", "real_model", "is_copy", "model_variant","production_year", "mileage_bucket"])

def _segment_motorcycles(df):
    """
    دسته‌بندی مدل‌های موتور سیکلت به بخش‌های Economic, High-Class, Copy/Clone و Medium.
    """
    df = df.copy()
    iranian_brands = IRANIAN_BRANDS

    ECONOMIC_MODELS = {
    "CG 125", "CG 150", "CG 200", "CDI 125", "Boxer 150", "HLX 150",
    "Rockz 125", "Wego 110", "Vino 50", "Pulsar 135", "Pulsar 180",
    "S2", "Daichi 150", "KPS 200", "Hank 150",
    }

    HIGH_CLASS_MODELS = {
        "CB 1300", "CRF 250", "ADV 160", "ADV 350", "PCX 160", "Forza 350",
        "NMAX 155", "XMAX 250", "R25", "MT 25", "MT 15", "R15", "WR 155",
        "WR 250", "Ninja 250", "Z 250", "ZX25R", "Galaxy JR 300",
        "Primavera 150", "Sprint 150", "GTS 250", "GTS 300", "GTV 300",
        "TNT 300", "TRK 251", "250NK", "300SR", "Dominar 250", "RS 200",
    }

    COPY_PRONE_MODELS = {
        "Click", "Click 150", "Click 160", "Aerox", "NMAX", "NVX",
        "ADV", "PCX", "Vario", "Galaxy J200", "Galaxy JR",
    }

    # Helpers
    def normalize_text(text):
        return str(text).strip().lower().replace("-", " ").replace("_", " ")

    def contains_any(target_model, model_set):
        return any(x in target_model for x in model_set)

    # پیش‌پردازش مجموعه‌ها (برای جلوگیری از تکرار این کار در هر سطر و افزایش چشمگیر سرعت)
    norm_economic = {normalize_text(x) for x in ECONOMIC_MODELS}
    norm_high_class = {normalize_text(x) for x in HIGH_CLASS_MODELS}
    norm_copy_prone = {normalize_text(x) for x in COPY_PRONE_MODELS}

    # منطق دسته‌بندی
    def determine_segment(row):
        model = normalize_text(row.get("real_model", ""))
        brand_raw = str(row.get("real_brand", "")).strip()

        # 1. Copy / Clone
        if contains_any(model, norm_copy_prone):
            if brand_raw in iranian_brands:
                return "Copy/Clone"
            if contains_any(model, norm_high_class):
                return "High-Class"
            return "Medium"

        # 2. High Class
        if contains_any(model, norm_high_class):
            return "High-Class"

        # 3. Economic
        if contains_any(model, norm_economic):
            return "Economic"

        # 4. Default
        return "Medium"

    # اعمال روی دیتافریم
    df["segment"] = df.apply(determine_segment, axis=1)
    
    return df


def _add_mileage_features(df, mileage_col="mileage", age_col="age", brand_col="real_brand", model_col="real_model", min_group_count=30):
    """
    Add mileage-related features without removing any rows.
    New features: mileage_clean, annual_mileage_rate, mileage_confidence, mileage_age_consistency, mileage_is_suspicious
    """
    df = df.copy()

    # 1) CLEAN MILEAGE
    mileage = pd.to_numeric(df[mileage_col], errors="coerce")
    suspicious_exact = {888888, 999999, 1000000, 111111, 1111}
    
    mileage_clean = mileage.mask(mileage.isin(suspicious_exact)).mask(mileage < 0)
    df["mileage_clean"] = mileage_clean

    # 2) AGE-SAFE VALUE
    age = pd.to_numeric(df[age_col], errors="coerce")
    age_for_rate = age.clip(lower=0.5)
    df["annual_mileage_rate"] = mileage_clean / age_for_rate

    # 3) GROUP STATISTICS (brand + model + age)
    temp = pd.DataFrame({"brand": df[brand_col], "model": df[model_col], "age": age, "mileage": mileage_clean})
    group = temp.groupby(["brand", "model", "age"], dropna=False)["mileage"]
    
    group_count = group.transform("count")
    group_median = group.transform("median")
    group_mad = group.transform(lambda x: np.median(np.abs(x.dropna() - np.median(x.dropna()))) if x.notna().any() else np.nan)

    # 4) FALLBACK GROUP (model + age)
    group2 = temp.groupby(["model", "age"], dropna=False)["mileage"]
    count2 = group2.transform("count")
    median2 = group2.transform("median")
    mad2 = group2.transform(lambda x: np.median(np.abs(x.dropna() - np.median(x.dropna()))) if x.notna().any() else np.nan)

    # 5) CHOOSE REFERENCE
    use_primary = group_count >= min_group_count
    ref_median = pd.Series(np.where(use_primary, group_median, median2), index=df.index, dtype=float)
    ref_mad = pd.Series(np.where(use_primary, group_mad, mad2), index=df.index, dtype=float)
    ref_count = pd.Series(np.where(use_primary, group_count, count2), index=df.index, dtype=float)

    # 6) MILEAGE DEVIATION FROM GROUP
    mileage_deviation = np.abs(mileage_clean - ref_median) / np.maximum(ref_median, 1000)
    robust_z = np.abs(mileage_clean - ref_median) / np.maximum(ref_mad * 1.4826, 1000)

    # 7) MILEAGE AGE CONSISTENCY
    consistency = np.exp(-np.minimum(robust_z, 10) / 3).where(mileage_clean.notna(), 0.0).clip(0, 1)
    df["mileage_age_consistency"] = consistency

    # 8) MILEAGE CONFIDENCE
    support_score = (np.log1p(ref_count) / np.log1p(100)).clip(0, 1)
    deviation_score = np.exp(-mileage_deviation / 0.75).clip(0, 1)
    confidence = (0.35 * support_score + 0.40 * deviation_score + 0.25 * consistency).where(mileage_clean.notna(), 0.0).clip(0, 1)
    df["mileage_confidence"] = confidence

    # 9) SUSPICIOUS FLAG
    suspicious = mileage.isin(suspicious_exact) | (mileage < 0) | (mileage_clean.notna() & (robust_z > 5))
    df["mileage_is_suspicious"] = suspicious.astype("int8")

    # 10) OPTIONAL DEBUG FEATURES
    df["mileage_reference_median"] = ref_median
    df["mileage_reference_mad"] = ref_mad
    df["mileage_reference_count"] = ref_count
    df["mileage_robust_z"] = robust_z

    # 11) SANITY CHECK
    assert len(df) == len(df), "Row count changed!"

    return df

class DataPreprocessor:
    def __init__(self, raw_df: pd.DataFrame, zero_df: pd.DataFrame = None,
                 enable_baseline: bool = False):
        """Preprocess without Audit or asof-dependent labeling.

        Existing copy/variant and sequential-IQR algorithms are preserved.
        Price baseline generation is optional (OFF by default).
        """
        self.df = raw_df.copy()
        self.today = dt.date.today()
        self.zero_df = zero_df.copy() if zero_df is not None else None
        self.enable_baseline = bool(enable_baseline)

    def _impute_missing_values(self):
        """Fill missing values only; preserve original ad condition for condition engineering.

        The *_raw columns are never group-imputed. Running this method again on the
        same DataFrame does not overwrite the stored original values.
        """
        df = self.df
        keys = ["real_brand", "real_model"]
        for key in keys:
            if key not in df.columns:
                raise KeyError(f"Missing grouping key: {key}")

        # Capture the actual information from the advertiser BEFORE filling gaps.
        for col in ["technical_score", "engine_condition", "body_condition"]:
            if col not in df.columns:
                df[col] = np.nan
            raw_col = f"{col}_raw"
            if raw_col not in df.columns:
                df[raw_col] = df[col].copy()
            if col in ["engine_condition", "body_condition"]:
                df[raw_col] = (df[raw_col].astype("string").str.strip()
                            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA,
                                        "NULL": pd.NA, "نامشخص": pd.NA}))
            else:
                df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce")
                df[raw_col] = df[raw_col].where(df[raw_col].between(1, 5))
            df[f"{col}_was_missing"] = df[raw_col].isna().astype("int8")

        # Fill categorical covariates for the rest of the pipeline without
        # overwriting nonmissing values. The condition builder uses *_raw.
        categorical = ["clutch_type", "brake_type", "start_type",
                    "engine_condition", "body_condition", "document_status",
                    "seller_type"]
        for col in categorical:
            if col not in df.columns:
                continue
            df[col] = (df[col].astype("string").str.strip()
                    .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NULL": pd.NA}))
            if not df[col].isna().any():
                continue
            def first_mode(series):
                m = series.mode(dropna=True)
                return m.iloc[0] if len(m) else pd.NA
            modes = df.groupby(keys, dropna=False)[col].transform(first_mode)
            df[col] = df[col].fillna(modes).fillna("نامشخص")

        # Again: only fill missing numeric values; do not change valid originals.
        for col in ["engine_volume", "technical_score"]:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            median = vals.groupby([df[k] for k in keys], dropna=False).transform("median")
            df[col] = vals.fillna(median).fillna(0)

        self.df = df
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

    def _clean_prepare_zero_df(self):
            self.zero_df['date'] = self.zero_df['jalali_date'].apply(_convert_jalali_to_greg)
            self.zero_df['date'] = self.zero_df['date'].fillna(pd.to_datetime(self.zero_df['insert_date']).dt.normalize())
            self.zero_df.loc[(self.zero_df.motor_class == 'کبیر موتور'),'brand_name'] = 'Kabir'
            zero_source = self.zero_df[['brand_name', 'model_name', 'date', 'price', 'raw_title', 'motor_class']].copy()
            zero_source.rename(columns={'brand_name': 'real_brand', 'model_name': 'real_model',\
                                         'raw_title':"title", 'motor_class':'model_variant'}, inplace=True)
            zero_source["mileage"] = 0 
            zero_source["mileage_bucket"] = 0 
            zero_source["age"] = 0
            zero_source["production_year"] = CURRENT_MARKET_YEAR
            # zero_source = _set_model_variant(zero_source, text_col='model_variant')
            # zero_source["model_variant"] = zero_source["model_variant"].fillna('DEFAULT')
            zero_source["is_copy"] = np.where(zero_source["real_brand"].str.strip().isin(IRANIAN_BRANDS), 1 , 0)
            zero_source["is_original"] = np.where(zero_source["real_brand"].str.strip().isin(IRANIAN_BRANDS), 0 , 1)
            zero_source["label_weight"] = 1.5
            zero_source["condition_tier"] = 1
            zero_source["source"] = "factory"
        
            zero_source = _segment_motorcycles(zero_source)
    
            return zero_source
    
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

        self.df['date'] = self.df['publish_date'].apply(_convert_jalali_to_greg)
        self.df = self.df.dropna(subset=['date']).copy()
        self.df['publish_year'] = self.df['publish_date'].str.split('-').str[0].astype(int)
        self.df["color"] = self.df["color"].fillna("unknown").apply(_surface_color_bucket)

        
        self.df['age'] = (self.df['publish_year'] - self.df['production_year']).clip(lower=0)
        self.df["label_weight"] = np.where(self.df["is_system_guess_correct"] == 1, 1.5, 1.0)

        self.df["expected_mileage"] = self.df["age"] * EXPECTED_KM_PER_YEAR
        self.df["mileage_gap"] = self.df["mileage"] - self.df["expected_mileage"]
        self.df["mileage_gap_ratio"] = self.df["mileage"] / (self.df["expected_mileage"] + 1.0)
        self.df["price_per_km_proxy"] = np.log1p(self.df["mileage"]) / (self.df["age"] + 1.0)

        # One-Hot Encoding برای باکت‌ها
        for i, name in enumerate(["zero_dry", "near_zero", "very_low", "low", "low_mid", "mid_low", "mid", "high", "very_high"]):
            self.df[f"m_cat_{name}"] = (self.df["mileage_bucket"] == i).astype(int)

        self.df["title_len"] = self.df["title"].astype(str).str.len()
        self.df["desc_len"] = self.df["description"].astype(str).str.len()

        self.df["source"]  = "ads"  
        

        self.df = _segment_motorcycles(self.df)
        price_reference = fit_price_reference(self.df)
        self.df  = _correct_is_copy_labels(self.df ,price_reference=price_reference)
        # Keep the existing specialized variant labeling exactly as implemented.
        # No asof, weekly refitting, or second variant-recovery system.
        self.df, registry = _add_model_variant(self.df, return_registry=True)

        self.df = _sequential_outlier_removal(self.df)
        self.df = _build_condition_features(self.df)

        if self.zero_df is not None:
            cleaned_zero_df = self._clean_prepare_zero_df()
            cleaned_zero_df = _set_model_variant(cleaned_zero_df,registry=registry,)
            return pd.concat([self.df, cleaned_zero_df])

        return self.df

    @staticmethod
    def _fill_factory_model_ids_only(data: pd.DataFrame) -> pd.DataFrame:
        """Only factory rows may inherit a missing model ID from known ads.

        Keep the previous (real_brand, real_model) lookup semantics. Never
        infer/fill IDs for used ads: those can represent unregistered models.
        """
        if data.empty or 'motorcycle_model_id' not in data:
            return data
        factory = data['source'].eq('factory')
        missing_factory = factory & data['motorcycle_model_id'].isna()
        if not missing_factory.any():
            return data
        known_ads = data.loc[
            data['source'].eq('ads') & data['motorcycle_model_id'].notna(),
            ['real_brand', 'real_model', 'motorcycle_model_id'],
        ]
        if known_ads.empty:
            return data
        # Equivalent to the original group-wise first non-null ID, except that
        # factory rows alone are fill targets and factory IDs are not references.
        lookup = known_ads.groupby(['real_brand', 'real_model'], dropna=False)[
            'motorcycle_model_id'
        ].first()
        # Positional vectorization also handles duplicate indices from concat(ads, factory).
        keys = pd.MultiIndex.from_frame(data[['real_brand', 'real_model']])
        resolved = pd.Series(lookup.reindex(keys).to_numpy(), index=data.index)
        data = data.copy()
        data['motorcycle_model_id'] = data['motorcycle_model_id'].where(
            ~missing_factory, resolved
        )
        return data

    def enrich_feature(self, enable_baseline: Optional[bool] = None) -> pd.DataFrame:
        """Return clean training data; optionally add expensive baseline profiles.

        Defaults to constructor's enable_baseline=False. Calling with True
        preserves the legacy baseline generation (no changes to its algorithm).
        """
        cleaned_combined_df = self.clean_and_prepare()
        if cleaned_combined_df.empty:
            return cleaned_combined_df
        cleaned_combined_df = self._fill_factory_model_ids_only(cleaned_combined_df)
        cleaned_combined_df = _add_mileage_features(cleaned_combined_df)

        if enable_baseline is None:
            enable_baseline = self.enable_baseline
        if enable_baseline:
            cleaned_combined_df = build_profiles_and_adaptive_price_baseline(
                cleaned_combined_df,
                min_price=50000000,
                min_monthly_support=7,
                min_weekly_support=3,
                momentum_clip=0.20,
            )

        mask_brand = cleaned_combined_df['real_brand'].str.contains('unknown', case=False, na=False)
        mask_model = cleaned_combined_df['real_model'].str.contains('unknown', case=False, na=False)
        return cleaned_combined_df.loc[~(mask_brand | mask_model)].copy()
