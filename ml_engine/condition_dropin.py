"""Drop-in functions for the motorcycle DataPreprocessor (October 2026).

- Put _impute_missing_values inside DataPreprocessor (indent that function there).
- Put _build_condition_features at module level (outside the class).
- Both functions must run on PRE-IMPUTATION raw ads, in that order.
- The evidence model is intentionally conservative and not a verification of true condition.
"""
import numpy as np
import pandas as pd
import re


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


# Strict patterns from the earlier tested Condition proxy. The code avoids
# counting common negated sales phrases such as "بدون روغن سوزی" as damage.
_ENGINE_PATTERN = (r"نیاز\s+به\s+تعمیر|موتور\s+خراب|گیربکس\s+خراب|"
                   r"تعمیر\s+اساسی|یاتاقان\s+زده|شاتون\s+زده|"
                   r"موتور\s+باز\s+شده|انجین\s+تعمیر\s+شده|"
                   r"موتور\s+تازه\s+تعمیر\s+شده|"
                   r"روغن[‌ ]?سوزی\s+(?:دارد|داره|زیاد)|"
                   r"ایراد\s+فنی\s+(?:دارد|داره)")
_COSMETIC_PATTERN = (r"خط[‌ \-]?و[‌ \-]?خش|خش[‌ ]?دار|خشیدگی|"
                     r"زمین[‌ ]?خوردگی|رنگ[‌ ]?شدگی|رنگ شده|"
                     r"صافکاری|ترک بدنه|فلاپ\s+شکسته")
_SEVERE_PATTERN = r"تصادف\s+شدید|تصادف\s+سنگین|شاسی\s+کج|شاسی\s+ضربه|چپ\s+کرده|واژگون\s+شده"
_GOOD_PATTERN = (r"کاملا سالم|کاملاً سالم|فنی سالم|سالم و پلمپ|"
                 r"بی[‌ ]?خط[‌ ]?و[‌ ]?خش|بدون خط[‌ ]?و[‌ ]?خش|بی[‌ ]?رنگ|بدون رنگ")


def _build_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild meaningful, AD-SPECIFIC condition tiers.

    condition_tier: 1 = healthy declared, 2 = cosmetic/minor issue,
                    3 = technical repair/severe damage, NA = unknown.
    condition_tier_model: 1/2/3 for known cases, 2 for unknown WITH separate
                          condition_unknown indicator. Never interpret unknown
                          as better than Tier 1 or impose a monotonic constraint
                          on an unknown=0 representation.

    Existing feature names are retained for compatibility. This is a proposed
    semantic builder, NOT ground-truth mechanical inspection.
    """
    d = df.copy()
    n = len(d)
    idx = d.index

    def col(name, default=np.nan):
        return d[name] if name in d.columns else pd.Series(default, index=idx)

    def orig(name):
        # IMPORTANT: original pre-imputation source wins over imputed column.
        return col(name + "_raw") if (name + "_raw") in d.columns else col(name)

    eng = orig("engine_condition").astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "نامشخص": pd.NA})
    body = orig("body_condition").astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "نامشخص": pd.NA})
    tech = pd.to_numeric(orig("technical_score"), errors="coerce")
    tech = tech.where(tech.between(1, 5))

    d["engine_condition_clean"] = eng
    d["body_condition_clean"] = body
    d["technical_score_condition_raw"] = tech

    eng_map = {"سالم و پلمپ": 5.0, "سالم": 4.0,
               "تعمیر شده": 2.5, "نیاز به تعمیر": 1.0}
    body_map = {"کاملا سالم": 5.0, "کاملاً سالم": 5.0,
                "سالم": 4.7, "خط و خش جزئی": 4.0,
                "خط و خش دار": 3.0, "زمین خوردگی": 2.0,
                "تصادفی": 1.0}
    d["mechanical_score"] = eng.map(eng_map).astype("float64")
    d["cosmetic_score"] = body.map(body_map).astype("float64")
    d["technical_component"] = (tech - 1.0) * 25.0
    mechanical_component = (d["mechanical_score"] - 1.0) * 25.0
    cosmetic_component = (d["cosmetic_score"] - 1.0) * 25.0

    weights = (mechanical_component.notna().astype(float) * .6
               + cosmetic_component.notna().astype(float) * .4)
    structured = (mechanical_component.fillna(0) * .6
                  + cosmetic_component.fillna(0) * .4).div(weights.replace(0, np.nan))
    tc = d["technical_component"]
    base = pd.Series(np.where(tc.notna() & structured.notna(),
        .65 * tc + .35 * structured,
        np.where(tc.notna(), tc, structured)), index=idx, dtype=float)

    # Free-text evidence; remove explicit negation, but do not attempt to infer
    # new facts from missing text or from the absence of defect words.
    full_text = (col("title", "").fillna("").astype("string") + " "
                 + col("description", "").fillna("").astype("string"))
    text = (full_text.str.replace("ي", "ی", regex=False)
                     .str.replace("ك", "ک", regex=False))
    positive_text = text.str.contains(_GOOD_PATTERN, regex=True, na=False)
    cleaned_text = text.str.replace(
        r"(?:بدون|بدونه|بدونِ|فاقد|بی)\s*(?:هیچ(?:\s*گونه)?\s*)?"
        r"(?:یک\s*قطره\s*)?(?:روغن\s*ریزی\s*(?:و\s*)?)?"
        r"(?:روغن\s*سوزی|روغن\s*سوز|تصادف|خرابی|ایراد\s*فنی|"
        r"مشکل\s*فنی|خط\s*و\s*خش|زمین\s*خوردگی|رنگ\s*شدگی|"
        r"نیاز\s*به\s*تعمیر)", "", regex=True)
    cleaned_text = cleaned_text.str.replace(
        r"(?:تصادف|خط\s*و\s*خش|خرابی|ایراد\s*فنی|نیاز\s*به\s*تعمیر)"
        r"\s*(?:ندارد|نداشته|نکرده|نخورده|نیست)", "", regex=True)
    engine_text = cleaned_text.str.contains(_ENGINE_PATTERN, regex=True, na=False)
    cosmetic_text = cleaned_text.str.contains(_COSMETIC_PATTERN, regex=True, na=False)
    severe_text = cleaned_text.str.contains(_SEVERE_PATTERN, regex=True, na=False)

    def flag(name):
        return pd.to_numeric(col(name, 0), errors="coerce").fillna(0).gt(0)

    flag_clean = flag("flag_clean")
    flag_acc = flag("flag_accident")
    flag_engine = flag("flag_engine_issue")

    # Evidence-based severity. "تعمیر شده" means a known repair history,
    # NOT necessarily an ongoing mechanical fault; it is tier 2 by default.
    structural_hard = (eng.eq("نیاز به تعمیر").fillna(False)
                       | body.eq("تصادفی").fillna(False)
                       | tech.le(2).fillna(False))
    structural_minor = (eng.eq("تعمیر شده").fillna(False)
                        | body.isin(["خط و خش جزئی", "خط و خش دار", "زمین خوردگی"])
                        | tech.between(2, 4, inclusive="right").fillna(False))
    strong_good = (eng.isin(["سالم", "سالم و پلمپ"])
                   | body.isin(["سالم", "کاملا سالم", "کاملاً سالم"])
                   | tech.ge(4.5).fillna(False) | flag_clean | positive_text)
    hard = structural_hard | engine_text | severe_text | flag_engine
    minor = structural_minor | cosmetic_text | flag_acc
    conflict = strong_good & (hard | minor)
    known = hard | minor | strong_good

    # Compatible nullable output; unknown is NOT encoded as a healthy tier.
    tier = pd.Series(pd.NA, index=idx, dtype="Int64")
    tier.loc[strong_good] = 1
    tier.loc[minor] = 2
    tier.loc[hard] = 3
    d["condition_tier"] = tier
    d["condition_unknown"] = (~known).astype("int8")
    d["condition_tier_model"] = tier.fillna(2).astype("int8")
    d["condition_conflict"] = conflict.astype("int8")
    d["condition_evidence_conflict"] = d["condition_conflict"]
    d["condition_evidence_positive"] = strong_good.astype("int8")
    d["evidence_engine_issue"] = (structural_hard | engine_text | flag_engine).astype("int8")
    d["evidence_cosmetic"] = (structural_minor | cosmetic_text | flag_acc).astype("int8")
    d["evidence_accident"] = (body.isin(["تصادفی", "زمین خوردگی"]) | severe_text | flag_acc).astype("int8")

    # Confidence = heuristic evidence strength; NOT a calibrated probability.
    confidence = pd.Series(np.select(
        [structural_hard | engine_text | severe_text,
         structural_minor | cosmetic_text,
         flag_engine | flag_acc,
         (eng.notna() & body.notna()) | tech.notna(),
         strong_good],
        [.90, .75, .65, .70, .55], default=0.0), index=idx, dtype=float)
    confidence.loc[conflict] = np.minimum(confidence.loc[conflict], .50)
    d["condition_confidence"] = confidence
    d["condition_evidence_confidence"] = confidence

    d["engine_issue_severity"] = np.select(
        [eng.eq("نیاز به تعمیر").fillna(False) | engine_text | severe_text,
         flag_engine | eng.eq("تعمیر شده").fillna(False)], [3, 2], default=0).astype("int8")
    d["accident_severity"] = np.select(
        [body.eq("تصادفی").fillna(False) | severe_text,
         body.eq("زمین خوردگی").fillna(False) | flag_acc], [3, 2], default=0).astype("int8")
    d["accident_penalty"] = d["accident_severity"].map({0: 0., 2: 15., 3: 30.})
    d["engine_issue_penalty"] = d["engine_issue_severity"].map({0: 0., 2: 12., 3: 25.})

    # Keep continuous features coherent with semantic tiers: engine failure
    # should never coexist with a "perfect" numerical condition score.
    adjusted = (base - d["accident_penalty"] - d["engine_issue_penalty"])
    d["condition_score_raw"] = base
    d["condition_score_continuous"] = np.select(
        [tier.eq(1).fillna(False), tier.eq(2).fillna(False), tier.eq(3).fillna(False)],
        [adjusted.fillna(95).clip(88, 100),
         adjusted.fillna(78).clip(68, 87),
         adjusted.fillna(52).clip(0, 67)], default=np.nan)
    d["condition_deficit"] = 100 - d["condition_score_continuous"]
    d["mechanical_penalty"] = (5.0 - d["mechanical_score"]).clip(lower=0)
    d["cosmetic_penalty"] = (5.0 - d["cosmetic_score"]).clip(lower=0)

    # Derived features present in your original builder.
    km = pd.to_numeric(col("mileage"), errors="coerce")
    age = pd.to_numeric(col("production_age", np.nan), errors="coerce")
    if age.isna().all():
        age = pd.to_numeric(col("age", np.nan), errors="coerce")
    flag_new = flag("is_brand_new")
    new_unused = flag_new | (age.eq(0) & (km.isna() | km.le(100)))
    low_usage = (~new_unused) & km.notna() & km.le(3000)
    d["usage_state"] = np.select(
        [new_unused, low_usage, d["condition_unknown"].eq(1)],
        ["new_unused", "low_usage", "unknown"], default="used")

    d["condition_reason"] = np.select(
        [hard, minor, strong_good],
        ["technical_or_severe_evidence", "cosmetic_or_minor_evidence", "declared_healthy"],
        default="unknown")
    d["condition_tier_label"] = tier.map(
        {1: "Healthy (declared)", 2: "Cosmetic / minor", 3: "Technical / severe"}
    ).fillna("Unknown")

    # Intermediate columns can be retained for debugging or dropped.
    d.drop(columns=["technical_component", "condition_score_raw",
                    "accident_penalty", "engine_issue_penalty"], inplace=True)
    return d
