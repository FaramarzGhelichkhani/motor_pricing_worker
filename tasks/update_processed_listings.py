"""Update processed_listings fields; temporary identity comes from Django DB.

Keeps original target filter, variant/condition logic and parameterized batch
update. The V10 copy-cleaner sees authoritative motorcycle_model_id/is_original
only after an in-memory (real_brand, real_model) JOIN. They are NEVER written
to processed_listings and are NEVER inferred from is_copy.
"""

from ml_engine._01_data_prep import _add_model_variant, _build_condition_features
from ml_engine.copy_label_cleaning import (
    _correct_is_copy_labels,
    fit_price_reference,
)

import jdatetime
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from core.database import SessionLocal, DjangoSession
import time

CURRENT_JALALI_YEAR = 1405
DRY_RUN = False
BATCH_SIZE = 1000

# The same reference must be used for every partial/60-day scoring batch.
REFERENCE_LOOKBACK_DAYS = 180
# Set True whenever you explicitly want to rebuild the reference.


def _jalali_date_from_db(values: pd.Series) -> pd.Series:
    """Return YYYY-MM-DD Jalali, the date format expected by V10.

    A string already beginning with a Jalali year is preserved; Gregorian
    timestamps, including created_at, are converted with jdatetime.
    """
    values = values.copy()
    as_text = values.astype("string").str.strip()
    jalali = as_text.str.match(r"^1[34]\d{2}[-/]\d{1,2}[-/]\d{1,2}", na=False)
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    result.loc[jalali] = as_text.loc[jalali].str.replace("/", "-", regex=False).str.slice(0, 10)

    need_convert = ~jalali & values.notna()
    if need_convert.any():
        greg = pd.to_datetime(values.loc[need_convert], errors="coerce")
        result.loc[need_convert] = greg.map(
            lambda dt: jdatetime.date.fromgregorian(date=dt.date()).strftime("%Y-%m-%d")
            if pd.notna(dt) else pd.NA
        ).astype("string")
    return result


def _set_publish_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Use the original listing publication date when available.

    Falling back to created_at is deliberate and disclosed: it makes V10
    price scoring possible when the database does not store publish_date.
    """
    frame = frame.copy()
    if "publish_date" in frame:
        converted = _jalali_date_from_db(frame["publish_date"])
        missing = converted.isna()
        if missing.any():
            converted.loc[missing] = _jalali_date_from_db(frame.loc[missing, "created_at"])
        frame["publish_date"] = converted
    else:
        frame["publish_date"] = _jalali_date_from_db(frame["created_at"])
    return frame


def _jalali_month_index(s: pd.Series) -> pd.Series:
    parts = s.astype("string").str.extract(r"^(\d{4})-(\d{1,2})")
    year = pd.to_numeric(parts[0], errors="coerce")
    month = pd.to_numeric(parts[1], errors="coerce")
    return (year * 12 + month).where(month.between(1, 12))


def _get_price_reference(
    db: Session,
    target: pd.DataFrame,
    has_publish_date: bool,
    has_flag_installment: bool,
):
    publication = (
        "p.publish_date AS publish_date,"
        if has_publish_date else ""
    )

    finance = (
        "p.flag_installment AS flag_installment,"
        if has_flag_installment else ""
    )

    history_sql = text(f"""
        SELECT
            p.real_brand,
            p.real_model,
            p.production_year,
            p.price,
            p.created_at,
            {publication}
            {finance}
            1 AS is_real_price,
            'ads' AS source
        FROM processed_listings AS p
        WHERE p.is_valid_ad = 1
          AND p.is_real_price = 1
          AND p.status = 'PROCESSED_OK'
          AND p.created_at >= NOW() - (
              :lookback_days * INTERVAL '1 day'
          )
    """)

    history = pd.read_sql(
        history_sql,
        db.bind,
        params={
            "lookback_days": REFERENCE_LOOKBACK_DAYS
        }
    )

    history = _set_publish_date(history)

    print(
        f"V10 history rows: {len(history):,}"
    )

    # ساخت مرجع قیمت فقط در حافظه
    price_reference = fit_price_reference(history)

    if not price_reference.get("families"):
        raise RuntimeError(
            "No reliable price families were fitted."
        )

    print(
        "Price reference ready in memory | "
        f"Families: {len(price_reference['families'])}"
    )

    return price_reference

def _identity_key(s: pd.Series) -> pd.Series:
    """Normalize superficial casing/whitespace, never infer brand/model aliases."""
    return (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )


def _attach_django_model_identity(df: pd.DataFrame, django_db: Session) -> pd.DataFrame:
    """In-memory many-to-one join; DB writes remain token-based only.

    Missing mappings are intentionally left as NA so V10 freezes copy changes
    for those rows. Conflicting identities for one normalized brand/model pair
    are an error rather than a nondeterministic join.
    """
    query_ids = """
        SELECT
            m.id AS motorcycle_model_id,
            b.name_en AS real_brand,
            m.name_en AS real_model,
            m.is_original AS is_original,
            m.parent_id AS parent_id
        FROM motor_motormodel AS m
        JOIN motor_brand AS b ON m.brand_id = b.id
        WHERE m.parent_id IS NOT NULL
    """
    models = pd.read_sql(text(query_ids), django_db.bind)
    required = {"motorcycle_model_id", "real_brand", "real_model", "is_original"}
    if not required.issubset(models.columns):
        raise RuntimeError(f"Django model query lacks: {sorted(required - set(models.columns))}")
    if models.empty:
        raise RuntimeError("No Django motorcycle models returned; refusing to update labels.")

    models = models.copy()
    models["__brand_key"] = _identity_key(models["real_brand"])
    models["__model_key"] = _identity_key(models["real_model"])
    # Incomplete dictionary rows cannot provide an authoritative model identity.
    valid_models = (
        models["__brand_key"].notna()
        & models["__brand_key"].ne("")
        & models["__model_key"].notna()
        & models["__model_key"].ne("")
        & pd.to_numeric(models["motorcycle_model_id"], errors="coerce").notna()
        & pd.to_numeric(models["is_original"], errors="coerce").isin([0, 1])
    )
    models = models.loc[valid_models, [
        "__brand_key", "__model_key", "motorcycle_model_id", "is_original"
    ]].copy()
    if models.empty:
        raise RuntimeError("No Django motorcycle identity rows have valid id and is_original.")

    pair = ["__brand_key", "__model_key"]
    # Duplicates of the same exact ID are harmless; conflicting IDs are not.
    duplicate_pairs = models.duplicated(pair, keep=False)
    if duplicate_pairs.any():
        conflicts = models.loc[duplicate_pairs].groupby(pair, dropna=False).filter(
            lambda g: len(g.drop_duplicates(["motorcycle_model_id", "is_original"])) > 1
        )
        if not conflicts.empty:
            preview = conflicts.head(8).to_dict("records")
            raise ValueError(f"Django model mapping is ambiguous by (brand, model): {preview}")
        models = models.drop_duplicates(pair, keep="first")

    original = df.copy()
    original["__brand_key"] = _identity_key(original["real_brand"])
    original["__model_key"] = _identity_key(original["real_model"])
    joined = original.merge(models, on=pair, how="left", validate="many_to_one", sort=False)
    if len(joined) != len(df) or not joined["token"].reset_index(drop=True).equals(
        df["token"].reset_index(drop=True)
    ):
        raise RuntimeError("Identity JOIN changed row count/order; refusing database update.")
    joined = joined.drop(columns=pair)
    matched = joined["motorcycle_model_id"].notna() & joined["is_original"].notna()
    print(f"Django identity mapping: {int(matched.sum()):,}/{len(joined):,} matched; "
          f"{int((~matched).sum()):,} unmatched (V10 freezes their is_copy).")
    if not matched.any():
        raise RuntimeError("Zero (real_brand, real_model) matches. Refusing database update.")
    return joined


def update_main():
    db: Session = SessionLocal()
    django_db: Session | None = None
    try:
        columns = {c["name"]: c["type"] for c in inspect(db.bind).get_columns("processed_listings")}
        required = {"brand_model_raw", "created_at"}
        missing = required - set(columns)
        if missing:
            raise RuntimeError(
                "processed_listings missing required fields: " + ", ".join(sorted(missing))
            )
        has_publish_date = "publish_date" in columns
        publish_sql = "p.publish_date AS publish_date," if has_publish_date else ""
        weight_sql = "p.label_weight AS label_weight," if "label_weight" in columns else "1.0 AS label_weight,"
        installment_sql = "p.flag_installment AS flag_installment," if "flag_installment" in columns else "0 AS flag_installment,"

        # ============================================================
        # 1) READ ONLY VALID / REAL-PRICE ADS: ORIGINAL FILTER UNCHANGED
        # ============================================================
        SQL = f"""
        SELECT
            p.token,
            p.title,
            p.description,
            p.real_brand,
            p.real_model,
            p.is_copy,
            p.brand_model_raw,
            p.created_at,
            {publish_sql}
            {weight_sql}
            {installment_sql}
            p.production_year,
            p.mileage,
            p.mileage_bucket,
            p.price,
            p.engine_condition,
            p.body_condition,
            p.technical_score,
            p.flag_accident,
            p.flag_engine_issue,
            p.flag_new,
            p.is_real_price
        FROM processed_listings AS p
        WHERE p.is_valid_ad = 1
          AND p.is_real_price = 1
          AND p.status = 'PROCESSED_OK'
          AND p.created_at >= NOW() - INTERVAL '60 days'
        """

        df = pd.read_sql(text(SQL), db.bind)
        print(f"Rows loaded: {len(df):,}")
        if df.empty:
            print("No matching ads. Nothing to update.")
            return

        # Identity is fetched separately from PRODUCTION Django DB.
        # It exists only in-memory and is NEVER part of processed_listings UPDATE.
        django_db = DjangoSession()
        df = _attach_django_model_identity(df, django_db)
        django_db.close()
        django_db = None

        # V10 price scoring requires a *Jalali* publish_date, not a Gregorian date.
        df = _set_publish_date(df)
        price_reference = _get_price_reference(db, df, has_publish_date, "flag_installment" in columns)

        # ============================================================
        # 2) TEMP FEATURES + V10 CORRECTION BEFORE MODEL VARIANT
        # ============================================================
        df["production_year"] = pd.to_numeric(df["production_year"], errors="coerce")
        df["age"] = (CURRENT_JALALI_YEAR - df["production_year"]).clip(lower=0)
        df["production_age"] = df["age"]
        df["is_brand_new"] = (
            pd.to_numeric(df["flag_new"], errors="coerce").fillna(0).astype(bool)
        )
        df["is_copy"] = pd.to_numeric(df["is_copy"], errors="coerce").fillna(0).astype(int)

        old_copy = df["is_copy"].copy()
        old_is_original = df["is_original"].copy(deep=True)
        df = _correct_is_copy_labels(df, price_reference=price_reference)
        pd.testing.assert_series_equal(df["is_original"], old_is_original, check_dtype=True)
        print("\nIS_COPY V10")
        print("Changed labels:", int(df["is_copy"].ne(old_copy).sum()))
        print(df["copy_label_decision_reason"].value_counts(dropna=False).head(25))
        print("\nPRICE REFERENCE STATUS")
        print(df["copy_price_evidence_status"].value_counts(dropna=False).head(15))

        # ============================================================
        # 3) MODEL VARIANT: ORIGINAL FUNCTION UNCHANGED
        # ============================================================
        df = _add_model_variant(df)

        # ============================================================
        # 4) CONDITION TIER: ORIGINAL FUNCTION UNCHANGED
        # ============================================================
        df = _build_condition_features(df)

        result = df[["token", "model_variant", "condition_tier", "is_copy"]].copy()
        result["is_copy"] = pd.to_numeric(result["is_copy"], errors="coerce").astype("Int64")

        # ============================================================
        # 5) PREVIEW
        # ============================================================
        print("\nMODEL VARIANT")
        print(result["model_variant"].value_counts(dropna=False).head(30))
        print("\nCONDITION TIER")
        print(result["condition_tier"].value_counts(dropna=False).sort_index())
        print("\nIS_COPY")
        print(result["is_copy"].value_counts(dropna=False))

        # ============================================================
        # 6) DRY RUN
        # ============================================================
        if DRY_RUN:
            print("\nDRY_RUN=True -> DATABASE NOT UPDATED")
            return

        # ============================================================
        # 7) SAFE BATCH UPDATE: ADDED is_copy, NO OTHER FIELD UPDATES
        # ============================================================
        print("\n" + "=" * 70)
        print("SAFE BATCH UPDATE")
        print("=" * 70)

        UPDATE_SQL_TEMPLATE = """
        UPDATE processed_listings AS p
        SET
            model_variant = v.model_variant,
            condition_tier = v.condition_tier,
            is_copy = v.is_copy
        FROM (
            VALUES
                {values}
        ) AS v(token, model_variant, condition_tier, is_copy)
        WHERE p.token = v.token
          AND (
              p.model_variant IS DISTINCT FROM v.model_variant
              OR p.condition_tier IS DISTINCT FROM v.condition_tier
              OR p.is_copy IS DISTINCT FROM v.is_copy
          )
        """
        # If the DB column is Boolean, transmit actual Boolean values.
        from sqlalchemy.sql.sqltypes import Boolean
        copy_is_boolean = isinstance(columns["is_copy"], Boolean)

        total = len(result)
        updated_total = 0
        for start in range(0, total, BATCH_SIZE):
            batch = result.iloc[start:start + BATCH_SIZE]
            values_sql = []
            params = {}
            for i, row in enumerate(batch.itertuples(index=False, name=None)):
                token, model_variant, condition_tier, is_copy = row
                values_sql.append(f"(:token_{i}, :variant_{i}, :tier_{i}, :copy_{i})")
                params[f"token_{i}"] = token
                params[f"variant_{i}"] = model_variant if pd.notna(model_variant) else None
                params[f"tier_{i}"] = int(condition_tier) if pd.notna(condition_tier) else None
                params[f"copy_{i}"] = (
                    (bool(is_copy) if copy_is_boolean else int(is_copy))
                    if pd.notna(is_copy) else None
                )
            update_sql = text(UPDATE_SQL_TEMPLATE.format(values=",\n".join(values_sql)))
            try:
                result_update = db.execute(update_sql, params)
                db.commit()
                updated_total += result_update.rowcount
            except Exception:
                db.rollback()
                print(f"\nERROR IN BATCH {start:,} - {min(start + BATCH_SIZE, total):,}")
                raise
            done = min(start + BATCH_SIZE, total)
            time.sleep(0.5)
            print(f"Processed: {done:,} / {total:,} | DB rows changed: {updated_total:,}")
        print("\n" + "=" * 70)
        print("BATCH UPDATE FINISHED")
        print(f"Input rows : {total:,}")
        print(f"DB updated : {updated_total:,}")
        print("=" * 70)

        # ============================================================
        # 8) VERIFY: ORIGINAL QUERY UNCHANGED
        # ============================================================
        VERIFY_SQL = """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (
                WHERE is_valid_ad = 1 AND is_real_price = 1 AND status = 'PROCESSED_OK'
            ) AS eligible,
            COUNT(*) FILTER (
                WHERE is_valid_ad = 1 AND is_real_price = 1 AND status = 'PROCESSED_OK'
                  AND model_variant IS NULL
            ) AS null_variant,
            COUNT(*) FILTER (
                WHERE is_valid_ad = 1 AND is_real_price = 1 AND status = 'PROCESSED_OK'
                  AND condition_tier IS NULL
            ) AS null_condition,
            COUNT(DISTINCT model_variant) AS variant_count,
            COUNT(DISTINCT condition_tier) AS condition_count
        FROM processed_listings;
        """
        check = pd.read_sql(text(VERIFY_SQL), db.bind)
        print("\n" + "=" * 70)
        print("DATABASE VERIFICATION")
        print("=" * 70)
        print(check.to_string(index=False))
    except Exception:
        db.rollback()
        print("\nDATABASE UPDATE FAILED.")
        raise
    finally:
        if django_db is not None:
            django_db.close()
        db.close()
