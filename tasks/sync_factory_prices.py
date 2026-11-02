
import logging
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import islice
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.database import DjangoSession, SessionLocal
from ml_engine.variant_labeling_v2 import _set_model_variant


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TEHRAN = ZoneInfo("Asia/Tehran")
TABLE_NAME = "motor_factory_prices"
BATCH_SIZE = 500


# ==========================================
# 1. SQL Queries
# ==========================================

# دریافت 15 روز اخیر و آخرین رکورد قدیمی
# برای هر ترکیب متنی از موتور و عنوان
SOURCE_SQL = text("""
WITH ranked AS (
    SELECT
        id,
        brand_name,
        model_name,
        motor_class,
        price,
        jalali_date,
        insert_date,
        source,
        price_provider,
        raw_title,

        ROW_NUMBER() OVER (
            PARTITION BY
                brand_name,
                model_name,
                COALESCE(NULLIF(BTRIM(motor_class), ''), ''),
                COALESCE(NULLIF(BTRIM(raw_title), ''), '')

            ORDER BY
                jalali_date DESC,
                insert_date DESC NULLS LAST,
                id DESC
        ) AS latest_rank

    FROM motorcycle_zero_prices

    WHERE is_standardized IS TRUE
      AND price > 0
      AND jalali_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
      AND jalali_date <= :today_jalali
)

SELECT
    id,
    brand_name,
    model_name,
    motor_class,
    price,
    jalali_date,
    insert_date,
    source,
    price_provider,
    raw_title

FROM ranked

WHERE jalali_date >= :cutoff_jalali
   OR latest_rank = 1
""")


# همان ساختار اتصال مدل‌ها در estimator_v2
MODEL_SQL = text("""
SELECT
    m.id AS motorcycle_model_id,
    b.name_en AS brand_name,
    m.name_en AS model_name

FROM motor_motormodel m

JOIN motor_brand b
    ON m.brand_id = b.id

WHERE m.parent_id IS NOT NULL
""")


# ==========================================
# 2. Helper Functions
# ==========================================

def normalize(value):
    if value is None or pd.isna(value):
        return ""

    s = unicodedata.normalize("NFKC", str(value))

    s = s.replace("ي", "ی").replace("ك", "ک")

    return " ".join(s.strip().casefold().split())


def build_catalog(rows):
    """
    ساخت Mapping برند و مدل به motorcycle_model_id

    اگر یک نام به چند ID اشاره کند،
    آن را مبهم در نظر می‌گیریم.
    """
    catalog = {}
    ambiguous = set()

    for row in rows:
        key = (
            normalize(row["brand_name"]),
            normalize(row["model_name"]),
        )

        if not all(key):
            continue

        model_id = int(row["motorcycle_model_id"])

        if key in catalog and catalog[key] != model_id:
            ambiguous.add(key)
        else:
            catalog[key] = model_id

    for key in ambiguous:
        catalog.pop(key, None)

    return catalog, ambiguous


def jalali_to_gregorian(value):
    import jdatetime

    try:
        year, month, day = map(
            int, str(value).split("-")
        )

        return jdatetime.date(
            year, month, day
        ).togregorian()

    except (ValueError, TypeError, OverflowError):
        return None


def batches(items, size=BATCH_SIZE):
    iterator = iter(items)

    while True:
        chunk = list(islice(iterator, size))

        if not chunk:
            break

        yield chunk


# ==========================================
# 3. Extract Model Variant
# ==========================================

def extract_model_variants(matched_rows):
    """
    استخراج model_variant با همان منطق
    ml_engine.variant_labeling_v2

    ورودی متن:
        raw_title + motor_class
    """

    if not matched_rows:
        return []

    variant_texts = []

    for row in matched_rows:
        parts = []

        raw_title = row.get("raw_title")
        motor_class = row.get("motor_class")

        if isinstance(raw_title, str) and raw_title.strip():
            parts.append(raw_title.strip())

        if isinstance(motor_class, str) and motor_class.strip():
            parts.append(motor_class.strip())

        variant_texts.append(" ".join(parts))

    frame = pd.DataFrame({
        "real_brand": [
            row["brand_name"] for row in matched_rows
        ],
        "real_model": [
            row["model_name"] for row in matched_rows
        ],
        "motorcycle_model_id": [
            row["_model_id"] for row in matched_rows
        ],
        "is_copy": [0] * len(matched_rows),
        "factory_variant_text": variant_texts,
    })

    labelled = _set_model_variant(
        frame,
        text_col="factory_variant_text",
        alignment="preserve",
    )

    return [
        (str(variant), bool(conflict))
        for variant, conflict in zip(
            labelled["model_variant"],
            labelled["variant_conflict"],
        )
    ]


# ==========================================
# 4. Prepare Factory Prices
# ==========================================

def prepare_rows(
    source_rows,
    model_rows,
    cutoff_jalali,
    today_jalali,
):
    """
    1. تطبیق مدل موتور
    2. استخراج model_variant
    3. حذف Variantهای متناقض
    4. نگهداری 15 روز اخیر
    5. نگهداری آخرین قیمت قدیمی هر مدل/کلاس/تیپ
    """

    catalog, ambiguous = build_catalog(model_rows)

    stats = defaultdict(int)
    matched = []

    # Match motorcycle identity
    for row in source_rows:
        row = dict(row)

        price = row.get("price")

        if not price or int(price) <= 0:
            stats["invalid_price"] += 1
            continue

        key = (
            normalize(row.get("brand_name")),
            normalize(row.get("model_name")),
        )

        model_id = catalog.get(key)

        if model_id is None:
            reason = (
                "ambiguous"
                if key in ambiguous
                else "unmatched"
            )
            stats[reason] += 1
            continue

        jalali_date = str(
            row.get("jalali_date") or ""
        )

        if jalali_date > today_jalali:
            stats["future_date"] += 1
            continue

        price_date = jalali_to_gregorian(
            jalali_date
        )

        if price_date is None:
            stats["invalid_date"] += 1
            continue

        row["_model_id"] = model_id
        row["_price_date"] = price_date

        matched.append(row)

    # Extract variants
    variant_results = extract_model_variants(
        matched
    )

    # Remove duplicate quotes per product/day
    per_day = {}

    for row, (variant, conflict) in zip(
        matched, variant_results
    ):

        if conflict:
            stats["conflicting_variant"] += 1
            continue

        motor_class = (
            row.get("motor_class") or ""
        ).strip()

        class_key = normalize(motor_class)

        key = (
            row["_model_id"],
            class_key,
            variant,
            str(row["jalali_date"]),
        )

        rank = (
            str(row.get("insert_date") or ""),
            int(row["id"]),
        )

        if key not in per_day or rank > per_day[key][0]:
            per_day[key] = (
                rank,
                row,
                variant,
                motor_class,
                class_key,
            )

    # Group by model + class + variant
    by_product = defaultdict(list)

    for (
        model_id,
        class_key,
        variant,
        day,
    ), item in per_day.items():

        product_key = (
            model_id,
            class_key,
            variant,
        )

        by_product[product_key].append(
            (day, item)
        )

    selected = []

    for items in by_product.values():

        items.sort(
            key=lambda x: (x[0], x[1][0]),
            reverse=True,
        )

        for index, (
            day,
            (
                rank,
                row,
                variant,
                motor_class,
                class_key,
            ),
        ) in enumerate(items):

            # Keep:
            # - all recent quotes
            # - latest old quote if no recent price
            if index > 0 and day < cutoff_jalali:
                continue

            selected.append({
                "crawler_id": int(row["id"]),

                "motorcycle_model_id": int(
                    row["_model_id"]
                ),

                "motor_class": motor_class or None,
                "motor_class_key": class_key,
                "model_variant": variant,

                "price": int(row["price"]),

                "jalali_date": day,
                "price_date": row["_price_date"],

                "source": row["source"],

                "price_provider": row.get(
                    "price_provider"
                ),

                "raw_title": row.get("raw_title"),

                "source_insert_date": row.get(
                    "insert_date"
                ),
            })

    stats["matched"] = len(matched)
    stats["selected"] = len(selected)
    stats["unique_products"] = len(by_product)

    return selected, dict(stats)


# ==========================================
# 5. Daily Sync
# ==========================================

def sync_factory_prices(asof: date | None = None):
    """
    انتقال روزانه قیمت کارخانه از Crawler
    به دیتابیس Production.

    فقط is_standardized=True

    Batch Upsert در یک تراکنش.
    """

    import jdatetime

    today = (
        asof
        or datetime.now(TEHRAN).date()
    )

    # Today + previous 14 calendar days
    cutoff = today - timedelta(days=14)

    today_jalali = jdatetime.date.fromgregorian(
        date=today
    ).strftime("%Y-%m-%d")

    cutoff_jalali = jdatetime.date.fromgregorian(
        date=cutoff
    ).strftime("%Y-%m-%d")

    now = datetime.now(TEHRAN)
    run_id = uuid4()

    crawler_db = SessionLocal()
    django_db = DjangoSession()

    try:
        # ------------------------------
        # Read standardized factory data
        # ------------------------------
        source_rows = crawler_db.execute(
            SOURCE_SQL,
            {
                "cutoff_jalali": cutoff_jalali,
                "today_jalali": today_jalali,
            },
        ).mappings().all()

        # ------------------------------
        # Production transaction
        # ------------------------------
        with django_db.begin():

            # Prevent concurrent sync runs
            locked = django_db.execute(
                text(
                    "SELECT pg_try_advisory_xact_lock(88421023)"
                )
            ).scalar()

            if not locked:
                raise RuntimeError(
                    "Factory sync is already running"
                )

            model_rows = django_db.execute(
                MODEL_SQL
            ).mappings().all()

            rows, stats = prepare_rows(
                source_rows=source_rows,
                model_rows=model_rows,
                cutoff_jalali=cutoff_jalali,
                today_jalali=today_jalali,
            )

            # Never wipe production if source
            # or matching returned nothing.
            if not rows:
                raise RuntimeError(
                    "No matched factory quotes. "
                    "Production remains unchanged."
                )

            table = Table(
                TABLE_NAME,
                MetaData(),
                autoload_with=django_db.connection(),
            )

            required_columns = {
                "crawler_id",
                "model_variant",
                "sync_run_id",
            }

            if not required_columns.issubset(
                set(table.c.keys())
            ):
                raise RuntimeError(
                    "Run Django migrations first."
                )

            # ------------------------------
            # Batch Upsert
            # ------------------------------
            for batch in batches(rows):

                values = [
                    {
                        **row,
                        "sync_run_id": run_id,
                        "synced_at": now,
                    }
                    for row in batch
                ]

                statement = pg_insert(
                    table
                ).values(values)

                changes = {
                    key: getattr(
                        statement.excluded, key
                    )
                    for key in values[0]
                    if key != "crawler_id"
                }

                statement = (
                    statement.on_conflict_do_update(
                        index_elements=[
                            table.c.crawler_id
                        ],
                        set_=changes,
                    )
                )

                django_db.execute(statement)

            # ------------------------------
            # Clean old redundant quotes
            # ------------------------------
            # All selected quotes received run_id.
            # Old quotes not selected are removed.
            # Latest old quote per product is retained.
            deleted = django_db.execute(
                text(f"""
                    DELETE FROM {TABLE_NAME}
                    WHERE sync_run_id IS DISTINCT FROM
                        CAST(:run_id AS uuid)
                """),
                {"run_id": str(run_id)},
            ).rowcount

        result = {
            "status": "ok",
            "asof": today.isoformat(),
            "upserted": len(rows),
            "deleted": deleted,
            **stats,
        }

        logger.info(
            "Factory price sync finished: %s",
            result,
        )

        return result

    except Exception:
        logger.exception(
            "Factory price sync failed. "
            "Transaction rolled back."
        )
        raise

    finally:
        crawler_db.close()
        django_db.close()

