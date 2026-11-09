"""Crawler -> Django canonical factory prices, one identity per model + variant.

Use as the daily factory sync. Requires the existing:
  core.database.SessionLocal / DjangoSession
  ml_engine.variant_labeling_v2._set_model_variant
  dictionary.mappings.BRAND_NAME_MAP (optional but recommended)
  jdatetime, SQLAlchemy/PostgreSQL.

No used/ML tables are touched. No factory-class dimension is used in published
prices. If old Django columns motor_class / motor_class_key still exist,
motor_class_key is written as empty string solely for schema compatibility.
"""
from __future__ import annotations

import logging
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from itertools import islice
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ml_engine.factory_variant import brand_for_row, normalize, variant_from_sources

try:
    from dictionary.mappings import BRAND_NAME_MAP
except ImportError:
    # The production repo has this module; factory_variant still knows the two
    # explicit company labels found in the attached factory data.
    BRAND_NAME_MAP = {}

logger = logging.getLogger(__name__)
TEHRAN = ZoneInfo('Asia/Tehran')
TABLE_NAME = 'motor_factory_prices'
BATCH_SIZE = 500

SOURCE_SQL = text("""
WITH ranked AS (
    SELECT id, brand_name, model_name, motor_class, price, jalali_date,
           insert_date, source, price_provider, raw_title,
           ROW_NUMBER() OVER (
               PARTITION BY brand_name, model_name,
                   COALESCE(NULLIF(BTRIM(motor_class), ''), ''),
                   COALESCE(NULLIF(BTRIM(raw_title), ''), '')
               ORDER BY jalali_date DESC, insert_date DESC NULLS LAST, id DESC
           ) AS latest_rank
    FROM motorcycle_zero_prices
    WHERE is_standardized IS TRUE
      AND price > 0
      AND jalali_date ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
      AND jalali_date <= :today_jalali
)
SELECT id, brand_name, model_name, motor_class, price, jalali_date,
       insert_date, source, price_provider, raw_title
FROM ranked
WHERE jalali_date >= :cutoff_jalali OR latest_rank = 1
""")

MODEL_SQL = text("""
SELECT m.id AS motorcycle_model_id, b.name_en AS brand_name,
       m.name_en AS model_name
FROM motor_motormodel m
JOIN motor_brand b ON m.brand_id = b.id
WHERE m.parent_id IS NOT NULL
""")


def build_catalog(model_rows, aliases=None):
    """No fuzzy matches: price must not be attached to a wrong model."""
    catalog, ambiguous = {}, set()
    for row in model_rows:
        from ml_engine.factory_variant import normalize_brand
        key = (normalize_brand(row['brand_name'], aliases), normalize(row['model_name']))
        if not all(key): continue
        mid = int(row['motorcycle_model_id'])
        if key in catalog and catalog[key] != mid: ambiguous.add(key)
        else: catalog[key] = mid
    for key in ambiguous: catalog.pop(key,None)
    return catalog, ambiguous


def jalali_to_gregorian(value):
    import jdatetime
    try:
        year, month, day = map(int, str(value).split('-'))
        return jdatetime.date(year, month, day).togregorian()
    except (ValueError, TypeError, OverflowError):
        return None


def batches(items, size=BATCH_SIZE):
    it = iter(items)
    while batch := list(islice(it,size)):
        yield batch


def extract_model_variants(rows):
    """Call user's EXISTING v2 classifier, then canonicalize factory-only classes."""
    if not rows:return []
    from ml_engine.variant_labeling_v2 import _set_model_variant
    frame = pd.DataFrame({
        'real_brand':[r['_canonical_brand'] for r in rows],
        'real_model':[r['model_name'] for r in rows],
        'motorcycle_model_id':[r['_model_id'] for r in rows],
        'is_copy':[0]*len(rows),
        'factory_variant_text':[
            ' '.join(str(p).strip() for p in (r.get('raw_title'),r.get('motor_class'))
                     if p is not None and str(p).strip()) for r in rows
        ],
    })
    labelled = _set_model_variant(frame,text_col='factory_variant_text',alignment='preserve')
    return [variant_from_sources(row,var,conflict,BRAND_NAME_MAP)
            for row,var,conflict in zip(rows,labelled.model_variant,labelled.variant_conflict)]


def _quote_rank(row):
    """Newest effective price date, source insertion time, then crawler ID."""
    insert = row.get('insert_date')
    # All known incoming date times are strings/datetimes, with nulls last.
    if isinstance(insert,datetime): insert = insert.isoformat()
    return (str(row.get('jalali_date') or ''),str(insert or ''),int(row['id']))


def _source_priority(row, canonical_brand):
    """Ties: prefer a manufacturer's direct feed when brand matches its maker.

    This only distinguishes Niroo Motor's own niroomotor source from bama.
    Other source names must NOT be claimed to be official by default.
    """
    return int(canonical_brand=='niroo motor' and str(row.get('source')).casefold()=='niroomotor')


def prepare_rows(source_rows, model_rows, cutoff_jalali, today_jalali, *, strict_price_conflicts=True):
    """Return canonical quotes + auditable stats; never use factory class as identity.

    - Resolve company from class BEFORE matching model id, reject nonexistent IDs.
    - Use variant_v2 to parse features and factory class-specific vocabulary.
    - Collapse alias spelling and retain newest quote per canonical product/day.
    - If two prices disagree by >20% in same day, use only a trustworthy
      unambiguous source priority (native-brand vs remapped; known maker feed);
      otherwise quarantine that product rather than silently choosing a price.
    - Keep every day of last 15 calendar dates, else latest old day only.
    """
    catalog, ambiguous = build_catalog(model_rows, BRAND_NAME_MAP)
    stats=Counter(); matched=[]
    for original in source_rows:
        r=dict(original)
        try: valid_price=int(r.get('price') or 0)>0
        except (ValueError,TypeError,OverflowError):valid_price=False
        if not valid_price: stats['invalid_price']+=1;continue
        day=str(r.get('jalali_date') or '')
        if day>today_jalali: stats['future_date']+=1;continue
        if len(day)!=10 or day[4]!='-' or day[7]!='-':
            stats['invalid_date']+=1;continue
        price_date=jalali_to_gregorian(day)
        if price_date is None: stats['invalid_date']+=1;continue
        canonical_brand=brand_for_row(r,BRAND_NAME_MAP)
        key=(canonical_brand,normalize(r.get('model_name')))
        mid=catalog.get(key)
        if mid is None:
            stats['unmatched_reassigned_brand' if canonical_brand != normalize(r.get('brand_name')) else
                  'ambiguous_catalog' if key in ambiguous else 'unmatched_catalog']+=1
            continue
        r['_model_id']=mid
        r['_price_date']=price_date
        r['_canonical_brand']=canonical_brand
        r['_brand_reassigned']=(canonical_brand!=normalize(r.get('brand_name')))
        matched.append(r)
        if r['_brand_reassigned']:stats['matched_reassigned_brand']+=1

    labels=extract_model_variants(matched)
    buckets=defaultdict(list)
    for row,label in zip(matched,labels):
        if label.status != 'ok': stats[label.status]+=1;continue
        if label.brand != row['_canonical_brand']:
            stats['brand_resolution_mismatch']+=1;continue
        variant=label.model_variant
        if len(variant)>255:stats['variant_too_long']+=1;continue
        if label.note:stats[label.note]+=1
        key=(row['_model_id'],variant,str(row['jalali_date']))
        buckets[key].append(row)

    per_product=defaultdict(list)
    conflict_examples=[]
    conflicting_keys=set()
    for (mid,variant,day),quotes in buckets.items():
        if len(quotes)>1: stats['same_day_alias_collisions']+=1
        prices=[int(q['price']) for q in quotes]
        high_spread=max(prices)>1.2*min(prices)
        # Provenance matters: a vendor-brand name in class is weaker evidence
        # than source model's native brand, especially when prices diverge.
        native=[q for q in quotes if not q['_brand_reassigned']]
        if high_spread and native and len(native)!=len(quotes):
            quotes=native
            stats['reassigned_brand_price_conflict_days']+=1
        # Prefer official Niroo Motor feed over bama when the two disagree.
        official=[q for q in quotes if _source_priority(q,q['_canonical_brand'])]
        if high_spread and official and len(official)!=len(quotes):
            quotes=official
            stats['official_feed_wins_conflict_days']+=1
        prices=[int(q['price']) for q in quotes]
        if strict_price_conflicts and len(quotes)>1 and max(prices)>1.2*min(prices):
            stats['unresolved_conflict_days']+=1
            conflicting_keys.add((mid,variant))
            if len(conflict_examples)<25:
                conflict_examples.append({'model_id':mid,'variant':variant,'date':day,
                                          'prices':sorted(set(prices)),'crawler_ids':[q['id'] for q in quotes]})
            continue
        # Always choose most recently inserted quote after safe disambiguation.
        chosen=max(quotes,key=_quote_rank)
        stats['deduplicated_source_rows']+=len(quotes)-1
        per_product[(mid,variant)].append((day,chosen))

    selected=[]
    stats['quarantined_products']=len(conflicting_keys)
    for (mid,variant),days in per_product.items():
        if (mid,variant) in conflicting_keys:
            continue
        days.sort(key=lambda t:_quote_rank(t[1]),reverse=True)
        for i,(day,row) in enumerate(days):
            if i and day<cutoff_jalali:continue
            selected.append({
                'crawler_id':int(row['id']),
                'motorcycle_model_id':int(mid),
                'model_variant':variant,
                'price':int(row['price']),
                'jalali_date':day,
                'price_date':row['_price_date'],
                'source':row['source'],
                'price_provider':row.get('price_provider'),
                'raw_title':row.get('raw_title'),
                'source_insert_date':row.get('insert_date'),
            })
    selected.sort(key=lambda r:(r['motorcycle_model_id'],r['model_variant'],r['price_date'],str(r['source_insert_date'] or ''),r['crawler_id']))
    unique_cells = {(r['motorcycle_model_id'],r['model_variant'],r['price_date']) for r in selected}
    if len(unique_cells) != len(selected):
        raise RuntimeError('Factory canonical product/day uniqueness broken')
    stats['matched_pre_variant']=len(matched)
    stats['selected_quotes']=len(selected)
    stats['published_product_variants']=len({(r['motorcycle_model_id'],r['model_variant']) for r in selected})
    stats['conflict_examples']=conflict_examples
    return selected,dict(stats)


def sync_factory_prices(asof:date|None=None):
    """Daily production sync; atomic upsert+delete in ONE Django DB transaction."""
    from core.database import DjangoSession,SessionLocal
    import jdatetime
    today=asof or datetime.now(TEHRAN).date()
    cutoff=today-timedelta(days=14)
    today_jalali=jdatetime.date.fromgregorian(date=today).strftime('%Y-%m-%d')
    cutoff_jalali=jdatetime.date.fromgregorian(date=cutoff).strftime('%Y-%m-%d')
    now=datetime.now(TEHRAN);run_id=uuid4()
    crawler_db=SessionLocal(); django_db=DjangoSession()
    try:
        source=crawler_db.execute(SOURCE_SQL,{'cutoff_jalali':cutoff_jalali,'today_jalali':today_jalali}).mappings().all()
        with django_db.begin():
            locked=django_db.execute(text('SELECT pg_try_advisory_xact_lock(88421023)')).scalar()
            if not locked:raise RuntimeError('Factory sync is already running')
            models=django_db.execute(MODEL_SQL).mappings().all()
            rows,stats=prepare_rows(source,models,cutoff_jalali,today_jalali)
            if not rows:raise RuntimeError('No safe canonical factory prices; production unchanged')
            table=Table(TABLE_NAME,MetaData(),autoload_with=django_db.connection())
            required={'crawler_id','motorcycle_model_id','model_variant','sync_run_id'}
            if not required.issubset(table.c.keys()):raise RuntimeError('Run model_variant migration first')
            # Remove stale crawler IDs *before* upsert. This is necessary when
            # optional UNIQUE(model,variant,date) is installed: an old quote may
            # occupy the canonical slot while the newer crawler_id differs.
            # The deletion and all subsequent writes are in this SAME transaction;
            # any exception rolls everything back. Keep unchanged crawler IDs.
            django_db.execute(text('''CREATE TEMP TABLE _factory_keep_ids (
                crawler_id BIGINT PRIMARY KEY
            ) ON COMMIT DROP'''))
            for keep_batch in batches(rows):
                django_db.execute(
                    text('INSERT INTO _factory_keep_ids(crawler_id) VALUES (:cid)'),
                    [{'cid': int(r['crawler_id'])} for r in keep_batch],
                )
            deleted = django_db.execute(text(f'''DELETE FROM {TABLE_NAME} q
                WHERE NOT EXISTS (
                    SELECT 1 FROM _factory_keep_ids k
                    WHERE k.crawler_id = q.crawler_id
                )''')).rowcount
            # Retain backward schema compatibility, but DO NOT store or query the
            # raw class: model_variant alone uniquely identifies the product.
            def db_row(r):
                x={**r,'synced_at':now,'sync_run_id':run_id}
                if 'motor_class_key' in table.c:x['motor_class_key']=''
                if 'motor_class' in table.c:x['motor_class']=None
                return x
            for batch in batches(rows):
                values=[db_row(r) for r in batch]
                stmt=pg_insert(table).values(values)
                changes={k:getattr(stmt.excluded,k) for k in values[0] if k!='crawler_id'}
                django_db.execute(stmt.on_conflict_do_update(index_elements=[table.c.crawler_id],set_=changes))
        result={'status':'ok','asof':today.isoformat(),'upserted':len(rows),'deleted':deleted,**stats}
        logger.info('Factory sync completed: %s',result)
        return result
    except Exception:
        logger.exception('Factory sync rolled back');raise
    finally:
        crawler_db.close();django_db.close()
