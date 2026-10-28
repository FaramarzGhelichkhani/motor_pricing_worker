"""Fast PostgreSQL persistence for MotorcyclePriceSurface v1.5.

COPY -> transaction-local staging -> set-based history UPSERT -> scoped
surface DELETE -> set-based INSERT. No changes to core.py or ORM parameters.

Each database commits independently. Crawler keeps full analytical payload;
Production keeps only approved cells and its existing columns.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, inspect, text

log = logging.getLogger(__name__)

GROUP = ["motorcycle_model_id", "real_brand", "real_model", "is_copy", "model_variant"]
HISTORY_NAME = "estimator_modelindexhistory"
SURFACE_NAME = "estimator_pricesurface"


def _num(x):
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _mid(x):
    f = _num(x)
    return int(f) if f is not None and f.is_integer() and f > 0 else None


def _bool(x):
    return isinstance(x, (bool, np.bool_)) and bool(x) or str(x).lower().strip() in ("true", "1", "1.0", "yes", "t")


def _int(x):
    v = _num(x)
    return int(v) if v is not None and v.is_integer() else None


def _money(x):
    v = _num(x)
    if v is None or v <= 0:
        raise ValueError(f"Invalid price: {x!r}")
    return int(round(v))


def _safe(x):
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_safe(v) for v in x]
    if isinstance(x, (datetime, date, pd.Timestamp)):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return _num(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(x, (str, int)):
        return x
    return str(x)


def _identity(row):
    mid = _mid(row["motorcycle_model_id"])
    values = [
        str(mid) if mid is not None else "UNKNOWN",
        str(row["real_brand"]), str(row["real_model"]),
        "1" if _bool(row["is_copy"]) else "0",
        str(row["model_variant"]),
    ]
    return hashlib.md5("\x1f".join(values).encode("utf-8")).hexdigest()


def _cell_key(row):
    values = [str(_int(row["age"])), str(_int(row["mileage_bucket"])),
              str(_int(row["condition_tier"])), "مشکی"]
    return hashlib.md5("\x1f".join(values).encode("utf-8")).hexdigest()


def _as_copy_field(value):
    """CSV COPY uses quoted strings and a dedicated NULL marker."""
    if value is None:
        return "\\N"
    if isinstance(value, (dict, list)):
        return json.dumps(_safe(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(value)


def _copy_rows(sa_conn, target_table, columns, records, buffer_rows=10000):
    """Stream batches over ONE COPY cursor and ONE database transaction.

    Supports psycopg2 and psycopg3; does not call DBAPI commit/rollback.
    """
    if not records:
        return 0
    dbapi_conn = sa_conn.connection.driver_connection
    driver = dbapi_conn.__class__.__module__
    cols_sql = ", ".join('"' + col + '"' for col in columns)
    statement = f"COPY {target_table} ({cols_sql}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    count = 0

    def chunks():
        nonlocal count
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for item in records:
            # Force quoting actual string '\\N' to distinguish it from NULL.
            row = []
            for col in columns:
                value = item.get(col)
                if value is None:
                    row.append(None)
                else:
                    value = _as_copy_field(value)
                    row.append(value)
            # csv.writer writes None as empty field; use explicit NULL marker.
            # The marker is reserved and ordinary string collisions are rejected.
            serialized = []
            for value in row:
                if value is None:
                    serialized.append("\\N")
                elif value == "\\N":
                    raise ValueError("COPY payload contains reserved NULL marker")
                else:
                    serialized.append(value)
            writer.writerow(serialized)
            count += 1
            if count % buffer_rows == 0:
                yield output.getvalue()
                output = io.StringIO(newline="")
                writer = csv.writer(output, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        if output.tell():
            yield output.getvalue()

    # 10k-row COPY commands: small enough to limit connection/statement risk,
    # far fewer round-trips than hundreds of INSERT batches.
    with dbapi_conn.cursor() as cursor:
        if hasattr(cursor, "copy_expert"):  # psycopg2
            for part in chunks():
                cursor.copy_expert(statement, io.StringIO(part))
        elif hasattr(cursor, "copy"):  # psycopg3
            for part in chunks():
                with cursor.copy(statement) as cp:
                    cp.write(part)
        else:
            raise RuntimeError(f"COPY requires psycopg2 or psycopg3, got {driver}")
    return count


class SurfaceDatabaseMixin:
    def _db_tables(self, conn, target):
        md = MetaData()
        h = Table(HISTORY_NAME, md, autoload_with=conn)
        s = Table(SURFACE_NAME, md, autoload_with=conn)
        hc, sc = set(h.c.keys()), set(s.c.keys())
        base_h = {"id", "date", "motorcycle_model_id", "is_copy", "model_variant", "price_low", "price_mid", "price_high", "algorithm", "is_reliable"}
        base_s = {"snapshot_id", "motorcycle_model_id", "year", "mileage_bucket", "condition_tier", "color", "is_copy", "model_variant", "price_low", "price_mid", "price_high"}
        if target == "crawler":
            base_h |= {"identity_key", "real_brand", "real_model", "market_payload", "total_surface_cells", "publishable_cells"}
            base_s |= {"real_brand", "real_model", "age", "cell_key", "reliability_tier", "is_reliable", "publish_state", "is_factory", "estimated_mape_pct", "estimated_interval_coverage_pct", "isotonic_change_pct", "mileage_clean"}

        missing = (base_h - hc, base_s - sc)
        if any(missing):
            raise RuntimeError(f"Incomplete {target} schema; history={sorted(missing[0])}, surface={sorted(missing[1])}")
        # Upsert requires a matching PostgreSQL unique constraint/index.
        uniques = set()
        insp = inspect(conn)
        for item in insp.get_unique_constraints(HISTORY_NAME):
            uniques.add(frozenset(item["column_names"]))
        for item in insp.get_indexes(HISTORY_NAME):
            if item.get("unique"):
                uniques.add(frozenset(item["column_names"]))
        identity = ({"date", "identity_key"} if target == "crawler" else {"date", "motorcycle_model_id", "is_copy", "model_variant"})
        if frozenset(identity) not in uniques:
            raise RuntimeError(f"{target} history needs UNIQUE index on {sorted(identity)}")
        return h, s

    def _prepare_frames(self, target):
        if self._surface_df is None or self._model_index_history_data is None:
            raise RuntimeError("Run fit() and create_surface() before database persistence")
        s = self._surface_df.copy()
        h = self._model_index_history_data.copy()
        s["_gid"] = s.apply(_identity, axis=1)
        h["_gid"] = h.apply(_identity, axis=1)
        if target == "production":
            reliable = s["is_reliable"].map(_bool)
            s = s.loc[reliable].copy()
            if "isotonic_block" in s:
                s = s.loc[~s["isotonic_block"].map(_bool) | s["is_factory"].map(_bool)].copy()
            ids = s["motorcycle_model_id"].map(_mid)
            years = pd.to_numeric(s["production_year"], errors="coerce")
            s = s.loc[ids.notna() & years.between(1300, 1600) & years.mod(1).eq(0)].copy()
        h = h[h["_gid"].isin(s["_gid"])].copy()
        if h["_gid"].duplicated().any():
            raise ValueError("Duplicate history identities; fix core.py normalization, do not silently deduplicate")
        if not set(s["_gid"]).issubset(set(h["_gid"])):
            raise ValueError("Surface group without matching ModelIndexHistory")
        if target == "production" and not h.empty:
            keys = h.apply(lambda r: (_mid(r["motorcycle_model_id"]), _bool(r["is_copy"]), str(r["model_variant"])), axis=1)
            if keys.duplicated().any():
                raise ValueError("Production model ID/copy/variant identity collision")
        return h, s

    def _history_records(self, target, h, s):
        count = s.groupby("_gid").size().to_dict()
        publish = s.loc[s["is_reliable"].map(_bool)].groupby("_gid").size().to_dict()
        tiers = (s.groupby(["_gid", "reliability_tier"]).size().unstack(fill_value=0).to_dict("index") if target == "crawler" else {})
        records = []
        for r in h.to_dict("records"):
            gid = r["_gid"]
            prices = [_num(r.get(c)) for c in ("price_p10", "price_p50", "price_p90")]
            if any(v is None or v <= 0 for v in prices):
                local = pd.to_numeric(s.loc[s["_gid"].eq(gid), "adjusted_price"], errors="coerce").dropna()
                if local.empty:
                    raise ValueError(f"Missing history market prices: {gid}")
                prices = [float(local.quantile(q)) for q in (0.1, 0.5, 0.9)]
            record = dict(
                motorcycle_model_id=_mid(r["motorcycle_model_id"]), date=self.today,
                price_low=_money(prices[0]), price_mid=_money(prices[1]), price_high=_money(prices[2]),
                color_coefficients={}, mape=_num(r.get("mape")),
                coverage_percent=_num(r.get("coverage_percent")),
                shift=_num(r.get("optimal_shift")) or 0.0,
                sample_count=_int(r.get("total_samples")) or 0,
                algorithm=self.algo, is_reliable=publish.get(gid, 0) > 0,
                is_copy=_bool(r["is_copy"]), model_variant=str(r["model_variant"]),
            )
            if target == "crawler":
                record.update(identity_key=gid, real_brand=str(r["real_brand"]), real_model=str(r["real_model"]),
                              total_surface_cells=count.get(gid, 0), publishable_cells=publish.get(gid, 0),
                              market_payload={"statistics": {k: _safe(v) for k, v in r.items() if not k.startswith("_")},
                                              "tier_counts": tiers.get(gid, {})})
            records.append(record)
        return records

    def _surface_records(self, target, s):
        for row in s.to_dict("records"):
            record = dict(motorcycle_model_id=_mid(row["motorcycle_model_id"]),
                          year=_int(row.get("production_year")),
                          mileage_bucket=_int(row.get("mileage_bucket")),
                          condition_tier=_int(row.get("condition_tier")),
                          color="مشکی", is_copy=_bool(row["is_copy"]),
                          model_variant=str(row["model_variant"]),
                          price_low=_money(row["suggested_lower"]),
                          price_mid=_money(row["adjusted_price"]),
                          price_high=_money(row["suggested_upper"]))
            if target == "crawler":
                record.update(_gid=row["_gid"], real_brand=str(row["real_brand"]),
                              real_model=str(row["real_model"]), cell_key=_cell_key(row),
                              age=_int(row.get("age")),
                              reliability_tier=str(row["reliability_tier"]),
                              is_reliable=_bool(row["is_reliable"]),
                              publish_state=str(row.get("publish_state", "")),
                              is_factory=_bool(row.get("is_factory")),
                              estimated_mape_pct=_num(row.get("estimated_mape_pct")),
                              estimated_interval_coverage_pct=_num(row.get("estimated_interval_coverage_pct")),
                              isotonic_change_pct=_num(row.get("isotonic_change_pct")),
                              mileage_clean=_num(row.get("mileage_clean"))
                              )
            yield record

    def _save_target(self, db, target):
        if db is None:
            return None
        t0 = time.perf_counter()
        history_df, surface_df = self._prepare_frames(target)
        if surface_df.empty:
            log.info("[DB][%s] No target cells: no DELETE, no writes", target)
            return {"target": target, "history_count": 0, "deleted": 0, "inserted": 0}
        history_records = self._history_records(target, history_df, surface_df)
        surface_records = self._surface_records(target, surface_df)
        # Session has a single active transaction for all operations.
        try:
            conn = db.connection()
            h, s = self._db_tables(conn, target)
            hcols = list(history_records[0])
            scols = (list(next(self._surface_records(target, surface_df))) if not surface_df.empty else [])
            # Temporary tables only exist on the session's PostgreSQL connection.
            conn.exec_driver_sql(f"CREATE TEMP TABLE _mps_hist_stage ON COMMIT DROP AS SELECT * FROM {HISTORY_NAME} WITH NO DATA")
            conn.exec_driver_sql(f"CREATE TEMP TABLE _mps_surface_stage ON COMMIT DROP AS SELECT * FROM {SURFACE_NAME} WITH NO DATA")
            if target == "crawler":
                conn.exec_driver_sql("ALTER TABLE _mps_surface_stage ADD COLUMN _gid text")
            n_h = _copy_rows(conn, "_mps_hist_stage", hcols, history_records)
            n_s = _copy_rows(conn, "_mps_surface_stage", scols, surface_records)
            log.info("[DB][%s] COPY staged history=%s surfaces=%s (%.1fs)", target, n_h, n_s, time.perf_counter()-t0)
            # For ON CONFLICT after SELECT, WHERE TRUE resolves parser ambiguity.
            conflict = (["date", "identity_key"] if target == "crawler"
                        else ["date", "motorcycle_model_id", "is_copy", "model_variant"])
            upd = [c for c in hcols if c not in conflict]
            if "updated_at" in h.c:
                upd.append("updated_at")
            insert_cols = ", ".join('"' + c + '"' for c in hcols)
            conflict_cols = ", ".join('"' + c + '"' for c in conflict)
            updates = ", ".join(f'"{c}" = ' + ("now()" if c == "updated_at" else f'EXCLUDED."{c}"') for c in upd)
            conn.exec_driver_sql(f"INSERT INTO {HISTORY_NAME} ({insert_cols}) SELECT {insert_cols} FROM _mps_hist_stage WHERE TRUE ON CONFLICT ({conflict_cols}) DO UPDATE SET {updates}")
            if target == "crawler":
                matched = "h.identity_key = st._gid AND h.date = :asof"
            else:
                matched = ("h.motorcycle_model_id = st.motorcycle_model_id AND h.is_copy = st.is_copy "
                           "AND h.model_variant = st.model_variant AND h.date = :asof")
            absent = conn.execute(text(f"SELECT count(*) FROM _mps_surface_stage st LEFT JOIN {HISTORY_NAME} h ON {matched} WHERE h.id IS NULL"), {"asof": self.today}).scalar_one()
            if absent:
                raise ValueError(f"{target}: {absent} surfaces lack matching history snapshots")
            deleted = conn.exec_driver_sql(
                f"DELETE FROM {SURFACE_NAME} p WHERE p.motorcycle_model_id IN "
                "(SELECT DISTINCT motorcycle_model_id FROM _mps_surface_stage WHERE motorcycle_model_id IS NOT NULL)"
            ).rowcount
            if target == "crawler":
                deleted += conn.exec_driver_sql(
                    f"DELETE FROM {SURFACE_NAME} p USING {HISTORY_NAME} oldh "
                    "WHERE p.snapshot_id = oldh.id AND p.motorcycle_model_id IS NULL "
                    "AND oldh.identity_key IN (SELECT DISTINCT _gid FROM _mps_surface_stage WHERE motorcycle_model_id IS NULL)"
                ).rowcount
            selected = [c for c in scols if c != "_gid"]
            selected_sql = ", ".join('"' + c + '"' for c in selected)
            target_cols = '"snapshot_id", ' + selected_sql
            source_cols = ", ".join('st."' + c + '"' for c in selected)
            conn.execute(text(
                f"INSERT INTO {SURFACE_NAME} ({target_cols}) "
                f"SELECT h.id, {source_cols} FROM _mps_surface_stage st "
                f"JOIN {HISTORY_NAME} h ON {matched}"
            ), {"asof": self.today})
            # All writes to this DB are committed together.
            db.commit()
            report = {"target": target, "history_count": n_h, "deleted": max(deleted, 0),
                      "inserted": n_s, "motorcycle_count": len({_mid(x) for x in surface_df["motorcycle_model_id"] if _mid(x) is not None}),
                      "seconds": round(time.perf_counter()-t0, 2)}
            log.info("[DB][%s] COMMIT %s", target, report)
            return report
        except Exception:
            db.rollback()
            log.exception("[DB][%s] transaction rolled back", target)
            raise

    def update_database(self):
        if self.crawler_db is None:
            raise RuntimeError("Crawler database Session is required")
        result = {"crawler": self._save_target(self.crawler_db, "crawler")}
        if self.production_db is not None:
            result["production"] = self._save_target(self.production_db, "production")
        return result
