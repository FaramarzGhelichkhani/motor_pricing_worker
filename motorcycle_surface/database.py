"""Bounded-memory PostgreSQL persistence for MotorcyclePriceSurface.

Batched history UPSERT -> resolved snapshot IDs -> scoped surface DELETE ->
streaming COPY directly into PriceSurface. A shared Engine uses one transaction;
separate PostgreSQL Engines use durable two-phase commit coordinated from the
crawler database. Crawler keeps full analytical payload; Production keeps
approved cells only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import time
import uuid
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, bindparam, func, inspect, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

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



def _identity_values(frame):
    """Vectorized-column identity hashing; avoids DataFrame.apply(axis=1)."""
    mids = pd.to_numeric(frame["motorcycle_model_id"], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan
    )
    brands = frame["real_brand"].astype(str).to_numpy(dtype=object)
    models = frame["real_model"].astype(str).to_numpy(dtype=object)
    copies = frame["is_copy"].to_numpy(dtype=object)
    variants = frame["model_variant"].astype(str).to_numpy(dtype=object)
    values = []
    for mid, brand, model, is_copy, variant in zip(
        mids, brands, models, copies, variants
    ):
        model_id = (
            str(int(mid))
            if np.isfinite(mid) and float(mid).is_integer() and mid > 0
            else "UNKNOWN"
        )
        copy_flag = "1" if _bool(is_copy) else "0"
        payload = "\x1f".join(
            (model_id, str(brand), str(model), copy_flag, str(variant))
        )
        values.append(hashlib.md5(payload.encode("utf-8")).hexdigest())
    return np.asarray(values, dtype=object)


def _rss_mb():
    """Best-effort current resident memory for operational logs."""
    try:
        with open("/proc/self/status", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return round(float(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _log_db_stage(target, stage, started, **fields):
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    rss = _rss_mb()
    suffix = f" rss_mb={rss}" if rss is not None else ""
    log.info(
        "[DB][%s] %s | elapsed=%.2fs%s%s",
        target, stage, time.perf_counter() - started, suffix,
        f" | {details}" if details else "",
    )


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





def _copy_rows(
    sa_conn, target_table, columns, records, buffer_rows=10000,
    *, progress_label=None,
):
    """Bounded-memory PostgreSQL COPY using at most buffer_rows records."""
    if records is None:
        return 0
    dbapi_conn = sa_conn.connection.driver_connection
    driver = dbapi_conn.__class__.__module__
    cols_sql = ", ".join('"' + col + '"' for col in columns)
    table_sql = '"' + str(target_table).replace('"', '""') + '"'
    statement = (
        f"COPY {table_sql} ({cols_sql}) "
        "FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    )
    count = 0

    def chunks():
        nonlocal count
        output = io.StringIO(newline="")
        writer = csv.writer(
            output, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
        )
        batch_rows = 0
        for item in records:
            serialized = []
            for col in columns:
                value = item.get(col)
                if value is None:
                    serialized.append("\\N")
                    continue
                value = _as_copy_field(value)
                if value == "\\N":
                    raise ValueError("COPY payload contains reserved NULL marker")
                serialized.append(value)
            writer.writerow(serialized)
            count += 1
            batch_rows += 1
            if batch_rows >= buffer_rows:
                yield output.getvalue(), batch_rows
                output = io.StringIO(newline="")
                writer = csv.writer(
                    output, lineterminator="\n", quoting=csv.QUOTE_MINIMAL
                )
                batch_rows = 0
        if batch_rows:
            yield output.getvalue(), batch_rows

    with dbapi_conn.cursor() as cursor:
        if hasattr(cursor, "copy_expert"):  # psycopg2
            for part, batch_count in chunks():
                cursor.copy_expert(statement, io.StringIO(part))
                if progress_label:
                    log.info(
                        "[DB][COPY] %s rows=%s total=%s rss_mb=%s",
                        progress_label, batch_count, count, _rss_mb(),
                    )
        elif hasattr(cursor, "copy"):  # psycopg3
            for part, batch_count in chunks():
                with cursor.copy(statement) as copy:
                    copy.write(part)
                if progress_label:
                    log.info(
                        "[DB][COPY] %s rows=%s total=%s rss_mb=%s",
                        progress_label, batch_count, count, _rss_mb(),
                    )
        else:
            raise RuntimeError(
                f"COPY requires psycopg2 or psycopg3, got {driver}"
            )
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
        """Build narrow persistence frames without copying the full surface."""
        if self._surface_df is None or self._model_index_history_data is None:
            raise RuntimeError(
                "Run fit() and create_surface() before database persistence"
            )
        if target not in {"crawler", "production"}:
            raise ValueError(f"Unknown database target: {target}")

        source_s = self._surface_df
        source_h = self._model_index_history_data
        if "is_factory" in source_s and source_s["is_factory"].map(_bool).any():
            raise ValueError(
                "Factory data must be stored in its own table, never PriceSurface"
            )

        surface_gids = _identity_values(source_s)
        history_gids = _identity_values(source_h)

        if target == "production":
            if not source_s["policy_version"].eq(
                "USED_RELEASE_FINAL_6_ALL_AGE_DUPLICATE_QUALITY_GUARD_2PCT"
            ).all():
                raise ValueError(
                    "Production requires USED_RELEASE_FINAL_6_ALL_AGE_DUPLICATE_QUALITY_GUARD_2PCT "
                    "decisions for every row"
                )
            if source_s["is_factory"].map(_bool).any():
                raise ValueError(
                    "Factory rows must not be persisted to PriceSurface"
                )
            reliable = source_s["is_reliable"].map(_bool)
            allowed = source_s["publication_allowed"].map(_bool)
            if not np.array_equal(reliable.to_numpy(), allowed.to_numpy()):
                raise ValueError("Inconsistent publication flags")
            ids = source_s["motorcycle_model_id"].map(_mid)
            years = pd.to_numeric(source_s["production_year"], errors="coerce")
            valid_year = (
                years.between(1300, 1600).fillna(False)
                & years.mod(1).eq(0).fillna(False)
            )
            keep_mask = (
                reliable & ids.notna() & valid_year
            ).to_numpy(dtype=bool)
        else:
            keep_mask = np.ones(len(source_s), dtype=bool)

        common = [
            "motorcycle_model_id", "production_year", "mileage_bucket",
            "condition_tier", "is_copy", "model_variant", "suggested_lower",
            "adjusted_price", "suggested_upper", "is_reliable",
        ]
        crawler_extra = [
            "real_brand", "real_model", "age", "reliability_tier",
            "publish_state", "is_factory", "estimated_mape_pct",
            "estimated_interval_coverage_pct", "isotonic_change_pct",
            "mileage_clean",
        ]
        surface_columns = common + (
            crawler_extra if target == "crawler" else []
        )
        missing = set(surface_columns) - set(source_s.columns)
        if missing:
            raise ValueError(
                f"{target} surface is missing persistence fields: {sorted(missing)}"
            )

        keep_positions = np.flatnonzero(keep_mask)
        surface_df = source_s.iloc[keep_positions][surface_columns].copy()
        selected_gids = surface_gids[keep_mask]
        surface_df["_gid"] = selected_gids

        history_mask = np.isin(history_gids, selected_gids)
        history_positions = np.flatnonzero(history_mask)
        history_df = source_h.iloc[history_positions].copy()
        history_df["_gid"] = history_gids[history_mask]

        if history_df["_gid"].duplicated().any():
            raise ValueError(
                "Duplicate history identities; fix core.py normalization, "
                "do not silently deduplicate"
            )
        if len(selected_gids):
            missing_history = set(selected_gids) - set(history_df["_gid"])
            if missing_history:
                raise ValueError(
                    "Surface group without matching ModelIndexHistory"
                )

        if target == "production" and not history_df.empty:
            keys = pd.DataFrame({
                "motorcycle_model_id": history_df["motorcycle_model_id"].map(_mid),
                "is_copy": history_df["is_copy"].map(_bool),
                "model_variant": history_df["model_variant"].astype(str),
            })
            if keys.duplicated().any():
                raise ValueError(
                    "Production model ID/copy/variant identity collision"
                )
        return history_df, surface_df

    def _history_records(self, target, h, s):
        """History payload is small; surface aggregates are computed once."""
        count = s.groupby("_gid", sort=False).size().to_dict()
        publish = (
            s.loc[s["is_reliable"].map(_bool)]
            .groupby("_gid", sort=False).size().to_dict()
        )
        tiers = (
            s.groupby(["_gid", "reliability_tier"], sort=False)
            .size().unstack(fill_value=0).to_dict("index")
            if target == "crawler" else {}
        )

        price_frame = s[["_gid", "adjusted_price"]].copy()
        price_frame["adjusted_price"] = pd.to_numeric(
            price_frame["adjusted_price"], errors="coerce"
        )
        price_frame = price_frame.loc[
            price_frame["adjusted_price"].gt(0)
            & np.isfinite(price_frame["adjusted_price"])
        ]
        quantiles = (
            price_frame.groupby("_gid", sort=False)["adjusted_price"]
            .quantile([0.1, 0.5, 0.9]).unstack()
            if not price_frame.empty else pd.DataFrame()
        )

        records = []
        columns = list(h.columns)
        for values in h.itertuples(index=False, name=None):
            r = dict(zip(columns, values))
            gid = r["_gid"]
            prices = [
                _num(r.get("price_p10")),
                _num(r.get("price_p50")),
                _num(r.get("price_p90")),
            ]
            if any(value is None or value <= 0 for value in prices):
                if gid not in quantiles.index:
                    raise ValueError(f"Missing history market prices: {gid}")
                prices = quantiles.loc[gid, [0.1, 0.5, 0.9]].to_list()

            record = dict(
                motorcycle_model_id=_mid(r["motorcycle_model_id"]),
                date=self.today,
                price_low=_money(prices[0]),
                price_mid=_money(prices[1]),
                price_high=_money(prices[2]),
                color_coefficients={},
                mape=_num(r.get("mape")),
                coverage_percent=_num(r.get("coverage_percent")),
                shift=_num(r.get("optimal_shift")) or 0.0,
                sample_count=_int(r.get("total_samples")) or 0,
                algorithm=self.algo,
                is_reliable=publish.get(gid, 0) > 0,
                is_copy=_bool(r["is_copy"]),
                model_variant=str(r["model_variant"]),
            )
            if target == "crawler":
                record.update(
                    identity_key=gid,
                    real_brand=str(r["real_brand"]),
                    real_model=str(r["real_model"]),
                    total_surface_cells=count.get(gid, 0),
                    publishable_cells=publish.get(gid, 0),
                    market_payload={
                        "statistics": {
                            key: _safe(value)
                            for key, value in r.items()
                            if not key.startswith("_")
                        },
                        "tier_counts": tiers.get(gid, {}),
                    },
                )
            records.append(record)
        return records


    def _surface_records(self, target, s, snapshot_by_gid):
        """Yield target-table rows with the resolved history foreign key."""
        positions = {name: s.columns.get_loc(name) for name in s.columns}
        p = positions
        for row in s.itertuples(index=False, name=None):
            gid = row[p["_gid"]]
            snapshot_id = snapshot_by_gid.get(gid)
            if snapshot_id is None:
                raise ValueError(f"Missing history snapshot ID for surface group: {gid}")
            age = row[p["age"]] if "age" in p else None
            bucket = row[p["mileage_bucket"]]
            condition = row[p["condition_tier"]]
            record = dict(
                snapshot_id=int(snapshot_id),
                motorcycle_model_id=_mid(row[p["motorcycle_model_id"]]),
                year=_int(row[p["production_year"]]),
                mileage_bucket=_int(bucket),
                condition_tier=_int(condition),
                color="مشکی",
                is_copy=_bool(row[p["is_copy"]]),
                model_variant=str(row[p["model_variant"]]),
                price_low=_money(row[p["suggested_lower"]]),
                price_mid=_money(row[p["adjusted_price"]]),
                price_high=_money(row[p["suggested_upper"]]),
            )
            if target == "crawler":
                cell_key = hashlib.md5(
                    "\x1f".join([
                        str(_int(age)), str(_int(bucket)),
                        str(_int(condition)), "مشکی",
                    ]).encode("utf-8")
                ).hexdigest()
                record.update(
                    real_brand=str(row[p["real_brand"]]),
                    real_model=str(row[p["real_model"]]),
                    cell_key=cell_key,
                    age=_int(age),
                    reliability_tier=str(row[p["reliability_tier"]]),
                    is_reliable=_bool(row[p["is_reliable"]]),
                    publish_state=str(row[p["publish_state"]]),
                    is_factory=_bool(row[p["is_factory"]]),
                    estimated_mape_pct=_num(row[p["estimated_mape_pct"]]),
                    estimated_interval_coverage_pct=_num(
                        row[p["estimated_interval_coverage_pct"]]
                    ),
                    isotonic_change_pct=_num(row[p["isotonic_change_pct"]]),
                    mileage_clean=_num(row[p["mileage_clean"]]),
                )
            yield record


    @staticmethod
    def _engine_fingerprint(engine):
        url = getattr(engine, "url", None)
        if url is not None and hasattr(url, "render_as_string"):
            return url.render_as_string(hide_password=True)
        return str(url)

    @staticmethod
    def _server_fingerprint(engine):
        url = engine.url
        return (
            getattr(engine.dialect, "name", ""),
            getattr(url, "host", None),
            getattr(url, "port", None) or 5432,
        )

    def _preflight_two_phase(self, participants):
        """Fail before writes unless every PostgreSQL server supports 2PC."""
        required_by_server = {}
        for participant in participants:
            key = self._server_fingerprint(participant["engine"])
            required_by_server[key] = required_by_server.get(key, 0) + 1

        checked = {}
        for participant in participants:
            engine = participant["engine"]
            required_methods = (
                "do_begin_twophase", "do_prepare_twophase",
                "do_commit_twophase", "do_rollback_twophase",
                "do_recover_twophase",
            )
            if (
                getattr(engine.dialect, "name", None) != "postgresql"
                or not all(
                    callable(getattr(engine.dialect, method, None))
                    for method in required_methods
                )
            ):
                raise RuntimeError(
                    "Atomic crawler/production persistence requires a "
                    "SQLAlchemy PostgreSQL dialect with two-phase commit support"
                )
            key = self._server_fingerprint(engine)
            if key in checked:
                continue
            with engine.connect() as conn:
                maximum = int(
                    conn.exec_driver_sql(
                        "SHOW max_prepared_transactions"
                    ).scalar_one()
                )
            required = required_by_server[key]
            if maximum < required:
                raise RuntimeError(
                    "Two-phase commit is disabled or undersized: "
                    f"PostgreSQL max_prepared_transactions={maximum}, "
                    f"required={required} for this server. Configure "
                    "max_prepared_transactions accordingly and restart PostgreSQL. "
                    "No price rows have been modified."
                )
            checked[key] = maximum
            log.info(
                "[DB][2PC] preflight server=%s max_prepared_transactions=%s required=%s",
                key, maximum, required,
            )

    @staticmethod
    def _acquire_coordinator_lock(engine):
        conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            conn.exec_driver_sql(
                "SELECT pg_advisory_lock(724319, 2)"
            ).scalar()
            return conn
        except Exception:
            conn.close()
            raise

    @staticmethod
    def _release_coordinator_lock(conn):
        if conn is None:
            return
        try:
            conn.exec_driver_sql(
                "SELECT pg_advisory_unlock(724319, 2)"
            ).scalar()
        except Exception:
            log.exception("[DB][2PC] Could not release coordinator advisory lock")
        finally:
            conn.close()

    @staticmethod
    def _ensure_coordinator_table(engine):
        with engine.begin() as conn:
            conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS _mps_transaction_coordinator (
                    run_id varchar(32) PRIMARY KEY,
                    decision varchar(16) NOT NULL,
                    xids text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )

    @staticmethod
    def _coordinator_record(engine, run_id, decision, xid_items):
        payload = json.dumps(
            xid_items, ensure_ascii=False, separators=(",", ":")
        )
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO _mps_transaction_coordinator
                        (run_id, decision, xids)
                    VALUES (:run_id, :decision, :xids)
                    ON CONFLICT (run_id) DO UPDATE SET
                        decision = EXCLUDED.decision,
                        xids = EXCLUDED.xids,
                        updated_at = now()
                    """
                ),
                {
                    "run_id": run_id,
                    "decision": decision,
                    "xids": payload,
                },
            )

    @staticmethod
    def _coordinator_decision(engine, run_id):
        with engine.connect() as conn:
            return conn.execute(
                text(
                    "SELECT decision FROM _mps_transaction_coordinator "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            ).scalar_one_or_none()

    @staticmethod
    def _resolve_prepared(engine, xid, decision):
        """Idempotently finish a prepared transaction via SQLAlchemy's 2PC API."""
        if decision not in {"COMMIT", "ROLLBACK"}:
            raise ValueError(f"Invalid prepared transaction decision: {decision}")
        with engine.connect() as conn:
            recovered = conn.recover_twophase()
            match = None
            for candidate in recovered:
                gtrid = getattr(candidate, "gtrid", None)
                if candidate == xid or gtrid == xid or str(candidate) == xid:
                    match = candidate
                    break
            if match is None:
                # It was already committed/rolled back, or never reached PREPARE.
                return True
            if decision == "COMMIT":
                conn.commit_prepared(match, recover=True)
            else:
                conn.rollback_prepared(match, recover=True)
            return True

    def _recover_pending_two_phase(self, coordinator_engine, engine_by_target):
        """Recover interrupted 2PC runs from the durable coordinator decision."""
        with coordinator_engine.connect() as conn:
            pending = conn.execute(
                text(
                    """
                    SELECT run_id, decision, xids
                    FROM _mps_transaction_coordinator
                    WHERE decision IN ('PREPARING', 'COMMIT')
                    ORDER BY created_at
                    """
                )
            ).fetchall()

        for row in pending:
            run_id, decision, payload = row[0], row[1], row[2]
            xid_items = json.loads(payload)
            action = "ROLLBACK" if decision == "PREPARING" else "COMMIT"
            for item in xid_items:
                target = item["target"]
                engine = engine_by_target.get(target)
                if engine is None:
                    raise RuntimeError(
                        f"Cannot recover pending 2PC run {run_id}: "
                        f"missing current engine for target={target}"
                    )
                if self._engine_fingerprint(engine) != item["engine"]:
                    raise RuntimeError(
                        f"Cannot recover pending 2PC run {run_id}: "
                        f"database endpoint changed for target={target}"
                    )
                if not self._resolve_prepared(engine, item["xid"], action):
                    raise RuntimeError(
                        f"Prepared transaction remains unresolved: "
                        f"target={target}, xid={item['xid']}, action={action}"
                    )

            final_state = "ROLLED_BACK" if action == "ROLLBACK" else "COMMITTED"
            self._coordinator_record(
                coordinator_engine, run_id, final_state, xid_items
            )
            log.warning(
                "[DB][2PC] recovered interrupted run=%s decision=%s targets=%s",
                run_id, final_state,
                [item["target"] for item in xid_items],
            )


    def _save_target(self, conn, target, run_id):
        """Persist one target directly in its caller-owned transaction."""
        started = time.perf_counter()
        history_df, surface_df = self._prepare_frames(target)
        _log_db_stage(
            target, "frames prepared", started,
            history_rows=len(history_df), surface_rows=len(surface_df),
        )

        if surface_df.empty:
            if target == "production":
                deleted = conn.exec_driver_sql(
                    f'DELETE FROM "{SURFACE_NAME}"'
                ).rowcount
                report = {
                    "target": target, "history_count": 0,
                    "deleted": max(deleted, 0), "inserted": 0,
                    "seconds": round(time.perf_counter() - started, 2),
                }
                log.warning(
                    "[DB][%s] Empty approved release: production surface cleared",
                    target,
                )
                return report
            log.info(
                "[DB][%s] No target cells: no DELETE, no writes", target
            )
            return {
                "target": target, "history_count": 0,
                "deleted": 0, "inserted": 0,
                "seconds": round(time.perf_counter() - started, 2),
            }

        schema_started = time.perf_counter()
        h, _ = self._db_tables(conn, target)
        _log_db_stage(target, "schema validated", schema_started)

        history_records = self._history_records(
            target, history_df, surface_df
        )
        if not history_records:
            raise ValueError(
                f"{target}: non-empty surface has no matching history records"
            )
        history_columns = list(history_records[0])
        conflict = (
            ["date", "identity_key"] if target == "crawler"
            else ["date", "motorcycle_model_id", "is_copy", "model_variant"]
        )
        update_columns = [
            column for column in history_columns if column not in conflict
        ]

        upsert_started = time.perf_counter()
        for offset in range(0, len(history_records), 500):
            batch = history_records[offset:offset + 500]
            statement = pg_insert(h).values(batch)
            update_values = {
                column: statement.excluded[column]
                for column in update_columns
            }
            if "updated_at" in h.c:
                update_values["updated_at"] = func.now()
            conflict_args = {
                "index_elements": [h.c[column] for column in conflict]
            }
            if update_values:
                statement = statement.on_conflict_do_update(
                    **conflict_args, set_=update_values
                )
            else:
                statement = statement.on_conflict_do_nothing(**conflict_args)
            conn.execute(statement)
        _log_db_stage(
            target, "history UPSERT complete", upsert_started,
            history_rows=len(history_records),
        )

        map_started = time.perf_counter()
        if target == "crawler":
            rows = conn.execute(
                text(
                    f'SELECT id, identity_key FROM "{HISTORY_NAME}" '
                    'WHERE date = :asof'
                ),
                {"asof": self.today},
            ).fetchall()
            id_by_key = {str(row[1]): int(row[0]) for row in rows}
            snapshot_by_gid = {
                gid: id_by_key[gid]
                for gid in history_df["_gid"].to_numpy(dtype=object)
                if gid in id_by_key
            }
        else:
            rows = conn.execute(
                text(
                    f'SELECT id, motorcycle_model_id, is_copy, model_variant '
                    f'FROM "{HISTORY_NAME}" WHERE date = :asof'
                ),
                {"asof": self.today},
            ).fetchall()
            id_by_key = {
                (_mid(row[1]), _bool(row[2]), str(row[3])): int(row[0])
                for row in rows
            }
            history_columns_for_map = list(history_df.columns)
            positions = {
                name: history_columns_for_map.index(name)
                for name in (
                    "_gid", "motorcycle_model_id", "is_copy", "model_variant"
                )
            }
            snapshot_by_gid = {}
            for row in history_df.itertuples(index=False, name=None):
                key = (
                    _mid(row[positions["motorcycle_model_id"]]),
                    _bool(row[positions["is_copy"]]),
                    str(row[positions["model_variant"]]),
                )
                snapshot_id = id_by_key.get(key)
                if snapshot_id is not None:
                    snapshot_by_gid[row[positions["_gid"]]] = snapshot_id

        missing_snapshots = set(surface_df["_gid"]) - set(snapshot_by_gid)
        if missing_snapshots:
            raise ValueError(
                f"{target}: {len(missing_snapshots)} surface groups lack "
                "a persisted history snapshot ID"
            )
        _log_db_stage(
            target, "history snapshot IDs resolved", map_started,
            snapshot_groups=len(snapshot_by_gid),
        )

        delete_started = time.perf_counter()
        if target == "production":
            deleted = conn.exec_driver_sql(
                f'DELETE FROM "{SURFACE_NAME}"'
            ).rowcount
        else:
            deleted = 0
            model_ids = surface_df["motorcycle_model_id"].map(_mid)
            ids = sorted({value for value in model_ids.dropna().tolist()})
            null_gids = (
                surface_df.loc[model_ids.isna(), "_gid"]
                .drop_duplicates().tolist()
            )
            if ids:
                delete_ids = text(
                    f'DELETE FROM "{SURFACE_NAME}" '
                    'WHERE motorcycle_model_id IN :ids'
                ).bindparams(bindparam("ids", expanding=True))
                deleted += conn.execute(delete_ids, {"ids": ids}).rowcount
            if null_gids:
                delete_null_ids = text(
                    f'DELETE FROM "{SURFACE_NAME}" p '
                    f'USING "{HISTORY_NAME}" oldh '
                    'WHERE p.snapshot_id = oldh.id '
                    'AND p.motorcycle_model_id IS NULL '
                    'AND oldh.identity_key IN :gids'
                ).bindparams(bindparam("gids", expanding=True))
                deleted += conn.execute(
                    delete_null_ids, {"gids": null_gids}
                ).rowcount
        _log_db_stage(
            target, "old surface rows deleted", delete_started,
            deleted=max(deleted, 0),
        )

        copy_started = time.perf_counter()
        surface_columns = list(
            next(self._surface_records(
                target, surface_df, snapshot_by_gid
            )).keys()
        )
        surface_count = _copy_rows(
            conn, SURFACE_NAME, surface_columns,
            self._surface_records(target, surface_df, snapshot_by_gid),
            progress_label=f"{target}/surface/{run_id[:8]}",
        )
        if surface_count != len(surface_df):
            raise RuntimeError(
                f"{target}: surface COPY row-count mismatch "
                f"{surface_count}/{len(surface_df)}"
            )
        _log_db_stage(
            target, "surface COPY complete", copy_started,
            inserted=surface_count,
        )

        report = {
            "target": target,
            "history_count": len(history_records),
            "deleted": max(deleted, 0),
            "inserted": surface_count,
            "motorcycle_count": int(
                surface_df["motorcycle_model_id"].map(_mid).nunique()
            ),
            "seconds": round(time.perf_counter() - started, 2),
        }
        log.info("[DB][%s] READY TO COMMIT %s", target, report)
        return report


    def update_database(self):
        """Atomically publish crawler and production targets.

        One shared Engine uses a single transaction. Separate PostgreSQL
        Engines use durable two-phase commit with a coordinator record in the
        crawler database. The coordinator also recovers interrupted runs.
        """
        if self.crawler_db is None:
            raise RuntimeError("Crawler database Session is required")

        target_sessions = [("crawler", self.crawler_db)]
        if self.production_db is not None:
            target_sessions.append(("production", self.production_db))

        engine_by_target = {}
        participants = []
        by_engine_id = {}
        for target, session in target_sessions:
            engine = (
                session.get_bind()
                if hasattr(session, "get_bind")
                else getattr(session, "bind", None)
            )
            if engine is None or not hasattr(engine, "connect"):
                raise RuntimeError(
                    f"Cannot resolve SQLAlchemy Engine for target={target}"
                )
            engine_by_target[target] = engine
            key = id(engine)
            if key not in by_engine_id:
                participant = {
                    "engine": engine, "targets": [], "conn": None,
                    "tx": None, "xid": None, "prepared": False,
                    "resolved": False,
                }
                by_engine_id[key] = participant
                participants.append(participant)
            by_engine_id[key]["targets"].append(target)

        two_phase = len(participants) > 1
        run_id = uuid.uuid4().hex
        coordinator_engine = engine_by_target["crawler"]
        coordinator_lock = None
        coordinator_created = False
        commit_decision = False
        decision_unknown = False
        xid_items = []
        result = {}
        participant_by_target = {
            target: by_engine_id[id(engine_by_target[target])]
            for target, _ in target_sessions
        }
        started = time.perf_counter()

        try:
            if two_phase:
                coordinator_lock = self._acquire_coordinator_lock(
                    coordinator_engine
                )
                self._ensure_coordinator_table(coordinator_engine)
                self._recover_pending_two_phase(
                    coordinator_engine, engine_by_target
                )
                self._preflight_two_phase(participants)

                for index, participant in enumerate(participants):
                    participant["xid"] = f"mps_{run_id}_{index}"
                    xid_items.append({
                        "target": participant["targets"][0],
                        "xid": participant["xid"],
                        "engine": self._engine_fingerprint(
                            participant["engine"]
                        ),
                    })
                self._coordinator_record(
                    coordinator_engine, run_id, "PREPARING", xid_items
                )
                coordinator_created = True
                log.info(
                    "[DB][2PC] coordinator intent persisted run_id=%s xids=%s",
                    run_id, xid_items,
                )

            for participant in participants:
                participant["conn"] = participant["engine"].connect()
                if two_phase:
                    participant["tx"] = participant["conn"].begin_twophase(
                        xid=participant["xid"]
                    )
                    log.info(
                        "[DB][2PC] BEGIN target=%s xid=%s",
                        participant["targets"], participant["xid"],
                    )
                else:
                    participant["tx"] = participant["conn"].begin()
                    log.info(
                        "[DB][TX] BEGIN targets=%s",
                        participant["targets"],
                    )

            for target, _ in target_sessions:
                participant = participant_by_target[target]
                result[target] = self._save_target(
                    participant["conn"], target, run_id
                )

            if two_phase:
                for participant in participants:
                    prepare_started = time.perf_counter()
                    participant["tx"].prepare()
                    participant["prepared"] = True
                    log.info(
                        "[DB][2PC] PREPARED target=%s xid=%s elapsed=%.2fs",
                        participant["targets"], participant["xid"],
                        time.perf_counter() - prepare_started,
                    )

                try:
                    self._coordinator_record(
                        coordinator_engine, run_id, "COMMIT", xid_items
                    )
                except Exception:
                    try:
                        observed = self._coordinator_decision(
                            coordinator_engine, run_id
                        )
                    except Exception:
                        decision_unknown = True
                        raise RuntimeError(
                            "Could not determine whether the durable 2PC "
                            "COMMIT decision was recorded. Prepared transactions "
                            "were left untouched for recovery; do not rerun blindly."
                        )
                    if observed != "COMMIT":
                        raise
                commit_decision = True
                log.info(
                    "[DB][2PC] COMMIT DECISION durable run_id=%s",
                    run_id,
                )

                commit_errors = []
                for participant in participants:
                    try:
                        participant["tx"].commit()
                        participant["resolved"] = True
                        log.info(
                            "[DB][2PC] COMMIT target=%s xid=%s",
                            participant["targets"], participant["xid"],
                        )
                    except Exception as exc:
                        participant["commit_error"] = str(exc)
                        try:
                            participant["conn"].invalidate()
                        except Exception:
                            pass
                        commit_errors.append(participant)
                        log.exception(
                            "[DB][2PC] COMMIT response failed target=%s xid=%s; "
                            "durable decision remains COMMIT",
                            participant["targets"], participant["xid"],
                        )

                if commit_errors:
                    try:
                        self._recover_pending_two_phase(
                            coordinator_engine, engine_by_target
                        )
                        for participant in participants:
                            participant["resolved"] = True
                    except Exception as recovery_error:
                        unresolved = [
                            f"{p['targets']}:{p['xid']}"
                            for p in participants if not p["resolved"]
                        ]
                        raise RuntimeError(
                            "Two-phase COMMIT decision is durable but recovery "
                            f"is incomplete. Unresolved participants={unresolved}. "
                            "Inspect _mps_transaction_coordinator and "
                            "pg_prepared_xacts before retrying."
                        ) from recovery_error
                else:
                    try:
                        self._coordinator_record(
                            coordinator_engine, run_id, "COMMITTED", xid_items
                        )
                    except Exception:
                        log.exception(
                            "[DB][2PC] Participants committed but coordinator "
                            "final status update failed; recovery will finalize it"
                        )
            else:
                commit_decision = True
                participant = participants[0]
                participant["tx"].commit()
                participant["resolved"] = True
                log.info(
                    "[DB][TX] COMMIT targets=%s",
                    participant["targets"],
                )

            log.info(
                "[DB][ATOMIC] SUCCESS run_id=%s mode=%s targets=%s "
                "seconds=%.2f rss_mb=%s",
                run_id, "2PC" if two_phase else "single-transaction",
                list(result), time.perf_counter() - started, _rss_mb(),
            )
            return result

        except Exception:
            if two_phase and coordinator_created and not commit_decision and not decision_unknown:
                rollback_ok = True
                for participant in reversed(participants):
                    participant_rollback_ok = True
                    tx = participant.get("tx")
                    if tx is not None:
                        try:
                            if tx.is_active:
                                tx.rollback()
                        except Exception:
                            log.exception(
                                "[DB][2PC] rollback call failed; checking prepared XID "
                                "target=%s xid=%s",
                                participant["targets"], participant["xid"],
                            )
                    try:
                        if not self._resolve_prepared(
                            participant["engine"], participant["xid"], "ROLLBACK"
                        ):
                            participant_rollback_ok = False
                    except Exception:
                        participant_rollback_ok = False
                        log.exception(
                            "[DB][2PC] prepared rollback recovery failed "
                            "target=%s xid=%s",
                            participant["targets"], participant["xid"],
                        )
                    participant["resolved"] = participant_rollback_ok
                    rollback_ok = rollback_ok and participant_rollback_ok
                if rollback_ok:
                    try:
                        self._coordinator_record(
                            coordinator_engine, run_id, "ROLLED_BACK", xid_items
                        )
                    except Exception:
                        log.exception(
                            "[DB][2PC] rollback completed but coordinator "
                            "status update failed; PREPARING is safe to retry"
                        )
                else:
                    log.critical(
                        "[DB][2PC] rollback incomplete; PREPARING coordinator "
                        "record retained for next-run recovery run_id=%s",
                        run_id,
                    )
            elif not two_phase and not commit_decision:
                for participant in participants:
                    tx = participant.get("tx")
                    if tx is not None:
                        try:
                            if tx.is_active:
                                tx.rollback()
                                participant["resolved"] = True
                        except Exception:
                            log.exception(
                                "[DB][TX] rollback failed targets=%s",
                                participant["targets"],
                            )
            log.exception(
                "[DB][ATOMIC] FAILED run_id=%s mode=%s elapsed=%.2fs",
                run_id, "2PC" if two_phase else "single-transaction",
                time.perf_counter() - started,
            )
            raise
        finally:
            for participant in participants:
                conn = participant.get("conn")
                if conn is not None:
                    if (
                        two_phase and participant.get("xid")
                        and not participant.get("resolved")
                        and (commit_decision or decision_unknown)
                    ):
                        try:
                            conn.invalidate()
                        except Exception:
                            pass
                    try:
                        conn.close()
                    except Exception:
                        log.exception(
                            "[DB] connection close failed targets=%s",
                            participant["targets"],
                        )
            self._release_coordinator_lock(coordinator_lock)
