import csv
import io
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from motorcycle_surface.database import (
    SurfaceDatabaseMixin,
    _copy_rows,
    _identity,
    _identity_values,
)


class DatabasePersistenceTests(unittest.TestCase):
    def test_vectorized_identity_matches_legacy_identity(self):
        frame = pd.DataFrame({
            "motorcycle_model_id": [203, None, 12.5, 17.0],
            "real_brand": ["Honda", "Brand X", "Brand Y", "Brand Z"],
            "real_model": ["Click 150", "Model A", "Model B", "Model C"],
            "is_copy": [0, "true", False, "1.0"],
            "model_variant": ["DEFAULT", "ABS", "DEFAULT", "TWIN_CYLINDER"],
        })
        expected = [_identity(row) for _, row in frame.iterrows()]
        self.assertEqual(_identity_values(frame).tolist(), expected)

    def test_surface_records_iterate_without_dataframe_records_materialization(self):
        class NoToDictFrame(pd.DataFrame):
            def to_dict(self, *args, **kwargs):
                raise AssertionError("full DataFrame to_dict must not be used")

        frame = NoToDictFrame({
            "_gid": ["group-1"],
            "motorcycle_model_id": [203],
            "production_year": [1403],
            "mileage_bucket": [2],
            "condition_tier": [1],
            "is_copy": [False],
            "model_variant": ["DEFAULT"],
            "suggested_lower": [650_000_000],
            "adjusted_price": [700_000_000],
            "suggested_upper": [750_000_000],
        })

        class Dummy(SurfaceDatabaseMixin):
            pass

        records = list(Dummy()._surface_records("production", frame))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["motorcycle_model_id"], 203)
        self.assertEqual(records[0]["price_mid"], 700_000_000)
        self.assertEqual(records[0]["_gid"], "group-1")

    def test_copy_rows_writes_bounded_batches_and_injects_run_id(self):
        class FakeCursor:
            def __init__(self):
                self.calls = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def copy_expert(self, statement, stream):
                self.calls.append((statement, stream.read()))

        cursor = FakeCursor()

        class FakeDBAPI:
            def cursor(self):
                return cursor

        sa_conn = SimpleNamespace(
            connection=SimpleNamespace(driver_connection=FakeDBAPI())
        )
        records = ({"value": index} for index in range(5))

        count = _copy_rows(
            sa_conn,
            "_mps_test_stage",
            ["value", "_mps_run_id"],
            records,
            buffer_rows=2,
            run_id="run-123",
            progress_label="unit-test",
        )

        self.assertEqual(count, 5)
        self.assertEqual(len(cursor.calls), 3)
        rows = []
        for statement, payload in cursor.calls:
            self.assertIn("COPY", statement)
            rows.extend(csv.reader(io.StringIO(payload)))
        self.assertEqual([row[0] for row in rows], ["0", "1", "2", "3", "4"])
        self.assertEqual({row[1] for row in rows}, {"run-123"})

    def test_two_phase_commit_prepares_every_database_before_any_commit(self):
        events = []

        class FakeTransaction:
            def __init__(self, xid):
                self.xid = xid
                self.is_active = True

            def prepare(self):
                events.append(("prepare", self.xid))

            def commit(self):
                events.append(("commit", self.xid))
                self.is_active = False

            def rollback(self):
                events.append(("rollback", self.xid))
                self.is_active = False

        class FakeConnection:
            def begin_twophase(self, xid):
                events.append(("begin", xid))
                return FakeTransaction(xid)

            def close(self):
                events.append(("close",))

            def invalidate(self):
                events.append(("invalidate",))

        class FakeURL:
            def __init__(self, endpoint):
                self.host = endpoint
                self.port = 5432
                self.endpoint = endpoint

            def render_as_string(self, hide_password=True):
                return f"postgresql://{self.endpoint}/prices"

        class FakeEngine:
            def __init__(self, endpoint):
                self.dialect = SimpleNamespace(
                    name="postgresql",
                    **{
                        method: (lambda *args, **kwargs: None)
                        for method in (
                            "do_begin_twophase", "do_prepare_twophase",
                            "do_commit_twophase", "do_rollback_twophase",
                            "do_recover_twophase",
                        )
                    },
                )
                self.url = FakeURL(endpoint)

            def connect(self):
                return FakeConnection()

        class FakeSession:
            def __init__(self, engine):
                self.engine = engine

            def get_bind(self):
                return self.engine

        class Dummy(SurfaceDatabaseMixin):
            def __init__(self, crawler_engine, production_engine):
                self.crawler_db = FakeSession(crawler_engine)
                self.production_db = FakeSession(production_engine)

            def _save_target(self, conn, target, run_id):
                events.append(("save", target))
                return {"target": target}

        runner = Dummy(FakeEngine("crawler-db"), FakeEngine("production-db"))
        with ExitStack() as stack:
            stack.enter_context(patch.object(
                SurfaceDatabaseMixin, "_acquire_coordinator_lock",
                return_value=object(),
            ))
            stack.enter_context(patch.object(
                SurfaceDatabaseMixin, "_release_coordinator_lock"
            ))
            stack.enter_context(patch.object(
                SurfaceDatabaseMixin, "_ensure_coordinator_table"
            ))
            stack.enter_context(patch.object(
                SurfaceDatabaseMixin, "_recover_pending_two_phase"
            ))
            stack.enter_context(patch.object(
                SurfaceDatabaseMixin, "_preflight_two_phase"
            ))
            stack.enter_context(patch.object(
                SurfaceDatabaseMixin, "_coordinator_record"
            ))
            result = runner.update_database()

        prepare_positions = [
            index for index, event in enumerate(events) if event[0] == "prepare"
        ]
        commit_positions = [
            index for index, event in enumerate(events) if event[0] == "commit"
        ]
        self.assertEqual(set(result), {"crawler", "production"})
        self.assertEqual(len(prepare_positions), 2)
        self.assertEqual(len(commit_positions), 2)
        self.assertLess(max(prepare_positions), min(commit_positions))


if __name__ == "__main__":
    unittest.main()
