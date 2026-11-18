"""Synthetic equivalence tests for stage-1 performance-only changes."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from motorcycle_surface.age_market import AgeMarketBuilder, _expand_age_span
from motorcycle_surface.core import (
    AGE_HISTORY_COLUMNS,
    MotorcyclePriceSurface,
    SurfaceConfig,
    _concat_age_training_history,
)
from motorcycle_surface.local_bias_v7 import downward_monotone_envelope
from motorcycle_surface.monotone_utils import product_order_min_closure
from motorcycle_surface.weighted_isotonic_v3 import weighted_ragged_isotonic


def _legacy_product_order_min_closure(values, coordinates):
    """Pre-optimization implementation retained as a test oracle."""
    result = np.asarray(values, float).copy()
    coords = np.asarray(coordinates, int)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    for position, index in enumerate(order):
        predecessors = order[:position]
        predecessors = predecessors[
            np.all(coords[predecessors] <= coords[index], axis=1)
        ]
        if len(predecessors):
            result[index] = min(result[index], np.min(result[predecessors]))
    return result


def _legacy_pava_down(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    averages, masses, counts = [], [], []
    for value, weight in zip(values, weights):
        averages.append(float(value))
        masses.append(float(weight))
        counts.append(1)
        while len(averages) > 1 and averages[-2] < averages[-1]:
            mass = masses[-2] + masses[-1]
            averages[-2] = (
                masses[-2] * averages[-2] + masses[-1] * averages[-1]
            ) / mass
            masses[-2] = mass
            counts[-2] += counts[-1]
            averages.pop()
            masses.pop()
            counts.pop()
    return np.repeat(averages, counts)


def _legacy_weighted_ragged_isotonic(
    prices, coordinates, weights=None, fixed_anchor=None, max_iter=300, tol=1e-9
):
    p = np.asarray(prices, float)
    c = np.asarray(coordinates, int)
    if len(p) == 0:
        return p.copy()
    weights = np.ones(len(p)) if weights is None else np.asarray(weights, float)

    slices = []
    for axis in range(3):
        bins = {}
        for index, coord in enumerate(c):
            key = tuple(coord[j] for j in range(3) if j != axis)
            bins.setdefault(key, []).append(index)
        slices.append([
            np.asarray(sorted(indices, key=lambda i: c[i, axis]), dtype=int)
            for indices in bins.values() if len(indices) > 1
        ])

    y = np.log(p)
    residual = np.zeros((3, len(p)), dtype=float)
    converged = False
    for _ in range(max_iter):
        old = y.copy()
        for axis in range(3):
            z = y + residual[axis]
            projected = z.copy()
            for indices in slices[axis]:
                projected[indices] = _legacy_pava_down(
                    z[indices], weights[indices]
                )
            residual[axis] = z - projected
            y = projected
        if np.max(np.abs(y - old)) < tol:
            converged = True
            break

    if not converged:
        for _ in range(8):
            previous = y.copy()
            for axis in range(3):
                for indices in slices[axis]:
                    y[indices] = _legacy_pava_down(y[indices], weights[indices])
            if np.max(np.abs(y - previous)) < 1e-10:
                break

    if fixed_anchor is not None:
        anchor_log = np.log(float(fixed_anchor))
        mask = np.all(c == [0, 0, 1], axis=1)
        if mask.sum() != 1:
            raise ValueError("Missing/duplicated factory anchor coordinate (0,0,1)")
        y = np.minimum(y, anchor_log)
        y[mask] = anchor_log

    y = _legacy_product_order_min_closure(y, c)
    return np.exp(y)


def _legacy_downward_envelope(baseline_prices, desired_prices, coordinates):
    xyz = np.asarray(coordinates, int)
    result = np.asarray(desired_prices, float).copy()
    ordering = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    for step, index in enumerate(ordering):
        previous = ordering[:step]
        previous = previous[np.all(xyz[previous] <= xyz[index], axis=1)]
        if len(previous):
            result[index] = min(result[index], float(result[previous].min()))
    return result


def _legacy_unweighted_projection(prices, coordinates, fixed_anchor=None):
    y = np.log(np.asarray(prices, float))
    coords = np.asarray(coordinates, int)
    for _ in range(9):
        previous = y.copy()
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            groups = {}
            for index, coord in enumerate(coords):
                key = tuple(int(coord[j]) for j in other)
                groups.setdefault(key, []).append(index)
            for indices in groups.values():
                indices = np.asarray(indices, int)
                indices = indices[np.argsort(coords[indices, axis])]
                if len(indices) > 1:
                    y[indices] = _legacy_pava_down(
                        y[indices], np.ones(len(indices))
                    )
        if np.max(np.abs(previous - y)) < 1e-10:
            break
    if fixed_anchor is not None:
        anchor_log = np.log(float(fixed_anchor))
        y = np.minimum(y, anchor_log)
        mask = np.all(coords == [0, 0, 1], axis=1)
        if mask.sum() != 1:
            raise ValueError("Factory anchor coordinate missing or duplicated")
        y[mask] = anchor_log
    return np.exp(_legacy_product_order_min_closure(y, coords))


def _unique_ragged_coordinates(rng, count):
    full = np.asarray([
        (age, bucket, condition)
        for age in range(9)
        for bucket in range(9)
        for condition in (1, 2, 3)
    ], dtype=int)
    return full[rng.choice(len(full), size=count, replace=False)]


class Stage1PerformanceEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261118)

    def test_prefix_closure_matches_legacy_on_sparse_unique_grids(self):
        for count in (1, 4, 20, 80, 180):
            coords = _unique_ragged_coordinates(self.rng, count)
            values = self.rng.normal(size=count)
            expected = _legacy_product_order_min_closure(values, coords)
            actual = product_order_min_closure(values, coords)
            np.testing.assert_array_equal(actual, expected)

    def test_duplicate_coordinates_keep_legacy_sequential_semantics(self):
        coords = np.asarray([
            (0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 1, 2)
        ])
        values = np.asarray([9.0, 5.0, 7.0, 4.0])
        np.testing.assert_array_equal(
            product_order_min_closure(values, coords),
            _legacy_product_order_min_closure(values, coords),
        )

    def test_non_finite_values_keep_legacy_closure_semantics(self):
        coords = np.asarray([
            (0, 0, 1), (1, 0, 1), (1, 1, 2)
        ])
        values = np.asarray([8.0, np.nan, 4.0])
        actual = product_order_min_closure(values, coords)
        expected = _legacy_product_order_min_closure(values, coords)
        np.testing.assert_array_equal(actual, expected)

    def test_weighted_projection_and_final_surface_discount_are_unchanged(self):
        coords = _unique_ragged_coordinates(self.rng, 100)
        if not np.any(np.all(coords == [0, 0, 1], axis=1)):
            coords[0] = [0, 0, 1]
        prices = np.exp(self.rng.normal(np.log(700_000_000), 0.22, len(coords)))
        weights = self.rng.uniform(0.5, 5.0, len(coords))

        expected_base = _legacy_weighted_ragged_isotonic(
            prices, coords, weights=weights, fixed_anchor=800_000_000
        )
        actual_base = weighted_ragged_isotonic(
            prices, coords, weights=weights, fixed_anchor=800_000_000
        )
        np.testing.assert_array_equal(actual_base, expected_base)

        discount = -self.rng.uniform(0.0, 0.03, len(coords))
        desired = actual_base * (1.0 + discount)
        expected_final = _legacy_downward_envelope(
            expected_base, desired, coords
        )
        actual_final = downward_monotone_envelope(
            actual_base, desired, coords
        )
        np.testing.assert_array_equal(actual_final, expected_final)

    def test_unweighted_core_projection_matches_legacy(self):
        coords = _unique_ragged_coordinates(self.rng, 90)
        prices = np.exp(self.rng.normal(np.log(500_000_000), 0.3, len(coords)))
        expected = _legacy_unweighted_projection(prices, coords)
        actual = MotorcyclePriceSurface._ragged_monotone_projection(
            prices, coords
        )
        np.testing.assert_array_equal(actual, expected)

    def test_vectorized_age_expansion_matches_apply_and_explode(self):
        span = pd.DataFrame({
            "gid": [2, 4, 8, 9],
            "min": [1, 2, 0, 4],
            "max": [3, 5, 2, 7],
            "size": [12, 18, 7, 9],
            "grid_min_age": [0, 1, 0, 3],
            "parent_age_support": [0, 2, 0, 4],
        })
        expected = span.copy()
        expected["age"] = expected.apply(
            lambda row: list(range(
                int(row["grid_min_age"]), int(row["max"]) + 1
            )),
            axis=1,
        )
        expected = expected.explode("age", ignore_index=True)
        expected["age"] = expected["age"].astype(int)
        actual = _expand_age_span(span)
        actual["age"] = actual["age"].astype(int)
        assert_frame_equal(actual, expected, check_exact=True)

    def test_diagnostic_profiles_do_not_change_production_snapshot_fields(self):
        dates = pd.date_range("2026-07-20", "2026-11-15", freq="5D")
        rows = []
        groups = [
            (203, "Honda", "Click 150", "DEFAULT", 900_000_000),
            (204, "Honda", "Click 150", "ABS", 940_000_000),
            (305, "Yamaha", "NMAX 155", "DEFAULT", 1_150_000_000),
        ]
        for model_id, brand, model, variant, base in groups:
            for age in (1, 2, 3):
                for index, date in enumerate(dates):
                    rows.append({
                        "motorcycle_model_id": model_id,
                        "real_brand": brand,
                        "real_model": model,
                        "is_copy": 0,
                        "model_variant": variant,
                        "segment": "Medium",
                        "date": date,
                        "age": age,
                        "price": base * (1.0 - 0.045 * age) * (1.0 + index * 0.0003),
                    })
        ads = pd.DataFrame(rows)
        with_diagnostics = AgeMarketBuilder(
            ads, include_diagnostic_profiles=True
        ).build_snapshot("2026-11-16")
        production = AgeMarketBuilder(
            ads, include_diagnostic_profiles=False
        ).build_snapshot("2026-11-16")

        diagnostic_columns = {
            *(f"blend{strength}" for strength in (0, 1, 3, 7)),
            *(f"iso{strength}" for strength in (0, 1, 3, 7)),
            "raw14", "raw30", "raw60", "raw120",
        }
        production_columns = [
            column for column in with_diagnostics.columns
            if column not in diagnostic_columns
        ]
        assert_frame_equal(
            with_diagnostics[production_columns].reset_index(drop=True),
            production[production_columns].reset_index(drop=True),
            check_exact=True,
        )
        self.assertFalse(diagnostic_columns.intersection(production.columns))

    def test_age_model_fit_receives_identical_features_from_full_or_narrow_history(self):
        origins = pd.to_datetime([
            "2026-10-01", "2026-10-08", "2026-10-15"
        ])
        rows = []
        for group_id in range(10):
            for age in (1, 2, 3):
                for origin_index, origin in enumerate(origins):
                    base = 900_000_000 - group_id * 3_000_000 - age * 35_000_000
                    row = {
                        "motorcycle_model_id": str(200 + group_id),
                        "real_brand": "Honda",
                        "real_model": f"Model {group_id}",
                        "is_copy": "0",
                        "model_variant": "DEFAULT",
                        "segment": "Medium",
                        "age": age,
                        "min_age": 1,
                        "max_age": 3,
                        "n_group_all": 15,
                        "med14": base * 0.99,
                        "med30": base,
                        "med60": base * 1.01,
                        "med120": base * 1.02,
                        "n14": 3,
                        "n30": 5,
                        "n60": 7,
                        "n120": 9,
                        "group_med14": base * 0.99,
                        "group_med30": base,
                        "group_med120": base * 1.02,
                        "group_n14": 8,
                        "group_n30": 12,
                        "group_n120": 20,
                        "curve_hist": base,
                        "curve_offset": 0.0,
                        "curve_anchor_n": 10,
                        "curve_slope": -0.05,
                        "pool_age_median": base,
                        "pool_age_n": 50,
                        "asof": origin,
                        "future_med": base * (1.0 + origin_index * 0.002),
                        "future_n": 5,
                        "unused_snapshot_payload": f"unused-{group_id}-{age}",
                    }
                    rows.append(row)
        full_history = pd.DataFrame(rows)
        narrow_history = full_history[AGE_HISTORY_COLUMNS].copy()
        captures = []

        class SpyCatBoost:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def fit(self, features, target, cat_features=None):
                captures.append((
                    features.copy(),
                    np.asarray(target, dtype=float).copy(),
                    tuple(cat_features or ()),
                ))
                return self

        config = SurfaceConfig(
            min_age_training_cells=40,
            age_target_mode="direct_rate",
            show_progress=False,
        )
        runner = MotorcyclePriceSurface(config=config)
        cutoff = pd.Timestamp("2026-11-16")
        with patch("motorcycle_surface.core.CatBoostRegressor", SpyCatBoost):
            runner._fit_age_from_past(full_history, cutoff)
            runner._fit_age_from_past(narrow_history, cutoff)

        self.assertEqual(len(captures), 2)
        assert_frame_equal(captures[0][0], captures[1][0], check_exact=True)
        np.testing.assert_array_equal(captures[0][1], captures[1][1])
        self.assertEqual(captures[0][2], captures[1][2])

    def test_narrow_age_history_is_identical_to_full_history_projection(self):
        frames = []
        for snapshot_index in range(3):
            frame = pd.DataFrame({
                column: [f"{column}-{snapshot_index}-{row}" for row in range(4)]
                for column in AGE_HISTORY_COLUMNS
            })
            frame["unused_wide_column"] = range(4)
            frames.append(frame)
        expected = pd.concat(frames, ignore_index=True)[AGE_HISTORY_COLUMNS]
        actual = _concat_age_training_history(frames)
        assert_frame_equal(actual, expected, check_exact=True)


if __name__ == "__main__":
    unittest.main()
