"""Exact helpers for sparse monotone product-order operations.

The prefix-minimum implementation is equivalent to the previous ordered
predecessor scan when coordinates are unique. Duplicate coordinates retain
the historical sequential semantics through a compatibility fallback.
"""
from __future__ import annotations

import numpy as np


def product_order_min_closure(values, coordinates):
    """Return the product-order lower envelope on a sparse 3-D coordinate set.

    For each point, take the minimum of its value and all values at coordinates
    that are componentwise less than or equal to it. Coordinates are compressed
    to their sorted unique levels, so the dense working array is bounded by the
    coordinate cardinalities rather than the numeric coordinate ranges.
    """
    values = np.asarray(values, dtype=float)
    coords = np.asarray(coordinates, dtype=int)
    if values.ndim != 1 or coords.shape != (len(values), 3):
        raise ValueError("values/coordinates must have shapes (n,) and (n, 3)")
    if len(values) == 0:
        return values.copy()

    # The old ordered scan treats equal coordinates sequentially: the first
    # duplicate is not lowered by later duplicates. Keep that exact behavior
    # for malformed/non-unique inputs rather than silently changing semantics.
    if (
        not np.isfinite(values).all()
        or np.unique(coords, axis=0).shape[0] != len(coords)
    ):
        result = values.copy()
        order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        for position, index in enumerate(order):
            predecessors = order[:position]
            predecessors = predecessors[
                np.all(coords[predecessors] <= coords[index], axis=1)
            ]
            if len(predecessors):
                result[index] = min(
                    result[index], float(np.min(result[predecessors]))
                )
        return result

    levels = tuple(np.unique(coords[:, axis]) for axis in range(3))
    positions = tuple(
        np.searchsorted(levels[axis], coords[:, axis]) for axis in range(3)
    )
    prefix = np.full(tuple(len(axis) for axis in levels), np.inf, dtype=float)
    np.minimum.at(prefix, positions, values)
    for axis in range(3):
        prefix = np.minimum.accumulate(prefix, axis=axis)
    return np.minimum(values, prefix[positions])
