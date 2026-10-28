"""Weighted least-squares isotonic projection of valid ragged 3D cells.

Dykstra's cyclic projections, in log price, with exact weighted 1D PAVA.
Unit-tested in tests/test_weighted_isotonic_v3.py.
"""
from __future__ import annotations
import numpy as np


def _weighted_pava_down(values, weights):
    """Weighted PAVA enforcing a nonincreasing sequence (pool adjacent violators)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if len(v) != len(w):
        raise ValueError('values/weights length mismatch')
    if np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError('Invalid PAVA weights: must be finite and positive')
    if not np.isfinite(v).all():
        raise ValueError('Invalid PAVA values: must be finite')
    avg, mass, counts = [], [], []
    for a, b in zip(v, w):
        avg.append(float(a)); mass.append(float(b)); counts.append(1)
        while len(avg) > 1 and avg[-2] < avg[-1]:
            weight = mass[-2] + mass[-1]
            avg[-2] = (mass[-2] * avg[-2] + mass[-1] * avg[-1]) / weight
            mass[-2] = weight
            counts[-2] += counts[-1]
            avg.pop(); mass.pop(); counts.pop()
    return np.repeat(avg, counts)


def weighted_ragged_isotonic(prices, coords, weights=None, fixed_anchor=None,
                              max_iter=300, tol=1e-9):
    """Project prices onto the nonincreasing-on-each-axis cone for a ragged 3D grid.

    `coords` holds integer (age, mileage_bucket, condition_tier) coordinates;
    the grid need not be rectangular. Returns prices (not logs), always finite
    and positive, satisfying: for every axis and every fixed value of the other
    two coordinates, price is nonincreasing as that axis increases.

    If `fixed_anchor` is given, it is the price at coordinate (0,0,1) (used
    for a factory-anchored group); that cell is clamped to this exact value
    and nothing may exceed it.
    """
    p = np.asarray(prices, float)
    c = np.asarray(coords, int)
    if len(p) == 0:
        return p.copy()
    if c.shape != (len(p), 3):
        raise ValueError('coords must be (n,3)')
    if not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError('Invalid ragged price coordinates: must be finite and positive')
    weights = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if len(weights) != len(p) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError('Invalid weights: must be finite, positive, matching length')

    # Partition cells into disjoint totally-ordered chains per axis.
    slices = []
    for axis in range(3):
        bins = {}
        for i, v in enumerate(c):
            key = tuple(v[j] for j in range(3) if j != axis)
            bins.setdefault(key, []).append(i)
        slices.append([np.asarray(sorted(ix, key=lambda t: c[t, axis]), dtype=int)
                       for ix in bins.values() if len(ix) > 1])

    y = np.log(p)
    resid = np.zeros((3, len(p)), dtype=float)
    converged = False
    for _ in range(max_iter):
        old = y.copy()
        for axis in range(3):
            z = y + resid[axis]
            proj = z.copy()
            for ix in slices[axis]:
                proj[ix] = _weighted_pava_down(z[ix], weights[ix])
            resid[axis] = z - proj
            y = proj
        if np.max(np.abs(y - old)) < tol:
            converged = True
            break
    if not converged:
        # Fallback: plain cyclic PAVA without Dykstra correction terms.
        # Still feasible (monotone on every axis), just not the exact
        # minimum-weighted-L2 projection in pathological tie-heavy inputs.
        for _ in range(8):
            prev = y.copy()
            for axis in range(3):
                for ix in slices[axis]:
                    y[ix] = _weighted_pava_down(y[ix], weights[ix])
            if np.max(np.abs(y - prev)) < 1e-10:
                break

    if fixed_anchor is not None:
        anchor_log = np.log(float(fixed_anchor))
        mask = np.all(c == [0, 0, 1], axis=1)
        if mask.sum() != 1:
            raise ValueError('Missing/duplicated factory anchor coordinate (0,0,1)')
        y = np.minimum(y, anchor_log)
        y[mask] = anchor_log

    # Final exact monotone closure over the full product order (defensive;
    # guards against any residual float drift from the iterative solver).
    order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
    for pos, i in enumerate(order):
        prior = order[:pos]
        prior = prior[np.all(c[prior] <= c[i], axis=1)]
        if len(prior):
            y[i] = min(y[i], np.min(y[prior]))
    return np.exp(y)