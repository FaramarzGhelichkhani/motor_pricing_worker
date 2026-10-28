"""Age-1..N publication completeness certificate for ragged motorcycle grids.

Designed to operate on the PRE-FILTER full grid, after genuine model prediction,
monotone projection, identity gate, and V10 publication decision.
This module never modifies prices, is_copy, is_original, or publication_allowed.
A missing/untrusted coordinate cannot be certified through interpolation of labels.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GROUP = ['motorcycle_model_id', 'real_brand', 'real_model', 'is_copy', 'model_variant']
CELL = GROUP + ['age', 'mileage_bucket', 'condition_tier']


def apply_age_window_certificate(
    grid: pd.DataFrame,
    *,
    max_required_age: int = 6,
    mileage_caps: dict | None = None,
    bucket_levels: tuple = tuple(range(9)),
    condition_levels: tuple = (1, 2, 3),
    publication_mode: str = 'audit',
) -> pd.DataFrame:
    """Attach group-wide readiness metadata; `strict` vetoes non-complete publication.

    Policy: age 0 exempt; group maximum age exempt; ages before its source min_age
    not demanded. Bucket 0 only condition 1; others follow config.
    Completeness concerns PUBLISHED (publication_allowed) prices, never the full
    diagnostic grid.
    `audit` leaves `publication_allowed` untouched (metadata only); `strict`
    revokes `publication_allowed` on all cells of eligible-but-incomplete
    groups. Rows remain in the frame either way, for diagnostics.
    """
    if publication_mode not in ('audit', 'strict'):
        raise ValueError('publication_mode must be audit or strict')
    if max_required_age < 1:
        raise ValueError('max_required_age must be at least 1')
    if grid.empty:
        return grid.copy()
    required = set(GROUP + ['age', 'mileage_bucket', 'condition_tier', 'min_age', 'max_age',
                             'publication_allowed'])
    if not required.issubset(grid.columns):
        raise ValueError(f'Missing columns for age-window certification: {sorted(required - set(grid.columns))}')
    if grid.duplicated(CELL).any():
        raise ValueError('Duplicate full-grid coordinates: certificate cannot be trusted')
    z = grid.copy()
    age = pd.to_numeric(z['age'], errors='coerce')
    min_age = pd.to_numeric(z['min_age'], errors='coerce')
    max_age = pd.to_numeric(z['max_age'], errors='coerce')
    f = z.get('is_factory', pd.Series(False, index=z.index)).fillna(False).astype(bool)
    obligation = (~f) & age.ge(1) & age.ge(min_age) & age.lt(max_age) & age.le(max_required_age)
    trusted = z['publication_allowed'].fillna(False).astype(bool)
    by_group = z[GROUP].copy()
    by_group['_required'] = obligation.to_numpy(bool)
    by_group['_trusted'] = (obligation & trusted).to_numpy(bool)
    counts = by_group.groupby(GROUP, dropna=False, as_index=False, sort=False).agg(
        coverage_predicted_required=('_required', 'sum'),
        coverage_reliable_required=('_trusted', 'sum'),
    )
    bounds = z.groupby(GROUP, dropna=False, as_index=False).agg(
        coverage_source_min_age=('min_age', 'min'),
        coverage_source_max_age=('max_age', 'max'),
    )
    counts = counts.merge(bounds, on=GROUP, validate='one_to_one')
    caps = mileage_caps or {}

    def combos(cap):
        return sum(1 if b == 0 else len(condition_levels)
                   for b in bucket_levels if int(b) <= int(cap))

    def expected(row):
        lo = row.coverage_source_min_age; hi = row.coverage_source_max_age
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 0
        a0 = max(1, int(np.ceil(lo))); a1 = min(int(max_required_age), int(np.ceil(hi)) - 1)
        return sum(combos(caps.get(age, 8)) for age in range(a0, a1 + 1)) if a0 <= a1 else 0

    counts['coverage_expected_required'] = counts.apply(expected, axis=1).astype(int)
    counts['coverage_missing_grid'] = ((counts.coverage_expected_required - counts.coverage_predicted_required).clip(lower=0)).astype(int)
    counts['coverage_unreliable_required'] = ((counts.coverage_predicted_required - counts.coverage_reliable_required).clip(lower=0)).astype(int)
    counts['coverage_window_applicable'] = counts.coverage_expected_required.gt(0)
    counts['coverage_full_to_age6'] = (counts.coverage_window_applicable
                                        & counts.coverage_reliable_required.eq(counts.coverage_expected_required)
                                        & counts.coverage_missing_grid.eq(0))
    counts['coverage_reliable_share'] = (counts.coverage_reliable_required /
                                          counts.coverage_expected_required.replace(0, np.nan))
    z = z.merge(counts, on=GROUP, how='left', validate='many_to_one')
    z['coverage_publication_mode'] = publication_mode
    z['coverage_publish_ready'] = z.coverage_full_to_age6 | ~z.coverage_window_applicable
    if publication_mode == 'strict':
        veto = z.coverage_window_applicable & ~z.coverage_full_to_age6
        z.loc[veto, 'publication_allowed'] = False
        z.loc[veto, 'is_reliable'] = False
        if 'publish_state' in z.columns:
            z.loc[veto & ~f, 'publish_state'] = 'GROUP_AGE6_INCOMPLETE'
    return z
