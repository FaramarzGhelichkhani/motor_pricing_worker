"""Motorcycle price surface package.

Public entry point: MotorcyclePriceSurface.fit(df).create_surface().

V10 is the single, final reliability/publication gate. There is no V8/V9
hierarchical reliability layer anymore: it was dead weight because V10's
`apply()` unconditionally recomputed `is_reliable` / `publication_allowed`
and made every earlier decision irrelevant. Tier A/B/C/D labels are now
derived directly from V10 evidence for display/debugging only; they never
gate publication by themselves.
"""
from .core import MotorcyclePriceSurface, SurfaceConfig, FEATURES_24, GROUP, CELL

__all__ = ["MotorcyclePriceSurface", "SurfaceConfig", "FEATURES_24", "GROUP", "CELL"]
__version__ = "11.0.0"