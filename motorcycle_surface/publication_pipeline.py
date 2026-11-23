"""One shared publication decision path for production and walk-forward folds.

Do not duplicate this chain: all guards must see only evidence available asof.
"""
from __future__ import annotations

from .publication_policy import apply_publication, PublicationConfig
from .recent_market_guard import apply_recent_market_guard
from .hierarchical_risk_guard import apply_hierarchical_oof_guard, HierarchicalRiskConfig


def apply_complete_publication(surface, ads, asof, earlier_oof,
                               publication_config: PublicationConfig | None = None,
                               model_risk_config: HierarchicalRiskConfig | None = None):
    local = apply_publication(surface, asof, earlier_oof,
                              publication_config or PublicationConfig())
    market = apply_recent_market_guard(local, ads, asof)
    return apply_hierarchical_oof_guard(
        market, earlier_oof, asof,
        model_risk_config or HierarchicalRiskConfig())
