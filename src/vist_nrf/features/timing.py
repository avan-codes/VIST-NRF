"""Timing / mask feature helpers (thin wrappers over history module)."""

from __future__ import annotations

from vist_nrf.features.history import (
    FeatureConfig,
    assert_causal,
    assert_no_future_label,
    build_feature_matrix,
    compute_patient_features,
)

__all__ = [
    "FeatureConfig",
    "assert_causal",
    "assert_no_future_label",
    "build_feature_matrix",
    "compute_patient_features",
]
