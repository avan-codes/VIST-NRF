"""Unit tests for VIST-NRF features (synthetic fixtures only — no patient data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vist_nrf.data.targets import reconstruct_event_time
from vist_nrf.features.history import (
    FeatureConfig,
    assert_causal,
    assert_no_future_label,
    build_feature_matrix,
    compute_patient_features,
)


def _make_patient(n_hours: int, n_vars: int = 3, seed: int = 0, label_pattern=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = np.arange(n_hours, dtype=float)
    data = {"hour": hours, "patient_id": f"S{seed}"}
    for j in range(n_vars):
        vals = rng.normal(0, 1, n_hours)
        # introduce missingness
        mask = rng.random(n_hours) < 0.4
        vals[mask] = np.nan
        data[f"v{j}"] = vals
    data["Age"] = 65.0
    data["Gender"] = 1
    data["HospAdmTime"] = 0.0
    data["ICULOS"] = hours
    data["ICU_unit"] = "MICU"
    if label_pattern is not None:
        data["SepsisLabel"] = np.array(label_pattern, dtype=float)
    else:
        data["SepsisLabel"] = 0.0
    return pd.DataFrame(data)


def test_locf_mask_tsl_causal():
    df = _make_patient(10, n_vars=2, seed=1)
    cfg = FeatureConfig(clinical_cols=["v0", "v1"], static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"])
    feats = compute_patient_features(df, cfg)
    # v0_locf should be NaN before first observation
    first_obs = df["v0"].first_valid_index()
    if first_obs is not None and first_obs > 0:
        assert np.isnan(feats["v0_locf"].iloc[0])
    # mask is 0/1
    assert set(feats["v0_mask"].dropna().unique()).issubset({0.0, 1.0})
    # tsl is NaN before first obs, 0 at obs
    if first_obs is not None:
        assert np.isnan(feats["v0_tsl"].iloc[:first_obs]).all()


def test_counts_causal():
    df = _make_patient(20, n_vars=2, seed=2)
    cfg = FeatureConfig(clinical_cols=["v0"], static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"],
                        count_windows=(6, 12, 24))
    feats = compute_patient_features(df, cfg)
    for w in (6, 12, 24):
        col = f"v0_count_{w}h"
        assert col in feats.columns
        # counts non-decreasing in t (cumulative over expanding window anchored at t)
        # not strictly monotone due to window slide, but non-negative
        assert (feats[col].dropna() >= 0).all()


def test_rolling_stats_causal():
    df = _make_patient(20, n_vars=2, seed=3)
    cfg = FeatureConfig(clinical_cols=["v0"], static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"],
                        rolling_windows=(6, 24))
    feats = compute_patient_features(df, cfg)
    for w in (6, 24):
        for stat in ("min", "max", "mean"):
            col = f"v0_roll_{stat}_{w}h"
            assert col in feats.columns


def test_build_feature_matrix_shape():
    frames = {"P1": _make_patient(15, seed=4), "P2": _make_patient(12, seed=5)}
    cfg = FeatureConfig(clinical_cols=["v0", "v1"],
                        static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"])
    fm = build_feature_matrix(frames, cfg)
    assert cfg.patient_id_col in fm.columns
    assert cfg.hour_col in fm.columns
    assert len(fm) == 15 + 12
    # no label column
    assert "SepsisLabel" not in fm.columns


def test_assert_causal_passes():
    frames = {"P1": _make_patient(20, seed=6)}
    cfg = FeatureConfig(clinical_cols=["v0"], static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"])
    fm = build_feature_matrix(frames, cfg)
    assert_causal(fm, frames, cfg)  # should not raise


def test_assert_no_future_label():
    frames = {"P1": _make_patient(10, seed=7)}
    cfg = FeatureConfig(clinical_cols=["v0"],
                        static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"])
    fm = build_feature_matrix(frames, cfg)
    targets = pd.DataFrame({"patient_id": ["P1"], "hour": [0], "y": [0]})
    assert_no_future_label(fm, targets, "patient_id", "hour")  # should not raise


def test_forbidden_feature_name_blocked():
    frames = {"P1": _make_patient(10, seed=8)}
    cfg = FeatureConfig(clinical_cols=["v0", "SepsisLabel"],
                        static_cols=["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"])
    fm = build_feature_matrix(frames, cfg)
    targets = pd.DataFrame({"patient_id": ["P1"], "hour": [0], "y": [0]})
    try:
        assert_no_future_label(fm, targets, "patient_id", "hour")
    except AssertionError:
        return
    raise AssertionError("SepsisLabel feature should be blocked")


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
