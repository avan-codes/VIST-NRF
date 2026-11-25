"""VIST-NRF feature engineering: causal, leakage-free features for irregular clinical time series.

Specification section 6.2 allowed features at time t:
  - Last observed value per variable (LOCF)
  - Current observation mask per variable
  - Time since last observation per variable
  - Measurement counts over previous 6, 12, 24 hours
  - Causal rolling min/max/mean over previous 6 and 24 hours (observed values only)
  - Age, gender, ICU-unit indicators, HospAdmTime, ICULOS at t

Specification section 6.3 forbidden features:
  - SepsisLabel as input
  - Any future value/mask/label/timestamp/statistic
  - Total ICU stay length / future record length
  - Patient ID as predictive feature
  - Hospital identity as predictive feature in primary model (may be env label for eval)

All functions are deterministic and causal: features at t depend ONLY on history <= t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Default clinical variable columns (40 inputs from PhysioNet/CinC 2019).
# The actual column names are read from the downloaded files; these are the
# canonical names from the challenge documentation.
DEFAULT_CLINICAL_COLS = [
    "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "K", "Na",
    "Cl", "Ca", "Mg", "Glucose", "Lactate", "Bilirubin_direct", "Hb", "Hct",
    "WBC", "Platelets", "PTT", "PT", "Albumin", "AST", "ALT", "ALP", "BUN",
    "Creatinine", "Glucose_serum", "TroponinI", "BNP", "D_dimer", "Lactate_serum",
    "TroponinT", "Bilirubin_total", "Fibrinogen", "Troponin", "Creatine_kinase",
    "Amylase", "Lipase", "GGT", "LDH",
]
STATIC_COLS = ["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"]


@dataclass
class FeatureConfig:
    clinical_cols: list[str]
    static_cols: list[str]
    label_col: str = "SepsisLabel"
    hour_col: str = "hour"
    patient_id_col: str = "patient_id"
    rolling_windows: tuple[int, ...] = (6, 24)
    count_windows: tuple[int, ...] = (6, 12, 24)
    min_history_hours: int = 6


def _causal_locf_mask_time_since(
    df: pd.DataFrame, cols: list[str], hour_col: str
) -> pd.DataFrame:
    """Compute LOCF, observation mask, and time-since-last-observation.

    For each row at hour t:
      - locf_j(t) = most recent observed value of j at hour <= t (NaN if never observed)
      - mask_j(t) = 1 if j observed at hour t else 0
      - tsl_j(t) = t - hour of most recent observation of j (NaN if never observed)

    Purely causal: uses expanding operations, never future rows.
    """
    hours = df[hour_col].to_numpy(dtype=float)
    blocks = []

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        observed = ~np.isnan(vals)
        mask = observed.astype(float)

        # LOCF: forward-fill observed values within the patient's history.
        # Use pandas ffill which is causal (forward fill only looks backward).
        s = pd.Series(vals, index=df.index)
        locf = s.where(observed).ffill().to_numpy(dtype=float)

        # Time since last observation: for each row, find most recent observed hour <= t.
        obs_hours = np.where(observed, hours, np.nan)
        obs_hour_series = pd.Series(obs_hours, index=df.index).ffill()
        last_obs_hour = obs_hour_series.to_numpy(dtype=float)
        tsl = hours - last_obs_hour  # NaN where never observed yet

        blocks.append(
            pd.DataFrame(
                {
                    f"{col}_locf": locf,
                    f"{col}_mask": mask,
                    f"{col}_tsl": tsl,
                },
                index=df.index,
            )
        )

    return pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)


def _causal_counts(df: pd.DataFrame, cols: list[str], hour_col: str,
                   windows: tuple[int, ...]) -> pd.DataFrame:
    """Measurement counts per variable over previous W hours (causal, vectorized)."""
    out = pd.DataFrame(index=df.index)
    hours = df[hour_col].to_numpy(dtype=float)
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        observed = ~df[col].isna().to_numpy()
        obs_hours = hours[observed]  # sorted because hours increasing
        for w in windows:
            # elements in (t-w, t]: #<=t minus #<=(t-w)
            hi = np.searchsorted(obs_hours, hours, side="right")
            lo = np.searchsorted(obs_hours, hours - w, side="right")
            out[f"{col}_count_{w}h"] = (hi - lo).astype(float)
    return out


def _causal_rolling_stats(df: pd.DataFrame, cols: list[str], hour_col: str,
                          windows: tuple[int, ...]) -> pd.DataFrame:
    """Causal rolling min/max/mean over observed values only, previous W hours."""
    out = pd.DataFrame(index=df.index)
    hours = df[hour_col].to_numpy(dtype=float)
    n = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        observed = ~np.isnan(vals)
        obs_hours = hours[observed]
        obs_vals = vals[observed]
        for w in windows:
            hi = np.searchsorted(obs_hours, hours, side="right")
            lo = np.searchsorted(obs_hours, hours - w, side="right")
            for stat in ("min", "max", "mean"):
                res = np.full(n, np.nan, dtype=float)
                for i in range(n):
                    sl = slice(int(lo[i]), int(hi[i]))
                    if sl.start >= sl.stop:
                        continue
                    v = obs_vals[sl]
                    if stat == "min":
                        res[i] = np.min(v)
                    elif stat == "max":
                        res[i] = np.max(v)
                    else:
                        res[i] = np.mean(v)
                out[f"{col}_roll_{stat}_{w}h"] = res
    return out


def compute_patient_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute all causal features for one patient's DataFrame (sorted by hour).

    Returns a DataFrame aligned to df's index with feature columns only
    (no label, no patient id). Purely causal.
    """
    df = df.sort_values(cfg.hour_col).reset_index(drop=True)

    parts = []
    parts.append(_causal_locf_mask_time_since(df, cfg.clinical_cols, cfg.hour_col))
    parts.append(_causal_counts(df, cfg.clinical_cols, cfg.hour_col, cfg.count_windows))
    parts.append(_causal_rolling_stats(df, cfg.clinical_cols, cfg.hour_col, cfg.rolling_windows))

    # Static features repeated at each hour (available at t): age, gender, ICU unit, adm time, ICULOS
    static_parts = []
    for col in cfg.static_cols:
        if col in df.columns:
            static_parts.append(pd.Series(df[col].to_numpy(), index=df.index, name=col))
        else:
            static_parts.append(pd.Series(np.nan, index=df.index, name=col))
    if static_parts:
        parts.append(pd.concat(static_parts, axis=1))

    feats = pd.concat(parts, axis=1)
    feats.index = df.index
    return feats


def build_feature_matrix(patient_frames: dict[str, pd.DataFrame], cfg: FeatureConfig) -> pd.DataFrame:
    """Build a long feature matrix across all patients.

    Returns DataFrame with columns: patient_id, hour, plus all feature columns.
    No labels, no future information.
    """
    frames = []
    for pid, df in patient_frames.items():
        feats = compute_patient_features(df, cfg)
        feats = feats.copy()
        feats.insert(0, cfg.patient_id_col, pid)
        feats.insert(1, cfg.hour_col, df[cfg.hour_col].to_numpy())
        frames.append(feats)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def assert_causal(features: pd.DataFrame, patient_frames: dict[str, pd.DataFrame],
                  cfg: FeatureConfig) -> None:
    """Leakage test: features at t unchanged when computed from truncated history vs full record.

    For each patient, recompute features at representative hours (early / mid / late)
    using only rows up to t and verify they match the full-record computation. This
    empirically confirms features are causal (no future leakage) without the O(n^2)
    cost of testing every hour of every patient.
    """
    for pid, full_df in patient_frames.items():
        n = len(full_df)
        if n <= cfg.min_history_hours:
            continue
        full_feats = compute_patient_features(full_df, cfg)
        # Representative hours: min history, midpoint, last hour
        test_times = {
            cfg.min_history_hours,
            min(n - 1, cfg.min_history_hours + (n - 1 - cfg.min_history_hours) // 2),
            n - 1,
        }
        for t in sorted(test_times):
            if t < cfg.min_history_hours:
                continue
            truncated = full_df.iloc[: t + 1].copy()
            trunc_feats = compute_patient_features(truncated, cfg)
            # compare the last row of truncated computation to row t of full
            full_row = full_feats.iloc[t]
            trunc_row = trunc_feats.iloc[-1]
            for col in trunc_row.index:
                if col in (cfg.patient_id_col, cfg.hour_col):
                    continue
                a = full_row[col]
                b = trunc_row[col]
                # skip non-numeric (static string) columns
                if not (isinstance(a, (int, float, np.floating, np.integer)) and
                        isinstance(b, (int, float, np.floating, np.integer))):
                    continue
                if pd.isna(a) and pd.isna(b):
                    continue
                if not np.isclose(a, b, equal_nan=True):
                    raise AssertionError(
                        f"future leakage at patient={pid} hour={t} col={col}: "
                        f"full={a} truncated={b}"
                    )


def assert_no_future_label(features: pd.DataFrame, targets: pd.DataFrame,
                           patient_id_col: str, hour_col: str) -> None:
    """Ensure no feature column is the label or contains future info by name."""
    forbidden = {"sepsislabel", "patientid"}
    for col in features.columns:
        cl = str(col).lower()
        if any(f in cl for f in forbidden):
            raise AssertionError(f"forbidden feature name: {col}")
