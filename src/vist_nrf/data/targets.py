"""Target construction for sepsis forecasting.

Implements specification section 5:
  - Reconstruct operational onset t_event = t_first_positive + 6 hours.
  - Horizon labels y(t, H) = 1 when t < t_event <= t + H.
  - Censoring: incomplete negative horizons are censored, not negative.
  - Exclude prediction times at or after t_event.
  - Require at least min_history_hours of past history.

All functions operate on in-memory pandas DataFrames and are deterministic,
so they can be unit-tested on hand-checked timelines.
"""

from __future__ import annotations

import pandas as pd


class TargetBuildError(ValueError):
    pass


def validate_label_pattern(label_series: pd.Series) -> dict:
    """Audit the SepsisLabel pattern for one patient.

    Returns a dict with:
      first_positive_idx : int or None
      n_rows : int
      n_positive : int
      already_positive_at_start : bool
      malformed : bool  (label goes 1 -> 0 -> 1, or non-binary)
      left_censored : bool  (first available row already positive)
    """
    labels = label_series.astype(float)
    n_rows = int(len(labels))
    n_positive = int((labels == 1).sum())
    non_binary = bool(((labels != 0) & (labels != 1)).any())
    # check 1 -> 0 -> 1 pattern: once label hits 1 it must stay 1 (documented convention)
    first_positive_idx = None
    for i, v in enumerate(labels.tolist()):
        if v == 1:
            first_positive_idx = i
            break
    non_monotonic = False
    if first_positive_idx is not None:
        after_first = labels.tolist()[first_positive_idx:]
        non_monotonic = any(v == 0 for v in after_first)
    malformed = bool(non_binary or non_monotonic)
    already_positive_at_start = bool(n_rows > 0 and labels.iloc[0] == 1)
    left_censored = already_positive_at_start
    return {
        "first_positive_idx": first_positive_idx,
        "n_rows": n_rows,
        "n_positive": n_positive,
        "already_positive_at_start": already_positive_at_start,
        "malformed": malformed,
        "left_censored": left_censored,
    }


def reconstruct_event_time(
    patient_id: str,
    labels: pd.Series,
    hour_index: pd.Series | None = None,
) -> dict:
    """Reconstruct operational sepsis onset for one patient.

    t_event = t_first_positive + 6 hours (documented label shift).

    Returns dict:
      patient_id, t_first_positive (hour index or clock), t_event,
      eligible (has a valid event), status ('event','no_event','left_censored','ambiguous'),
      audit (from validate_label_pattern)
    """
    audit = validate_label_pattern(labels)

    if audit["malformed"]:
        return {
            "patient_id": patient_id,
            "t_first_positive": None,
            "t_event": None,
            "eligible": False,
            "status": "ambiguous",
            "audit": audit,
        }
    if audit["left_censored"] and audit["n_positive"] == len(labels):
        # every row positive from the start: fully left-censored
        return {
            "patient_id": patient_id,
            "t_first_positive": None,
            "t_event": None,
            "eligible": False,
            "status": "left_censored",
            "audit": audit,
        }
    if audit["left_censored"]:
        # first row positive but later rows may be 0: ambiguous pattern
        return {
            "patient_id": patient_id,
            "t_first_positive": None,
            "t_event": None,
            "eligible": False,
            "status": "ambiguous",
            "audit": audit,
    }
    if audit["n_positive"] == 0:
        return {
            "patient_id": patient_id,
            "t_first_positive": None,
            "t_event": None,
            "eligible": False,
            "status": "no_event",
            "audit": audit,
        }

    fp = audit["first_positive_idx"]
    if hour_index is not None:
        t_first = hour_index.iloc[fp]
        t_event = hour_index.iloc[fp] + 6  # hours
    else:
        t_first = fp
        t_event = fp + 6
    return {
        "patient_id": patient_id,
        "t_first_positive": t_first,
        "t_event": t_event,
        "eligible": True,
        "status": "event",
        "audit": audit,
    }


def make_horizon_labels(
    patient_id: str,
    n_hours: int,
    t_event: float | None,
    horizon_hours: int = 12,
    min_history_hours: int = 6,
) -> pd.DataFrame:
    """Build per-hour prediction rows for one patient.

    For each eligible prediction hour t (0-indexed hour of stay):
      y = 1 when t < t_event <= t + horizon_hours
      y = 0 only if full horizon observed (t + horizon_hours <= n_hours - 1)
      otherwise censored (exclude).

    Excludes t >= t_event (event already occurred).
    Requires t >= min_history_hours (at least 6h past history).

    Returns DataFrame with columns:
      patient_id, hour, y, censored, eligible_for_scoring
    """
    rows = []
    for t in range(n_hours):
        # require minimum history
        if t < min_history_hours:
            continue
        # exclude times at or after the event
        if t_event is not None and t >= t_event:
            continue
        horizon_end = t + horizon_hours
        if t_event is None:
            # non-event patient: negative only if full horizon observed
            if horizon_end < n_hours:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "hour": t,
                        "y": 0,
                        "censored": False,
                        "eligible_for_scoring": True,
                    }
                )
            else:
                rows.append(
                    {
                        "patient_id": patient_id,
                        "hour": t,
                        "y": None,
                        "censored": True,
                        "eligible_for_scoring": False,
                    }
                )
        else:
            if t < t_event <= horizon_end:
                y = 1
            else:
                y = 0
            # event patients: full future to t_event is observed by definition
            # (label exists through stay). Mark not censored.
            rows.append(
                {
                    "patient_id": patient_id,
                    "hour": t,
                    "y": y,
                    "censored": False,
                    "eligible_for_scoring": True,
                }
            )
    return pd.DataFrame(rows)


def build_targets(
    patient_frames: dict[str, pd.DataFrame],
    label_col: str = "SepsisLabel",
    hour_col: str | None = None,
    horizon_hours: int = 12,
    min_history_hours: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build targets for all patients.

    patient_frames: {patient_id: DataFrame with at least label_col, sorted by hour}
    Returns (targets_long, patient_summary).
    """
    target_rows = []
    summary_rows = []
    for pid, df in patient_frames.items():
        labels = df[label_col]
        hour_index = df[hour_col] if hour_col and hour_col in df.columns else None
        ev = reconstruct_event_time(pid, labels, hour_index)
        summary_rows.append(
            {
                "patient_id": pid,
                "status": ev["status"],
                "t_first_positive": ev["t_first_positive"],
                "t_event": ev["t_event"],
                "eligible": ev["eligible"],
                "n_rows": int(len(df)),
                **ev["audit"],
            }
        )
        if not ev["eligible"]:
            continue
        n_hours = int(len(df))
        te = ev["t_event"]
        # use hour positions (0-indexed) if no hour_col; else map clock to position
        if hour_col and hour_col in df.columns:
            # positions are row indices; t_event is a clock -> convert to position
            # simplest: treat t_event as hour-position offset from first_positive
            t_first_pos = ev["audit"]["first_positive_idx"]
            te_pos = t_first_pos + 6
        else:
            te_pos = ev["t_event"]
        tl = make_horizon_labels(
            pid, n_hours, te_pos, horizon_hours=horizon_hours,
            min_history_hours=min_history_hours,
        )
        target_rows.append(tl)

    targets_long = (
        pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
    )
    patient_summary = pd.DataFrame(summary_rows)
    return targets_long, patient_summary
