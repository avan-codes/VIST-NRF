"""VIST-NRF censoring & acceptance tests (§5.3).

Censoring rules:
  - Exclude prediction times at or after t_event (event already occurred).
  - Do not exclude a time merely because SepsisLabel is already 1 (label can turn
    positive 6h before t_event).
  - Require >= 6h of past history.
  - If a record ends before a complete negative horizon is observed, that example
    is censored, not automatically negative. Exclude via complete-horizon rule or
    use a censoring-aware method. Report affected patients/rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def complete_horizon_mask(targets: pd.DataFrame) -> pd.Series:
    """Boolean mask of scoring-eligible rows (not censored)."""
    return targets["eligible_for_scoring"].astype(bool)


def censored_count(targets: pd.DataFrame) -> int:
    return int((~targets["eligible_for_scoring"]).sum())


def censored_patients(targets: pd.DataFrame) -> list:
    return sorted(targets.loc[~targets["eligible_for_scoring"], "patient_id"].unique().tolist())


def assert_no_early_event_prediction(targets: pd.DataFrame) -> None:
    """No scoring-eligible row at or after t_event."""
    ineligible_after_event = targets[
        (targets["eligible_for_scoring"]) & (targets["hour"] >= targets.get("t_event", np.inf))
    ]
    if len(ineligible_after_event) > 0:
        raise AssertionError(
            f"{len(ineligible_after_event)} eligible rows at/after t_event (future leakage)"
        )


def assert_censoring_not_negative(targets: pd.DataFrame) -> None:
    """Censored rows must not be labeled 0."""
    bad = targets[(~targets["eligible_for_scoring"]) & (targets["y"] == 0)]
    if len(bad) > 0:
        raise AssertionError(f"{len(bad)} censored rows incorrectly labeled y=0")


def censoring_report(targets: pd.DataFrame, patient_summary: pd.DataFrame) -> dict:
    n_censored = censored_count(targets)
    n_eligible = int(targets["eligible_for_scoring"].sum())
    return {
        "n_censored_rows": n_censored,
        "n_eligible_rows": n_eligible,
        "n_censored_patients": len(censored_patients(targets)),
        "censored_patients": censored_patients(targets),
        "left_censored_patients": int((patient_summary["status"] == "left_censored").sum()),
        "ambiguous_patients": int((patient_summary["status"] == "ambiguous").sum()),
        "no_event_patients": int((patient_summary["status"] == "no_event").sum()),
    }
