"""Unit tests for VIST-NRF data layer (synthetic fixtures only — no patient data)."""

from __future__ import annotations

import pandas as pd

from vist_nrf.data.splits import assert_splits_valid, split_by_environment
from vist_nrf.data.targets import (
    build_targets,
    make_horizon_labels,
    reconstruct_event_time,
    validate_label_pattern,
)


# ---------------------------------------------------------------------------
# targets.py
# ---------------------------------------------------------------------------

def test_event_reconstruction_basic():
    labels = pd.Series([0, 0, 0, 1, 1, 1], name="SepsisLabel")
    ev = reconstruct_event_time("P1", labels)
    assert ev["status"] == "event"
    assert ev["t_first_positive"] == 3
    assert ev["t_event"] == 9  # 3 + 6
    assert ev["eligible"] is True
    assert ev["audit"]["left_censored"] is False


def test_left_censored():
    labels = pd.Series([1, 1, 1], name="SepsisLabel")
    ev = reconstruct_event_time("P2", labels)
    assert ev["status"] == "left_censored"
    assert ev["eligible"] is False
    assert ev["t_event"] is None


def test_no_event():
    labels = pd.Series([0, 0, 0, 0], name="SepsisLabel")
    ev = reconstruct_event_time("P3", labels)
    assert ev["status"] == "no_event"
    assert ev["eligible"] is False


def test_malformed():
    labels = pd.Series([1, 0, 1], name="SepsisLabel")
    ev = reconstruct_event_time("P4", labels)
    assert ev["status"] == "ambiguous"
    assert ev["audit"]["malformed"] is True


def test_validate_label_pattern_non_binary():
    labels = pd.Series([0, 2, 0], name="SepsisLabel")
    pat = validate_label_pattern(labels)
    assert pat["malformed"] is True


def test_horizon_labels_event_patient():
    # t_first_positive=3, t_event=9, H=12, min_history=6, n_hours=20
    tl = make_horizon_labels("P1", n_hours=20, t_event=9, horizon_hours=12, min_history_hours=6)
    assert not tl.empty
    # excluded: t<6 (history), t>=9 (event occurred)
    assert set(tl["hour"]) == set(range(6, 9))  # 6,7,8
    assert (tl["y"] == 1).all()
    assert (tl["censored"] == False).all()
    assert (tl["eligible_for_scoring"] == True).all()


def test_horizon_labels_non_event_censoring():
    # non-event, n_hours=20, H=12: y=0 only when t+12 < 20 => t<8
    tl = make_horizon_labels("P3", n_hours=20, t_event=None, horizon_hours=12, min_history_hours=6)
    neg = tl[tl["y"] == 0]
    censored = tl[tl["censored"] == True]
    assert (neg["hour"] < 8).all()
    assert (censored["hour"] >= 8).all()
    assert (censored["eligible_for_scoring"] == False).all()


def test_build_targets_end_to_end():
    frames = {
        "P1": pd.DataFrame({"SepsisLabel": [0, 0, 0, 1, 1, 1, 1, 1]}),
        "P2": pd.DataFrame({"SepsisLabel": [1, 1, 1, 1]}),          # left-censored
        "P3": pd.DataFrame({"SepsisLabel": [0, 0, 0, 0]}),            # no event
    }
    targets, summary = build_targets(frames)
    assert len(summary) == 3
    assert summary[summary["patient_id"] == "P1"]["status"].iloc[0] == "event"
    assert summary[summary["patient_id"] == "P2"]["status"].iloc[0] == "left_censored"
    assert summary[summary["patient_id"] == "P3"]["status"].iloc[0] == "no_event"
    # only P1 contributes eligible prediction rows
    assert (targets["patient_id"] == "P1").all()
    assert (targets["eligible_for_scoring"] == True).all()


# ---------------------------------------------------------------------------
# splits.py
# ---------------------------------------------------------------------------

def _ever_pos(env_pids, frac_pos=0.5):
    return {p: (1 if i % 2 == 0 else 0) for i, p in enumerate(env_pids)}


def test_split_disjoint_and_covers():
    pids = [f"P{i:03d}" for i in range(100)]
    ep = _ever_pos(pids)
    splits = split_by_environment({"A": pids}, {"A": ep}, seed=7)
    sp = splits["A"]
    assert sp.is_disjoint(), "train/val/test must be disjoint"
    assert sp.covers_all(pids), "split must cover all patients"
    assert len(sp.train) + len(sp.val) + len(sp.test) == 100


def test_split_stratified():
    pids = [f"P{i:03d}" for i in range(100)]
    # 50 positive, 50 negative
    ep = {p: (1 if i < 50 else 0) for i, p in enumerate(pids)}
    splits = split_by_environment({"A": pids}, {"A": ep}, seed=3)
    sp = splits["A"]
    train_pos = sum(ep[p] for p in sp.train)
    val_pos = sum(ep[p] for p in sp.val)
    test_pos = sum(ep[p] for p in sp.test)
    assert train_pos > 0 and val_pos > 0 and test_pos > 0, "each split should have positives"


def test_cross_env_disjoint():
    pids_a = [f"A{i:03d}" for i in range(50)]
    pids_b = [f"B{i:03d}" for i in range(50)]
    ep_a = _ever_pos(pids_a)
    ep_b = _ever_pos(pids_b)
    splits = split_by_environment({"A": pids_a, "B": pids_b}, {"A": ep_a, "B": ep_b}, seed=1)
    assert_splits_valid(splits)  # should not raise


def test_split_reproducible():
    pids = [f"P{i:03d}" for i in range(50)]
    ep = _ever_pos(pids)
    s1 = split_by_environment({"A": pids}, {"A": ep}, seed=5)
    s2 = split_by_environment({"A": pids}, {"A": ep}, seed=5)
    assert s1["A"].train == s2["A"].train
    assert s1["A"].val == s2["A"].val
    assert s1["A"].test == s2["A"].test


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
