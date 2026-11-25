"""VIST-NRF training script.

Per specification §13: every experiment defined by YAML; every run needs a YAML
config, seed, saved split, run ID, code commit, and environment metadata. Main
experiment scripts refuse to run unless committed preregistration hash matches;
exploratory runs require --exploratory and go to a separate directory.

Usage:
  python scripts/train.py configs/baselines/e1_a_to_a.yaml
  python scripts/train.py configs/baselines/e1_a_to_a.yaml --exploratory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow running as `python scripts/train.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vist_nrf.utils.io import (
    environment_metadata,
    ensure_dir,
    git_commit,
    save_jsonl,
    sha256_file,
    utc_now,
)
from vist_nrf.utils.prereg import PreregistrationError, enforce_prereg, prereg_hash
from vist_nrf.utils.seed import set_seed
from vist_nrf.data.physionet2019 import load_environment
from vist_nrf.data.splits import split_by_environment, assert_splits_valid
from vist_nrf.data.targets import build_targets
from vist_nrf.features.history import FeatureConfig, build_feature_matrix, assert_causal
from vist_nrf.eval.censoring import censoring_report, assert_no_early_event_prediction, assert_censoring_not_negative
from vist_nrf.eval.metrics import evaluate
from vist_nrf.models import build_model


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_id(cfg: dict) -> str:
    base = cfg.get("run_id") or cfg.get("experiment", "run")
    seed = cfg.get("seed", 0)
    return f"{base}_seed{seed}"


def run(cfg_path: str | Path, exploratory: bool = False) -> dict:
    cfg = load_config(cfg_path)
    repo_root = Path(__file__).resolve().parents[1]

    # --- seeds ---
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    # --- preregistration enforcement ---
    expected_hash = cfg.get("prereg_hash")
    if not exploratory and expected_hash is None:
        # require the committed hash to be recorded in config
        committed = prereg_hash(repo_root)
        if committed is None:
            raise PreregistrationError(
                "preregistration.md not found. Commit+freeze preregistration before confirmatory runs."
            )
        expected_hash = committed
    try:
        enforce_prereg(repo_root, expected_hash=expected_hash, allow_exploratory=exploratory)
    except PreregistrationError as e:
        print(f"PREREGISTRATION BLOCK: {e}")
        raise

    run_id = build_run_id(cfg)
    print(f"RUN {run_id} | config={cfg_path} | seed={seed} | exploratory={exploratory}")

    # --- environment metadata + commit ---
    commit = git_commit(repo_root)
    meta = environment_metadata()
    meta["commit"] = commit
    meta["run_id"] = run_id
    meta["config"] = str(cfg_path)
    meta["exploratory"] = bool(exploratory)

    # --- data ---
    data_cfg = cfg.get("dataset", {})
    raw_root = data_cfg.get("raw_root")  # path to downloaded physionet2019 training/
    train_env = cfg.get("train_env", "A")
    test_envs = cfg.get("test_envs", ["B"])
    horizon_hours = int(cfg.get("horizon_hours", 12))
    min_history = int(cfg.get("min_history_hours", 6))

    # Build synthetic fixture data if no real data (for CI / smoke test).
    if raw_root is None or not Path(raw_root).exists():
        print("No real PhysioNet data found — using synthetic fixture (smoke test only).")
        patient_frames = {
            f"{train_env}P{i:03d}": _synthetic_patient(48, seed=100 + i, ever_pos=(i % 2 == 0))
            for i in range(40)
        }
        for env in test_envs:
            patient_frames.update(
                {
                    f"{env}P{i:03d}": _synthetic_patient(48, seed=200 + i, ever_pos=(i % 3 == 0))
                    for i in range(30)
                }
            )
    else:
        print(f"Loading real PhysioNet data from {raw_root}")
        patient_frames = {}
        for env in [train_env] + test_envs:
            env_frames = load_environment(raw_root, env)
            patient_frames.update(env_frames)
            print(f"  env {env}: {len(env_frames)} patients")

    # --- targets ---
    targets_long, patient_summary = build_targets(
        patient_frames,
        label_col="SepsisLabel",
        horizon_hours=horizon_hours,
        min_history_hours=min_history,
    )

    # censoring acceptance checks
    assert_no_early_event_prediction(targets_long)
    assert_censoring_not_negative(targets_long)
    cens_report = censoring_report(targets_long, patient_summary)
    print(f"Censoring: {cens_report['n_censored_rows']} rows censored, "
          f"{cens_report['n_eligible_rows']} eligible")

    # --- features ---
    clinical_cols = [c for c in patient_frames[next(iter(patient_frames))].columns
                     if c not in ("SepsisLabel", "hour", "patient_id", "Age", "Gender",
                                  "HospAdmTime", "ICULOS", "ICU_unit", "_source_file")]
    static_cols = [c for c in ("Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit")
                   if c in patient_frames[next(iter(patient_frames))].columns]
    cfg_feat = FeatureConfig(
        clinical_cols=clinical_cols,
        static_cols=static_cols,
        min_history_hours=min_history,
        patient_id_col="patient_id",
        hour_col="hour",
    )
    feature_matrix = build_feature_matrix(patient_frames, cfg_feat)

    # leakage test — this BLOCKS model experiments if it fails (spec §6.4)
    assert_causal(feature_matrix, patient_frames, cfg_feat)
    print("Causality / leakage test: PASSED")

    # --- splits (patient-level, per env) ---
    patients_by_env = {}
    ever_pos_by_env = {}
    for env in [train_env] + test_envs:
        env_pids = [p for p in patient_frames if p.startswith(env)]
        patients_by_env[env] = env_pids
        ever_pos_by_env[env] = {
            p: int(patient_summary[patient_summary["patient_id"] == p]["n_positive"].iloc[0] > 0)
            if p in patient_summary["patient_id"].values else 0
            for p in env_pids
        }
    splits = split_by_environment(patients_by_env, ever_pos_by_env, seed=seed)
    assert_splits_valid(splits)
    print(f"Splits: { {e: (len(s.train), len(s.val), len(s.test)) for e, s in splits.items()} }")

    # --- model ---
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "logistic_regression")
    model_params = dict(model_cfg.get("params", {}))
    model = build_model(model_name, seed=seed, **model_params)

    # --- train on source env train split ---
    train_pids = splits[train_env].train
    # join targets with features
    tr = targets_long.merge(
        feature_matrix[["patient_id", "hour"] + [c for c in feature_matrix.columns
                                                   if c not in ("patient_id", "hour")]],
        on=["patient_id", "hour"], how="inner",
    )
    tr = tr[tr["eligible_for_scoring"]]
    feature_cols = [c for c in tr.columns if c not in ("patient_id", "hour", "y", "censored", "eligible_for_scoring")]
    Xtr = tr[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ytr = tr["y"].to_numpy(dtype=int)
    env_tr = np.zeros(len(ytr), dtype=int)  # source env index 0

    print(f"Training {model_name} on {len(Xtr)} rows from {len(train_pids)} patients (env {train_env})")
    model.fit(Xtr, ytr, envs=env_tr)

    # --- evaluate on each test env ---
    results = []
    for test_env in test_envs:
        test_pids = splits[test_env].test
        te_mask = feature_matrix["patient_id"].isin(test_pids)
        te_fm = feature_matrix.loc[te_mask]
        te = targets_long.merge(
            te_fm[["patient_id", "hour"] + [c for c in te_fm.columns
                                             if c not in ("patient_id", "hour")]],
            on=["patient_id", "hour"], how="inner",
        )
        te = te[te["eligible_for_scoring"]]
        Xte = te[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        yte = te["y"].to_numpy(dtype=int)
        env_te = np.ones(len(yte), dtype=int) * (1 if test_env != train_env else 0)
        p = model.predict_proba(Xte)[:, 1]
        m = evaluate(yte, p, threshold=0.5)
        rec = {
            "run_id": run_id,
            "train_env": train_env,
            "test_env": test_env,
            "model": model_name,
            "horizon_hours": horizon_hours,
            "seed": seed,
            "commit": commit,
            "n_test_patients": len(test_pids),
            "n_test_rows": len(Xte),
            **m.__dict__,
            "censored_rows": cens_report["n_censored_rows"],
            "utc": utc_now(),
        }
        rec.update(meta)
        results.append(rec)
        print(f"  {test_env}: AUROC={m.auroc:.3f} AUPRC={m.auprc:.3f} "
              f"Brier={m.brier:.3f} ECE={m.ece:.3f} prev={m.prevalence:.3f} n={m.n}")

    # --- save results JSONL ---
    results_dir = cfg.get("results_dir", "results")
    ensure_dir(results_dir)
    for rec in results:
        save_jsonl(Path(results_dir) / "runs" / f"{run_id}.jsonl", rec)
    # also a summary
    summary_path = Path(results_dir) / "summary.jsonl"
    for rec in results:
        save_jsonl(summary_path, rec)
    print(f"Results saved to {results_dir}/runs/{run_id}.jsonl and {summary_path}")
    return results


def _synthetic_patient(n_hours, seed, ever_pos=False):
    """Generate a single synthetic patient frame for smoke tests."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n_hours, dtype=float)
    cols = {
        "hour": hours,
        "patient_id": f"S{seed}",
        "Age": float(rng.integers(40, 90)),
        "Gender": int(rng.integers(0, 2)),
        "HospAdmTime": 0.0,
        "ICULOS": hours,
        "ICU_unit": "MICU",
        "SepsisLabel": 0.0,
    }
    for j in range(8):
        v = rng.normal(0, 1, n_hours)
        v[rng.random(n_hours) < 0.4] = np.nan
        cols[f"v{j}"] = v
    if ever_pos:
        # label turns 1 at mid-stay and stays (documented convention: +6h onset)
        event_hour = n_hours // 2
        cols["SepsisLabel"] = (hours >= event_hour).astype(float)
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="YAML config path")
    ap.add_argument("--exploratory", action="store_true",
                    help="Allow run without frozen preregistration (goes to exploratory dir)")
    args = ap.parse_args()
    run(args.config, exploratory=args.exploratory)


if __name__ == "__main__":
    main()
