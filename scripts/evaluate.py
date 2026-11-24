"""VIST-NRF evaluate script.

Loads a trained model checkpoint and a saved split, evaluates on test env(s),
and appends JSONL results. Reuses the same config-driven interface as train.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vist_nrf.utils.io import environment_metadata, ensure_dir, git_commit, save_jsonl, utc_now
from vist_nrf.utils.prereg import PreregistrationError, enforce_prereg, prereg_hash
from vist_nrf.utils.seed import set_seed
from vist_nrf.data.splits import split_by_environment, assert_splits_valid
from vist_nrf.data.targets import build_targets
from vist_nrf.features.history import FeatureConfig, build_feature_matrix
from vist_nrf.eval.censoring import assert_no_early_event_prediction, assert_censoring_not_negative, censoring_report
from vist_nrf.eval.metrics import evaluate
from vist_nrf.models import build_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_run_id(cfg):
    return f"{cfg.get('run_id') or cfg.get('experiment','run')}_seed{cfg.get('seed',0)}"


def _synthetic_patient(n_hours, seed, ever_pos=False):
    import pandas as pd
    rng = np.random.default_rng(seed)
    hours = np.arange(n_hours, dtype=float)
    cols = {"hour": hours, "patient_id": f"S{seed}", "Age": float(rng.integers(40, 90)),
            "Gender": int(rng.integers(0, 2)), "HospAdmTime": 0.0, "ICULOS": hours,
            "ICU_unit": "MICU", "SepsisLabel": 0.0}
    for j in range(8):
        v = rng.normal(0, 1, n_hours)
        v[rng.random(n_hours) < 0.4] = np.nan
        cols[f"v{j}"] = v
    if ever_pos:
        cols["SepsisLabel"] = (hours >= n_hours // 2).astype(float)
    return pd.DataFrame(cols)


def run(cfg_path, checkpoint=None, exploratory=False):
    cfg = load_config(cfg_path)
    repo_root = Path(__file__).resolve().parents[1]
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    expected_hash = cfg.get("prereg_hash") or prereg_hash(repo_root)
    try:
        enforce_prereg(repo_root, expected_hash=expected_hash, allow_exploratory=exploratory)
    except PreregistrationError as e:
        print(f"PREREGISTRATION BLOCK: {e}")
        raise

    run_id = build_run_id(cfg)
    print(f"EVAL {run_id} | config={cfg_path} | checkpoint={checkpoint}")

    commit = git_commit(repo_root)
    meta = environment_metadata()
    meta.update({"commit": commit, "run_id": run_id, "config": cfg_path, "utc": utc_now()})

    train_env = cfg.get("train_env", "A")
    test_envs = cfg.get("test_envs", ["B"])
    horizon_hours = int(cfg.get("horizon_hours", 12))
    min_history = int(cfg.get("min_history_hours", 6))

    raw_root = cfg.get("dataset", {}).get("raw_root")
    if raw_root is None or not Path(raw_root).exists():
        patient_frames = {
            f"{train_env}P{i:03d}": _synthetic_patient(48, seed=100 + i, ever_pos=(i % 2 == 0))
            for i in range(40)
        }
        for env in test_envs:
            patient_frames.update(
                {f"{env}P{i:03d}": _synthetic_patient(48, seed=200 + i, ever_pos=(i % 3 == 0))
                 for i in range(30)}
            )
    else:
        raise NotImplementedError("Real PhysioNet path not automated.")

    targets_long, patient_summary = build_targets(
        patient_frames, label_col="SepsisLabel",
        horizon_hours=horizon_hours, min_history_hours=min_history,
    )
    assert_no_early_event_prediction(targets_long)
    assert_censoring_not_negative(targets_long)
    cens_report = censoring_report(targets_long, patient_summary)

    clinical_cols = [c for c in patient_frames[next(iter(patient_frames))].columns
                     if c not in ("SepsisLabel", "hour", "patient_id", "Age", "Gender",
                                  "HospAdmTime", "ICULOS", "ICU_unit")]
    static_cols = [c for c in ("Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit")
                   if c in patient_frames[next(iter(patient_frames))].columns]
    cfg_feat = FeatureConfig(clinical_cols=clinical_cols, static_cols=static_cols,
                             min_history_hours=min_history,
                             patient_id_col="patient_id", hour_col="hour")
    feature_matrix = build_feature_matrix(patient_frames, cfg_feat)

    splits = split_by_environment(
        {e: [p for p in patient_frames if p.startswith(e)] for e in [train_env] + test_envs},
        {e: {p: 1 for p in patient_frames if p.startswith(e)} for e in [train_env] + test_envs},
        seed=seed,
    )
    assert_splits_valid(splits)

    model_cfg = cfg.get("model", {})
    model = build_model(model_cfg.get("name", "logistic_regression"), seed=seed,
                        **dict(model_cfg.get("params", {})))
    if checkpoint:
        model = type(model).load(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")

    results_dir = cfg.get("results_dir", "results")
    ensure_dir(results_dir)
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
        feature_cols = [c for c in te.columns if c not in ("patient_id", "hour", "y",
                                                           "censored", "eligible_for_scoring")]
        Xte = te[feature_cols].apply(lambda col: pd.to_numeric(col, errors="coerce")).fillna(0.0).to_numpy(dtype=float)
        yte = te["y"].to_numpy(dtype=int)
        p = model.predict_proba(Xte)[:, 1]
        m = evaluate(yte, p)
        rec = {"run_id": run_id, "train_env": train_env, "test_env": test_env,
               "model": model_cfg.get("name"), "horizon_hours": horizon_hours,
               "seed": seed, "commit": commit, "n_test_patients": len(test_pids),
               "n_test_rows": len(Xte), **m.__dict__,
               "censored_rows": cens_report["n_censored_rows"], "utc": utc_now()}
        rec.update(meta)
        results.append(rec)
        print(f"  {test_env}: AUROC={m.auroc:.3f} AUPRC={m.auprc:.3f} "
              f"Brier={m.brier:.3f} ECE={m.ece:.3f} prev={m.prevalence:.3f} n={m.n}")

    for rec in results:
        save_jsonl(Path(results_dir) / "runs" / f"{run_id}.jsonl", rec)
        save_jsonl(Path(results_dir) / "summary.jsonl", rec)
    print(f"Results saved to {results_dir}/runs/{run_id}.jsonl")
    return results


import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--exploratory", action="store_true")
    args = ap.parse_args()
    run(args.config, checkpoint=args.checkpoint, exploratory=args.exploratory)


if __name__ == "__main__":
    main()
