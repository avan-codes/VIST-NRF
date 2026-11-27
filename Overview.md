# Overview — VIST-NRF (Complete Reference)

Visit-Invariant State Tracking under Nonstationary Recording and Follow-up (VIST-NRF).
**Research only — not a clinically validated diagnostic tool. Do not use for patient-care decisions.**

This document is the single source of truth for: what every folder does, what every file contains, every function signature and its purpose, and how the whole project fits together. Read this before modifying code.

---

## 1. Project in one line

VIST-NRF is a reproducible ML research system that forecasts sepsis onset from irregular, partially-observed ICU time series (PhysioNet/CinC Challenge 2019, two hospital systems A and B), and tests whether a model can learn patient-state information that stays reliable when the observation/follow-up policy changes (Hospital A → Hospital B).

Core research question: *Can a forecasting model learn useful patient-state information that remains reliable when the observation or follow-up policy changes?*

---

## 2. Repository layout (every folder)

```
/home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF/
├── README.md                      # One-page scope, install, rules, Colab workflow
├── Overview.md                    # This file — deep dive: folders, files, functions
├── PROJECT_REPORT.md              # Problem, solution approach, architecture, results
├── pyproject.toml                 # Package config (setuptools, src layout, deps)
├── requirements.txt               # pip dependencies (numpy, pandas, torch, sklearn, PyYAML, ...)
├── .gitignore                     # Never commit: data/, *.psv, checkpoints, .env, tokens
├── preregistration.md             # DRAFT preregistration (not frozen yet)
├── gap_log.md                     # Literature search log (template, entries pending)
├── logs/PROGRESS.md               # Progress log (task id, change, commit, tests, next step)
├── configs/
│   └── baselines/
│       └── e1_a_to_a.yaml         # Experiment E1 config: A→A logistic regression
├── docs/
│   ├── ENV_NOTES.md               # Colab/Drive topology, bootstrap skeleton
│   ├── DATA_NOTES.md              # PhysioNet download commands, acceptance checks
│   └── DECISIONS.md               # Decision log (dated deviations from prereg)
├── data/                          # GITIGNORED — raw/, processed/, splits/ (never commit)
├── notebooks/
│   ├── 00_bootstrap.ipynb         # Colab: Drive mount, git clone/pull, pip install -e
│   ├── 01_data_audit.ipynb        # Colab: PhysioNet data audit
│   ├── 02_baselines.ipynb         # Colab: Baselines
│   ├── 03_semisynthetic.ipynb     # Colab: Controlled policy-shift benchmark
│   ├── 04_method_and_ablations.ipynb  # Colab: VIST-NRF method + ablations
│   ├── 05_uncertainty_and_failures.ipynb   # Colab: Calibration + bootstrap
│   └── main.ipynb                 # End-to-end single notebook (Drive + clone + train + eval)
├── src/vist_nrf/                  # THE Python package (importable as vist_nrf)
│   ├── __init__.py                # Package marker, __version__ = "0.1.0"
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py                # set_seed — one-call Python/NumPy/PyTorch seeder
│   │   ├── io.py                  # environment_metadata, git_commit, save_jsonl, utc_now, sha256_file, ensure_dir
│   │   ├── prereg.py              # enforce_prereg, prereg_hash, PreregistrationError
│   │   └── logging.py             # Scrubbed JSONL run logger
│   ├── data/
│   │   ├── __init__.py            # Re-exports build_targets, split_by_environment, etc.
│   │   ├── physionet2019.py       # .psv parser, audit, hospital-system inference, load_environment
│   │   ├── targets.py             # Event reconstruction, horizon labels, censoring, build_targets
│   │   ├── splits.py              # Stratified 70/15/15 patient-level splits, assert_splits_valid
│   │   └── download.py            # download_physionet2019 (kagglehub), audit_tree, save_manifest, verify_counts
│   ├── features/
│   │   ├── __init__.py
│   │   ├── history.py             # FeatureConfig, build_feature_matrix, compute_patient_features, assert_causal, LOCF/mask/tsl/counts/rolling
│   │   └── timing.py              # Timing-specific feature helpers
│   ├── models/
│   │   ├── __init__.py            # Re-exports BaseModel, LogisticRegressionBaseline, XGBoostBaseline, GRUDModel, build_model, MODEL_REGISTRY
│   │   ├── baselines.py           # Logistic regression, XGBoost (values/timing), GRU-D
│   │   └── vist_nrf.py            # VIST-NRF method: state encoder + policy encoder + head + adversarial env head
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── irm.py                 # IRMv1 penalty
│   │   ├── vrex.py                # V-REx variance-of-risk penalty
│   │   ├── groupdro.py            # GroupDRO worst-environment risk
│   │   └── invariance.py          # InvarianceLoss wrapper + adversarial env penalty
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py             # AUROC, AUPRC, Brier, ECE, sensitivity, PPV, bootstrap, MetricsResult, evaluate
│   │   └── censoring.py           # complete_horizon_mask, censored_count, censoring_report, assert_*
│   ├── benchmarks/
│   │   ├── __init__.py
│   │   └── semisynthetic.py       # OU latent trajectories, policy-dependent observation, oracle_predict
│   └── theory_checks/
│       └── __init__.py            # Scale-ambiguity counterexample + anchor-breaks-ambiguity demo
├── scripts/
│   ├── train.py                   # Main training script (config-driven, end-to-end)
│   ├── evaluate.py                # Load checkpoint + split, evaluate on test env(s)
│   ├── run_grid.py                # Multi-seed runs via temp configs
│   ├── reproduce.py               # Clean re-run from summary.jsonl record
│   ├── make_tables.py             # Regenerate tables from saved JSONL
│   ├── make_figures.py            # Regenerate figures from saved JSONL
│   └── gen_notebooks.py           # Generates the 6 Colab notebooks
├── tests/
│   ├── test_data.py               # 12 unit tests (targets + splits), synthetic fixtures only
│   └── test_features.py           # 7 unit tests (causality + leakage)
├── results/                       # GITIGNORED large files; JSONL results, tables, figures only
├── figures/                       # Generated figures (gitignored large binaries)
├── reports/                       # gate_0..gate_3.md (human gate decisions)
└── paper/                         # claims.md, main.tex, references.bib (writing only)
```

---

## 3. Module-by-module deep dive (every file, every function)

### 3.1 `src/vist_nrf/utils/` — shared infrastructure

**`seed.py`**
- `set_seed(seed)` — Sets Python `random`, NumPy, and PyTorch seeds (CPU + CUDA if available) and enables deterministic algorithms. Call once at the start of every run so results are reproducible.

**`io.py`**
- `environment_metadata()` — Returns a dict with platform, Python version, torch version, CUDA availability/device name, deterministic flag. Recorded with every result so you know exactly what hardware/software produced a number.
- `git_commit(repo_root)` — Returns the short SHA of the current `HEAD`. Stored in every result record so every number is traceable to an exact code version.
- `save_jsonl(path, record)` — Appends one JSON line to a file. All experiment results are stored as JSONL (one line = one run) so they are append-only, easy to aggregate, and never overwritten by accident.
- `utc_now()` — Current UTC ISO-8601 timestamp.
- `sha256_file(path)` — SHA-256 of a file. Used by preregistration to hash `preregistration.md`.
- `ensure_dir(path)` — Creates a directory (and parents) if missing.

**`prereg.py`**
- `prereg_hash(repo_root)` — SHA-256 of committed `preregistration.md`, or `None` if missing.
- `enforce_prereg(repo_root, expected_hash, allow_exploratory, exploratory_dir)` — Refuses confirmatory runs unless the committed preregistration hash matches. Exploratory runs need `--exploratory` and write to a separate directory. Prevents p-hacking / moving goalposts after seeing results.
- `PreregistrationError` — Exception raised when enforcement fails.

**`logging.py`**
- Scrubbed JSONL run logger. Strips secrets/tokens before writing.

---

### 3.2 `src/vist_nrf/data/` — data layer (download, parse, targets, splits)

**`physionet2019.py`** — Parses PhysioNet/CinC Challenge 2019 `.psv` files (pipe-separated, hourly rows).
- `PhysioNetFile` (dataclass) — Structural audit record for one `.psv` file: path, patient_id, hospital_system (A/B/None), n_rows, n_cols, columns, sepsis_positive, n_positive, first_positive_hour, size_bytes. No patient data is leaked — only metadata.
- `parse_psv(path)` — Reads one `.psv` into a DataFrame. Inserts `patient_id` (from filename stem) and `hour` (0-indexed row index) at the front, adds `_source_file`. Sorted by Hour.
- `patient_id_from_filename(path)` — Returns `Path(path).stem` (e.g. `p00001.psv` → `p00001`).
- `infer_hospital_system(path)` — Infers 'A' or 'B' from parent directory name (`seta` → A, `setb` → B). Does **not** assume folder names — returns `None` if not inferable (spec §4.2).
- `audit_psv(path)` — Full structural audit of one file → `PhysioNetFile`.
- `list_psv_files(root)` — Recursively lists all `*.psv` under root, sorted.
- `load_environment(root, env_label)` — Loads all `.psv` files whose inferred hospital system == `env_label`. Returns `{patient_id: DataFrame}`. Raises `FileNotFoundError` if no files match (never silently uses the wrong environment).

**`targets.py`** — Builds sepsis forecasting targets (spec §5).
- `TargetBuildError(ValueError)` — Raised on malformed target construction.
- `validate_label_pattern(label_series)` — Audits one patient's SepsisLabel: returns `first_positive_idx`, `n_rows`, `n_positive`, `already_positive_at_start`, `malformed` (1→0→1 or non-binary), `left_censored`.
- `reconstruct_event_time(patient_id, labels, hour_index)` — Reconstructs operational onset `t_event = t_first_positive + 6 hours`. Returns dict with `patient_id`, `t_first_positive`, `t_event`, `eligible`, `status` ('event'/'no_event'/'left_censored'/'ambiguous'), `audit`.
- `make_horizon_labels(patient_id, n_hours, t_event, horizon_hours, min_history_hours)` — Per-hour prediction rows: `y = 1` when `t < t_event <= t + H`; `y = 0` only if full horizon observed; otherwise censored. Excludes `t >= t_event`. Requires `t >= min_history_hours`. Returns DataFrame: `patient_id, hour, y, censored, eligible_for_scoring`.
- `build_targets(patient_frames, label_col, hour_col, horizon_hours, min_history_hours)` — Loops all patients, calls `reconstruct_event_time` + `make_horizon_labels`. Returns `(targets_long, patient_summary)`.

**`splits.py`** — Patient-level train/val/test splits (spec §6.1).
- `PatientSplit` (dataclass) — `train`, `val`, `test`, `env`, `stratify_ever_positive`, `seed`. Methods: `to_dict()`, `all_patients()`, `is_disjoint()` (train/val/test pairwise disjoint), `covers_all(all_patients)`.
- `split_patients(patient_ids, ever_positive, env, seed, train_frac, val_frac, test_frac)` — Stratified split: train vs temp, then temp → val/test. Stratifies by ever-positive when both classes present. Raises if fractions don't sum to 1.
- `split_by_environment(patients_by_env, ever_positive_by_env, seed)` — Builds one `PatientSplit` per environment. Returns `dict[str, PatientSplit]`.
- `assert_splits_valid(splits)` — Checks within-env disjointness and cross-env uniqueness (no patient in two environments). Raises `AssertionError` on violation.

**`download.py`** — Download + manifest (no patient data committed).
- `DownloadManifest` (dataclass) — `source_url`, `access_date_utc`, `license`, `root`, `n_files_a`, `n_files_b`, `n_files_other`, `total_size_bytes`, `file_audits`. Has `.save(path)` to write JSON.
- `utc_now()` — UTC timestamp.
- `audit_tree(root, source_url)` — Audits the downloaded tree: counts .psv files per inferred env (A/B/other), per-file audits, total size. Expected counts: A=20,336, B=20,000 (from spec).
- `verify_counts(manifest)` — Returns list of problems; empty means counts match spec.
- `save_manifest(root, out_path, source_url)` — Audits + saves + verifies; raises `RuntimeError` if verification fails.
- `download_physionet2019(dataset_slug, dest_root)` — Downloads via `kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")`. Returns local cached path. Requires `pip install kagglehub` + Kaggle credentials.

---

### 3.3 `src/vist_nrf/features/` — causal feature engineering (spec §6.2)

**`history.py`**
- `DEFAULT_CLINICAL_COLS` — 40 canonical PhysioNet variable names (EtCO2, BaseExcess, HCO3, FiO2, pH, ...).
- `STATIC_COLS` — `["Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit"]`.
- `FeatureConfig` (dataclass) — `clinical_cols`, `static_cols`, `label_col="SepsisLabel"`, `hour_col="hour"`, `patient_id_col="patient_id"`, `rolling_windows=(6,24)`, `count_windows=(6,12,24)`, `min_history_hours=6`.
- `compute_patient_features(df, cfg)` — All causal features for one patient: LOCF, mask, time-since-last-observation, counts (6/12/24h), rolling min/max/mean (6/24h), static cols repeated per hour. Purely causal — uses only history ≤ t.
- `build_feature_matrix(patient_frames, cfg)` — Loops patients, computes features, inserts `patient_id` and `hour` columns. Returns long DataFrame: `patient_id, hour, <features...>`. No labels, no future info.
- `assert_causal(features, patient_frames, cfg)` — Leakage test: recomputes features at representative hours (min-history, midpoint, last) from truncated history vs full record; must match exactly. Raises `AssertionError` if any mismatch → blocks model training (spec §6.4).
- `assert_no_future_label(features, targets, patient_id_col, hour_col)` — Forbids feature columns named `sepsislabel` or `patientid`.

Internal helpers (not in `__all__` but used within module):
- `_causal_locf_mask_time_since(df, cols, hour_col)` — LOCF via pandas `ffill` (backward-only), mask = observed indicator, time-since = hour − last observed hour.
- `_causal_counts(df, cols, hour_col, windows)` — Measurement counts over previous W hours using `searchsorted` on observed timestamps (vectorized, O(n log n)).
- `_causal_rolling_stats(df, cols, hour_col, windows)` — Causal rolling min/max/mean over observed values only, previous W hours.

**`timing.py`** — Timing-specific feature helpers (inter-observation intervals, burstiness, etc.).

---

### 3.4 `src/vist_nrf/models/` — model layer (spec §9)

**`baselines.py`**
- `BaseModel` (ABC) — Abstract interface: `fit(X, y, **kwargs)`, `predict_proba(X)` → (n,2) probabilities, `save(path)`, `load(path)`, `name` property. All models share this so evaluation code is model-agnostic.
- `_save_sklearn_model` / `_load_sklearn_model` — joblib wrappers.
- `_proba_2col(scores_1d)` — Converts 1-D positive-class scores to (n, 2) probability matrix `[1-p, p]`.
- `LogisticRegressionBaseline(C=1.0, max_iter=1000, seed=0)` — sklearn `LogisticRegression` on values-only causal features. `name = "logistic_regression"`.
- `XGBoostBaseline(max_depth=6, n_estimators=400, learning_rate=0.05, seed=0, use_timing=False)` — `xgb.XGBClassifier`. `name = "xgboost_values"` or `"xgboost_timing"`. Raises `ImportError` if xgboost not installed.
- `GRUDModel(input_dim, hidden_dim=64, n_layers=1, dropout=0.1, lr=1e-3, epochs=20, batch_size=64, seed=0, device="cpu")` — PyTorch GRU-D: GRU over input → hidden=64 → dropout → Linear → 1 logit → sigmoid. Learnable decay params (`decay_w`, `decay_b`) for missing-value imputation toward data mean. `name = "grud"`.
- `MODEL_REGISTRY` — Dict: `{"logistic_regression": LogisticRegressionBaseline, "xgboost": XGBoostBaseline, "xgboost_values": lambda, "xgboost_timing": lambda, "grud": GRUDModel}`.
- `build_model(model_name, **kwargs)` — Factory. If `"vist_nrf"`, lazily imports `VISTNRFModel` (avoids circular import). Otherwise looks up `MODEL_REGISTRY`.

**`vist_nrf.py`** — VIST-NRF method (spec §9 Stage C). Built only if baselines leave a meaningful gap.
- `_GradientReversal` (torch.autograd.Function) — Forward = identity, backward = negates gradient. DANN-style adversarial environment confusion.
- `_StateEncoder(input_dim, hidden_dim=64, n_layers=1, dropout=0.1)` — GRU over `cat([values, mask, delta])` → last hidden → dropout → `z_state`.
- `_PolicyEncoder(input_dim, hidden_dim=32)` — MLP on `cat([obs_rate, mean_gap])` → `z_policy`. Summarizes mask/timing patterns.
- `_PredictionHead(state_dim, policy_dim, use_policy=False)` — MLP: `in_dim → in_dim//2 → ReLU → Dropout → 1 logit`. Primarily `z_state`; optionally `cat([z_state, z_policy])`.
- `VISTNRFModel(nn.Module, BaseModel)` — Full method:
  - `__init__(input_dim, n_envs=2, hidden_dim=64, policy_dim=32, use_policy=False, use_adversarial=False, invariance_method="erm", invariance_weight=1.0, lr=1e-3, epochs=20, batch_size=64, seed=0, device="cpu")`
  - Components: `state_encoder`, `policy_encoder`, `head`, optional `env_head = Linear(hidden_dim, n_envs)`, `InvarianceLoss`.
  - `_to_sequences(X)` — Reshapes flat feature matrix to `(B, T=1, F)`: values, mask (from NaN), delta (=1 for 1-hour step).
  - `fit(X, y, envs=None)` — Adam optimizer, BCEWithLogitsLoss. Per batch: `z = state_encoder(...)`, `logits = head(z)`, BCE loss. Optional invariance penalty on `(logits, y, env)`. Optional adversarial env loss via gradient reversal. `epochs` iterations.
  - `predict_proba(X)` — Sigmoid of logits → (n, 2) probabilities.
  - `save(path)` / `load(path)` — torch state_dict + config.
  - `name` — `"vist_nrf"` with suffixes: `_policy`, `_adv`, `<invariance_method>`.

---

### 3.5 `src/vist_nrf/losses/` — robustness/invariance losses (spec §9 Stage B)

**`irm.py`**
- IRMv1 penalty: gradient of a dummy classifier (w.r.t. its weight) must be zero within each environment. Encourages predictors optimal across environments.

**`vrex.py`**
- V-REx: variance of per-environment risk. Softer alternative to IRM; encourages equal risk across environments.

**`groupdro.py`**
- GroupDRO: minimizes the maximum per-environment risk (worst-environment risk). Improves worst-environment performance.

**`invariance.py`**
- `InvarianceLoss(method, weight)` — Wrapper dispatching to IRM / V-REx / GroupDRO / ERM.
- Includes spec §9 caution: an adversarial hospital classifier does NOT automatically identify patient state and may remove clinically useful hospital-correlated information. Ablate individually.

---

### 3.6 `src/vist_nrf/eval/` — evaluation layer (spec §10, §11)

**`metrics.py`**
- `auroc(y_true, y_score)` — ROC AUC (sklearn). Returns nan if single class.
- `auprc(y_true, y_score)` — Average precision.
- `brier(y_true, y_prob)` — Brier score.
- `ece(y_true, y_prob, n_bins=10)` — Expected calibration error (equal-width bins).
- `sensitivity_ppv(y_true, y_prob, threshold=0.5)` — Sensitivity, PPV, confusion counts.
- `prevalence(y_true)` — Mean of labels.
- `MetricsResult` (dataclass) — `auroc, auprc, brier, ece, prevalence, n, n_pos, sensitivity, ppv, threshold`.
- `evaluate(y_true, y_prob, threshold=0.5)` — Returns `MetricsResult`.
- `reliability_curve(y_true, y_prob, n_bins=10)` — `(prob_true, prob_pred, bin_counts)` for reliability diagram.
- `patient_bootstrap(y_true, y_score, patient_ids, metric_fn, n_resamples=1000, seed=0)` — Patient-level cluster bootstrap: resample patients, keep all their rows. Returns metric values per resample.
- `bootstrap_ci(values, alpha=0.05)` — Percentile CI from bootstrap values.
- `paired_bootstrap_diff(...)` — Paired patient-level bootstrap for `metric_a − metric_b`.
- `holm_correction(p_values)` — Holm step-down FWER correction.
- `split_conformal_threshold(cal_scores, cal_labels, alpha=0.1)` — Split conformal quantile threshold (secondary §11).
- `conformal_coverage(test_scores, test_labels, threshold)` — Empirical coverage.

**`censoring.py`**
- `complete_horizon_mask(targets)` — Boolean mask of scoring-eligible rows.
- `censored_count(targets)` — Number of censored rows.
- `censored_patients(targets)` — Sorted list of censored patient IDs.
- `assert_no_early_event_prediction(targets)` — No eligible row at/after `t_event`.
- `assert_censoring_not_negative(targets)` — Censored rows must not be labeled `y=0`.
- `censoring_report(targets, patient_summary)` — Dict: `n_censored_rows`, `n_eligible_rows`, `n_censored_patients`, `censored_patients`, `left_censored_patients`, `ambiguous_patients`, `no_event_patients`.

---

### 3.7 `src/vist_nrf/benchmarks/` — semi-synthetic benchmark (spec §8)

**`semisynthetic.py`**
- OU (Ornstein-Uhlenbeck) latent trajectory generator.
- Policy-dependent observation (different hospitals sample differently).
- Outcome generated from latent state **only** (policy must not directly change outcome).
- `oracle_predict` — Prediction using true latent state: the policy-invariant upper bound.
- Config-driven. Used for held-out-policy evidence (E4).

---

### 3.8 `src/vist_nrf/theory_checks/` — mathematical foundations (spec §3.2)

**`__init__.py`**
- Scale-ambiguity counterexample: `lambda(t) = a * exp(beta * s(t))` is scale-ambiguous — rescale `s` and inversely rescale `beta`, same intensity.
- Anchor-breaks-ambiguity demo: anchored measurements break the ambiguity.
- Every theory claim states its assumptions explicitly.

---

### 3.9 `scripts/` — orchestration (spec §13)

**`train.py`**
- `load_config(path)` — Reads YAML config.
- `build_run_id(cfg)` — `f"{run_id}_seed{seed}"`.
- `run(cfg_path, exploratory=False)` — Full end-to-end experiment:
  1. Seed, preregistration enforcement
  2. Commit + environment metadata
  3. Load data (real PhysioNet or synthetic fixture for CI smoke test)
  4. Build targets + censoring acceptance checks
  5. Build causal features + `assert_causal()` leakage test (BLOCKS if failed)
  6. Patient-level splits (70/15/15 per env)
  7. Build model via `build_model()`
  8. Train on source env train split
  9. Evaluate on each test env → `MetricsResult`
  10. Save JSONL results to `results/runs/<run_id>.jsonl` and `results/summary.jsonl`
- `_synthetic_patient(n_hours, seed, ever_pos)` — Synthetic patient frame for smoke tests.
- `main()` — CLI: `python scripts/train.py <config> [--exploratory]`.

**`evaluate.py`** — Load checkpoint + split, evaluate on test env(s). Separates training from evaluation.

**`run_grid.py`** — Multi-seed runs via temp configs (≥5 independent seeds for final configs; same tuning budget across methods).

**`reproduce.py`** — Clean re-run from `summary.jsonl` record at the committed code version.

**`make_tables.py`** — Regenerates tables from saved JSONL (never hand-type experimental numbers).

**`make_figures.py`** — Regenerates figures from saved JSONL.

**`gen_notebooks.py`** — Generates the 6 Colab notebooks.

---

### 3.10 `tests/` — unit tests (synthetic fixtures only, no patient data)

**`test_data.py`** — 12 tests:
- `test_event_reconstruction_basic`, `test_left_censored`, `test_no_event`, `test_malformed`, `test_validate_label_pattern_non_binary` (targets)
- `test_horizon_labels_event_patient`, `test_horizon_labels_non_event_censoring` (horizon labels)
- `test_build_targets_end_to_end` (targets pipeline)
- `test_split_disjoint_and_covers`, `test_split_stratified`, `test_cross_env_disjoint`, `test_split_reproducible` (splits)

**`test_features.py`** — 7 tests (causality + leakage).

All 19 tests must pass before model experiments run.

---

### 3.11 `notebooks/` — Colab-ready notebooks

- `00_bootstrap.ipynb` — Drive mount, git clone/pull with `GITHUB_TOKEN` askpass, `pip install -e`, import verification.
- `01_data_audit.ipynb` — PhysioNet data audit.
- `02_baselines.ipynb` — Baselines.
- `03_semisynthetic.ipynb` — Controlled policy-shift benchmark.
- `04_method_and_ablations.ipynb` — VIST-NRF method + ablations.
- `05_uncertainty_and_failures.ipynb` — Calibration + bootstrap.
- `main.ipynb` — Single end-to-end notebook: Drive + clone + install + kagglehub download + full pipeline → results to Drive.

---

### 3.12 `configs/`, `results/`, `figures/`, `reports/`, `paper/`

- `configs/baselines/e1_a_to_a.yaml` — Experiment E1: A→A, logistic regression, horizon=12h, min_history=6h, C=1.0, max_iter=1000, seed=0, exploratory=true.
- `results/` — GITIGNORED. JSONL runs (`results/runs/*.jsonl`), `summary.jsonl`, `table_comparison.md`, `fig_auroc_by_env.png`.
- `figures/` — Generated figures.
- `reports/gate_0..gate_3.md` — Human gate decisions (Gate 0: gap credible; Gate 1: H1/H2 supported; Gate 2: sharp claim; Gate 3: final claims-to-evidence map).
- `paper/` — Writing only: `claims.md`, `main.tex`, `references.bib`.

---

## 4. End-to-end data flow (how a run works)

```
1. Config YAML (e.g. e1_a_to_a.yaml) → seed, env, horizon, model, params
2. load_environment(raw_root, env) → {patient_id: DataFrame}  [physionet2019.py]
3. build_targets(...) → targets_long + patient_summary        [targets.py]
   - reconstruct_event_time: t_event = t_first_positive + 6h
   - make_horizon_labels: y(t,H), censoring, min_history
   - assert_no_early_event_prediction + assert_censoring_not_negative
4. FeatureConfig → build_feature_matrix → feature_matrix       [features/history.py]
   - LOCF, mask, tsl, counts, rolling stats — all causal
   - assert_causal(feature_matrix, ...) → leakage test (BLOCKS if fail)
5. split_by_environment(...) → splits (70/15/15 per env, stratified) [splits.py]
   - assert_splits_valid() → disjoint + cross-env unique
6. build_model(name) → model                                   [models/]
   - logistic_regression / xgboost / grud / vist_nrf
7. model.fit(Xtr, ytr, envs=env_tr) on train split
8. model.predict_proba(Xte) → p; evaluate(yte, p) → MetricsResult [eval/metrics.py]
   - AUROC, AUPRC, Brier, ECE, prevalence, sensitivity, PPV
9. Save JSONL to Drive: results/runs/<run_id>.jsonl + summary.jsonl
   - Each record: metrics + commit + env metadata + run_id + utc
```

---

## 5. Key design decisions (why)

| Decision | Reason |
|----------|--------|
| Patient-level splits (not row-level) | Same patient's early/late hours must not leak across train/test. Models deploy on new patients. |
| Censoring, not auto-negative | Incomplete negative horizons are censored, never confirmed negative. Prevents label bias. |
| `t_event = t_first_positive + 6h` | PhysioNet label turns positive 6h before actual event. VIST-NRF forecasts operational onset, not the shifted label. |
| `assert_causal()` blocks training | Future-data leakage is the #1 error in clinical TS research. Test must pass before any model trains. |
| Simple baselines first | Spec §9 stopping rule: if simpler method matches full architecture within uncertainty, use simpler. VIST-NRF is Stage C, evidence-driven. |
| Preregistration enforcement | Prevents p-hacking. Confirmatory runs need committed hash match; exploratory goes to separate dir. |
| No data/secrets in git | `.gitignore` excludes `data/`, `*.psv`, checkpoints, `.env`. Raw data lives on Drive (Colab workflow). |
| JSONL results + commit + env metadata | Every number traceable to exact code version + hardware. Regenerable tables/figures. |

---

## 6. How to run

**Local (Kali Linux):**
```bash
cd /home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF
python3 -m pip install -e . --break-system-packages
python3 -m pip install -e ".[dev]" --break-system-packages
python3 -m pytest tests/ -q          # expect 19 passed
python3 scripts/train.py configs/baselines/e1_a_to_a.yaml --exploratory
```

**Colab (T4 GPU recommended):**
1. Open `notebooks/main.ipynb` (or `00_bootstrap.ipynb` → `02_baselines.ipynb`).
2. Add `GITHUB_TOKEN` to Colab Secrets (left sidebar → key icon).
3. Run Cell 1 (Drive mount + clone + install).
4. Run Cell 2 (kagglehub download → `VIST_NRF_DATA_RAW`).
5. Run Cell 3 (full pipeline → results saved to Drive `results/runs/`).
6. `<OWNER>` in clone URL = your GitHub username.

**Experiments:**
- E1: A→A (within-env), logistic regression — `configs/baselines/e1_a_to_a.yaml`
- E2: A→B and B→A (cross-site) — copy config, change `train_env`/`test_envs`
- E3: random thinning (sanity)
- E4: semi-synthetic held-out policies
- E5: ICU-unit transfer
- E6: remove a lab block at test time

---

## 7. Current status

- ✅ Repo initialized, branch `main`, origin `https://github.com/avan-codes/VIST-NRF.git`
- ✅ `vist_nrf/data` subpackage committed (was previously git-ignored by `data/` rule; fixed with `!src/vist_nrf/data/`)
- ✅ `download_physionet2019` added to `data/download.py` (kagglehub)
- ✅ `notebooks/main.ipynb` committed — end-to-end Colab notebook
- ✅ 19 unit tests pass (`tests/test_data.py` + `tests/test_features.py`)
- ✅ Smoke test runs end-to-end (synthetic fixture), AUROC=1.000 on synthetic data
- 🔄 Real PhysioNet data downloaded via kagglehub (~45MB, 20,336 A + 20,000 B patients)
- ⏳ Real-data E1 training, E2 cross-site, Gate 1, semi-synthetic, VIST-NRF Stage C — pending

**Research only — not a clinically validated diagnostic tool.**
