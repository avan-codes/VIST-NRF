# Progress Log — VIST-NRF

Format per entry: Task ID | change | commit | tests | issues | next step.

## 2026-11-25 — Phase 0 Foundation (commit 37f5f3a, then a4d9b7a)

- Created repository skeleton: README.md (research-only scope warning), pyproject.toml, requirements.txt, .gitignore, preregistration.md (draft), gap_log.md, docs/{ENV_NOTES,DATA_NOTES,DECISIONS}.md, logs/PROGRESS.md.
- requirements.txt: numpy, pandas, scipy, scikit-learn, torch, PyYAML, pytest, nbformat (+ optional xgboost).
- .gitignore: excludes data/, *.psv, checkpoints (*.pt/*.pth/.ckpt), credentials (.env), large results/png/pt, Python/IDE/Colab cache. Commits small text manifests only.
- pyproject.toml: setuptools backend, package `vist_nrf` under src/, dev + xgboost extras.
- Git initialized (branch main). Commit 37f5f3a "Phase 0 foundation" (18 files). Commit a4d9b7a (PROGRESS.md update).
- Local env: Python 3.14.8, torch CPU 2.15.0+cpu; xgboost not installed locally (optional). Package installed editable (`pip install -e . --break-system-packages`). Colab bootstrap not yet executed.

## Phase 1 — Data pipeline, features, models, eval, training script

- src/vist_nrf/utils/: seed.py (one-call Python/NumPy/PyTorch seeder), io.py (JSONL/JSON, SHA-256, env metadata incl. torch/CUDA/commit), prereg.py (hash enforcement; exploratory bypass), logging.py (scrubbed JSONL run logger).
- src/vist_nrf/data/: physionet2019.py (.psv parser, audit, env inference — no folder-name assumptions), targets.py (event reconstruction t_event=t_first_positive+6h, horizon labels, censoring, malformed/1→0→1 detection), splits.py (stratified 70/15/15 patient-level, disjoint+coverage checks), download.py (manifest + count verification A=20336/B=20000).
- src/vist_nrf/features/: history.py (causal LOCF/mask/time-since/counts/rolling min-max-mean, vectorized with searchsorted), timing.py, __init__.py; leakage tests assert_causal + assert_no_future_label.
- src/vist_nrf/models/: baselines.py (logistic regression, XGBoost values/timing, GRU-D), vist_nrf.py (VIST-NRF state encoder + policy encoder + prediction head + gradient-reversal adversarial env head + invariance), build_model() registry. torch.load uses weights_only=False (own checkpoints).
- src/vist_nrf/losses/: irm.py, vrex.py, groupdro.py, invariance.py (InvarianceLoss wrapper + adversarial env penalty, with spec §9 cautions).
- src/vist_nrf/eval/: metrics.py (AUROC/AUPRC/Brier/ECE/sensitivity/PPV/prevalence, patient-level cluster bootstrap, paired bootstrap, Holm correction, reliability curve, split conformal), censoring.py (complete-horizon rule, censored≠negative, reports).
- src/vist_nrf/benchmarks/: semisynthetic.py (OU latent trajectories, policy-dependent observation, outcome from latent state only, oracle_predict).
- src/vist_nrf/theory_checks/__init__.py: intensity-only scale-ambiguity counterexample + anchor-breaks-ambiguity demo (§3.2).
- scripts/: train.py (config-driven run: seed, split, targets, features, causality test, model fit, evaluate, JSONL results + env metadata + commit), evaluate.py, run_grid.py (multi-seed via temp configs), reproduce.py (clean re-run from summary.jsonl), make_tables.py, make_figures.py, gen_notebooks.py.
- notebooks/: 00_bootstrap … 05_uncertainty_and_failures.ipynb (Colab-ready, valid nbformat).
- tests/: test_data.py (12 tests) + test_features.py (7 tests) = 19 tests, ALL PASS. Causality/leakage test blocks model experiments if it fails (spec §6.4).
- Smoke test: scripts/train.py runs end-to-end on synthetic fixture; results saved to results/runs/<run_id>.jsonl and results/summary.jsonl; table + figures regenerated (results/table_comparison.md, results/fig_auroc_by_env.png).

## Issues
- Real PhysioNet data DOWNLOADED via kagglehub to /home/pranjal/.cache/kagglehub/datasets/farjanayesmin/the-physionet-challenge-2019-dataset/versions/1 (20,336 A + 20,000 B patients, matches spec counts). raw_root set in configs/baselines/e1_a_to_a.yaml. Real-data path wired in train.py via load_environment; causality test + training running on real data (background).
- xgboost not installed locally (optional dep); XGBoostBaseline raises clear ImportError if missing.
- Colab bootstrap (00_bootstrap.ipynb) documented but not executed (needs GITHUB_TOKEN in Colab Secrets + OWNER set).

## Next step
- Real-data E1 training run in progress (background, notify_on_complete). Review results in results/runs/ and results/summary.jsonl when it finishes.
- Run E2 cross-site (A→B, B→A) with 5 seeds, then Gate 1 decision on H1/H2.
- If supported, build semi-synthetic held-out-policy evidence (E4) and VIST-NRF Stage C method; else pivot/report negative result per §16.
