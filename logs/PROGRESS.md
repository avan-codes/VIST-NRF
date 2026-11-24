# Progress Log — VIST-NRF

Format per entry: Task ID | change | commit | tests | issues | next step.

## 2026-11-25 — Phase 0 Foundation

- Created repository skeleton: README.md, pyproject.toml, requirements.txt, .gitignore, preregistration.md, gap_log.md, docs/{ENV_NOTES,DATA_NOTES,DECISIONS}.md, logs/PROGRESS.md.
- requirements.txt: numpy, pandas, scipy, scikit-learn, torch, PyYAML, pytest, nbformat (+ optional xgboost).
- .gitignore: excludes data/, *.psv, checkpoints (*.pt/*.pth/.ckpt), credentials (.env), large results, Python/IDE/Colab cache. Commits small text CSVs/manifests only (no blanket *.csv exclusion).
- pyproject.toml: setuptools backend, package `vist_nrf` under src/, dev + xgboost extras.
- NOT a git repo yet — run `git init` + first commit before training work.
- Git initialized (branch main). First commit 37f5f3a "Phase 0 foundation: repo skeleton, data layer, unit tests" (18 files, 888 insertions).
- Local env: Python 3.14.8, torch CPU (2.15.0+cpu); xgboost not installed locally (optional). Colab bootstrap not yet executed. Package installed editable (`pip install -e . --break-system-packages`).
- tests/test_data.py: 12 tests pass (targets event/left-censored/no-event/malformed, horizon labels + censoring, split disjoint/coverage/stratification/reproducibility). tests/test_run.ipynb already exists (agent connectivity test with matplotlib drawing cell).

## Next step
Phase 1 data pipeline: physionet2019 parser is written but not yet tested against real .psv files (no data downloaded). Next: features module (causal LOCF, mask, time-since-observation, rolling stats) and leakage-prevention unit tests (no future leakage, truncated-history stability). Then Stage A baselines.
