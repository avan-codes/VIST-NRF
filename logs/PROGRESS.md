# Progress Log — VIST-NRF

Format per entry: Task ID | change | commit | tests | issues | next step.

## 2026-11-25 — Phase 0 Foundation

- Created repository skeleton: README.md, pyproject.toml, requirements.txt, .gitignore, preregistration.md, gap_log.md, docs/{ENV_NOTES,DATA_NOTES,DECISIONS}.md, logs/PROGRESS.md.
- requirements.txt: numpy, pandas, scipy, scikit-learn, torch, PyYAML, pytest, nbformat (+ optional xgboost).
- .gitignore: excludes data/, *.psv, checkpoints (*.pt/*.pth/.ckpt), credentials (.env), large results, Python/IDE/Colab cache. Commits small text CSVs/manifests only (no blanket *.csv exclusion).
- pyproject.toml: setuptools backend, package `vist_nrf` under src/, dev + xgboost extras.
- NOT a git repo yet — run `git init` + first commit before training work.
- Local env: Python 3.14.8, torch CPU; xgboost not installed locally (optional). Colab bootstrap not yet executed.
- tests/test_run.ipynb already exists (agent connectivity test with matplotlib drawing cell).

## Next step
`git init`, stage foundation files, first commit. Then write src/vist_nrf package utils (seed, io, prereg) and unit tests on synthetic fixtures.
