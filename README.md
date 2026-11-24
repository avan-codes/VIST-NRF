# VIST-NRF

**Visit-Invariant State Tracking under Nonstationary Recording and Follow-up**

Research problem: robust forecasting from irregular, partially observed clinical time series. Primary case study: sepsis forecasting using PhysioNet/CinC Challenge 2019 (two hospital systems).

## Scope (read before using)

This is **research only**. Do not present any model here as a clinically validated diagnostic tool or for patient-care decisions.

## Repository structure

```
VIST-NRF/
├── README.md, pyproject.toml, requirements.txt, .gitignore
├── preregistration.md, gap_log.md
├── logs/PROGRESS.md
├── docs/{ENV_NOTES.md, DATA_NOTES.md, DECISIONS.md}
├── configs/{data, baselines, semisynthetic, method}/
├── data/              # gitignored: raw/, processed/, splits/
├── notebooks/         # 00_bootstrap ... 05_uncertainty_and_failures.ipynb (Colab-ready)
├── src/vist_nrf/      # package: data, features, models, losses, benchmarks, eval, theory_checks, utils
├── scripts/           # audit_data, train, evaluate, run_grid, reproduce, make_tables, make_figures
├── experiments/       # config + README per experiment
├── tests/             # unit tests (synthetic fixtures; no real patient data)
├── theory/            # mathematical statements and counterexamples
├── results/           # gitignored large files; small summaries only
├── figures/
├── reports/{gate_0..gate_3.md}
└── paper/{claims.md, main.tex, references.bib}
```

## Install (local)

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

## Colab workflow

Persistent data/checkpoints/results live on Google Drive; code syncs via Git. See `docs/ENV_NOTES.md` and `notebooks/00_bootstrap.ipynb`.

The Colab bootstrap cell mounts Drive, sets `VIST_NRF_DRIVE/DATA/RESULTS`, clones the repo using `GITHUB_TOKEN` from Colab Secrets (never embedded in a URL), and installs the package in editable mode.

## Non-negotiable rules (summary)

- No fabricated results, citations, dataset facts, URLs, or deadlines. Mark unknown facts `UNVERIFIED` and record how to check them.
- Every run needs a YAML config, seed, saved split, run ID, code commit, and environment metadata.
- Prevent patient overlap, future-data leakage, and target-environment tuning.
- A task is complete only after its tests/acceptance checks have actually run.
- Report negative results and strong baselines honestly. Never tune until the proposed method wins.
- Use small logical Git commits. Never commit raw data, checkpoints, credentials, or tokens.
- Start with simple baselines. Build the full method only if experiments justify it.

## Data

Primary data: PhysioNet/CinC Challenge 2019 (CC BY 4.0). Official: https://physionet.org/content/challenge-2019/1.0.0/

Download requires a PhysioNet credential and is NOT part of the repo. Record counts, schema, license, and access date in `docs/DATA_NOTES.md`.
