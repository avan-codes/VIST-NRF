# VIST-NRF — Main Execution Guide (End-to-End)

Ye document VIST-NRF project ka complete end-to-end execution guide hai. Local setup se lekar Colab (T4 GPU) par training tak, sab kuch yahan hai.

---

## 1. Project Kya Hai

**VIST-NRF** = Visit-Invariant State Tracking under Nonstationary Recording and Follow-up

Irregular clinical time series (ICU data) se sepsis forecast karna. Do hospital systems (A aur B) par PhysioNet/CinC Challenge 2019 data use hota hai.

**Research Question:** Kya model useful patient-state seekh sakta hai jo observation policy change hone par bhi reliable rahe?

**Research Only** — clinically validated diagnostic tool nahi hai, patient-care decisions ke liye use mat karna.

---

## 2. Repository Structure (Abhi ban chuja hai)

```
/home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF/
├── Overview.md              # In-depth project rationale
├── README.md                # Project scope & rules
├── pyproject.toml           # Package config
├── requirements.txt         # Python dependencies
├── .gitignore               # Never commit patient data/checkpoints/tokens
├── preregistration.md       # Draft preregistration (not frozen)
├── gap_log.md               # Literature search log
├── logs/PROGRESS.md         # Progress log (update after each task)
├── docs/
│   ├── ENV_NOTES.md         # Colab/Drive topology
│   ├── DATA_NOTES.md        # Data download & audit notes
│   └── DECISIONS.md         # Decision log (dated deviations)
├── configs/
│   └── baselines/
│       └── e1_a_to_a.yaml   # Experiment config (E1: A→A logistic regression)
├── notebooks/
│   ├── 00_bootstrap.ipynb   # Colab: Drive mount + repo clone + install
│   ├── 01_data_audit.ipynb  # Colab: PhysioNet data audit
│   ├── 02_baselines.ipynb   # Colab: Baselines
│   ├── 03_semisynthetic.ipynb  # Colab: Controlled policy-shift benchmark
│   ├── 04_method_and_ablations.ipynb  # Colab: VIST-NRF method
│   └── 05_uncertainty_and_failures.ipynb  # Colab: Calibration + bootstrap
├── src/vist_nrf/            # Python package
│   ├── utils/               # seed, io, prereg enforcement, logging
│   ├── data/                # physionet2019 parser, targets, splits, download manifest
│   ├── features/            # Causal LOCF, mask, time-since, counts, rolling stats
│   ├── models/              # Logistic regression, XGBoost, GRU-D, VIST-NRF
│   ├── losses/              # IRM, V-REx, GroupDRO, invariance
│   ├── eval/                # Metrics, bootstrap, censoring, conformal
│   ├── benchmarks/          # Semi-synthetic generator
│   └── theory_checks/       # Scale-ambiguity counterexample
├── scripts/
│   ├── train.py             # Main training script (config-driven)
│   ├── evaluate.py          # Evaluation script
│   ├── run_grid.py          # Multi-seed runs
│   ├── reproduce.py         # Clean re-run from summary.jsonl
│   ├── make_tables.py       # Regenerate tables from saved JSONL
│   ├── make_figures.py      # Regenerate figures from saved JSONL
│   └── gen_notebooks.py     # Generate Colab notebooks
├── tests/
│   ├── test_data.py         # 12 tests (targets + splits)
│   └── test_features.py     # 7 tests (causality + leakage)
├── results/                 # gitignored: JSONL results, tables, figures
├── figures/
├── reports/                 # gate_0..gate_3.md
└── paper/                   # claims.md, main.tex, references.bib
```

**Status:** 53 files committed to `main` (commit `f578f68`), 19 unit tests PASS.

---

## 3. Prerequisites

### Local (Kali Linux)
- Python 3.14.8
- torch 2.15.0+cpu (CUDA nahi hai local par)
- xgboost optional (Colab par install hoga)
- Package editable install: `pip install -e . --break-system-packages`

### Colab (T4 GPU recommended)
- Google Colab account
- Google Drive mount karne ke liye access
- Colab Secrets mein `GITHUB_TOKEN` (GitHub personal access token, repo access ke liye)
- GitHub repo owner name (`<OWNER>` apna GitHub username)

---

## 4. Step-by-Step Execution

### Phase A: Local Setup (Ek baar karna hai)

```bash
cd /home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF

# Package install (editable)
python3 -m pip install -e . --break-system-packages
python3 -m pip install -e ".[dev]" --break-system-packages

# Tests run karein (sab pass hone chahiye)
python3 -m pytest tests/ -q

# Git status check
git status --short
git log --oneline -3
```

**Expected:** `19 passed` in tests, working tree clean.

---

### Phase B: Colab Setup (T4 GPU)

Colab browser mein kholein: `https://colab.research.google.com`

**Runtime settings:** Runtime → Change runtime type → **T4 GPU** → Save

#### Cell 1 — Environment setup (Drive mount + repo clone + install)

```python
import os, stat, subprocess
from pathlib import Path
from google.colab import drive, userdata

drive.mount('/content/drive')
os.environ['VIST_NRF_DRIVE'] = '/content/drive/MyDrive/VIST-NRF'
os.environ['VIST_NRF_DATA'] = str(Path(os.environ['VIST_NRF_DRIVE']) / 'data')
os.environ['VIST_NRF_RESULTS'] = str(Path(os.environ['VIST_NRF_DRIVE']) / 'results')
for key in ('VIST_NRF_DATA', 'VIST_NRF_RESULTS'):
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

repo_url = 'https://github.com/<OWNER>/VIST-NRF.git'  # <-- apna GitHub username daalna
repo_dir = '/content/VIST-NRF'
token = userdata.get('GITHUB_TOKEN')
if not token:
    raise RuntimeError('Colab Secrets mein GITHUB_TOKEN add karein (left sidebar → key icon).')

askpass = Path('/tmp/vist_nrf_askpass.sh')
askpass.write_text('#!/bin/sh\ncase "$1" in\n*Username*) echo x-access-token ;;\n*Password*) printf "%s" "$GITHUB_TOKEN" ;;\nesac\n')
askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
env = os.environ.copy()
env.update(GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0', GITHUB_TOKEN=token)
try:
    if not Path(repo_dir).exists():
        subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, env=env)
    else:
        subprocess.run(['git', '-C', repo_dir, 'pull', '--ff-only'], check=True, env=env)
finally:
    askpass.unlink(missing_ok=True)
    env.pop('GITHUB_TOKEN', None)
    del token

subprocess.run(['python', '-m', 'pip', 'install', '-q', '-e', repo_dir], check=True)
import torch, vist_nrf
commit = subprocess.check_output(['git', '-C', repo_dir, 'rev-parse', '--short', 'HEAD'], text=True).strip()
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None', '| commit:', commit)
```

> **Important:** `GITHUB_TOKEN` ko Colab Secrets mein add karein bina jise clone fail hoga. `<OWNER>` apna GitHub username hai.

#### Cell 2 — Data download (kagglehub se)

```python
import kagglehub, os
path = kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")
print('Dataset path:', path)
os.environ['VIST_NRF_DATA_RAW'] = path
```

Yeh ~45MB download karega. Local par pehle hi download ho chuka hai:
`/home/pranjal/.cache/kagglehub/datasets/farjanayesmin/the-physionet-challenge-2019-dataset/versions/1`

Counts verify: Env A = 20,336 patients, Env B = 20,000 patients (spec se match karta hai).

---

### Phase C: Run Experiments

#### Experiment E1 — A → A (Within-environment, logistic regression)

Config: `configs/baselines/e1_a_to_a.yaml` (pehle se set hai `raw_root` par)

Colab Cell 3 — Train:

```python
import sys, os, json
from pathlib import Path
sys.path.insert(0, '/content/VIST-NRF')

import numpy as np
import pandas as pd
import yaml

from vist_nrf.utils.seed import set_seed
from vist_nrf.utils.io import environment_metadata, git_commit, save_jsonl, utc_now
from vist_nrf.utils.prereg import enforce_prereg, prereg_hash, PreregistrationError
from vist_nrf.data.physionet2019 import load_environment
from vist_nrf.data.splits import split_by_environment, assert_splits_valid
from vist_nrf.data.targets import build_targets
from vist_nrf.features.history import FeatureConfig, build_feature_matrix, assert_causal
from vist_nrf.eval.censoring import censoring_report, assert_no_early_event_prediction, assert_censoring_not_negative
from vist_nrf.eval.metrics import evaluate
from vist_nrf.models import build_model

set_seed(0)
repo_root = Path('/content/VIST-NRF')
commit = git_commit(repo_root)
meta = environment_metadata()
meta.update({'commit': commit, 'utc': utc_now()})

# Config load
cfg_path = '/content/VIST-NRF/configs/baselines/e1_a_to_a.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# Preregistration (exploratory mode since not frozen)
enforce_prereg(repo_root, allow_exploratory=True)

raw_root = os.environ.get('VIST_NRF_DATA_RAW', cfg['dataset']['raw_root'])
train_env = cfg.get('train_env', 'A')
test_envs = cfg.get('test_envs', ['A'])
horizon_hours = int(cfg.get('horizon_hours', 12))
min_history = int(cfg.get('min_history_hours', 6))

print(f"Loading real PhysioNet data from {raw_root}")
patient_frames = {}
for env in [train_env] + test_envs:
    env_frames = load_environment(raw_root, env)
    patient_frames.update(env_frames)
    print(f"  env {env}: {len(env_frames)} patients")

# Targets
targets_long, patient_summary = build_targets(
    patient_frames, label_col="SepsisLabel",
    horizon_hours=horizon_hours, min_history_hours=min_history,
)
assert_no_early_event_prediction(targets_long)
assert_censoring_not_negative(targets_long)
cens_report = censoring_report(targets_long, patient_summary)
print(f"Censoring: {cens_report['n_censored_rows']} rows censored, {cens_report['n_eligible_rows']} eligible")

# Features
clinical_cols = [c for c in patient_frames[next(iter(patient_frames))].columns
                 if c not in ("SepsisLabel", "hour", "patient_id", "Age", "Gender",
                              "HospAdmTime", "ICULOS", "ICU_unit", "_source_file")]
static_cols = [c for c in ("Age", "Gender", "HospAdmTime", "ICULOS", "ICU_unit")
               if c in patient_frames[next(iter(patient_frames))].columns]
cfg_feat = FeatureConfig(clinical_cols=clinical_cols, static_cols=static_cols,
                         min_history_hours=min_history,
                         patient_id_col="patient_id", hour_col="hour")

feature_matrix = build_feature_matrix(patient_frames, cfg_feat)
assert_causal(feature_matrix, patient_frames, cfg_feat)
print("Causality / leakage test: PASSED")

# Splits (patient-level, 70/15/15)
splits = split_by_environment(
    {e: [p for p in patient_frames if p.startswith(e)] for e in [train_env] + test_envs},
    {e: {p: 1 for p in patient_frames if p.startswith(e)} for e in [train_env] + test_envs},
    seed=0,
)
assert_splits_valid(splits)
print(f"Splits: {{e: (len(s.train), len(s.val), len(s.test)) for e, s in splits.items()}}")

# Model
model_cfg = cfg.get("model", {})
model = build_model(model_cfg.get("name", "logistic_regression"), seed=0,
                    **dict(model_cfg.get("params", {})))

# Train on source env train split
train_pids = splits[train_env].train
tr = targets_long.merge(
    feature_matrix[["patient_id", "hour"] + [c for c in feature_matrix.columns
                                               if c not in ("patient_id", "hour")]],
    on=["patient_id", "hour"], how="inner",
)
tr = tr[tr["eligible_for_scoring"]]
feature_cols = [c for c in tr.columns if c not in ("patient_id", "hour", "y", "censored", "eligible_for_scoring")]
Xtr = tr[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
ytr = tr["y"].to_numpy(dtype=int)
env_tr = np.zeros(len(ytr), dtype=int)

print(f"Training {model_cfg.get('name')} on {len(Xtr)} rows from {len(train_pids)} patients (env {train_env})")
model.fit(Xtr, ytr, envs=env_tr)

# Evaluate on test env(s)
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
    p = model.predict_proba(Xte)[:, 1]
    m = evaluate(yte, p)
    print(f"  {test_env}: AUROC={m.auroc:.3f} AUPRC={m.auprc:.3f} Brier={m.brier:.3f} ECE={m.ece:.3f} prev={m.prevalence:.3f} n={m.n}")
    
    rec = {"train_env": train_env, "test_env": test_env, "model": model_cfg.get("name"),
           "horizon_hours": horizon_hours, "seed": 0, "commit": commit,
           "n_test_patients": len(test_pids), "n_test_rows": len(Xte), **m.__dict__,
           "censored_rows": cens_report["n_censored_rows"]}
    rec.update(meta)
    results.append(rec)

# Save to Drive
results_dir = Path(os.environ['VIST_NRF_RESULTS']) / "runs"
results_dir.mkdir(parents=True, exist_ok=True)
for rec in results:
    with open(results_dir / f"e1_lr_a_to_a_seed0.jsonl", "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
print("Results saved to Drive:", results_dir)
```

**Output dekhna:** AUROC, AUPRC, Brier, ECE, prevalence — sab save ho jayega JSONL mein.

#### Experiment E2 — A → B aur B → A (Cross-site transfer)

Config copy karke `test_envs` aur `train_env` change karein:

```yaml
# configs/baselines/e2_a_to_b.yaml
run_id: e2_lr_a_to_b
dataset:
  raw_root: <kagglehub path>
train_env: A
test_envs: [B]
horizon_hours: 12
min_history_hours: 6
model:
  name: logistic_regression
  params:
    C: 1.0
    max_iter: 1000
seed: 0
```

Do cells run karein: ek `train_env: A, test_envs: [B]` ke liye, ek `train_env: B, test_envs: [A]` ke liye.

---

### Phase D: Results Review

```python
import json
from pathlib import Path

results_dir = Path(os.environ['VIST_NRF_RESULTS']) / "runs"
for f in sorted(results_dir.glob("*.jsonl")):
    print(f"=== {f.name} ===")
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            print(f"  {r['train_env']}→{r['test_env']} | {r['model']} | AUROC={r['auroc']:.3f} | AUPRC={r['auprc']:.3f} | Brier={r['brier']:.3f} | ECE={r['ece']:.3f} | prev={r['prevalence']:.3f} | n={r['n']}")
```

Tables regenerate karna:
```bash
python3 scripts/make_tables.py --summary results/summary.jsonl --out results/table_comparison.md
python3 scripts/make_figures.py --summary results/summary.jsonl --outdir results
```

---

### Phase E: Gate Decisions

| Gate | Kya decide karna hai | Document |
|------|---------------------|----------|
| Gate 0 | Gap credible hai? Real-data assumptions defensible hain? | `reports/gate_0.md` |
| Gate 1 | H1/H2 supported hain? (timing informative? cross-site degradation hai?) | `reports/gate_1.md` |
| Gate 2 | Ek sharp claim convincing evidence ke saath? | `reports/gate_2.md` |
| Gate 3 | Final claims-to-evidence map, reproduction clean runtime par | `reports/gate_3.md` |

**Rule:** Human researcher approve kare bina agle phase par mat jao. Preregistration deviate hona chahiye toh `docs/DECISIONS.md` mein date + reason likho.

---

### Phase F: Multi-Seed Run (Confirmatory)

Preregistration freeze hone ke baad:

```bash
python3 scripts/run_grid.py configs/baselines/e1_a_to_a.yaml --seeds 0 1 2 3 4
```

Ye 5 seeds run karega aur `results/runs/` mein JSONL append karega. Reproduce ke liye:

```bash
python3 scripts/reproduce.py results/summary.jsonl e1_lr_a_to_a_seed0
```

---

## 5. Key Rules (Non-Negotiable)

1. **No fabricated results** — sab kuch actual code se run hona chahiye. Unknown facts ko `UNVERIFIED` mark karo.
2. **Every run needs:** YAML config, seed, saved split, run ID, code commit, environment metadata.
3. **No patient overlap** across splits. No future-data leakage. No target-environment tuning.
4. **Tests must pass** before model experiments. `assert_causal()` blocks if leakage found.
5. **Report negative results honestly.** Simple baselines first. Complex VIST-NRF method sirf agar baselines justify karein.
6. **No secrets in git.** `.gitignore` exclude karta hai `data/`, `*.psv`, `*.pt`, `.env`, large results.
7. **Small logical commits.** Never commit raw data, checkpoints, credentials.

---

## 6. Troubleshooting

| Problem | Solution |
|---------|----------|
| `GITHUB_TOKEN` not found | Colab Secrets mein add karein (left sidebar → key icon → `GITHUB_TOKEN`) |
| `raw_root` path wrong | Kagglehub path check karein: `kagglehub.dataset_download(...)` se mila path use karein |
| xgboost import error | `pip install xgboost>=2.0` Colab cell mein, ya `logistic_regression` use karein |
| torch.load weights_only error | Pehle se fix hai (`weights_only=False`), latest code pull karein |
| Training slow on local CPU | Colab T4 GPU use karein (yehi recommended hai) |
| Causality test fails | Feature code mein future leakage hai — fix karna zaroori hai before model training |

---

## 7. Current Status (Abhi kya hai)

- ✅ Repo initialized, 53 files, 3 commits (branch `main`)
- ✅ 19 unit tests pass (`tests/test_data.py` + `tests/test_features.py`)
- ✅ Real PhysioNet data downloaded via kagglehub (20,336 A + 20,000 B)
- ✅ `load_environment` wired in `train.py`
- ✅ Colab notebooks ready (`00_bootstrap` … `05_uncertainty_and_failures`)
- ✅ `Overview.md` — in-depth design document
- ⏳ Real-data E1 training running on local CPU (~20+ min, faster on Colab T4)
- ⏳ E2 cross-site, Gate 1, semi-synthetic, VIST-NRF method — pending

---

## 8. Quick Command Reference

```bash
# Local test
python3 -m pytest tests/ -q

# Local train (smoke/synthetic)
python3 scripts/train.py configs/baselines/e1_a_to_a.yaml --exploratory

# Multi-seed grid
python3 scripts/run_grid.py configs/baselines/e1_a_to_a.yaml --seeds 0 1 2 3 4

# Regenerate tables
python3 scripts/make_tables.py --summary results/summary.jsonl --out results/table_comparison.md

# Regenerate figures
python3 scripts/make_figures.py --summary results/summary.jsonl --outdir results

# Reproduce a run
python3 scripts/reproduce.py results/summary.jsonl e1_lr_a_to_a_seed0
```

---

**Colab par shuru karne ke liye:** Cell 1 se shuru karein, `GITHUB_TOKEN` add karein, `<OWNER>` replace karein, phir Cell 2 aur 3 run karein. Koi error aaye toh message copy karke mujhe dikhayein, main fix kar dunga.
