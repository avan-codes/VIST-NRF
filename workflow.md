# WORKFLOW.md — VIST-NRF: Karna kaise kya hai (Step-by-Step)

Yeh guide batati hai ki shuru se end tak kya karna hai. Hinglish mein likha hai kyunki yeh padhne mein aasaan hai.

---

## 0. Pehle se check kar lo (Prerequisites)

- Google Colab account (T4 GPU recommended)
- Google Drive access (results aur data yahan save honge)
- GitHub Personal Access Token (PAT) — `repo` scope wala
- Kaggle account (PhysioNet dataset download ke liye)

---

## Step 1: Colab Secrets setup (Ek baar karna hai)

Colab notebook kholo → Left sidebar → Key icon (Secrets) → Add new secret:

| Name | Value |
|------|-------|
| `GITHUB_TOKEN` | Apna GitHub PAT (pehle se generate rakho: GitHub Settings → Developer settings → Personal access tokens → Fine-grained, repo read/write) |

> Bina `GITHUB_TOKEN` ke repo clone nahi hoga.

---

## Step 2: Notebook chalana (Colab)

Notebook: `notebooks/main.ipynb` (repo mein committed hai)

### Cell 1 — Drive mount + Repo clone + Install

```python
# Drive mount hoga
# /content/drive/MyDrive/VIST-NRF banega
# ENV vars set honge: VIST_NRF_DRIVE, VIST_NRF_DATA, VIST_NRF_RESULTS
# Repo clone hoga /content/VIST-NRF mein (GITHUB_TOKEN se)
# pip install -e . hoga
# torch, vist_nrf import verify hoga
```

Jab run karo:
- Agar `/content/VIST-NRF` pehle se hai → `git pull --ff-only` hoga (latest code milega)
- Agar nahi hai → fresh clone hoga
- Token askpass script ke through use hoga (token koi log mein nahi likha jata)

### Cell 2 — Data download (KaggleHub)

```python
import kagglehub, os
path = kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")
print('Dataset path:', path)
os.environ['VIST_NRF_DATA_RAW'] = path
```

- ~45MB download hoga (KaggleHub cache mein save rehta hai)
- Path `VIST_NRF_DATA_RAW` env variable mein save hoga, taaki agla cell use kar sake
- Kaggle credentials chahiye (pehle se login ho ya `~/.kaggle/kaggle.json`)

### Cell 3 — Full pipeline (Train + Evaluate + Save)

Ye sab kuch karta hai ek saath:

1. **sys.path clean** → `/content/VIST-NRF/src` front par
2. **vist_nrf import verify** → `print("vist_nrf imported from: ...")`
3. **Imports:** seed, io, prereg, data, features, eval, models
4. **Config load:** `configs/baselines/e1_a_to_a.yaml` (E1: A→A logistic regression)
5. **Preregistration:** `enforce_prereg(allow_exploratory=True)` — exploratory mode kyunki frozen nahi hai
6. **Data load:** `load_environment(raw_root, env)` — env A (aur test envs) ke liye patients load
7. **Targets:** `build_targets(...)` — horizon=12h, min_history=6h
8. **Censoring checks:** `assert_no_early_event_prediction`, `assert_censoring_not_negative`, `censoring_report`
9. **Features:** `FeatureConfig` + `build_feature_matrix` + `assert_causal()` (leakage test — PASS hona zaroori hai)
10. **Splits:** `split_by_environment` — 70/15/15 patient-level
11. **Model:** `build_model("logistic_regression", seed=0, C=1.0, max_iter=1000)`
12. **Train:** env A ke train split par
13. **Evaluate:** test env(s) par AUROC, AUPRC, Brier, ECE, prevalence
14. **Save:** Drive par `results/runs/e1_lr_a_to_a_seed0.jsonl`

Output example:
```
Loading real PhysioNet data from /root/.cache/kagglehub/...
  env A: 20336 patients
Censoring: X rows censored, Y eligible
Causality / leakage test: PASSED
Splits: {'A': (14235, 3050, 3051)}
Training logistic_regression on ... rows from ... patients (env A)
  A: AUROC=0.xxx AUPRC=0.xxx Brier=0.xxx ECE=0.xxx prev=0.xxx n=...
Results saved to Drive: /content/drive/MyDrive/VIST-NRF/results/runs
```

---

## Step 3: Local par chalana (Agar Colab nahi chahiye)

```bash
cd /home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF

# Package install
python3 -m pip install -e . --break-system-packages
python3 -m pip install -e ".[dev]" --break-system-packages

# Tests (zaroori hai pehle)
python3 -m pytest tests/ -q
# Expected: 19 passed

# Train run
python3 scripts/train.py configs/baselines/e1_a_to_a.yaml --exploratory
```

Result `results/runs/` aur `results/summary.jsonl` mein save hoga.

---

## Step 4: Results dekhna

### Colab (Drive se)

```python
import json
from pathlib Path
import os

results_dir = Path(os.environ['VIST_NRF_RESULTS']) / "runs"
for f in sorted(results_dir.glob("*.jsonl")):
    print(f"=== {f.name} ===")
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            print(f"  {r['train_env']}→{r['test_env']} | {r['model']} | AUROC={r['auroc']:.3f} | AUPRC={r['auprc']:.3f} | Brier={r['brier']:.3f} | ECE={r['ece']:.3f} | prev={r['prevalence']:.3f} | n={r['n']}")
```

### Local

```bash
cat results/summary.jsonl | python3 -m json.tool  # ya ek-ek line padho
```

Table regenerate karna:
```bash
python3 scripts/make_tables.py --summary results/summary.jsonl --out results/table_comparison.md
python3 scripts/make_figures.py --summary results/summary.jsonl --outdir results
```

---

## Step 5: Cross-site experiment (A→B, B→A)

Agar E1 A→A chal gaya, toh E2 ke liye:

1. `configs/baselines/` mein naya banao: `e2_a_to_b.yaml`
   ```yaml
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

2. `python3 scripts/train.py configs/baselines/e2_a_to_b.yaml --exploratory`

3. Isi tarah `e2_lr_b_to_a.yaml` mein `train_env: B`, `test_envs: [A]`

4. Compare karo: A→A AUROC vs A→B AUROC — agar gira toh cross-site degradation confirm (H2 supported).

---

## Troubleshooting (Aam problems)

| Problem | Solution |
|---------|----------|
| `GITHUB_TOKEN` not found | Colab Secrets mein add karo (left sidebar → key icon → `GITHUB_TOKEN`) |
| `ModuleNotFoundError: No module named 'vist_nrf.data'` | `git pull` kar lo `/content/VIST-NRF` mein (data subpackage latest commit mein hai) |
| `raw_root` path wrong | `kagglehub.dataset_download(...)` se mila path use karo, `VIST_NRF_DATA_RAW` set karo |
| `xgboost` import error | `pip install xgboost>=2.0` karo, ya `logistic_regression` use karo |
| Causality test fails | Feature code mein future leakage hai — fix karna zaroori hai before model training |
| Training slow on local CPU | Colab T4 GPU use karo |

---

## Rules (Follow karna zaroori hai)

1. **No fabricated results** — sab kuch actual code se run hona chahiye.
2. **Every run needs:** YAML config, seed, saved split, run ID, code commit, environment metadata.
3. **No patient overlap** across splits. No future-data leakage. No target-environment tuning.
4. **Tests must pass** (`pytest tests/ -q`) before model experiments.
5. **Report negative results honestly.** Simple baselines first.
6. **No secrets in git.** `GITHUB_TOKEN` kabhi code mein mat likho, `.gitignore` check karo.
7. **Small logical commits.** Raw data, checkpoints, credentials commit mat karo.

---

## Quick Command Reference

```bash
# Local test
python3 -m pytest tests/ -q

# Local train
python3 scripts/train.py configs/baselines/e1_a_to_a.yaml --exploratory

# Multi-seed grid (5 seeds)
python3 scripts/run_grid.py configs/baselines/e1_a_to_a.yaml --seeds 0 1 2 3 4

# Regenerate tables
python3 scripts/make_tables.py --summary results/summary.jsonl --out results/table_comparison.md

# Regenerate figures
python3 scripts/make_figures.py --summary results/summary.jsonl --outdir results

# Reproduce a run
python3 scripts/reproduce.py results/summary.jsonl e1_lr_a_to_a_seed0
```

---

## Colab chalane ka kram (Yaad rakhne ke liye)

1. Notebook kholo: `notebooks/main.ipynb`
2. `GITHUB_TOKEN` Colab Secrets mein add karo
3. Cell 1 run karo → Drive mount + clone + install
4. Cell 2 run karo → KaggleHub data download
5. Cell 3 run karo → Full pipeline → results Drive par
6. Results check karo JSONL se

Koi error aaye toh error message copy karke mujhe dikhao, main fix kar dunga.

---

**Research only — not a clinically validated diagnostic tool.**
