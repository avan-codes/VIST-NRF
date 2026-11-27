# PROJECT_REPORT — VIST-NRF

Visit-Invariant State Tracking under Nonstationary Recording and Follow-up.
**Research only — not a clinically validated diagnostic tool. Do not use for patient-care decisions.**

---

## 1. Kya problem hai (Problem Statement)

ICU mein patients ki clinical measurements (heart rate, blood pressure, lactate, etc.) ghante-ghante li jaati hain, lekin **irregular** hoti hain — sabhi variables har ghante nahi mapte, kuch missing hote hain, aur har hospital ka recording policy alag hota hai (kahan zyada baar measure karenge, kahan kam).

Isse do bade problems paida hote hain:

1. **Irregular + partially observed time series**: Standard forecasting models assume regularly-sampled, fully-observed input. Real ICU data is the opposite — missingness aur timing dono maujood hain, aur timing khud informative ho sakta hai (agar doctor baar-baar measure kar raha hai toh matlab condition worsen ho rahi hai).

2. **Nonstationary recording / follow-up policy (domain shift)**: Hospital A mein measurement policy alag hai hospital B se. Agar model train karte hain Hospital A par aur test Hospital B par, toh model jo "timing patterns" seekh liya tha wo policy-specific ho sakta hai, patient-state specific nahi. Cross-site performance degrade ho jaata hai.

**Core research question:**
> Kya ek forecasting model useful patient-state information seekh sakta hai jo observation/follow-up policy change hone par bhi reliable rahe?

**Primary case study:** Sepsis forecasting — 12 ghante pehle predict karna ki patient ko sepsis hoga ya nahi — use PhysioNet/CinC Challenge 2019 dataset (do hospital systems: A = 20,336 patients, B = 20,000 patients).

---

## 2. Kya solve karna hai (Objectives)

| Objective | Kaise pata chalega |
|-----------|-------------------|
| Sepsis onset forecast 12 ghante advance mein | AUROC@12h, AUPRC, Brier, ECE |
| Timing/missingness predictive value kya hai values se pare? | XGBoost values-only vs values+timing compare |
| Cross-site degradation hai kya? | A→A vs A→B aur B→A vs B→B |
| Invariance/robustness method degradation kam karta hai kya? | IRM / V-REx / GroupDRO / VIST-NRF vs ERM baseline |

---

## 3. Kaise solve karein ge (Solution Approach)

Pipeline 6 stages mein hai, har stage ek hi data leak nahi karta (causal guarantee):

### Stage 1 — Data Load
PhysioNet `.psv` files (pipe-separated, hourly rows) ko load kiya jaata hai. Har patient ka `patient_id` filename se aata hai, hospital system (A/B) folder name se infer hota hai. Folder name assume nahi kiya — inspect kiya jaata hai (spec §4.2).

### Stage 2 — Target Construction (spec §5)
- SepsisLabel positive hota hai `t_sepsis - 6` ghante pehle (challenge ka documented shift).
- **Reconstructed operational onset**: `t_event = t_first_positive + 6 hours` — yeh asli target hai.
- Horizon label: `y(t, H) = 1` jab `t < t_event <= t + H`; `y = 0` sirf jab full horizon observe ho; otherwise **censored** (negative nahi maana).
- `t >= t_event` waale rows exclude (event already occurred — future leakage).
- Min 6 ghante history chahiye prediction ke liye.
- Malformed pattern `1→0→1` ko ambiguous mark kiya jaata hai.

### Stage 3 — Causal Features (spec §6.2)
Har ghante `t` par features sirf history `<= t` use karte hain:
- LOCF (last observed carry forward) per variable
- Observation mask per variable (kya ye ghante measure hua?)
- Time-since-last-observation per variable
- Measurement counts pichle 6/12/24 ghante mein
- Causal rolling min/max/mean pichle 6/24 ghante mein
- Static: Age, Gender, HospAdmTime, ICULOS, ICU_unit

Forbidden (leak): SepsisLabel input, future values/masks/labels, total stay length, patient ID as predictor, hospital identity as predictor in primary model.

### Stage 4 — Causality Leakage Test (spec §6.4)
`assert_causal()` — har patient ke liye representative hours (early, mid, late) par features full-record se aur truncated history se compute karte hain; dono match hone chahiye. Agar mismatch → model training **blocked**. Future-data leakage rokne ke liye yeh non-negotiable hai.

### Stage 5 — Patient-Level Splits (spec §6.1)
70/15/15 train/val/test, **patient level** (row level nahi), stratified by ever-positive sepsis status. Ek patient do splits mein nahi aa sakta (disjointness check). Cross-env bhi koi patient repeat nahi hona chahiye.

### Stage 6 — Model + Evaluate (spec §9, §10)
Models share ek interface: `fit(X, y, envs)`, `predict_proba(X)`, `save`, `load`.

Available models:
1. **Logistic Regression** — values-only causal features (baseline)
2. **XGBoost** — values-only, aur values+timing variant (primary H1 comparison)
3. **GRU-D** — PyTorch time-aware recurrent baseline (learnable decay for missing values)
4. **VIST-NRF** — State encoder + policy encoder + prediction head + optional adversarial env head + invariance loss (Stage C, sirf agar baselines justify karein)

Evaluation metrics: AUROC, AUPRC, Brier, ECE, prevalence, sensitivity, PPV. Patient-level cluster bootstrap (1000 resamples) for CIs. Paired bootstrap for method differences. Holm correction across comparison families.

Results JSONL mein save: har run ka commit, env metadata, seed, run_id — taaki har number traceable ho exact code version se.

---

## 4. Architecture kya hai (Detailed Architecture)

### Data Layer (`src/vist_nrf/data/`)
```
physionet2019.py  → parse_psv, audit_psv, load_environment
targets.py        → reconstruct_event_time, make_horizon_labels, build_targets
splits.py         → split_patients, split_by_environment, assert_splits_valid
download.py       → download_physionet2019 (kagglehub), audit_tree, save_manifest
```

### Feature Layer (`src/vist_nrf/features/`)
```
history.py  → FeatureConfig, compute_patient_features, build_feature_matrix, assert_causal
timing.py   → timing-specific feature helpers
```

Feature pipeline per patient:
```
Raw .psv → [hour, patient_id, 40 clinical vars, SepsisLabel]
   ↓
compute_patient_features
   ├─ LOCF per variable (causal ffill)
   ├─ Observation mask per variable
   ├─ Time-since-last-observation per variable
   ├─ Counts over 6/12/24h windows (searchsorted, vectorized)
   ├─ Rolling min/max/mean over 6/24h windows (observed only)
   └─ Static cols repeated per hour
   ↓
build_feature_matrix → long DataFrame: patient_id, hour, <features>
   ↓
assert_causal() → leakage test (PASS required before training)
```

### Model Layer (`src/vist_nrf/models/`)
```
baselines.py  → LogisticRegressionBaseline, XGBoostBaseline, GRUDModel, build_model(), MODEL_REGISTRY
vist_nrf.py   → VISTNRFModel (state encoder + policy encoder + head + adversarial head)
```

VIST-NRF internal architecture:
```
Input: values (B,T,F), mask (B,T,F), delta (B,T,F)
   │
   ├─ State Encoder: GRU(cat[values, mask, delta]) → hidden=64 → dropout → z_state
   │
   ├─ Policy Encoder: MLP(cat[obs_rate, mean_gap]) → hidden=32 → z_policy
   │
   ├─ Prediction Head: MLP(z_state [+ z_policy if use_policy]) → 1 logit → sigmoid
   │
   ├─ (Optional) Adversarial Env Head: GradientReversal(z_state) → Linear(hidden, n_envs) → cross-entropy
   │
   └─ Loss = BCE(logits, y) + invariance_weight × [invariance loss + adversarial env loss]
```

### Loss Layer (`src/vist_nrf/losses/`)
```
irm.py   → IRMv1 penalty (dummy classifier gradient = 0 within each env)
vrex.py  → V-REx (variance of per-environment risk)
groupdro.py → GroupDRO (worst-environment risk)
invariance.py → InvarianceLoss wrapper (IRM/V-REx/GroupDRO/ERM) + adversarial penalty
```

### Eval Layer (`src/vist_nrf/eval/`)
```
metrics.py   → AUROC, AUPRC, Brier, ECE, sensitivity, PPV, bootstrap CI, paired bootstrap, Holm, conformal
censoring.py → complete_horizon_mask, censoring_report, assert_no_early_event_prediction, assert_censoring_not_negative
```

### Scripts (`scripts/`)
```
train.py    → End-to-end: seed → prereg → data → targets → features → causality → splits → model → evaluate → JSONL
evaluate.py → Load checkpoint, evaluate on test env(s)
run_grid.py → Multi-seed runs (≥5 seeds)
reproduce.py → Clean re-run from summary.jsonl
make_tables.py / make_figures.py → Regenerate tables/figures from saved JSONL
gen_notebooks.py → Generate Colab notebooks
```

---

## 5. Experiments (Plan)

| Experiment | Train Env | Test Env | Model | Purpose |
|------------|-----------|----------|-------|---------|
| E1 | A | A | logistic_regression | Within-environment baseline |
| E2 | A | B | logistic_regression | Cross-site transfer degradation |
| E2 | B | A | logistic_regression | Cross-site transfer degradation |
| E3 | A | A | xgboost_values | H1: timing informative? |
| E3 | A | A | xgboost_timing | |
| E4 | Semi-synthetic held-out policies | | VIST-NRF | Policy-invariance evidence |
| E5 | ICU-unit transfer | | — | If sample sizes allow |
| E6 | Remove lab block at test | | — | Robustness check |

**Current config** (`configs/baselines/e1_a_to_a.yaml`): E1, logistic regression, horizon=12h, min_history=6h, C=1.0, max_iter=1000, seed=0, exploratory=true.

**Stopping rule (spec §9):** Agar simpler augmentation/invariance method full architecture ke barabar result deta hai uncertainty ke andar, toh simpler method report kiya jaayega. VIST-NRF Stage C sirf agar baselines meaningful gap chhodte hain.

---

## 6. Current Results (Abhi tak kya hua)

| Run | Train | Test | Model | AUROC | AUPRC | Brier | ECE | Prevalence | N |
|-----|-------|------|-------|-------|-------|-------|-----|------------|---|
| e1_lr_a_to_a_seed0 | A | A | logistic_regression | 1.000 | 1.000 | ~0.000 | ~0.005 | 0.500 | 48 |

> Yeh synthetic fixture smoke test result hai (real PhysioNet data par abhi full training pending hai). Real data par AUROC shayad 1.000 na ho; synthetic perfect separable thi.

**Tests:** 19/19 pass (`tests/test_data.py` 12 + `tests/test_features.py` 7).

**Real data status:** PhysioNet/CinC 2019 download kiya jaaya hai kagglehub se (`farjanayesmin/the-physionet-challenge-2019-dataset`), ~45MB, 20,336 A + 20,000 B patients. `configs/baselines/e1_a_to_a.yaml` mein `raw_root` set hai.

---

## 7. Gates (Human Decision Points)

| Gate | Decision | Document |
|------|----------|----------|
| Gate 0 | Gap credible hai? Real-data assumptions defensible hain? | `reports/gate_0.md` |
| Gate 1 | H1/H2 supported hain? (timing informative? cross-site degradation hai?) | `reports/gate_1.md` |
| Gate 2 | Ek sharp claim convincing evidence ke saath? | `reports/gate_2.md` |
| Gate 3 | Final claims-to-evidence map, clean runtime par reproduction | `reports/gate_3.md` |

Rule: Human researcher approve kare bina agle phase par mat jao. Preregistration deviate hona toh `docs/DECISIONS.md` mein date + reason likho.

---

## 8. Non-Negotiable Rules

1. **No fabricated results** — sab kuch actual code se run hona chahiye. Unknown facts `UNVERIFIED` mark karo.
2. **Every run needs:** YAML config, seed, saved split, run ID, code commit, environment metadata.
3. **No patient overlap** across splits. No future-data leakage. No target-environment tuning.
4. **Tests must pass** before model experiments. `assert_causal()` blocks if leakage found.
5. **Report negative results honestly.** Simple baselines first. Complex VIST-NRF sirf agar justified.
6. **No secrets in git.** `.gitignore` excludes `data/`, `*.psv`, `*.pt`, `.env`, large results.
7. **Small logical commits.** Never commit raw data, checkpoints, credentials, tokens.

---

## 9. How to Run (Colab / Local)

**Local:**
```bash
cd /home/pranjal/Desktop/Projects/Github/Research_Projects/VIST-NRF
python3 -m pip install -e . --break-system-packages
python3 -m pip install -e ".[dev]" --break-system-packages
python3 -m pytest tests/ -q
python3 scripts/train.py configs/baselines/e1_a_to_a.yaml --exploratory
```

**Colab (`notebooks/main.ipynb`):**
1. Colab Secrets mein `GITHUB_TOKEN` add karein (left sidebar → key icon).
2. Cell 1: Drive mount + git clone/pull + `pip install -e`.
3. Cell 2: `kagglehub.dataset_download("farjanayesmin/the-physionet-challenge-2019-dataset")` → `VIST_NRF_DATA_RAW`.
4. Cell 3: Full pipeline → results Drive par `results/runs/e1_lr_a_to_a_seed0.jsonl`.

Results JSONL format per record: `auroc, auprc, brier, ece, prevalence, n, n_pos, sensitivity, ppv, threshold, train_env, test_env, model, horizon_hours, seed, commit, n_test_patients, n_test_rows, censored_rows, utc, platform, python, torch, cuda_available, ...`

---

## 10. Roadmap / Next Steps

1. Real-data E1 training run (A→A logistic regression) — results review.
2. E2 cross-site (A→B, B→A) with 5 seeds → Gate 1 decision on H1/H2.
3. Agar supported → semi-synthetic held-out-policy evidence (E4) + VIST-NRF Stage C method + ablations.
4. Agar nahi supported → negative result report (baseline hi best hai) per spec §16.
5. Tables/figures regenerate from saved JSONL (`make_tables.py`, `make_figures.py`).
6. Paper draft (`paper/claims.md`, `main.tex`).

---

**Repository:** https://github.com/avan-codes/VIST-NRF.git  
**Branch:** main  
**Latest commit:** `a23978c` (main.ipynb added), `192a0a8` (download fn), `ad6f8ca` (data subpackage tracked)

**Research only — not a clinically validated diagnostic tool.**
