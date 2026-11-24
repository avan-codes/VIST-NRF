# Overview — VIST-NRF

## 1. What this project is

**VIST-NRF** = **Visit-Invariant State Tracking under Nonstationary Recording and Follow-up**.

It is a reproducible machine-learning research system for **forecasting from irregular, partially observed clinical time series**, with the primary real-data case study being **sepsis forecasting using the PhysioNet/CinC Challenge 2019 dataset** (two hospital systems).

**Core research question:**
> Can a forecasting model learn useful patient-state information that remains reliable when the observation or follow-up policy changes?

Clinical measurements are not collected at random. They are taken because a patient's condition changed, because a care protocol required them, or because a hospital follows a different recording practice. Measurement timing therefore encodes both **patient-state information** and **hospital-policy information**. VIST-NRF investigates whether we can separate the two.

## 2. Why this project was chosen

### The problem is real and under-studied
Irregular clinical sampling is the norm, not the exception, in ICU data. Standard forecasting methods assume regularly sampled, fully observed inputs — an assumption that fails in practice. When the observation policy shifts (e.g., moving from Hospital A to Hospital B, or a protocol change within one hospital), models that have inadvertently learned policy-specific timing patterns degrade. VIST-NRF directly addresses this gap.

### Two hospital systems enable a controlled test
The PhysioNet/CinC Challenge 2019 provides hourly ICU data from **two hospital systems (A and B)**, with published counts of 20,336 and 20,000 patient records respectively. This natural two-environment structure allows:
- **In-environment evaluation** (A→A, B→B): does the model work where it was trained?
- **Cross-site transfer** (A→B, B→A): does it degrade when the observation policy changes?

### A controlled semi-synthetic benchmark is essential
Real data alone cannot prove that a model has learned invariant patient state, because the outcome-generating process may itself differ between hospitals (the sepsis label uses clinical suspicion of infection including antibiotics and blood cultures, which may differ by hospital). VIST-NRF therefore includes a **semi-synthetic benchmark** with known latent states where observation policies can be changed without changing the outcome generator — providing clean evidence about policy-shift robustness.

### Research scope, not clinical deployment
This is explicitly **research only**. The specification states: "Do not present the model as a clinically validated diagnostic tool or for patient-care decisions." The deliverable is a reproducible research system whose claims are supported by experiments and verified theory — not a clinical product.

## 3. Why each function / module was chosen

### 3.1 Data layer — `src/vist_nrf/data/`

| Module | Function | Why |
|--------|----------|-----|
| `physionet2019.py` | `.psv` parser, audit, hospital-system inference | Reads pipe-separated hourly PhysioNet files. Infers hospital system (A/B) from directory names rather than assuming folder names (spec §4.2: "Inspect the downloaded directory tree; do not assume folder names"). Patient ID comes from the filename, never treated as a predictive feature. |
| `targets.py` | Event reconstruction, horizon labels, censoring | Implements spec §5: `t_event = t_first_positive + 6 hours` (the documented label shift), horizon labels `y(t,H) = 1 when t < t_event <= t + H`, censoring of incomplete negative horizons (never turned into confirmed negatives), exclusion of times at/after `t_event`, minimum 6h history requirement. Detects malformed `1→0→1` label patterns as ambiguous. |
| `splits.py` | Stratified 70/15/15 patient-level splits | Spec §6.1: within each hospital system, split patients (not rows) by ever-positive sepsis status. Splits are saved once and never silently regenerated. `assert_splits_valid()` checks disjointness and cross-environment uniqueness. |
| `download.py` | Manifest + count verification | Records counts, schema, license, size, access date. Verifies against expected counts (A=20,336 / B=20,000) and **raises if mismatch** — no silent proceeding. |

### 3.2 Feature layer — `src/vist_nrf/features/`

| Module | Function | Why |
|--------|----------|-----|
| `history.py` | Causal LOCF, observation mask, time-since-observation, measurement counts (6/12/24h), rolling min/max/mean | Spec §6.2 allowed features. Every feature is **causal**: at time `t` it uses only history `≤ t`. LOCF via forward-fill; counts and rolling stats via vectorized `searchsorted` on observed timestamps. Time-since-last-observation captures the information in measurement gaps (H1 hypothesis: timing/missingness adds predictive information beyond values). |
| `assert_causal()` | Leakage test: truncated-history vs full-record equivalence | Spec §6.4: features at `t` must be unchanged whether computed from truncated history or full record. This test **blocks model experiments if it fails** — no model trains on leaky features. |

### 3.3 Model layer — `src/vist_nrf/models/`

| Module | Function | Why |
|--------|----------|-----|
| `baselines.py` | Logistic regression, XGBoost (values-only vs values+timing), GRU-D | Spec §9 Stage A: start simple. Logistic regression on values-only features tests whether raw values suffice. XGBoost values+timing is the primary H1 comparison (does timing add information?). GRU-D is a time-aware recurrent baseline if the data audit justifies deep sequence modeling. |
| `vist_nrf.py` | VIST-NRF method: state encoder + policy encoder + prediction head + adversarial env head | Spec §9 Stage C: built **only if** simpler methods leave a meaningful gap. State encoder (time-aware GRU over values/mask/gaps → `z_state`), policy encoder (summarizes mask/timing → `z_policy`), prediction head primarily on `z_state`. Optional gradient-reversal adversarial environment head on `z_state` (separation objective). |
| `build_model()` registry | Single interface: `fit`, `predict_proba`, `save`, `load` | All models share the same interface so experiments can swap models without changing evaluation code. |

### 3.4 Loss layer — `src/vist_nrf/losses/`

| Module | Function | Why |
|--------|----------|-----|
| `irm.py` | IRMv1 penalty (gradient of dummy classifier = 0 within each env) | Stage B robustness baseline (§9): invariance risk minimization encourages predictors that are optimal across environments. |
| `vrex.py` | V-REx: variance of per-environment risk | Encourages equal risk across environments — a softer alternative to IRM. |
| `groupdro.py` | GroupDRO: worst-environment risk | Minimizes the maximum per-environment risk, improving worst-environment performance. |
| `invariance.py` | `InvarianceLoss` wrapper + adversarial env penalty | Combined wrapper for Stage B methods. Includes the spec §9 caution: adversarial hospital classifiers do not automatically identify patient state and may remove clinically useful hospital-correlated information — ablate individually. |

### 3.5 Evaluation layer — `src/vist_nrf/eval/`

| Module | Function | Why |
|--------|----------|-----|
| `metrics.py` | AUROC, AUPRC, Brier, ECE, sensitivity, PPV, prevalence, bootstrap CIs, paired bootstrap, Holm correction, reliability curves, split conformal | Spec §10: primary AUROC@12h with in-environment + shifted-environment + degradation + worst-environment reporting. Patient-level cluster bootstrap (default 1,000 resamples) for CIs; paired bootstrap for method differences; Holm correction across comparison families. |
| `censoring.py` | Complete-horizon rule, censored≠negative, censoring reports | Spec §5.3: incomplete negative horizons are censored, never auto-negative. Reports affected patients and rows so they are never silently dropped. |

### 3.6 Benchmarks & theory

| Module | Function | Why |
|--------|----------|-----|
| `benchmarks/semisynthetic.py` | OU latent trajectories, policy-dependent observation, outcome from latent state only, oracle_predict | Spec §8: config-driven generator with known latent states. Outcome generated from latent state only; policy must not directly change outcome. Oracle using true latent state is the policy-invariant upper bound. |
| `theory_checks/__init__.py` | Scale-ambiguity counterexample + anchor-breaks-ambiguity demo | Spec §3.2: numerically demonstrates that `lambda(t) = a*exp(beta*s(t))` is scale-ambiguous (rescale `s` and inversely rescale `beta`, same intensity). Shows anchored measurements break the ambiguity. Every theory claim states its assumptions. |

### 3.7 Scripts & orchestration

| Script | Function | Why |
|--------|----------|-----|
| `train.py` | Full experiment: seed, targets, features, causality test, model fit, evaluate, JSONL results + commit + env metadata | Spec §13: every run needs YAML config, seed, saved split, run ID, code commit, environment metadata. Results appended as JSONL with stable `run_id`. |
| `evaluate.py` | Load checkpoint + split, evaluate on test env(s) | Separates training from evaluation; supports checkpoint loading. |
| `run_grid.py` | Multi-seed runs via temp configs | Spec §13: ≥5 independent seeds for final configurations; same tuning budget across methods. |
| `reproduce.py` | Clean re-run from `summary.jsonl` record | Reproducibility: regenerate any recorded run at its committed code version. |
| `make_tables.py` / `make_figures.py` | Regenerate tables/figures from saved JSONL | Spec §17: "Tables/figures regenerate from saved results on a clean runtime" — never hand-type experimental numbers. |
| `gen_notebooks.py` | Generates the 6 Colab notebooks | Colab/Drive workflow per spec §14. |

## 4. Why the design decisions matter

### Causality first
Every feature computation is causal. The `assert_causal()` test verifies that truncating the history does not change features at time `t`. This directly prevents **future-data leakage** (spec §6.3), which is the most common error in clinical time-series research.

### Patient-level splits, not row-level
Splits are at the patient level (spec §6.1). If you split rows, the same patient's early and late hours could appear in both train and test, leaking future information. Patient-level splits also correctly model the deployment scenario: you train on some patients and deploy on entirely new patients.

### Censoring, not auto-negative
A patient whose ICU stay ends before the full prediction horizon is observed is **censored**, not a confirmed negative. Treating censored rows as negatives biases the label and invalidates AUROC/AUPRC. VIST-NRF excludes them via the complete-horizon rule and reports how many are affected.

### Reconstructed operational onset, not the shifted label
The PhysioNet sepsis label turns positive 6 hours before the actual event (`t_sepsis - 6`). VIST-NRF reconstructs `t_event = t_first_positive + 6` as the operational onset target (spec §5.2). It does **not** define the target as "next positive label within H hours" — that would forecast an already-shifted label and misstate the clinical horizon.

### Simple baselines first, complex method only if justified
Spec §9 stopping rule: "if a simpler augmentation or invariance method matches the full architecture within uncertainty, use and report the simpler method." VIST-NRF's neural method is Stage C, built only after baselines are evaluated and Gate 1 approves. This prevents over-engineering and ensures any complexity is evidence-driven.

### Preregistration enforcement
`prereg.py` refuses confirmatory runs unless the committed `preregistration.md` hash matches. Exploratory runs require an explicit `--exploratory` flag and go to a separate directory. This prevents p-hacking and moving goalposts after seeing results.

### No secrets, no patient data in git
`.gitignore` excludes `data/`, `*.psv`, checkpoints (`*.pt/*.pth/.ckpt`), credentials (`.env`), and large results. Code, configs, tests, and small reports are committed; raw patient data and checkpoints live on Google Drive (Colab workflow, spec §14).

## 5. How to read the code for the paper

| Paper section | Where to look in code |
|---------------|----------------------|
| Problem formulation (§3) | `src/vist_nrf/data/targets.py`, `src/vist_nrf/theory_checks/__init__.py` |
| Data & splits (§4, §6) | `src/vist_nrf/data/physionet2019.py`, `src/vist_nrf/data/splits.py` |
| Features (§6.2) | `src/vist_nrf/features/history.py` |
| Baselines (§9 Stage A) | `src/vist_nrf/models/baselines.py` |
| Robustness baselines (§9 Stage B) | `src/vist_nrf/losses/` |
| VIST-NRF method (§9 Stage C) | `src/vist_nrf/models/vist_nrf.py` |
| Metrics & statistics (§10) | `src/vist_nrf/eval/metrics.py` |
| Censoring (§5.3) | `src/vist_nrf/eval/censoring.py` |
| Semi-synthetic (§8) | `src/vist_nrf/benchmarks/semisynthetic.py` |
| Training & reproducibility (§13) | `scripts/train.py`, `scripts/reproduce.py`, `scripts/make_tables.py`, `scripts/make_figures.py` |

## 6. Current status

- **Phase 0 Foundation:** complete (commit `37f5f3a`)
- **Phase 1 Data pipeline + baselines:** complete (commit `4fd2087`)
- **All 19 unit tests pass** (`tests/test_data.py` + `tests/test_features.py`)
- **Training smoke test runs end-to-end** and saves JSONL results
- **Real PhysioNet data not yet downloaded** — requires your PhysioNet credential; see `docs/DATA_NOTES.md` for the `aws s3 sync` command
- **Colab bootstrap documented** in `docs/ENV_NOTES.md` and `notebooks/00_bootstrap.ipynb` (needs `GITHUB_TOKEN` in Colab Secrets)

## 7. Summary statement for the paper

VIST-NRF is a reproducible research system for visit-invariant state tracking under nonstationary recording and follow-up. It addresses the clinical reality that ICU measurement timing encodes both patient state and hospital policy. The system combines causal feature engineering, patient-level disjoint splits, reconstructed operational-onset targets with proper censoring, simple strong baselines (logistic regression, XGBoost), robustness methods (IRM/V-REx/GroupDRO), and an optional neural state/policy-separation architecture (VIST-NRF). All experiments are preregistration-enforced, config-driven, seeded, and produce regenerable tables and figures from saved JSONL results. The code supports both in-environment (A→A, B→B) and cross-site transfer (A→B, B→A) evaluation on PhysioNet/CinC Challenge 2019, plus a controlled semi-synthetic benchmark with known latent states.

**Research only — not a clinically validated diagnostic tool.**
