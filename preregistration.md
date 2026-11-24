# Preregistration — VIST-NRF

Status: DRAFT (not frozen). Do not enable the confirmatory main-experiment grid until this is committed, hash-enforced, and Gate 1 approved.

## Identifying info
run_id_prefix: UNSET
researcher: UNSET
date_drafted: UNSET
date_frozen: UNSET

## Primary objective
Reconstructed operational sepsis onset forecasting at 12-hour horizon (primary), with 6- and 24-hour horizons as secondary.

## Primary metric
AUROC at 12 hours, reported for in-environment and shifted-environment tests, degradation (in-environment minus shifted), and worst-environment performance. Report with patient-level cluster bootstrap (default 1,000 resamples) 95% CIs.

## Secondary metrics
AUPRC (with prevalence), Brier score, ECE, sensitivity, positive predictive value at threshold chosen on source validation only.

## Hypotheses
H1: measurement timing/missingness adds predictive information beyond measured values.
H2: a model trained under one observation policy degrades under another.
H3: policy-aware/invariant training reduces degradation against equally tuned baselines.
H4 (secondary): calibration/uncertainty deteriorates under shift and can be improved by a justified method.

## Scenarios
E1: A to A and B to B (within-environment).
E2: A to B and B to A (cross-site transfer).
E3: random thinning at pre-defined rates (sanity check only).
E4: semi-synthetic held-out policies.
E5: ICU-unit transfer, if sample sizes allow.
E6: remove a laboratory block at test time.

## Data rules
- Dataset: PhysioNet/CinC Challenge 2019 (training sets A and B).
- Target: reconstructed operational onset = t_first_positive + 6 hours (documented label shift).
- Censoring: exclude incomplete negative horizons; never turn an incomplete negative into a confirmed negative.
- Splits: 70/15/15 patient-level per hospital, stratified by ever-positive; saved once; never regenerated silently.
- Forbidden features: SepsisLabel input, future values/masks/labels/timestamps, total ICU stay length, patient ID as predictor, hospital identity as predictor in primary model.

## Models (Stage A first)
- Logistic regression on values-only causal features.
- XGBoost on values-only features.
- XGBoost on values + timing/mask features (primary H1 comparison).
- GRU-D if data audit justifies deep sequence modeling.

## Tuning
Default 20 random-search trials per comparable method; same budget across methods; select only on source-domain validation; at least 5 independent seeds for final configs.

## Stopping rule
If a simpler augmentation/invariance method matches the full architecture within uncertainty, use and report the simpler method.

## Deviations
Any preregistration deviation must be dated and explained in docs/DECISIONS.md.

## Gate
Frozen only after Gate 1 approval. Main experiment scripts enforce that committed preregistration hash matches.
