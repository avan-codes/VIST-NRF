"""VIST-NRF evaluation metrics, patient-level bootstrap, calibration.

Specification section 10:
  - Primary: AUROC at 12h, in-environment + shifted, degradation, worst-environment.
  - Secondary: AUPRC with prevalence, Brier score, ECE, sensitivity, PPV at
    threshold chosen on source validation only.
  - Statistics: >=5 seeds, patient-level cluster bootstrap (default 1000),
    paired patient-level bootstrap for method differences, effect sizes + CIs,
    Holm correction across pre-defined comparison families.

Specification section 11 (secondary, only after core result stable):
  - Calibration: Brier, ECE, reliability curves per environment.
  - Split conformal prediction sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return float(brier_score_loss(y_true, y_prob))


def ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected calibration error (equal-width bins)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(sel):
            continue
        conf = y_prob[sel].mean()
        acc = y_true[sel].mean()
        ece_val += (sel.mean()) * abs(conf - acc)
    return float(ece_val)


def sensitivity_ppv(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"sensitivity": sens, "ppv": ppv, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prevalence(y_true) -> float:
    return float(np.asarray(y_true).astype(int).mean())


@dataclass
class MetricsResult:
    auroc: float
    auprc: float
    brier: float
    ece: float
    prevalence: float
    n: int
    n_pos: int
    sensitivity: float
    ppv: float
    threshold: float


def evaluate(y_true, y_prob, threshold: float = 0.5) -> MetricsResult:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return MetricsResult(
        auroc=auroc(y_true, y_prob),
        auprc=auprc(y_true, y_prob),
        brier=brier(y_true, y_prob),
        ece=ece(y_true, y_prob),
        prevalence=prevalence(y_true),
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
        sensitivity=sensitivity_ppv(y_true, y_prob, threshold)["sensitivity"],
        ppv=sensitivity_ppv(y_true, y_prob, threshold)["ppv"],
        threshold=threshold,
    )


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return (prob_true, prob_pred, bin_counts) for a reliability diagram."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    counts = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        sel = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        counts.append(int(sel.sum()))
    return prob_true, prob_pred, np.array(counts)


# ---------------------------------------------------------------------------
# Patient-level cluster bootstrap
# ---------------------------------------------------------------------------

def patient_bootstrap(y_true, y_score, patient_ids, metric_fn, n_resamples: int = 1000,
                      seed: int = 0):
    """Patient-level cluster bootstrap: resample patients, keep all their rows.

    Returns array of metric values (one per resample) for CI computation.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    out = np.full(n_resamples, np.nan)
    for r in range(n_resamples):
        sampled = rng.choice(unique_patients, size=n_patients, replace=True)
        mask = np.isin(patient_ids, sampled)
        if len(np.unique(y_true[mask])) < 2:
            out[r] = float("nan")
            continue
        out[r] = metric_fn(y_true[mask], y_score[mask])
    return out


def bootstrap_ci(values, alpha: float = 0.05):
    """Percentile CI from bootstrap values, ignoring NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    lo = np.percentile(v, 100 * alpha / 2)
    hi = np.percentile(v, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def paired_bootstrap_diff(y_true, score_a, score_b, patient_ids, metric_fn,
                          n_resamples: int = 1000, seed: int = 0):
    """Paired patient-level bootstrap for the difference metric_a - metric_b."""
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    patient_ids = np.asarray(patient_ids)
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    diffs = np.full(n_resamples, np.nan)
    for r in range(n_resamples):
        sampled = rng.choice(unique_patients, size=n_patients, replace=True)
        mask = np.isin(patient_ids, sampled)
        if len(np.unique(y_true[mask])) < 2:
            continue
        diffs[r] = metric_fn(y_true[mask], score_a[mask]) - metric_fn(y_true[mask], score_b[mask])
    return diffs


def holm_correction(p_values):
    """Holm step-down FWER correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    running_max = 0.0
    for i in range(n):
        adj_i = min(1.0, (n - i) * ranked[i])
        running_max = max(running_max, adj_i)
        adj[order[i]] = running_max
    return adj


# ---------------------------------------------------------------------------
# Conformal prediction (split conformal, secondary §11)
# ---------------------------------------------------------------------------

def split_conformal_threshold(cal_scores, cal_labels, alpha: float = 0.1):
    """Split conformal quantile threshold for score-based prediction sets.

    Uses the (1-alpha)(1 + 1/n) quantile of calibration scores.
    """
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(cal_scores, q_level))


def conformal_coverage(test_scores, test_labels, threshold):
    """Empirical coverage: fraction of test labels within the score threshold."""
    test_scores = np.asarray(test_scores)
    test_labels = np.asarray(test_labels).astype(int)
    # For binary outcome with score = P(y=1), prediction set = {0} if score<thr else {0,1}
    # Coverage = fraction where true label in set = 1 - P(score<thr & y=1)
    in_set = (test_scores >= threshold) | (test_labels == 1)
    return float(in_set.mean())
