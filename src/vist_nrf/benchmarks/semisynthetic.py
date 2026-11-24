"""VIST-NRF semi-synthetic benchmark generator.

Specification section 8: config-driven generator for controlled observation-policy
shift with known latent states. Outcome generated from latent state ONLY; policy
must not directly change the outcome generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SemiSyntheticConfig:
    n_patients: int = 200
    n_hours: int = 48
    n_vars: int = 8
    n_envs: int = 2
    seed: int = 0
    # latent progression: Ornstein-Uhlenbeck
    ou_theta: float = 0.1
    ou_sigma: float = 0.3
    # observation model: lambda_e(t | s) = w_e * protocol_e(t) + (1-w_e) * r_e * exp(beta_e^T s)
    protocol_schedule: list[float] = field(default_factory=lambda: [0.05] * 48)
    w: list[float] = field(default_factory=lambda: [0.5, 0.5])
    r: list[float] = field(default_factory=lambda: [0.2, 0.2])
    beta: list[list[float]] = field(default_factory=lambda: [[0.5] * 8, [0.5] * 8])
    # outcome generator from latent state only
    outcome_slope: float = 1.0
    outcome_intercept: float = -2.0
    noise_sd: float = 0.5
    clipping: tuple[float, float] = (0.0, 1.0)


def _ou_trajectory(n_hours, theta, sigma, rng):
    s = np.zeros(n_hours)
    s[0] = rng.normal(0, 1)
    for t in range(1, n_hours):
        s[t] = s[t - 1] + theta * (0 - s[t - 1]) + sigma * rng.standard_normal()
    return s


def generate(env_cfg: SemiSyntheticConfig, env_index: int, rng: np.random.Generator):
    """Generate one environment's semi-synthetic dataset.

    Returns list of patient dicts with keys:
      patient_id, env, s (latent trajectory), x (observed channels, NaN when unobserved),
      mask (observed), y (outcome label), hour
    """
    n_patients = env_cfg.n_patients
    n_hours = env_cfg.n_hours
    n_vars = env_cfg.n_vars
    patients = []
    for i in range(n_patients):
        patient_id = f"E{env_index}_P{i:04d}"
        # latent state trajectory (OU with patient-level random effect)
        patient_effect = rng.normal(0, 0.5)
        s = _ou_trajectory(n_hours, env_cfg.ou_theta, env_cfg.ou_sigma, rng) + patient_effect
        # outcome from latent state ONLY (logistic on mean state)
        mean_s = float(np.mean(s))
        logit_p = env_cfg.outcome_intercept + env_cfg.outcome_slope * mean_s
        p_event = 1.0 / (1.0 + np.exp(-logit_p))
        y = int(rng.random() < p_event)

        # observation policy per hour
        beta_e = np.array(env_cfg.beta[env_index], dtype=float)
        w_e = env_cfg.w[env_index]
        r_e = env_cfg.r[env_index]
        protocol = np.array(env_cfg.protocol_schedule, dtype=float)

        x = np.full((n_hours, n_vars), np.nan)
        mask = np.zeros((n_hours, n_vars), dtype=float)
        for t in range(n_hours):
            lam = w_e * protocol[t] + (1 - w_e) * r_e * np.exp(np.clip(beta_e @ s[t], -10, 10))
            lam = float(np.clip(lam, env_cfg.clipping[0], env_cfg.clipping[1]))
            for j in range(n_vars):
                if rng.random() < lam:
                    x[t, j] = s[t] + rng.normal(0, env_cfg.noise_sd)
                    mask[t, j] = 1.0
        patients.append(
            {
                "patient_id": patient_id,
                "env": env_index,
                "s": s,
                "x": x,
                "mask": mask,
                "y": y,
                "hour": np.arange(n_hours, dtype=float),
            }
        )
    return patients


def generate_all(cfg: SemiSyntheticConfig):
    """Generate all environments. Returns {env_index: [patients]}."""
    rng = np.random.default_rng(cfg.seed)
    data = {}
    for e in range(cfg.n_envs):
        # per-environment rng stream
        env_rng = np.random.default_rng(cfg.seed + 1000 + e)
        # override env-specific params if lists provided
        env_cfg = SemiSyntheticConfig(
            n_patients=cfg.n_patients,
            n_hours=cfg.n_hours,
            n_vars=cfg.n_vars,
            seed=cfg.seed,
            ou_theta=cfg.ou_theta,
            ou_sigma=cfg.ou_sigma,
            protocol_schedule=cfg.protocol_schedule,
            w=cfg.w[e] if isinstance(cfg.w, list) else cfg.w,
            r=cfg.r[e] if isinstance(cfg.r, list) else cfg.r,
            beta=cfg.beta[e] if isinstance(cfg.beta, list) and len(cfg.beta) > e else cfg.beta,
            outcome_slope=cfg.outcome_slope,
            outcome_intercept=cfg.outcome_intercept,
            noise_sd=cfg.noise_sd,
            clipping=cfg.clipping,
        )
        data[e] = generate(env_cfg, e, env_rng)
    return data


def to_long_table(patients_by_env):
    """Flatten to a long DataFrame-like structure (list of row dicts) with features."""
    rows = []
    for env, patients in patients_by_env.items():
        for p in patients:
            n_hours = len(p["hour"])
            for t in range(n_hours):
                row = {
                    "patient_id": p["patient_id"],
                    "env": env,
                    "hour": p["hour"][t],
                    "y": p["y"],
                    "s_true": p["s"][t],
                }
                for j in range(p["x"].shape[1]):
                    row[f"x{j}"] = p["x"][t, j]
                    row[f"mask{j}"] = p["mask"][t, j]
                rows.append(row)
    return rows


def oracle_predict(patients_by_env):
    """Oracle that uses true latent state s (upper bound, policy-invariant)."""
    scores = {}
    truths = {}
    for env, patients in patients_by_env.items():
        for p in patients:
            scores[p["patient_id"]] = float(np.mean(p["s"]))
            truths[p["patient_id"]] = p["y"]
    return scores, truths
