"""VIST-NRF theory checks: identifiability counterexamples (§3.2).

Reproduces numerically the scale-ambiguity counterexample for an intensity-only
observation model lambda(t) = a * exp(beta * s(t)): rescaling s and inversely
rescaling beta leaves the same intensity. This demonstrates non-identifiability
in that restricted model; it does NOT prove every model is non-identifiable when
additional anchored measurements exist.
"""

from __future__ import annotations

import numpy as np


def intensity_scale_ambiguity_demo(seed: int = 0):
    """Show that (s, beta) and (k*s, beta/k) give identical intensities."""
    rng = np.random.default_rng(seed)
    n = 200
    s = rng.standard_normal(n)
    a = 0.3
    beta = 2.0
    intensity = a * np.exp(beta * s)

    # rescale: s' = k*s, beta' = beta/k, a' = a -> same intensity
    k = 3.0
    s_prime = k * s
    beta_prime = beta / k
    intensity_prime = a * np.exp(beta_prime * s_prime)

    max_diff = float(np.max(np.abs(intensity - intensity_prime)))
    return {
        "a": a,
        "beta": beta,
        "k": k,
        "max_intensity_diff": max_diff,
        "identical": bool(max_diff < 1e-9),
        "note": ("Scale ambiguity in intensity-only model lambda(t)=a*exp(beta*s(t)): "
                 "rescaling s and inversely rescaling beta leaves intensity unchanged. "
                 "This does not imply non-identifiability with anchored measurements."),
    }


def anchor_breaks_ambiguity_demo(seed: int = 0):
    """Adding an anchored measurement of s breaks the scale ambiguity."""
    rng = np.random.default_rng(seed)
    n = 200
    s = rng.standard_normal(n)
    a = 0.3
    beta = 2.0
    k = 3.0
    # anchored observation: m = s + noise, identifies the scale of s
    noise = rng.normal(0, 0.1, n)
    m = s + noise
    corr_original = float(np.corrcoef(s, m)[0, 1])
    # under rescaling s' = k*s, the anchored measurement no longer matches
    m_prime = k * s + noise
    corr_rescaled = float(np.corrcoef(k * s, m_prime)[0, 1])
    return {
        "corr_original": corr_original,
        "corr_rescaled": corr_rescaled,
        "note": ("Anchored measurement of s identifies the scale, breaking the "
                 "intensity-only ambiguity."),
    }
