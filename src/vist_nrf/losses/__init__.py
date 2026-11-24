"""VIST-NRF losses: invariance / robustness penalties for Stage B baselines.

Penalties are added to the base prediction loss (e.g. BCE) with a weight.
They take environment indices so penalties are computed within environments.
"""

from vist_nrf.losses.invariance import adversarial_env_penalty, InvarianceLoss
from vist_nrf.losses.groupdro import groupdro_penalty
from vist_nrf.losses.irm import irm_penalty
from vist_nrf.losses.vrex import vrex_penalty

__all__ = [
    "irm_penalty",
    "vrex_penalty",
    "groupdro_penalty",
    "adversarial_env_penalty",
    "InvarianceLoss",
]
