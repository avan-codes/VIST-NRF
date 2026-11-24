"""Adversarial environment penalty + combined InvarianceLoss wrapper.

Spec warning (§9): an adversarial hospital classifier does NOT automatically
identify patient state and may remove clinically useful hospital-correlated
information. Ablate individually and test environment leakage + representation
collapse + predictive performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def adversarial_env_penalty(env_logits, envs, weight=1.0):
    """Negative environment classification loss.

    Returned as a penalty to *minimize*: minimizing the negative env loss means
    the encoder maximizes env classification loss (environment confusion).
    A separate env classifier minimizes env loss normally.
    """
    env_loss = F.cross_entropy(env_logits, envs.long())
    return -weight * env_loss


class InvarianceLoss(nn.Module):
    """Combined invariance loss wrapper for Stage B methods.

    method: 'erm' | 'irm' | 'vrex' | 'groupdro' | 'adversarial'
    """

    def __init__(self, method="erm", weight=1.0):
        super().__init__()
        self.method = method
        self.weight = weight

    def forward(self, logits, targets, envs, env_logits=None):
        if self.method == "erm":
            return logits.new_zeros(())
        if self.method == "irm":
            from vist_nrf.losses.irm import irm_penalty
            return irm_penalty(logits, targets, envs, self.weight)
        if self.method == "vrex":
            from vist_nrf.losses.vrex import vrex_penalty
            return vrex_penalty(logits, targets, envs, self.weight)
        if self.method == "groupdro":
            from vist_nrf.losses.groupdro import groupdro_penalty
            return groupdro_penalty(logits, targets, envs, self.weight)
        if self.method == "adversarial":
            if env_logits is None:
                return logits.new_zeros(())
            return adversarial_env_penalty(env_logits, envs, self.weight)
        raise ValueError(f"Unknown invariance method: {self.method}")
