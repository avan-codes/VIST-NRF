"""V-REx (Variance-Risk Extrapolation) penalty.

Encourages equal risk across environments by penalizing the variance of
per-environment risks.
"""

import torch
import torch.nn.functional as F


def _bce(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def vrex_penalty(logits, targets, envs, weight=1.0):
    risks = []
    for e in torch.unique(envs):
        sel = envs == e
        if sel.sum() == 0:
            continue
        risks.append(_bce(logits[sel], targets[sel]))
    if len(risks) <= 1:
        return logits.new_zeros(())
    return weight * torch.stack(risks).var(unbiased=False)
