"""IRM (Invariant Risk Minimization) penalty.

IRMv1 regularizer: gradient of loss wrt a dummy classifier weight should be zero
within each environment. Added to base BCE with a weight.
"""

import torch
import torch.nn.functional as F


def _bce(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def irm_penalty(logits, targets, envs, weight=1.0):
    device = logits.device
    total_penalty = logits.new_zeros(())
    n_envs = 0
    for e in torch.unique(envs):
        sel = envs == e
        if sel.sum() == 0:
            continue
        scale = torch.tensor([1.0], device=device, requires_grad=True)
        loss_e = _bce(logits[sel] * scale, targets[sel])
        grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
        total_penalty = total_penalty + torch.sum(grad ** 2)
        n_envs += 1
    if n_envs == 0:
        return logits.new_zeros(())
    return weight * total_penalty / n_envs
