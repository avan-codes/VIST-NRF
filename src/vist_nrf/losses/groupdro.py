"""GroupDRO penalty: worst-environment risk.

Returns the maximum per-environment BCE risk; the outer loop minimizes this,
which corresponds to GroupDRO when group weights concentrate on the worst group.
"""

import torch
import torch.nn.functional as F


def _bce(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets.float())


def groupdro_penalty(logits, targets, envs, weight=1.0):
    risks = []
    for e in torch.unique(envs):
        sel = envs == e
        if sel.sum() == 0:
            continue
        risks.append(_bce(logits[sel], targets[sel]))
    if not risks:
        return logits.new_zeros(())
    return weight * torch.stack(risks).max()
