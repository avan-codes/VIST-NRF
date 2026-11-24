"""Centralized seeding for reproducibility.

Set Python, NumPy, and PyTorch seeds in one call. Document GPU nondeterminism
where it cannot be eliminated.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Set Python/NumPy/PyTorch seeds. Returns the seed used.

    When deterministic=True, enables torch deterministic algorithms where
    available and sets CUBLAS_WORKSPACE_CONFIG. Note: some GPU ops remain
    nondeterministic even with these flags; document that in run metadata.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older PyTorch may not support this; ignore.
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return seed
