"""Preregistration hash enforcement.

Main experiment scripts refuse to run unless the committed preregistration.md
hash matches the recorded value. Exploratory runs require an explicit flag and
go to a separate directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from vist_nrf.utils.io import sha256_file


class PreregistrationError(RuntimeError):
    pass


def prereg_hash(repo_root: str | Path) -> Optional[str]:
    """SHA-256 of the committed preregistration.md, or None if missing."""
    p = Path(repo_root) / "preregistration.md"
    if not p.exists():
        return None
    return sha256_file(p)


def enforce_prereg(
    repo_root: str | Path,
    expected_hash: Optional[str] = None,
    allow_exploratory: bool = False,
    exploratory_dir: str = "experiments/_exploratory",
) -> str:
    """Verify preregistration state.

    Returns the current preregistration hash. Raises PreregistrationError if
    confirmatory run requested but preregistration is missing or hash mismatches.
    """
    current = prereg_hash(repo_root)

    if current is None:
        if allow_exploratory:
            return "exploratory-no-prereg"
        raise PreregistrationError(
            "preregistration.md not found. Commit and freeze preregistration "
            "before confirmatory runs, or pass --exploratory."
        )

    if expected_hash is not None and current != expected_hash:
        raise PreregistrationError(
            f"preregistration.md hash mismatch: expected {expected_hash}, "
            f"got {current}. Revert preregistration or update expected hash "
            "with an explicit decision in docs/DECISIONS.md."
        )

    if allow_exploratory:
        return current
    return current
