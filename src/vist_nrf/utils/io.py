"""I/O helpers: atomic JSONL append, config/split hashing, environment metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    """Append one JSON record per line. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_json(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)


def environment_metadata() -> Dict[str, Any]:
    """Capture device, package versions, timestamps for reproducibility."""
    meta: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": None,
        "deterministic_algorithms": None,
    }
    if torch.cuda.is_available():
        meta["cuda_device_name"] = torch.cuda.get_device_name(0)
    try:
        meta["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception:
        meta["deterministic_algorithms"] = None
    return meta


def git_commit(repo_root: str | Path) -> Optional[str]:
    """Return short git commit hash, or None if not a repo / unavailable."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
