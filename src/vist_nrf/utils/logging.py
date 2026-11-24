"""Small structured logger: writes run metadata + results as JSONL, never secrets."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_SENSITIVE_KEYS = {"token", "password", "secret", "github_token", "access_key"}


def _scrub(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in record.items():
        kl = str(k).lower()
        if any(s in kl for s in _SENSITIVE_KEYS):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_run(results_dir: str | Path, run_id: str, record: Dict[str, Any]) -> None:
    """Append a run record as JSONL under results_dir/runs/<run_id>.jsonl."""
    from vist_nrf.utils.io import ensure_dir, save_jsonl

    p = Path(results_dir) / "runs" / f"{run_id}.jsonl"
    ensure_dir(p.parent)
    entry = {"run_id": run_id, "timestamp_utc": utc_now(), **record}
    save_jsonl(p, _scrub(entry))


def log_run(results_dir: str | Path, run_id: str, **kwargs: Any) -> None:
    append_run(results_dir, run_id, kwargs)
