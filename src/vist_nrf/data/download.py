"""Download/manifest helpers for PhysioNet/CinC Challenge 2019.

Actual download requires PhysioNet credentials and is performed by the
researcher (spec section 4.1), e.g.:

  aws s3 sync --no-sign-request \
    s3://physionet-open/challenge-2019/1.0.0/training/ \
    "$VIST_NRF_DATA/raw/physionet2019/"

This module audits the downloaded tree and writes a download manifest so later
runs use the same data snapshot. It does NOT download automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import kagglehub

from vist_nrf.data.physionet2019 import audit_psv, list_psv_files, infer_hospital_system


@dataclass
class DownloadManifest:
    source_url: str
    access_date_utc: str
    license: str
    root: str
    n_files_a: int
    n_files_b: int
    n_files_other: int
    total_size_bytes: int
    file_audits: list[dict]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_physionet2019(
    dataset_slug: str = "farjanayesmin/the-physionet-challenge-2019-dataset",
    dest_root: str | Path | None = None,
) -> str:
    """Download the PhysioNet/CinC Challenge 2019 dataset via KaggleHub.

    Returns the local directory path KaggleHub downloaded the dataset to.
    KaggleHub caches the download, so repeated calls are fast once cached.

    Requires: pip install kagglehub  and a Kaggle account with the dataset
    accepted (kagglehub handles auth via ~/.kaggle/kaggle.json or the
    KAGGLE_USERNAME/KAGGLE_KEY env vars).
    """
    path = kagglehub.dataset_download(dataset_slug)
    root = Path(path)
    if not root.exists():
        raise RuntimeError(f"KaggleHub download did not produce a path: {root}")
    dest = Path(dest_root) if dest_root else root
    if dest != root and dest.exists():
        raise FileExistsError(f"dest_root already exists: {dest}")
    # KaggleHub already placed files at `path`; just return it (caller may
    # symlink/copy into dest_root if they want a canonical location).
    return str(root)


def audit_tree(root: str | Path, source_url: str = "UNVERIFIED") -> DownloadManifest:
    """Audit the downloaded PhysioNet tree and build a manifest.

    Counts .psv files per inferred environment (A / B / other) and records
    size + per-file audit. Expected counts: A=20336, B=20000 (from spec);
    if they differ, stop and investigate.
    """
    root = Path(root)
    files = list_psv_files(root)
    audits = [asdict(audit_psv(p)) for p in files]
    total_size = sum(a["size_bytes"] for a in audits)

    n_a = sum(1 for a in audits if a["hospital_system"] == "A")
    n_b = sum(1 for a in audits if a["hospital_system"] == "B")
    n_other = len(audits) - n_a - n_b

    return DownloadManifest(
        source_url=source_url,
        access_date_utc=utc_now(),
        license="CC BY 4.0 (UNVERIFIED against downloaded LICENSE file)",
        root=str(root),
        n_files_a=n_a,
        n_files_b=n_b,
        n_files_other=n_other,
        total_size_bytes=int(total_size),
        file_audits=audits,
    )


def verify_counts(manifest: DownloadManifest) -> list[str]:
    """Return list of problems; empty means counts match spec expectations."""
    problems = []
    expected_a, expected_b = 20336, 20000
    if manifest.n_files_a != expected_a:
        problems.append(
            f"env A file count {manifest.n_files_a} != expected {expected_a}"
        )
    if manifest.n_files_b != expected_b:
        problems.append(
            f"env B file count {manifest.n_files_b} != expected {expected_b}"
        )
    if manifest.n_files_other > 0:
        problems.append(
            f"{manifest.n_files_other} files not inferable as A/B — inspect folder names"
        )
    return problems


def save_manifest(root: str | Path, out_path: str | Path, source_url: str = "UNVERIFIED") -> DownloadManifest:
    manifest = audit_tree(root, source_url=source_url)
    manifest.save(out_path)
    problems = verify_counts(manifest)
    if problems:
        raise RuntimeError("Manifest verification failed:\n" + "\n".join(problems))
    return manifest
