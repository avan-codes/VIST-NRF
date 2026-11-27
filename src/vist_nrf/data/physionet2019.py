"""PhysioNet/CinC Challenge 2019 .psv parser.

Each patient file is pipe-separated with hourly rows:
  PatientID (in filename, not a column), Hour, 40 clinical inputs, SepsisLabel.
Missing measurements are NaN. License CC BY 4.0. Download is NOT automated here
(requires PhysioNet credentials); this module only parses already-downloaded files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PhysioNetFile:
    path: Path
    patient_id: str
    hospital_system: Optional[str]  # 'A' or 'B', inferred from parent dir name
    n_rows: int
    n_cols: int
    columns: list[str]
    sepsis_positive: bool
    n_positive: int
    first_positive_hour: Optional[int]
    size_bytes: int


def parse_psv(path: str | Path) -> pd.DataFrame:
    """Read a single .psv file into a DataFrame sorted by Hour.

    Expected columns: Hour + clinical variables + SepsisLabel.
    PatientID is taken from the filename (e.g. 'p00001.psv' -> 'p00001').
    """
    p = Path(path)
    df = pd.read_csv(p, sep="|")
    # Hour is implicit (row index = hourly index), patient ID from filename.
    df.insert(0, "hour", np.arange(len(df), dtype=float))
    df.insert(0, "patient_id", p.stem)
    df["_source_file"] = str(p)
    return df


def patient_id_from_filename(path: str | Path) -> str:
    return Path(path).stem


def infer_hospital_system(path: str | Path) -> Optional[str]:
    """Infer 'A' or 'B' from the parent directory name if it contains A/B.

    Do not assume folder names; inspect the actual directory tree first
    (spec section 4.2). Returns None if not inferable.
    """
    name = Path(path).parent.name.upper()
    lower = name.lower()
    if "seta" in lower:
        return "A"
    if "setb" in lower:
        return "B"
    return None


def audit_psv(path: str | Path) -> PhysioNetFile:
    """Produce a structural audit record for one .psv file (no patient data leaked)."""
    p = Path(path)
    df = parse_psv(p)
    pid = patient_id_from_filename(p)
    label_col = "SepsisLabel" if "SepsisLabel" in df.columns else None
    n_positive = int(df[label_col].eq(1).sum()) if label_col else 0
    first_pos_hour = None
    if label_col and n_positive > 0:
        first_pos_hour = int(df.index[df[label_col] == 1][0])
    return PhysioNetFile(
        path=p,
        patient_id=pid,
        hospital_system=infer_hospital_system(p),
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=list(df.columns),
        sepsis_positive=bool(n_positive > 0),
        n_positive=n_positive,
        first_positive_hour=first_pos_hour,
        size_bytes=int(p.stat().st_size),
    )


def list_psv_files(root: str | Path) -> list[Path]:
    """Recursively list .psv files under root. Inspect; do not assume folder names."""
    return sorted(Path(root).rglob("*.psv"))


def load_environment(root: str | Path, env_label: str) -> dict[str, pd.DataFrame]:
    """Load all .psv files whose inferred hospital system == env_label.

    Returns {patient_id: DataFrame}. Raises if no files match (never silently
    use the wrong environment).
    """
    frames: dict[str, pd.DataFrame] = {}
    for p in list_psv_files(root):
        if infer_hospital_system(p) == env_label:
            df = parse_psv(p)
            frames[patient_id_from_filename(p)] = df
    if not frames:
        raise FileNotFoundError(f"No .psv files inferred as environment {env_label} under {root}")
    return frames
