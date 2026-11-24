"""VIST-NRF run_grid: run one config across multiple seeds.

Writes a per-seed temp config (seed overridden) and calls scripts/train.py for
each. Appends JSONL results per (config, seed). Enforces preregistration unless
--exploratory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vist_nrf.utils.prereg import PreregistrationError, enforce_prereg, prereg_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--exploratory", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    expected = None
    if not args.exploratory:
        expected = prereg_hash(repo_root)
        if expected is None:
            raise PreregistrationError("preregistration.md not found.")
    try:
        enforce_prereg(repo_root, expected_hash=expected, allow_exploratory=args.exploratory)
    except PreregistrationError as e:
        print(f"PREREGISTRATION BLOCK: {e}")
        raise

    with open(args.config) as f:
        base = yaml.safe_load(f)

    for seed in args.seeds:
        cfg = dict(base)
        cfg["seed"] = seed
        cfg["run_id"] = f"{base.get('run_id','run')}_seed{seed}"
        tmp = Path(tempfile.mkdtemp()) / "grid_config.yaml"
        with open(tmp, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"=== seed {seed} -> {tmp} ===")
        cmd = ["python", "scripts/train.py", str(tmp)]
        if args.exploratory:
            cmd.append("--exploratory")
        subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
