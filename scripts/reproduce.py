"""VIST-NRF reproduce: clean re-run of a recorded experiment.

Reads a summary.jsonl record (run_id, config, seed, commit) and re-runs
scripts/train.py at the recorded commit to verify reproducibility.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vist_nrf.utils.prereg import PreregistrationError, enforce_prereg, prereg_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", help="results/summary.jsonl path")
    ap.add_argument("run_id", help="run_id to reproduce")
    ap.add_argument("--exploratory", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    expected = None if args.exploratory else prereg_hash(repo_root)
    if expected is None and not args.exploratory:
        raise PreregistrationError("preregistration.md not found.")
    try:
        enforce_prereg(repo_root, expected_hash=expected, allow_exploratory=args.exploratory)
    except PreregistrationError as e:
        print(f"PREREGISTRATION BLOCK: {e}")
        raise

    record = None
    with open(args.summary) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("run_id") == args.run_id:
                record = rec
                break
    if record is None:
        raise SystemExit(f"run_id {args.run_id} not found in {args.summary}")

    print(f"Reproducing {args.run_id} (commit {record.get('commit')}, config {record.get('config')})")
    cmd = ["python", "scripts/train.py", record["config"]]
    if args.exploratory:
        cmd.append("--exploratory")
    subprocess.run(cmd, check=True, cwd=repo_root)
    print("Reproduce complete.")


if __name__ == "__main__":
    main()
