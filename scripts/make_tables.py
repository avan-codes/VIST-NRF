"""VIST-NRF make_tables: regenerate markdown tables from saved JSONL results.

Never hand-type experimental numbers — everything comes from results/summary.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_results(summary_path="results/summary.jsonl"):
    rows = []
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def markdown_table(rows, group_cols=("train_env", "test_env", "model")):
    """Build a markdown comparison table grouped by the key columns."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c) for c in group_cols)
        groups[key].append(r)

    lines = []
    lines.append("| " + " | ".join(group_cols) + " | n_seeds | AUROC (mean) | AUROC (95% CI) | AUPRC (mean) | Brier (mean) | ECE (mean) | prevalence |")
    lines.append("| " + " | ".join(["---"] * (len(group_cols) + 8)) + " |")
    import numpy as np
    for key in sorted(groups):
        rs = groups[key]
        aurocs = [r["auroc"] for r in rs if r.get("auroc") == r["auroc"]]
        auprcs = [r["auprc"] for r in rs if r.get("auprc") == r["auprc"]]
        bri = [r["brier"] for r in rs if r.get("brier") == r["brier"]]
        ece = [r["ece"] for r in rs if r.get("ece") == r["ece"]]
        prev = [r["prevalence"] for r in rs]
        def mean(x): return float(np.mean(x)) if x else float("nan")
        def ci(x):
            if not x:
                return "(nan, nan)"
            lo = float(np.percentile(x, 2.5)); hi = float(np.percentile(x, 97.5))
            return f"({lo:.3f}, {hi:.3f})"
        lines.append("| " + " | ".join(str(k) for k in key) +
                     f" | {len(rs)} | {mean(aurocs):.3f} | {ci(aurocs)} | {mean(auprcs):.3f} | {mean(bri):.3f} | {mean(ece):.3f} | {mean(prev):.3f} |")
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.jsonl")
    ap.add_argument("--out", default="results/table_comparison.md")
    args = ap.parse_args()
    rows = load_results(args.summary)
    if not rows:
        print("No results found.")
        return
    table = markdown_table(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table)
    print(table)
    print(f"\nSaved table to {args.out}")


if __name__ == "__main__":
    main()
