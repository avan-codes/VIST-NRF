"""VIST-NRF make_figures: regenerate figures from saved JSONL results.

Never hand-type experimental numbers — everything comes from results/summary.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys_path_inserted = False


def load_results(summary_path="results/summary.jsonl"):
    rows = []
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig_auroc_by_env(rows, out="results/fig_auroc_by_env.png"):
    """AUROC by test environment and model, with per-seed points."""
    import itertools
    models = sorted({r["model"] for r in rows})
    test_envs = sorted({r["test_env"] for r in rows})
    x = np.arange(len(test_envs))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        means, lo, hi = [], [], []
        for env in test_envs:
            vals = [r["auroc"] for r in rows if r["test_env"] == env and r["model"] == model
                    and r.get("auroc") == r["auroc"]]
            if vals:
                means.append(np.mean(vals))
                lo.append(np.percentile(vals, 2.5))
                hi.append(np.percentile(vals, 97.5))
            else:
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        ax.errorbar(x + i * width, means,
                    yerr=[np.array(means) - np.array(lo), np.array(hi) - np.array(means)],
                    fmt="o-", label=model, capsize=4)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(test_envs)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Test environment")
    ax.set_title("AUROC by test environment and model (per-seed mean + 95% CI)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def fig_degradation(rows, out="results/fig_degradation.png"):
    """In-environment minus shifted-environment AUROC degradation."""
    # degradation per (train_env, model, seed): in-env AUROC - cross-env AUROC
    in_env = {}
    cross = {}
    for r in rows:
        key = (r["train_env"], r["model"], r["seed"])
        if r["test_env"] == r["train_env"]:
            in_env[key] = r["auroc"]
        else:
            cross.setdefault(key, []).append(r["auroc"])
    degradations = []
    labels = []
    for key, ieval in in_env.items():
        cvals = cross.get(key, [])
        if not cvals:
            continue
        degradations.append(ieval - float(np.mean(cvals)))
        labels.append(f"{key[0]}->{list({r['test_env'] for r in rows if r['train_env']==key[0] and r['model']==key[1]})[0]} ({key[1]})")
    if not degradations:
        print("No degradation pairs found.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(degradations)), degradations)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Degradation (in-env AUROC - shifted AUROC)")
    ax.set_title("Cross-site degradation per model/seed")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.jsonl")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    rows = load_results(args.summary)
    if not rows:
        print("No results found.")
        return
    fig_auroc_by_env(rows, f"{args.outdir}/fig_auroc_by_env.png")
    fig_degradation(rows, f"{args.outdir}/fig_degradation.png")
    print("Figures done.")


if __name__ == "__main__":
    main()
