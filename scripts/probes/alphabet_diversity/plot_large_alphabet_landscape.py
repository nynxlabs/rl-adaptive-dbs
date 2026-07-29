#!/usr/bin/env python3
"""Plot open-loop P_beta landscape for large burst alphabets (from diversity JSON).

Reads ``artifacts/ddpg/alphabet_diversity_large_n.json`` and writes a side-by-side
45 Hz skip_regular ranking for burst n=41 / 128 / 256 — no training, no PTQ nudges.

  uv run python -m rl_adaptive_dbs.run \\
    scripts/probes/alphabet_diversity/plot_large_alphabet_landscape.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN_JSON = Path("artifacts/ddpg/alphabet_diversity_large_n.json")
OUT_DIR = Path("figures/mehregan/images/6a")
OUT_STEM = "large_alphabet_landscape"


def _rows_for(key: str, mean_hz: float, payload: dict) -> dict:
    for run in payload["runs"]:
        if run["key"] == key and float(run["mean_hz"]) == mean_hz:
            return run
    raise KeyError(f"missing run key={key!r} mean_hz={mean_hz}")


def main() -> int:
    payload = json.loads(IN_JSON.read_text())
    keys = ["burst", "burst_n128", "burst_n256"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), dpi=150, sharey=True)
    no_stim = None
    for ax, key in zip(axes, keys, strict=True):
        run = _rows_for(key, 45.0, payload)
        no_stim = float(run["no_stim_p_beta"])
        rows = [r for r in run["one_step"] if r["action"] != 0 and np.isfinite(r["p_beta_raw"])]
        xs = np.array([r["action"] for r in rows], dtype=float)
        ys = np.array([r["p_beta_raw"] for r in rows], dtype=float)
        order = np.argsort(ys)
        ax.scatter(xs, ys, s=10, c="#4c78a8", alpha=0.85, label="irregular")
        ax.axhline(no_stim, color="#333333", linestyle="--", linewidth=1.0, label="no stim")
        # highlight near-best (within 2%)
        best = float(ys.min())
        near = ys <= best * 1.02
        ax.scatter(xs[near], ys[near], s=28, c="#2ca02c", zorder=3, label="near-best (±2%)")
        skip = run["skip_regular"]
        ax.set_title(
            f"{key}\nn={run['n_actions']} uniq={run.get('n_unique_traces')} "
            f"near={skip['n_near_best']} ok={skip['diversity_ok']}",
            fontsize=9,
        )
        ax.set_xlabel("pattern index")
        ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    axes[0].set_ylabel(r"$P_\beta$ (1-step)")
    axes[0].legend(loc="upper right", fontsize=7, framealpha=0.95)
    fig.suptitle("45 Hz skip_regular open-loop landscape (large alphabet probe)", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # version manually: overwrite stem with _preview for vault eyeball
    out = OUT_DIR / f"{OUT_STEM}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    # also land in vault if repo path is symlink dir of files
    print(f"wrote {out}", flush=True)
    print(f"no_stim Pβ={no_stim:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
