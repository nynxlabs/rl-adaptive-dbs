"""Plot Mehregan Fig 3 screen runs: Fig 4a training traces vs the digitized paper.

Left: per-step training $P_\\beta/1000$ (10-step running mean) of each 45 Hz
screen JSON, with the refined Fig 4a digitization. Right: spread (std) of the
noise-free actor's logits at a reference state after each episode, log scale.
Diagnostic only.

  uv run python -m rl_adaptive_dbs.run scripts/probes/mehregan_fig3_screen_plot.py \
      artifacts/probes/mehregan_fig3/screen/r1-*.json --out artifacts/probes/mehregan_fig3/screen_v1.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "digitization"))
from paper_gates import load_refined, refined_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = [(p.stem, json.loads(p.read_text())) for p in args.runs]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=150)
    paper = load_refined(refined_path("4a"))
    px, py = paper["training" if "training" in paper else next(iter(paper))]
    ax.plot(px, py, "k--", linewidth=2.0, label="Paper (digitized)", zorder=5)
    for name, d in runs:
        if d["mean_hz"] == 45.0:
            y = np.asarray(d["beta_trace"], dtype=float)
            ax.plot(np.convolve(y, np.ones(10) / 10, mode="valid"), linewidth=1.0, label=name)
        std = [r["0.50"]["logit_std"] for r in d["actor_diag"]]
        ax2.semilogy(range(-1, len(std) - 1), std, marker="o", markersize=3, linewidth=1.0,
                     label=f"{name} ({d['mean_hz']:g} Hz)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("PSD (x10³), 10-step running mean")
    ax.set_title("Fig 4a training trace, 45 Hz, seed 0")
    ax.legend(fontsize=6, loc="lower left")
    ax2.set_xlabel("After episode (−1 = init)")
    ax2.set_ylabel("Logit std at state 0.50")
    ax2.set_title("Noise-free actor logit spread")
    ax2.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
