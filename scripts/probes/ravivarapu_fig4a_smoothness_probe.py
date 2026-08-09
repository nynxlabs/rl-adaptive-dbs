#!/usr/bin/env python3
"""Fig 4a smoothness exploration — display-only rolling means vs paper digitization.

Loads a cached ``series.json`` (default: ship v42), plots raw per-episode PSD plus
optional rolling means. **Does not affect gates or ship PNGs** — diagnostic only.

Example::

  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig4a_smoothness_probe.py
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig4a_smoothness_probe.py \\
    --windows 5 10 15 --series artifacts/figures/papers/ravivarapu/4/series.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIG_PATH = ROOT / "artifacts/figures/papers/ravivarapu/paper_digitization/curves_fig4a.json"
DEFAULT_SERIES = ROOT / "artifacts/figures/papers/ravivarapu/4/series.json"
OUT_DIR = ROOT / "artifacts/probes"
PAPER_LABELS = {"baseline": "Baseline", "paper": "SEA-DBS"}


def _paper_curve(label: str, n: int) -> np.ndarray:
    dig = json.loads(DIG_PATH.read_text(encoding="utf-8"))
    xy = dig["series"][label]["xy"]
    xs = np.asarray(xy["x"], dtype=float)
    ys = np.asarray(xy["y"], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    eps = np.arange(1, n + 1, dtype=float)
    return np.interp(eps, xs, ys)


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(np.asarray(y, dtype=float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def _roughness_stats(y: np.ndarray) -> dict[str, float]:
    a = np.asarray(y, dtype=float)
    d = np.diff(a)
    return {
        "std": float(np.std(a)),
        "diff_std": float(np.std(d)),
        "max_abs_diff": float(np.max(np.abs(d))) if len(d) else float("nan"),
        "large_jumps_gt_0_03": int(np.sum(np.abs(d) > 0.03)),
    }


def _plot_overlay(
    series: dict[str, Any],
    *,
    windows: list[int],
    out_png: Path,
) -> None:
    n = min(len(series["variants"]["baseline"]["episode_psd"]), len(series["variants"]["paper"]["episode_psd"]))
    eps = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, variant, title in zip(axes, ("baseline", "paper"), ("Baseline (DDPG)", "SEA-DBS"), strict=True):
        raw = np.asarray(series["variants"][variant]["episode_psd"][:n], dtype=float)
        paper = _paper_curve(PAPER_LABELS[variant], n)
        ax.plot(eps, raw, color="C0", alpha=0.35, lw=1.0, label="raw (ship metric)")
        for i, w in enumerate(windows):
            sm = _rolling_mean(raw, w)
            ax.plot(eps, sm, lw=1.8, label=f"rolling mean w={w}")
        ax.plot(eps, paper, color="k", ls="--", lw=1.2, alpha=0.75, label="paper dig")
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("Mean beta PSD (norm)")
    seed = series.get("seed", "?")
    fig.suptitle(f"Fig 4a smoothness probe (seed {seed}) — display only, gates use raw")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 15])
    args = parser.parse_args()

    series = json.loads(args.series.read_text(encoding="utf-8"))
    n = min(len(series["variants"]["baseline"]["episode_psd"]), len(series["variants"]["paper"]["episode_psd"]))
    paper_b = _paper_curve("Baseline", n)
    paper_s = _paper_curve("SEA-DBS", n)
    raw_b = np.asarray(series["variants"]["baseline"]["episode_psd"][:n], float)
    raw_s = np.asarray(series["variants"]["paper"]["episode_psd"][:n], float)

    report: dict[str, Any] = {
        "series": str(args.series),
        "seed": series.get("seed"),
        "n_episodes": n,
        "windows": args.windows,
        "roughness": {
            "raw_baseline": _roughness_stats(raw_b),
            "raw_sea": _roughness_stats(raw_s),
            "paper_baseline": _roughness_stats(paper_b),
            "paper_sea": _roughness_stats(paper_s),
        },
        "rolling": {},
    }
    for w in args.windows:
        sm_b = _rolling_mean(raw_b, w)
        sm_s = _rolling_mean(raw_s, w)
        report["rolling"][f"w{w}"] = {
            "baseline": _roughness_stats(sm_b),
            "sea": _roughness_stats(sm_s),
        }

    out_json = OUT_DIR / "ravivarapu_fig4a_smoothness_probe.json"
    out_png = OUT_DIR / "ravivarapu_fig4a_smoothness_probe.png"
    _plot_overlay(series, windows=args.windows, out_png=out_png)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
