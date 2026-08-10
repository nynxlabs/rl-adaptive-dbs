#!/usr/bin/env python3
"""Nguyen et al. Figure 6 — GPi α–β and DBS parameters over training.

Reads the shared Fig. 4 train cache. Panel (a) per-episode mean α–β; (b) amplitude,
frequency, and pulse width at episode end.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/6/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/6/plot.py --refresh-train
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from nguyen_gates import attach_digitization, fig6_training_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

import matplotlib.pyplot as plt
import numpy as np

from controllers.snn.config import (
    INIT_AMPLITUDE_NA_PER_CM2,
    INIT_FREQUENCY_HZ,
    INIT_PULSE_WIDTH_MS,
    BIOMARKER_THRESHOLD,
)

FIG4_CACHE = Path("artifacts/figures/papers/nguyen/4")
FIG4_SERIES = FIG4_CACHE / "series.json"
FIG4_MANIFEST = FIG4_CACHE / "manifest.json"
FIGURES_DIR = Path("figures/nguyen/images/6")
CACHE_DIR = Path("artifacts/figures/papers/nguyen/6")
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "alpha_beta_params"
SMOOTH_WINDOW = 20

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "font.size": 10,
}


def _vault_backed_png(path: Path) -> Path:
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    vault_dir = paper.resolve().parent
    vault_target = vault_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    if not vault_target.exists():
        vault_target.touch()
    path.symlink_to(vault_target)
    return path


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    window = max(1, int(window))
    if window == 1 or y.size == 1:
        return y.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(y.astype(float), (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: y.size]


def load_series(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Fig 4 series cache: {path}")
    series = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "episode_alpha_beta_means",
        "episode_amplitudes",
        "episode_frequencies",
        "episode_pulse_widths",
    )
    n = len(series.get("episode_rewards", []))
    for key in required:
        if key not in series or len(series[key]) != n:
            msg = (
                f"series missing {key!r}; re-run Fig 4 train:\n"
                "  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py"
            )
            raise ValueError(msg)
    return series


def evaluate_gates(series: dict[str, Any], *, fig4_manifest: dict[str, Any] | None) -> dict[str, Any]:
    n = int(series.get("num_episodes", 0))
    shared_train = bool(
        fig4_manifest is not None
        and fig4_manifest.get("gates", {}).get("pass")
        and n == int(fig4_manifest.get("gates", {}).get("n_episodes", -1))
    )
    heuristic = {
        "n_episodes": n,
        "shared_train": shared_train,
        "pass": shared_train,
    }
    if not shared_train:
        heuristic["reason"] = "fig4_train_not_passing"
        return heuristic

    dig = fig6_training_gates(
        series["episode_alpha_beta_means"],
        series["episode_amplitudes"],
        series["episode_frequencies"],
        series["episode_pulse_widths"],
    )
    return attach_digitization(heuristic, dig)


def plot_series(series: dict[str, Any], out_path: Path, *, smooth_window: int) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    ab = np.asarray(series["episode_alpha_beta_means"], dtype=float)
    amp = np.asarray(series["episode_amplitudes"], dtype=float)
    freq = np.asarray(series["episode_frequencies"], dtype=float)
    pw = np.asarray(series["episode_pulse_widths"], dtype=float)
    episodes = np.arange(ab.size, dtype=float)

    ab_smooth = moving_average(ab, smooth_window)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True, constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(episodes, ab, color="#9ecae1", linewidth=0.8, alpha=0.85, label="Raw")
    ax0.plot(episodes, ab_smooth, color="#08519c", linewidth=2.0, label="Smoothed")
    ax0.axhline(BIOMARKER_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2, label="θ=150")
    ax0.set_ylabel("α–β Power")
    ax0.set_title("GPi α–β Oscillation Power")
    ax0.grid(True, linestyle="--", alpha=0.6)

    ax1 = axes[1]
    ax1.plot(episodes, amp, color="#e41a1c", linewidth=1.2, label="Amplitude")
    ax1.plot(episodes, freq, color="#377eb8", linewidth=1.2, label="Frequency")
    ax1.plot(episodes, pw, color="#4daf4a", linewidth=1.2, label="Pulse width")
    _paper_overlay.overlay_nguyen_fig6(ax0, axes[1])
    _paper_overlay.place_legend(ax0, fontsize=8)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("DBS Parameters")
    ax1.set_title("DBS Parameters")
    ax1.grid(True, linestyle="--", alpha=0.6)
    _paper_overlay.place_legend(ax1, fontsize=8, ncol=3)

    last_ep = max(0, ab.size - 1)
    for ax in axes:
        ax.set_xlim(0.0, float(last_ep))
        if last_ep >= 100:
            ax.set_xticks(np.arange(0, last_ep + 1, 100))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "n_episodes": int(ab.size),
        "alpha_beta_final": float(ab[-1]) if ab.size else float("nan"),
        "amp_final": float(amp[-1]) if amp.size else float("nan"),
        "freq_final": float(freq[-1]) if freq.size else float("nan"),
        "pw_final": float(pw[-1]) if pw.size else float("nan"),
        "init_amp": INIT_AMPLITUDE_NA_PER_CM2,
        "init_freq": INIT_FREQUENCY_HZ,
        "init_pw": INIT_PULSE_WIDTH_MS,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=FIG4_SERIES)
    parser.add_argument("--fig4-manifest", type=Path, default=FIG4_MANIFEST)
    parser.add_argument("--refresh-train", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-update-docs", action="store_true")
    args = parser.parse_args(argv)

    if args.refresh_train:
        fig4_path = Path(__file__).resolve().parent.parent / "4" / "plot.py"
        spec = importlib.util.spec_from_file_location("nguyen_fig4_plot", fig4_path)
        assert spec and spec.loader
        fig4_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fig4_mod)
        rc = int(fig4_mod.main())
        if rc != 0:
            return rc

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    t0 = time.perf_counter()
    series = load_series(args.series)
    fig4_manifest = None
    if args.fig4_manifest.is_file():
        fig4_manifest = json.loads(args.fig4_manifest.read_text(encoding="utf-8"))

    gates = evaluate_gates(series, fig4_manifest=fig4_manifest)
    panel = plot_series(series, args.out, smooth_window=args.smooth_window)

    caption = (
        f"Fig 4 shared train {series['num_episodes']} ep; "
        f"αβ_late={panel.get('alpha_beta_final', float('nan')):.1f}, "
        f"amp={panel.get('amp_final', float('nan')):.0f}; pass={gates['pass']}"
    )
    manifest = {
        "panel": "2/6",
        "out": args.out.as_posix(),
        "series": args.series.as_posix(),
        "fig4_manifest": args.fig4_manifest.as_posix(),
        "gates": gates,
        "panel_stats": panel,
        "elapsed_s": time.perf_counter() - t0,
        "png_version": png_version,
        "caption": caption,
    }
    write_json(args.manifest, manifest)

    if not args.no_update_docs:
        updated = _figure_promote.promote_nguyen_6(
            manifest=manifest,
            png_path=args.out,
        )
        print(f"updated comparison doc: {updated['doc']}", flush=True)

    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
