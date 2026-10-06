#!/usr/bin/env python3
"""Mehregan et al.  Figure 4a — training GPi beta power vs step.

Paper §IV.A.1: DDPG on the **45 Hz** fixed-mean pattern alphabet, 10 episodes ×
30 steps (300 env steps). Y-axis **PSD(x10³)** = raw $P_\\beta$ / 1000.

This run *is* the paper's 45 Hz model: the same training feeds Fig 4b (episode
view), Fig 5a (efficacy eval) and Fig 6a (fp32 + PTQ). Training follows Alg. 1
as written via ``mehregan/paper_protocol.py``; only paper-silent knobs are
settable (``--set FIELD=VALUE``: alphabet, jitter_fraction, logit_noise_std,
init_bias_scale, gamma, tau).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/4a/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/4a/plot.py --plot-only

Writes ``figures/mehregan/images/4a/training_beta_vN.png`` (N auto-increments),
``artifacts/figures/papers/mehregan/4a/series.json`` and ``checkpoint.pt``
(fp32 actor for Fig 5a / 6a).
"""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import sys
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig4a_gates  # noqa: E402

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_PP = load_script_module(
    "mehregan_paper_protocol", Path(__file__).resolve().parents[1] / "paper_protocol.py"
)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import = load_script_module("figure_overlay_import", _OVERLAY_IMPORT)
_paper_overlay = _overlay_import.load_paper_overlay()

FIGURES_DIR = Path("figures/mehregan/images/4a")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/4a")
DEFAULT_SERIES = CACHE_DIR / "series.json"
DEFAULT_CHECKPOINT = CACHE_DIR / "checkpoint.pt"
OUT_STEM = "training_beta"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"

MEAN_HZ = 45.0
NUM_EPISODES = _PP.NUM_EPISODES
STEPS_PER_EPISODE = _PP.STEPS_PER_EPISODE
DEFAULT_SEED = 0
EARLY_END = 130
LATE_START = 150
WINDOW = 30

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
}


def _linked_png(path: Path) -> Path:
    """Ensure ``path`` is a symlink into the linked figure dir when ``paper.png`` is.

    Fig assets under ``figures/<paper>/images/`` may be symlinks into an external notes folder. New
    versioned PNGs must follow the same pattern so the main checkout and docs see them.
    """
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    linked_dir = paper.resolve().parent
    linked_target = linked_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    # Create/overwrite the linked file on savefig through this symlink.
    if not linked_target.exists():
        linked_target.touch()
    path.symlink_to(linked_target)
    return path


def _window_mean(trace: list[float], start: int, end: int) -> float:
    chunk = np.asarray(trace[start:end], dtype=float)
    if chunk.size == 0:
        return float("nan")
    return float(chunk.mean())


def _gate_summary(beta_trace: list[float]) -> dict[str, Any]:
    """Digitization-anchored gates (paper early/late x windows + drop ratio).

    Absolute early/late bands are intentionally dropped: seed changes level;
    paper is one realization. Require trend down and a drop that tracks the
    digitized paper drop / late-to-early ratio.
    """
    n = len(beta_trace)
    early = _window_mean(beta_trace, 0, min(EARLY_END, n))
    late = _window_mean(beta_trace, min(LATE_START, n), n)
    start_w = _window_mean(beta_trace, 0, min(WINDOW, n))
    end_w = _window_mean(beta_trace, max(0, n - WINDOW), n)
    mid = _window_mean(beta_trace, min(120, n), min(150, n))
    dig = fig4a_gates(beta_trace, n_expected=NUM_EPISODES * STEPS_PER_EPISODE)
    gates = dict(dig["gates"])
    return {
        "n_steps": n,
        "early_mean_0_130": early,
        "late_mean_150_end": late,
        "start_window_mean": start_w,
        "end_window_mean": end_w,
        "delta_end_minus_start": end_w - start_w,
        "mid_mean_120_150": mid,
        "trend_down": gates.get("overall_trend_down"),
        "paper_gate_metrics": dig["metrics"],
        "paper_ref": dig["paper_ref"],
        "gates": gates,
        "gates_pass": all(gates.values()),
    }


def _ylim_for_trace(y: np.ndarray) -> tuple[float, float, list[float]]:
    """Y limits that include every sample (paper panel uses ~0.3–0.6; we extend if needed)."""
    if y.size == 0 or not np.isfinite(y).any():
        return 0.3, 0.6, [0.3, 0.4, 0.5, 0.6]
    pad = 0.02
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    # Keep the paper's usual window when possible, but never clip the trace.
    lo = min(0.3, y_min - pad)
    hi = max(0.6, y_max + pad)
    lo = float(np.floor(lo * 20.0) / 20.0)  # 0.05 grid
    hi = float(np.ceil(hi * 20.0) / 20.0)
    if hi <= lo:
        hi = lo + 0.3
    ticks = [float(t) for t in np.arange(lo, hi + 1e-9, 0.1)]
    if not ticks or ticks[-1] < hi - 1e-9:
        ticks.append(hi)
    return lo, hi, ticks


def plot_fig4a(cache: dict[str, Any], *, out_path: Path) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    y = np.asarray(cache["beta_norm_trace"], dtype=float)
    y_plot = y
    x = np.arange(y.size)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(x, y_plot, color="#1f6f6f", linewidth=1.1, label="Replication")
    paper_y = _paper_overlay.overlay_mehregan_fig4a(ax)
    paper_vals = np.concatenate([arr[0] for arr in paper_y.values() if arr[0].size]) if paper_y else np.array([])
    y_combined = np.concatenate([y_plot, paper_vals]) if paper_vals.size else y_plot
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    y0, y1, yticks = _ylim_for_trace(y_combined)
    ax.set_ylim(y0, y1)
    ax.set_yticks(yticks)
    ax.set_xlabel("Steps")
    ax.set_ylabel(r"PSD($x10^3$)")
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    fig.tight_layout()
    _paper_overlay.place_legend(ax, loc="upper right", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "n_steps": int(y.size),
        "y_min": float(y_plot.min()) if y_plot.size else float("nan"),
        "y_max": float(y_plot.max()) if y_plot.size else float("nan"),
        "y_mean": float(y_plot.mean()) if y_plot.size else float("nan"),
        "ylim": [y0, y1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="fp32 actor written after training (Fig 5a / 6a eval source)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output PNG (default: next {FIGURES_DIR.as_posix()}/{OUT_STEM}_vN.png)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plot-only", action="store_true", help="Re-plot from --series")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Paper-silent knob override (repeatable)",
    )
    parser.add_argument(
        "--no-update-docs",
        dest="update_docs",
        action="store_false",
        help="Skip the tracker caption / image link refresh",
    )
    parser.set_defaults(update_docs=True)
    _resume_cli.add_export_notes_arg(parser)
    _resume_cli.add_update_report3_arg(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
        args.out = _linked_png(args.out)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
        args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        if not args.series.exists():
            print(f"missing series cache: {args.series}", file=sys.stderr)
            return 2
        cache = json.loads(args.series.read_text())
        print(f"loaded traces from {args.series}", flush=True)
    else:
        knobs = _PP.apply_overrides(_PP.PaperKnobs(), args.overrides)
        print(
            f"Fig 4a train — paper Alg. 1, {NUM_EPISODES} ep x {STEPS_PER_EPISODE}, "
            f"{MEAN_HZ:g} Hz, seed {args.seed}, knobs {knobs.as_dict()}",
            flush=True,
        )
        t0 = time.time()
        record = _PP.train_paper(
            mean_hz=MEAN_HZ,
            seed=args.seed,
            knobs=knobs,
            checkpoint_path=args.checkpoint,
            extra={"figure": "mehregan_fig4a", "png_version": png_version},
        )
        elapsed = time.time() - t0
        versioned_ckpt = args.checkpoint.with_name(
            f"{args.checkpoint.stem}_v{png_version}{args.checkpoint.suffix}"
        )
        shutil.copyfile(args.checkpoint, versioned_ckpt)
        cache = {
            "figure": "mehregan_fig4a",
            "seed": args.seed,
            "num_episodes": NUM_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "mean_hz": MEAN_HZ,
            "state_length": _PP.STATE_LENGTH,
            "plant_dt_ms": _PP.PAPER_DT_MS,
            "protocol": "paper_alg1",
            "knobs": record["knobs"],
            "paper_stated": record["paper_stated"],
            "elapsed_s": elapsed,
            "beta_norm_trace": record["beta_trace"],
            "actions": record["actions"],
            "step_rewards": record["step_rewards"],
            "episode_rewards": record["episode_rewards"],
            "reset_beta": record["reset_beta"],
            "training": {
                "episode_rewards": record["episode_rewards"],
                "unique_actions": record["unique_actions"],
                "action_counts": record["action_counts"],
            },
            "checkpoint": str(args.checkpoint),
            "checkpoint_versioned": str(versioned_ckpt),
        }
        args.series.parent.mkdir(parents=True, exist_ok=True)
        args.series.write_text(json.dumps(cache, indent=2) + "\n")
        versioned_series = args.series.with_name(f"{args.series.stem}_v{png_version}.json")
        versioned_series.write_text(json.dumps(cache, indent=2) + "\n")
        print(f"wrote {args.series} and {versioned_series} ({elapsed:.0f}s)", flush=True)

    print(f"output PNG: {args.out} (version={png_version})", flush=True)
    panel = plot_fig4a(cache, out_path=args.out)
    print(f"wrote {args.out}", flush=True)

    summary = _gate_summary(cache["beta_norm_trace"])
    manifest = {
        "figure": "mehregan_fig4a",
        "seed": cache.get("seed", args.seed),
        "num_episodes": cache.get("num_episodes", NUM_EPISODES),
        "steps_per_episode": cache.get("steps_per_episode", STEPS_PER_EPISODE),
        "mean_hz": cache.get("mean_hz", MEAN_HZ),
        "state_length": cache.get("state_length", _PP.STATE_LENGTH),
        "plant_dt_ms": cache.get("plant_dt_ms", _PP.PAPER_DT_MS),
        "protocol": cache.get("protocol"),
        "knobs": cache.get("knobs"),
        "paper_stated": cache.get("paper_stated"),
        "elapsed_s": cache.get("elapsed_s"),
        "png_version": png_version,
        "summary": summary,
        "training": cache.get("training"),
        "panel": panel,
        "output_png": str(args.out),
        "series": str(args.series),
        "checkpoint": cache.get("checkpoint"),
    }
    print(
        f"gates: {summary['gates']} gates_pass={summary['gates_pass']}",
        flush=True,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = _panel.stamp_manifest(manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.manifest}", flush=True)

    if args.update_docs:
        updated = _figure_promote.promote_4a(
            manifest=manifest,
            series_path=args.series,
            png_path=args.out,
            update_docs=True,
        )
        print(f"updated docs caption: {updated.get('caption')}", flush=True)

    return _panel.exit_code(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
