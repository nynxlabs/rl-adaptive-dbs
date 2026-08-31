#!/usr/bin/env python3
"""Ravivarapu Fig 7 — ablation PSD over 10 stimulation steps."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, SEADBSConfig
from rl_adaptive_dbs.parallel_workers import TrainSeedJob, train_seed_worker

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_spec = importlib.util.spec_from_file_location("figure_resume_cli", _RESUME_CLI)
assert _resume_spec and _resume_spec.loader
_resume_cli = importlib.util.module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(_resume_cli)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_spec = importlib.util.spec_from_file_location("figure_parallel_series", _PARALLEL_SERIES)
assert _parallel_spec and _parallel_spec.loader
_parallel_series = importlib.util.module_from_spec(_parallel_spec)
_parallel_spec.loader.exec_module(_parallel_series)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_fig7_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/7")
FIGURES_DIR = Path("figures/ravivarapu/images/7")
OUT_STEM = "ablation_psd"
VARIANTS = ("baseline", "baseline-pm", "baseline-gs", "paper")
LABELS = {
    "baseline": "Baseline",
    "baseline-pm": "Baseline+PM",
    "baseline-gs": "Baseline+GS",
    "paper": "SEA-DBS",
}


def ensure_checkpoints(
    seed: int,
    *,
    smoke: bool,
    resume: bool = False,
    checkpoint_interval: int = 50,
    parallel_series: int = 0,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[TrainSeedJob] = []
    for variant in VARIANTS:
        ckpt = CACHE_DIR / f"{variant}_train{seed}.pt"
        if ckpt.is_file() and not resume:
            continue
        cfg = SEADBSConfig(variant=variant, seed=seed, log_episodes=True)
        if smoke:
            cfg = cfg.for_smoke(episodes=2, max_steps=5)
        else:
            cfg = replace(cfg, num_episodes=150)
        jobs.append(
            TrainSeedJob(
                controller="sea_dbs",
                variant=variant,
                seed=seed,
                episodes=cfg.num_episodes if not smoke else cfg.num_episodes,
                checkpoint_dir=CACHE_DIR,
                smoke=smoke,
                resume_path=ckpt if ckpt.is_file() and resume else None,
                checkpoint_interval=checkpoint_interval,
            )
        )
    if not jobs:
        return
    _parallel_series.run_series_parallel(jobs, train_seed_worker, parallel_series)


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_fig7_gates(traces)
    n = min(len(v) for v in traces.values()) if traces else 0
    return merge_gate_report(dig, {"n_steps": n})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Resume training from existing per-variant checkpoints in cache",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=_resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N episodes (default {_resume_cli.DEFAULT_CHECKPOINT_INTERVAL})",
    )
    _parallel_series.add_parallel_series_argument(parser)
    args = parser.parse_args()
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    if not args.plot_only:
        ensure_checkpoints(
            args.seed,
            smoke=args.smoke,
            resume=bool(args.retrain),
            checkpoint_interval=args.checkpoint_interval,
            parallel_series=args.parallel_series,
        )

    eval_jobs = [
        _parallel_series.RavivarapuAblationEvalJob(
            variant=variant,
            seed=args.seed,
            checkpoint=str(CACHE_DIR / f"{variant}_train{args.seed}.pt"),
            n_steps=steps,
        )
        for variant in VARIANTS
    ]
    eval_results = _parallel_series.run_series_parallel(
        eval_jobs,
        _parallel_series.ravivarapu_ablation_eval_worker,
        args.parallel_series,
    )
    traces = {variant: trace for variant, trace in eval_results}

    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in VARIANTS:
        y = np.asarray(traces[variant], dtype=float)
        ax.plot(np.arange(y.size, dtype=float), y, label=LABELS[variant], linewidth=1.5)
    paper = _paper_overlay.overlay_ravivarapu_fig7(ax)
    ys = [np.asarray(v, dtype=float) for v in traces.values()]
    ys.extend(v[0] for v in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([0.0, 1.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Stimulation step")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=8)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    gates = {"pass": True, "smoke_override": True} if args.smoke else evaluate_gates(traces)
    manifest = {
        "panel": "7",
        "variants": list(VARIANTS),
        "n_steps": steps,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
