#!/usr/bin/env python3
"""Ravivarapu Fig 7 — ablation PSD over 10 stimulation steps."""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
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
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_series = load_script_module("figure_parallel_series", _PARALLEL_SERIES)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_fig7_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import = load_script_module("figure_overlay_import", _OVERLAY_IMPORT)
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


def _ckpt(variant: str, seed: int) -> Path:
    if variant in ("baseline", "paper"):
        p4 = Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{seed}.pt"
        if p4.is_file():
            return p4
    p7 = CACHE_DIR / f"{variant}_train{seed}.pt"
    if p7.is_file():
        return p7
    return Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{seed}.pt"


def _linked_png(path: Path) -> Path:
    path = Path(path)
    roots: list[Path] = []
    main_root = getattr(_figure_promote, "REPO_ROOT", None)
    if isinstance(main_root, Path):
        roots.append(main_root)
    roots.append(Path.cwd())
    for root in roots:
        paper = root / path.parent / "paper.png"
        if not paper.is_symlink():
            continue
        linked_dir = paper.resolve().parent
        linked_target = linked_dir / path.name
        local = path if path.is_absolute() else Path.cwd() / path
        local.parent.mkdir(parents=True, exist_ok=True)
        if not linked_target.exists():
            linked_target.parent.mkdir(parents=True, exist_ok=True)
            linked_target.touch()
        if local.exists() or local.is_symlink():
            if local.resolve() != linked_target.resolve():
                return linked_target
            return local
        try:
            local.symlink_to(linked_target)
        except OSError:
            return linked_target
        return local
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def main() -> int:
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
    _resume_cli.add_export_notes_arg(parser)
    _resume_cli.add_update_report3_arg(parser)
    _parallel_series.add_parallel_series_argument(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    series_path = CACHE_DIR / "series.json"
    if args.plot_only:
        if not series_path.is_file():
            raise SystemExit(f"missing series cache: {series_path}")
        payload = json.loads(series_path.read_text(encoding="utf-8"))
        traces = payload["traces"]
        steps = int(payload.get("steps", len(next(iter(traces.values()))) - 1))
    else:
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
                checkpoint=str(_ckpt(variant, args.seed)),
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
    scale = float(_paper_overlay.RAVI_INFERENCE_PAPER_Y_TO_NORM)
    ys: list[np.ndarray] = []
    for variant in VARIANTS:
        y = np.asarray(traces[variant], dtype=float) * scale
        ax.plot(np.arange(y.size, dtype=float), y, label=LABELS[variant], linewidth=1.5)
        ys.append(y)
    paper = _paper_overlay.overlay_ravivarapu_fig7(ax)
    ys.extend(py for _px, py in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([300.0, 480.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_xlim(0, 10)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD")
    ax.set_title("Ablation study (50 Hz)")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=8)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(_linked_png(png_path), dpi=150)
    plt.close(fig)
    gates = {"pass": False, "smoke_override": True} if args.smoke else evaluate_gates(traces)
    if not args.plot_only:
        series_path.write_text(
            json.dumps(
                {
                    "traces": traces,
                    "steps": steps,
                    "n_psd_samples": steps + 1,
                },
                indent=2,
            )
            + "\n"
        )
    caption = (
        f"Ablation study GPi beta PSD over 10 steps (seed {args.seed}); pass={gates.get('pass')}; "
        f"Baseline vs +PM vs +GS vs SEA-DBS."
    )
    manifest = {
        "panel": "7",
        "variants": list(VARIANTS),
        "n_steps": steps,
        "n_psd_samples": steps + 1,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "caption": caption,
        "series_cache": series_path.as_posix(),
    }
    manifest = _panel.stamp_manifest(manifest)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if hasattr(_figure_promote, "promote_ravivarapu_7"):
        _figure_promote.promote_ravivarapu_7(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))

if __name__ == "__main__":
    raise SystemExit(main())
