#!/usr/bin/env python3
"""Ravivarapu Fig 7 — ablation PSD over 10 stimulation steps.

All four variants train with the same Fig 4 config (``fig4_ravivarapu_config``):
Baseline and SEA-DBS reuse the Fig 4 checkpoints; Baseline+PM and Baseline+GS
train here with the identical config. Eval is the Fig 5a rollout (training
plant setup, 50 Hz, untreated t=0 plus 10 argmax actions).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/7/plot.py --export-notes --update-report
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/7/plot.py --plot-only
"""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, INFERENCE_CARRIER_50HZ

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_series = load_script_module("figure_parallel_series", _PARALLEL_SERIES)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_COMMON = Path(__file__).resolve().parents[1] / "common.py"
_common = load_script_module("ravivarapu_common", _COMMON)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_fig7_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import = load_script_module("figure_overlay_import", _OVERLAY_IMPORT)
_paper_overlay = _overlay_import.load_paper_overlay()

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/7")
FIG5A_SERIES = Path("artifacts/figures/papers/ravivarapu/5a/series.json")
FIGURES_DIR = Path("docs/controllers/sea_dbs/figures/images/7")
OUT_STEM = "ablation_psd"
VARIANTS = ("baseline", "baseline-pm", "baseline-gs", "paper")
FIG4_VARIANTS = ("baseline", "paper")
LABELS = {
    "baseline": "Baseline",
    "baseline-pm": "Baseline+PM",
    "baseline-gs": "Baseline+GS",
    "paper": "SEA-DBS",
}


def _ckpt(variant: str, seed: int) -> Path:
    if variant in FIG4_VARIANTS:
        return _common.fig4_checkpoint(variant, seed)
    return CACHE_DIR / f"{variant}_train{seed}.pt"


@dataclass(frozen=True)
class _TrainJob:
    variant: str
    seed: int
    smoke: bool
    checkpoint: str
    resume: bool
    checkpoint_interval: int


def _train_worker(job: _TrainJob) -> dict[str, Any]:
    _parallel_series.bootstrap_worker_threads()
    from controllers.sea_dbs.config import fig4_ravivarapu_config
    from controllers.sea_dbs.trainer import train_sea_dbs

    cfg = fig4_ravivarapu_config(seed=job.seed, variant=job.variant)
    if job.smoke:
        cfg = cfg.for_smoke(episodes=2, max_steps=5)
    ckpt = Path(job.checkpoint)
    result = train_sea_dbs(
        config=cfg,
        checkpoint_path=ckpt,
        resume_path=ckpt if job.resume and ckpt.is_file() else None,
        checkpoint_interval=job.checkpoint_interval,
    )
    return {
        "variant": job.variant,
        "episode_psd": result.episode_psd,
        "episode_rewards": result.episode_rewards,
        "episode_stim_frac": result.episode_stim_frac,
    }


def _checkpoint_current(variant: str, seed: int) -> bool:
    """True when the cached checkpoint was trained with today's shared config.

    A checkpoint from an older ``fig4_ravivarapu_config`` (e.g. another
    ``n_obs``) must be retrained, not silently evaluated.
    """
    from dataclasses import asdict

    from controllers.sea_dbs.checkpoint import load_checkpoint
    from controllers.sea_dbs.config import config_from_dict, fig4_ravivarapu_config

    path = _ckpt(variant, seed)
    if not path.is_file():
        return False
    saved = asdict(config_from_dict(load_checkpoint(path)["sea_dbs_config"]))
    return saved == asdict(fig4_ravivarapu_config(seed=seed, variant=variant))


def ensure_checkpoints(
    seed: int,
    *,
    smoke: bool,
    resume: bool,
    checkpoint_interval: int,
    parallel_series: int,
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        _TrainJob(
            variant=variant,
            seed=seed,
            smoke=smoke,
            checkpoint=str(_ckpt(variant, seed)),
            resume=resume,
            checkpoint_interval=checkpoint_interval,
        )
        for variant in VARIANTS
        if variant not in FIG4_VARIANTS and (resume or smoke or not _checkpoint_current(variant, seed))
    ]
    if jobs:
        _parallel_series.run_series_parallel(jobs, _train_worker, parallel_series)
    for variant in FIG4_VARIANTS:
        if not smoke and _ckpt(variant, seed).is_file() and not _checkpoint_current(variant, seed):
            raise SystemExit(
                f"{_ckpt(variant, seed)} was trained with another config; retrain Fig 4 first "
                "(scripts/figures/papers/ravivarapu/4a/plot.py)"
            )
        if not _ckpt(variant, seed).is_file():
            raise SystemExit(
                f"missing {_ckpt(variant, seed)}; train Fig 4 first "
                "(scripts/figures/papers/ravivarapu/4a/plot.py)"
            )


def _fig5a_traces() -> dict[str, list[float]] | None:
    """Fig 5a rollout (the paper plots the same Baseline / SEA-DBS curves in Figs 5a, 6, 7)."""
    if not FIG5A_SERIES.is_file():
        return None
    return json.loads(FIG5A_SERIES.read_text(encoding="utf-8")).get("traces")


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_fig7_gates(traces, fig5a_traces=_fig5a_traces())
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
        help="Resume training from existing +PM / +GS checkpoints in the Fig 7 cache",
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
    _common.add_no_promote_arg(parser)
    args = parser.parse_args()
    if args.no_promote:
        args.export_notes = False
        args.update_report = False
    _resume_cli.configure_promote_publish(args, _figure_promote)
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    series_path = CACHE_DIR / "series.json"
    actions: dict[str, list[int]] = {}
    eval_meta: dict[str, Any] = {}
    if args.plot_only:
        if not series_path.is_file():
            raise SystemExit(f"missing series cache: {series_path}")
        payload = json.loads(series_path.read_text(encoding="utf-8"))
        traces = payload["traces"]
        actions = payload.get("actions") or {}
        eval_meta = payload.get("eval_meta") or {}
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
            _parallel_series.RavivarapuInferenceEvalJob(
                variant=variant,
                seed=args.seed,
                checkpoint=str(_ckpt(variant, args.seed)),
                max_steps=steps,
                carrier_hz=INFERENCE_CARRIER_50HZ,
            )
            for variant in VARIANTS
        ]
        results = _parallel_series.run_series_parallel(
            eval_jobs,
            _parallel_series.ravivarapu_inference_eval_worker,
            args.parallel_series,
        )
        traces = {}
        for result in results:
            traces[result.variant] = result.trace
            actions[result.variant] = result.actions
            eval_meta[result.variant] = result.eval_meta

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
    fig.savefig(_common.linked_png(png_path, _figure_promote, promote=not args.no_promote), dpi=150)
    plt.close(fig)
    gates = {"pass": False, "smoke_override": True} if args.smoke else evaluate_gates(traces)
    if not args.plot_only:
        series_path.write_text(
            json.dumps(
                {
                    "traces": traces,
                    "actions": actions,
                    "eval_meta": eval_meta,
                    "steps": steps,
                    "n_psd_samples": steps + 1,
                },
                indent=2,
            )
            + "\n"
        )
    caption = (
        f"Ablation GPi beta PSD over 10 steps @ 50 Hz (seed {args.seed}, argmax actions); "
        f"pass={gates.get('pass')}; Baseline vs +PM vs +GS vs SEA-DBS, one shared training config."
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
        "actions": {k: list(v) for k, v in actions.items()},
        "eval_meta": eval_meta,
    }
    manifest = _panel.stamp_manifest(manifest)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_promote and hasattr(_figure_promote, "promote_ravivarapu_7"):
        _figure_promote.promote_ravivarapu_7(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))


if __name__ == "__main__":
    raise SystemExit(main())
