#!/usr/bin/env python3
"""Nguyen et al. Figure 7 — 50-episode eval α–β trace (25 steps each).

Loads the Fig. 4 checkpoint and rolls out greedy policy; compares mean step trace
to digitized paper Fig. 7 and Fig. 3 PD On reference.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/7/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/7/plot.py --plot-only
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
from nguyen_gates import attach_digitization, fig7_eval_gates  # noqa: E402

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

from controllers.snn.config import BIOMARKER_THRESHOLD, EVAL_EPISODES, EVAL_MAX_STEPS, SNNConfig
from controllers.snn.eval import evaluate

FIG4_CACHE = Path("artifacts/figures/papers/nguyen/4")
FIG4_CHECKPOINT = FIG4_CACHE / "checkpoint.pt"
FIG3_MANIFEST = Path("artifacts/figures/papers/nguyen/3/manifest.json")
FIGURES_DIR = Path("figures/nguyen/images/7")
CACHE_DIR = Path("artifacts/figures/papers/nguyen/7")
DEFAULT_EVAL = CACHE_DIR / "eval.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "eval_50ep"

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


def _fig3_pd_on_median() -> float | None:
    if not FIG3_MANIFEST.is_file():
        return None
    manifest = json.loads(FIG3_MANIFEST.read_text(encoding="utf-8"))
    samples_path = Path(manifest.get("samples", ""))
    if not samples_path.is_file():
        return None
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    pd_on = np.asarray(samples.get("pd_on", []), dtype=float)
    if pd_on.size == 0:
        return None
    return float(np.median(pd_on))


def evaluate_gates(
    eval_payload: dict[str, Any],
    *,
    fig4_manifest: dict[str, Any] | None,
    fig3_pd_on_median: float | None,
) -> dict[str, Any]:
    checkpoint_ok = bool(
        fig4_manifest is not None
        and int(fig4_manifest.get("gates", {}).get("n_episodes", -1)) == 500
    )
    heuristic = {
        "checkpoint_lineage_ok": checkpoint_ok,
        "pass": checkpoint_ok,
    }
    if not checkpoint_ok:
        heuristic["reason"] = "fig4_train_not_passing"

    trajectories = eval_payload["alpha_beta_trajectories"]
    max_len = max(len(tr) for tr in trajectories) if trajectories else 0
    pad_trajectories = [
        tr + [tr[-1]] * (max_len - len(tr)) if len(tr) < max_len else tr[:max_len]
        for tr in trajectories
    ]
    dig = fig7_eval_gates(
        pad_trajectories,
        fig3_pd_on_median=fig3_pd_on_median,
    )
    return attach_digitization(heuristic, dig)


def plot_eval(eval_payload: dict[str, Any], out_path: Path) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    trajectories: list[list[float]] = eval_payload["alpha_beta_trajectories"]
    max_len = max(len(tr) for tr in trajectories)
    pad_trajectories = [
        tr + [tr[-1]] * (max_len - len(tr)) if len(tr) < max_len else tr[:max_len]
        for tr in trajectories
    ]
    step_means = []
    step_lowers = []
    step_uppers = []
    for step in range(max_len):
        vals = [float(tr[step]) for tr in pad_trajectories]
        if vals:
            step_means.append(float(np.mean(vals)))
            step_lowers.append(float(np.percentile(vals, 2.5)))
            step_uppers.append(float(np.percentile(vals, 97.5)))
        else:
            step_means.append(float("nan"))
            step_lowers.append(float("nan"))
            step_uppers.append(float("nan"))
    steps = np.arange(len(step_means), dtype=float)
    mean_trace = np.asarray(step_means, dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.fill_between(
        steps,
        step_lowers,
        step_uppers,
        color="#9ecae1",
        alpha=0.30,
        edgecolor="#08519c",
        linewidth=0.8,
        label="Replication 95% CI",
        zorder=2,
    )
    ax.plot(steps, mean_trace, color="#08519c", linewidth=2.2, label="Replication Mean", zorder=3)
    ax.axhline(BIOMARKER_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2, label="θ=150 (threshold)")
    _paper_overlay.overlay_nguyen_fig7(ax, show_confidence=True)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("α–β Power")
    ax.set_title("Evaluation α–β (50 episodes)")
    ax.grid(True, linestyle="--", alpha=0.6)
    _paper_overlay.place_legend(ax, fontsize=8)
    ax.set_xlim(0.0, max(24.0, float(steps[-1]) if steps.size else 24.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "n_episodes": len(trajectories),
        "n_steps": max_len,
        "overall_mean": float(np.nanmean(mean_trace)) if mean_trace.size else float("nan"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=FIG4_CHECKPOINT)
    parser.add_argument("--fig4-manifest", type=Path, default=FIG4_CACHE / "manifest.json")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--max-steps", type=int, default=EVAL_MAX_STEPS)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-update-docs", action="store_true")
    parser.add_argument(
        "--push-kb",
        action="store_true",
        help="After promote, copy replication PNGs to the knowledge-base vault",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="After promote, refresh Report 3 gallery image links in the knowledge-base",
    )
    args = parser.parse_args(argv)
    _figure_promote.set_push_kb_images(args.push_kb)
    _figure_promote.set_update_report3(args.update_report)

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    t0 = time.perf_counter()
    if args.plot_only:
        if not args.eval_cache.is_file():
            print(f"missing eval cache: {args.eval_cache}", file=sys.stderr)
            return 2
        eval_payload = json.loads(args.eval_cache.read_text(encoding="utf-8"))
    else:
        if not args.checkpoint.is_file():
            print(f"missing checkpoint: {args.checkpoint}", file=sys.stderr)
            return 2
        cfg = SNNConfig().with_variant_defaults()
        eval_payload = evaluate(
            args.checkpoint,
            config=cfg,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
        write_json(args.eval_cache, eval_payload)

    fig4_manifest = None
    if args.fig4_manifest.is_file():
        fig4_manifest = json.loads(args.fig4_manifest.read_text(encoding="utf-8"))
    fig3_median = _fig3_pd_on_median()

    gates = evaluate_gates(
        eval_payload,
        fig4_manifest=fig4_manifest,
        fig3_pd_on_median=fig3_median,
    )
    panel = plot_eval(eval_payload, args.out)

    caption = (
        f"eval {panel['n_episodes']}×{panel['n_steps']} steps; "
        f"mean αβ={panel['overall_mean']:.1f}; pass={gates['pass']}"
    )
    manifest = {
        "panel": "2/7",
        "out": args.out.as_posix(),
        "eval_cache": args.eval_cache.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "gates": gates,
        "panel_stats": panel,
        "fig3_pd_on_median": fig3_median,
        "elapsed_s": time.perf_counter() - t0,
        "png_version": png_version,
        "caption": caption,
    }
    write_json(args.manifest, manifest)

    if not args.no_update_docs:
        updated = _figure_promote.promote_nguyen_7(
            manifest=manifest,
            png_path=args.out,
        )
        print(f"updated comparison doc: {updated['doc']}", flush=True)

    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
