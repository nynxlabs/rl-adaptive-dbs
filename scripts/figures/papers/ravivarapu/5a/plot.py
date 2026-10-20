#!/usr/bin/env python3
"""Ravivarapu Fig 5a — inference @ 50 Hz carrier (post-train eval).

Eval-only panel on the Fig 4 checkpoints (``baseline_train0.pt`` /
``paper_train0.pt``). Each actor runs on the plant / observation setup it was
trained on (read from the checkpoint); the only override is the paper's 50 Hz
stimulation frequency. Untreated t=0 plus 10 deterministic (argmax) actions.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/5a/plot.py --export-notes --update-report
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/5a/plot.py --plot-only
"""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import (
    ABLATION_EVAL_STEPS,
    INFERENCE_CARRIER_50HZ,
    INFERENCE_PSD_SAMPLES,
)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_series = load_script_module("figure_parallel_series", _PARALLEL_SERIES)

_COMMON = Path(__file__).resolve().parents[1] / "common.py"
_common = load_script_module("ravivarapu_common", _COMMON)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_inference_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import = load_script_module("figure_overlay_import", _OVERLAY_IMPORT)
_paper_overlay = _overlay_import.load_paper_overlay()

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/5a")
FIGURES_DIR = Path("docs/controllers/sea_dbs/figures/images/5a")
OUT_STEM = "inference_50hz"
VARIANTS = ("baseline", "paper")
CARRIER_HZ = INFERENCE_CARRIER_50HZ


def evaluate_gates(traces: dict[str, list[float]], *, n_expected: int) -> dict[str, Any]:
    dig = ravivarapu_inference_gates(
        traces["baseline"],
        traces["paper"],
        carrier_hz=CARRIER_HZ,
        n_expected=n_expected,
    )
    return merge_gate_report(dig, {"n_psd_samples": len(traces["baseline"])})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached series.json (no eval)")
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
    n_expected = steps + 1 if args.smoke else INFERENCE_PSD_SAMPLES
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
        n_expected = int(payload.get("n_psd_samples", len(next(iter(traces.values())))))
    else:
        eval_jobs = [
            _parallel_series.RavivarapuInferenceEvalJob(
                variant=variant,
                seed=args.seed,
                checkpoint=str(_common.fig4_checkpoint(variant, args.seed)),
                max_steps=steps,
                carrier_hz=CARRIER_HZ,
            )
            for variant in VARIANTS
        ]
        eval_results = _parallel_series.run_series_parallel(
            eval_jobs,
            _parallel_series.ravivarapu_inference_eval_worker,
            args.parallel_series,
        )
        traces = {}
        for result in eval_results:
            traces[result.variant] = result.trace
            actions[result.variant] = result.actions
            eval_meta[result.variant] = result.eval_meta

    fig, ax = plt.subplots(figsize=(6, 4))
    scale = float(_paper_overlay.RAVI_INFERENCE_PAPER_Y_TO_NORM)
    ys: list[np.ndarray] = []
    for variant, label in (("baseline", "Baseline 50Hz"), ("paper", "SEA-DBS 50Hz")):
        y = np.asarray(traces[variant], dtype=float) * scale
        ax.plot(np.arange(y.size, dtype=float), y, label=label, linewidth=1.5)
        ys.append(y)
    paper = _paper_overlay.overlay_ravivarapu_fig5a(ax)
    ys.extend(py for _px, py in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([300.0, 480.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_xlim(0, 10)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD")
    ax.set_title("Beta stimulation freq. 50 Hz")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=8)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(_common.linked_png(png_path, _figure_promote, promote=not args.no_promote), dpi=150)
    plt.close(fig)

    gates = {"pass": False, "smoke_override": True} if args.smoke else evaluate_gates(
        traces, n_expected=n_expected
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        series_path.write_text(
            json.dumps(
                {
                    "traces": traces,
                    "actions": actions,
                    "steps": steps,
                    "n_psd_samples": n_expected,
                    "eval_meta": eval_meta,
                },
                indent=2,
            )
            + "\n"
        )
    caption = (
        f"Inference GPi beta PSD vs step @ 50 Hz (seed {args.seed}, argmax actions, "
        f"training plant setup); pass={gates.get('pass')}; Baseline vs SEA-DBS."
    )
    manifest = {
        "panel": "5a",
        "carrier_hz": CARRIER_HZ,
        "n_steps": steps,
        "n_psd_samples": n_expected,
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
    if not args.no_promote and hasattr(_figure_promote, "promote_ravivarapu_5a"):
        _figure_promote.promote_ravivarapu_5a(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))


if __name__ == "__main__":
    raise SystemExit(main())
