#!/usr/bin/env python3
"""Ravivarapu Fig 6 — FP16 post-training quantization @ 50 Hz.

Four series on the Fig 4 checkpoints: Baseline / SEA-DBS at fp32 and after
FP16 PTQ (a plain ``.half()`` cast of the trained actor — no weight noise).
Same rollout as Fig 5a: training plant setup, 50 Hz, untreated t=0 plus 10
deterministic (argmax) actions. Model size is the actor's parameter bytes at
fp32 vs fp16 (plus the full actor+critic+predictive-model set for reference).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/6/plot.py --export-notes --update-report
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/6/plot.py --plot-only
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
os.environ.setdefault("RL_DBS_MAX_THREADS", "1")

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
from ravivarapu_gates import merge_gate_report, ravivarapu_fig6_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import = load_script_module("figure_overlay_import", _OVERLAY_IMPORT)
_paper_overlay = _overlay_import.load_paper_overlay()

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/6")
FIGURES_DIR = Path("figures/ravivarapu/images/6")
OUT_STEM = "ptq_fp16_50hz"
SERIES = (
    ("baseline", "Baseline", False),
    ("baseline", "Baseline + PTQ(fp16)", True),
    ("paper", "SEA-DBS", False),
    ("paper", "SEA-DBS + PTQ(fp16)", True),
)
PLOT_STYLE = {
    "Baseline": {"color": "#1f77b4", "linewidth": 1.5},
    "Baseline + PTQ(fp16)": {"color": "#ff7f0e", "linewidth": 1.5, "linestyle": "--"},
    "SEA-DBS": {"color": "#2ca02c", "linewidth": 1.5},
    "SEA-DBS + PTQ(fp16)": {"color": "#d62728", "linewidth": 1.5, "linestyle": "--"},
}


def _param_bytes(ckpt: Path) -> dict[str, int]:
    """Parameter bytes of the saved networks at fp32 and after an fp16 cast."""
    import torch

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_actor = sum(int(t.numel()) for t in payload["actor_state_dict"].values())
    n_all = n_actor + sum(int(t.numel()) for t in payload["critic_state_dict"].values())
    if payload.get("predictive_state_dict"):
        n_all += sum(int(t.numel()) for t in payload["predictive_state_dict"].values())
    return {
        "actor_params": n_actor,
        "actor_bytes_fp32": 4 * n_actor,
        "actor_bytes_fp16": 2 * n_actor,
        "all_networks_bytes_fp32": 4 * n_all,
        "all_networks_bytes_fp16": 2 * n_all,
    }


def _max_fp16_logit_error(ckpt: Path, states: np.ndarray) -> float:
    """Largest |fp32 − fp16| actor logit over the states the fp32 rollout visited."""
    import torch

    from controllers.sea_dbs.checkpoint import load_actor_from_payload, load_checkpoint
    from controllers.sea_dbs.quantization import FP16ActorWrapper, apply_fp16_ptq

    actor, _cfg = load_actor_from_payload(load_checkpoint(ckpt))
    half = FP16ActorWrapper(apply_fp16_ptq(actor))
    x = torch.as_tensor(states.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        return float((actor(x) - half(x).float()).abs().max())


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_fig6_gates(traces)
    n = min(len(v) for v in traces.values())
    return merge_gate_report(dig, {"n_steps": n})


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
    sizes: dict[str, Any] = {}

    if args.plot_only:
        if not series_path.is_file():
            raise SystemExit(f"missing series cache: {series_path}")
        payload = json.loads(series_path.read_text(encoding="utf-8"))
        traces = payload["traces"]
        actions = payload.get("actions") or {}
        eval_meta = payload.get("eval_meta") or {}
        sizes = payload.get("model_sizes") or {}
        steps = int(payload.get("steps", len(next(iter(traces.values()))) - 1))
        n_expected = int(payload.get("n_psd_samples", len(next(iter(traces.values())))))
    else:
        jobs = [
            _parallel_series.RavivarapuInferenceEvalJob(
                variant=variant,
                seed=args.seed,
                checkpoint=str(_common.fig4_checkpoint(variant, args.seed)),
                max_steps=steps,
                carrier_hz=INFERENCE_CARRIER_50HZ,
                use_fp16_ptq=use_ptq,
                label=label,
            )
            for variant, label, use_ptq in SERIES
        ]
        results = _parallel_series.run_series_parallel(
            jobs,
            _parallel_series.ravivarapu_inference_eval_worker,
            args.parallel_series,
        )
        traces = {}
        for result in results:
            label = str(result.label)
            traces[label] = result.trace
            actions[label] = result.actions
            eval_meta[label] = result.eval_meta
        for variant, label, use_ptq in SERIES:
            if use_ptq:
                continue
            ckpt = _common.fig4_checkpoint(variant, args.seed)
            sizes[variant] = _param_bytes(ckpt)
            sizes[variant]["max_fp16_logit_error"] = _max_fp16_logit_error(
                ckpt, np.asarray(traces[label][:-1], dtype=float)
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    scale = float(_paper_overlay.RAVI_INFERENCE_PAPER_Y_TO_NORM)
    ys: list[np.ndarray] = []
    for _, label, _use_ptq in SERIES:
        y = np.asarray(traces[label], dtype=float) * scale
        ax.plot(np.arange(y.size, dtype=float), y, label=label, **PLOT_STYLE[label])
        ys.append(y)
    paper = _paper_overlay.overlay_ravivarapu_fig6(ax)
    ys.extend(py for _px, py in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([300.0, 480.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_xlim(0, 10)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD")
    ax.set_title("FP16 PTQ @ 50 Hz")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=7)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(_common.linked_png(png_path, _figure_promote, promote=not args.no_promote), dpi=150)
    plt.close(fig)

    gates = {"pass": False, "smoke_override": True} if args.smoke else evaluate_gates(traces)
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
                    "model_sizes": sizes,
                },
                indent=2,
            )
            + "\n"
        )
    sea_sizes = sizes.get("paper") or {}
    kb32 = sea_sizes.get("all_networks_bytes_fp32", 0) / 1024
    kb16 = sea_sizes.get("all_networks_bytes_fp16", 0) / 1024
    caption = (
        f"FP16 PTQ inference GPi beta PSD vs step @ 50 Hz (seed {args.seed}, argmax actions, "
        f"plain .half() cast); pass={gates.get('pass')}; Baseline/SEA-DBS fp32 + PTQ; "
        f"SEA-DBS networks {kb32:.0f} KB → {kb16:.0f} KB."
    )
    manifest = {
        "panel": "6",
        "carrier_hz": INFERENCE_CARRIER_50HZ,
        "n_steps": steps,
        "n_psd_samples": n_expected,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "caption": caption,
        "series_cache": series_path.as_posix(),
        "actions": {k: list(v) for k, v in actions.items()},
        "eval_meta": eval_meta,
        "model_sizes": sizes,
    }
    manifest = _panel.stamp_manifest(manifest)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_promote and hasattr(_figure_promote, "promote_ravivarapu_6"):
        _figure_promote.promote_ravivarapu_6(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))


if __name__ == "__main__":
    raise SystemExit(main())
