#!/usr/bin/env python3
"""Ravivarapu Fig 6 — FP16 PTQ @ 50 Hz (10-step eval).

Eval-only panel. Resume training from Fig 4a / Fig 7 checkpoints as above.

Paper Fig. 6 has **four** series: Baseline, Baseline+PTQ(fp16), SEA-DBS,
SEA-DBS+PTQ(fp16).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, INFERENCE_CARRIER_50HZ, SEADBSConfig
from controllers.sea_dbs.eval import evaluate

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_fig6_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
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


def _ckpt(variant: str, seed: int) -> Path:
    path = Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{seed}.pt"
    if path.is_file():
        return path
    return Path("artifacts/sea_dbs") / f"{variant}_train{seed}.pt"


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_fig6_gates(traces)
    n = min(len(v) for v in traces.values())
    return merge_gate_report(dig, {"n_steps": n})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    traces: dict[str, list[float]] = {}
    for variant, label, use_ptq in SERIES:
        payload = evaluate(
            _ckpt(variant, args.seed),
            config=SEADBSConfig(variant=variant, seed=args.seed),
            max_steps=steps,
            carrier_hz=INFERENCE_CARRIER_50HZ,
            use_fp16_ptq=use_ptq,
        )
        traces[label] = payload["p_beta_trajectories"][0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for _, label, _ in SERIES:
        y = np.asarray(traces[label], dtype=float)
        ax.plot(np.arange(y.size, dtype=float), y, label=label, linewidth=1.5)
    paper = _paper_overlay.overlay_ravivarapu_fig6(ax)
    ys = [np.asarray(v, dtype=float) for v in traces.values()]
    ys.extend(v[0] for v in paper.values())
    all_y = np.concatenate(ys) if ys else np.array([0.0, 1.0])
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Stimulation step")
    ax.set_ylabel("Mean beta PSD (norm)")
    _paper_overlay.place_legend(ax, fontsize=8)
    ax.grid(True, alpha=0.3)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    gates = {"pass": True, "smoke_override": True} if args.smoke else evaluate_gates(traces)
    manifest = {
        "panel": "6",
        "n_steps": steps,
        "carrier_hz": INFERENCE_CARRIER_50HZ,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "series": list(traces.keys()),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
