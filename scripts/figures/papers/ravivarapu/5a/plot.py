#!/usr/bin/env python3
"""Ravivarapu Fig 5a — inference @ 50 Hz carrier (post-train eval).

Paper panel: 10 stimulation steps; SEA-DBS below Baseline; stronger than 30 Hz.
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
from ravivarapu_gates import merge_gate_report, ravivarapu_inference_gates  # noqa: E402

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/5a")
FIGURES_DIR = Path("figures/ravivarapu/images/5a")
OUT_STEM = "inference_50hz"
VARIANTS = ("baseline", "paper")


def _ckpt(variant: str, seed: int) -> Path:
    path = Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{seed}.pt"
    if path.is_file():
        return path
    return Path("artifacts/sea_dbs") / f"{variant}_train{seed}.pt"


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_inference_gates(
        traces["baseline"],
        traces["paper"],
        carrier_hz=INFERENCE_CARRIER_50HZ,
    )
    return merge_gate_report(dig, {"n_steps": len(traces["baseline"])})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    traces: dict[str, list[float]] = {}
    for variant in VARIANTS:
        payload = evaluate(
            _ckpt(variant, args.seed),
            config=SEADBSConfig(variant=variant, seed=args.seed),
            max_steps=steps,
            carrier_hz=INFERENCE_CARRIER_50HZ,
        )
        traces[variant] = payload["p_beta_trajectories"][0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, label in (("baseline", "Baseline 50Hz"), ("paper", "SEA-DBS 50Hz")):
        ax.plot(traces[variant], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD (norm)")
    ax.set_title("Beta stimulation freq. 50 Hz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    gates = {"pass": True, "smoke_override": True} if args.smoke else evaluate_gates(traces)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series_path = CACHE_DIR / "series.json"
    series_path.write_text(json.dumps({"traces": traces, "steps": steps}, indent=2) + "\n")
    manifest = {
        "panel": "5a",
        "carrier_hz": INFERENCE_CARRIER_50HZ,
        "n_steps": steps,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "series_cache": series_path.as_posix(),
    }
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
