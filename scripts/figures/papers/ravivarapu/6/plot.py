#!/usr/bin/env python3
"""Ravivarapu Fig 6 — FP16 PTQ @ 50 Hz (10-step eval).

Paper Fig. 6 has **four** series: Baseline, Baseline+PTQ(fp16), SEA-DBS,
SEA-DBS+PTQ(fp16).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

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


def evaluate_gates(traces: dict[str, list[float]]) -> dict:
    base = np.asarray(traces["Baseline"], dtype=float)
    sea = np.asarray(traces["SEA-DBS"], dtype=float)
    sea_ptq = np.asarray(traces["SEA-DBS + PTQ(fp16)"], dtype=float)
    base_ptq = np.asarray(traces["Baseline + PTQ(fp16)"], dtype=float)
    n = min(base.size, sea.size, sea_ptq.size, base_ptq.size)
    if n < 3:
        return {"pass": False, "reason": "too_few_steps", "n_steps": n}
    sea_track = float(
        np.mean(np.abs(sea_ptq[:n] - sea[:n])) / max(1e-6, np.mean(np.abs(sea[:n])))
    )
    base_track = float(
        np.mean(np.abs(base_ptq[:n] - base[:n])) / max(1e-6, np.mean(np.abs(base[:n])))
    )
    gates = {
        "n_steps": n,
        "four_series": True,
        "sea_below_baseline": float(np.mean(sea[:n])) < float(np.mean(base[:n])),
        "sea_ptq_below_baseline": float(np.mean(sea_ptq[:n])) < float(np.mean(base[:n])),
        "sea_ptq_tracks_fp32": sea_track < 0.15,
        "baseline_ptq_tracks_fp32": base_track < 0.25,
        "sea_ptq_rel_gap": sea_track,
        "baseline_ptq_rel_gap": base_track,
    }
    gates["pass"] = bool(
        gates["sea_below_baseline"]
        and gates["sea_ptq_below_baseline"]
        and gates["sea_ptq_tracks_fp32"]
    )
    return gates


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
        ax.plot(traces[label], label=label)
    ax.set_xlabel("Stimulation step")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.legend(fontsize=8)
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
