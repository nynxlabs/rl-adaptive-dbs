#!/usr/bin/env python3
"""Ravivarapu Fig 6 — FP16 PTQ @ 50 Hz (10-step eval)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    ckpt = Path("artifacts/figures/papers/ravivarapu/4") / f"paper_train{args.seed}.pt"
    if not ckpt.is_file():
        ckpt = Path("artifacts/sea_dbs") / f"paper_train{args.seed}.pt"
    cfg = SEADBSConfig(variant="paper", seed=args.seed)
    fp32 = evaluate(ckpt, config=cfg, max_steps=steps, carrier_hz=INFERENCE_CARRIER_50HZ)
    ptq = evaluate(
        ckpt,
        config=cfg,
        max_steps=steps,
        carrier_hz=INFERENCE_CARRIER_50HZ,
        use_fp16_ptq=True,
    )
    baseline_ckpt = Path("artifacts/figures/papers/ravivarapu/4") / f"baseline_train{args.seed}.pt"
    if not baseline_ckpt.is_file():
        baseline_ckpt = Path("artifacts/sea_dbs") / f"baseline_train{args.seed}.pt"
    base = evaluate(
        baseline_ckpt,
        config=SEADBSConfig(variant="baseline", seed=args.seed),
        max_steps=steps,
        carrier_hz=INFERENCE_CARRIER_50HZ,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(base["p_beta_trajectories"][0], label="Baseline")
    ax.plot(fp32["p_beta_trajectories"][0], label="SEA-DBS FP32")
    ax.plot(ptq["p_beta_trajectories"][0], label="SEA-DBS FP16 PTQ")
    ax.set_xlabel("Stimulation step")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    manifest = {
        "panel": "6",
        "n_steps": steps,
        "carrier_hz": INFERENCE_CARRIER_50HZ,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": {"pass": args.smoke, "smoke_override": args.smoke},
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
