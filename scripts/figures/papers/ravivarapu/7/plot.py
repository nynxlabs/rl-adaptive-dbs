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
from controllers.sea_dbs.eval import evaluate_ablation_steps
from controllers.sea_dbs.trainer import train_sea_dbs

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import merge_gate_report, ravivarapu_fig7_gates  # noqa: E402

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


def ensure_checkpoints(seed: int, *, smoke: bool) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        ckpt = CACHE_DIR / f"{variant}_train{seed}.pt"
        if ckpt.is_file():
            continue
        cfg = SEADBSConfig(variant=variant, seed=seed, log_episodes=True)
        if smoke:
            cfg = cfg.for_smoke(episodes=2, max_steps=5)
        else:
            cfg = replace(cfg, num_episodes=150)
        train_sea_dbs(config=cfg, checkpoint_path=ckpt)


def evaluate_gates(traces: dict[str, list[float]]) -> dict[str, Any]:
    dig = ravivarapu_fig7_gates(traces)
    n = min(len(v) for v in traces.values()) if traces else 0
    return merge_gate_report(dig, {"n_steps": n})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    steps = 5 if args.smoke else ABLATION_EVAL_STEPS
    if not args.plot_only:
        ensure_checkpoints(args.seed, smoke=args.smoke)

    traces: dict[str, list[float]] = {}
    for variant in VARIANTS:
        ckpt = CACHE_DIR / f"{variant}_train{args.seed}.pt"
        payload = evaluate_ablation_steps(
            ckpt,
            config=SEADBSConfig(variant=variant, seed=args.seed),
            n_steps=steps,
        )
        traces[variant] = payload["p_beta_trajectories"][0]

    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in VARIANTS:
        ax.plot(traces[variant], label=LABELS[variant])
    ax.set_xlabel("Stimulation step")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
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
