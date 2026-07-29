#!/usr/bin/env python3
"""Ravivarapu Fig 5b — inference @ 30 Hz carrier (post-train eval)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from controllers.sea_dbs.config import INFERENCE_CARRIER_30HZ, SEADBSConfig
from controllers.sea_dbs.eval import evaluate

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/5b")
FIGURES_DIR = Path("figures/ravivarapu/images/5b")
OUT_STEM = "inference_30hz"
VARIANTS = ("baseline", "paper")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    steps = 5 if args.smoke else 30
    traces: dict[str, list[float]] = {}
    for variant in VARIANTS:
        ckpt = Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{args.seed}.pt"
        if not ckpt.is_file():
            ckpt = Path("artifacts/sea_dbs") / f"{variant}_train{args.seed}.pt"
        payload = evaluate(
            ckpt,
            config=SEADBSConfig(variant=variant, seed=args.seed),
            max_steps=steps,
            carrier_hz=INFERENCE_CARRIER_30HZ,
        )
        traces[variant] = payload["p_beta_trajectories"][0]

    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, label in (("baseline", "Baseline"), ("paper", "SEA-DBS")):
        ax.plot(traces[variant], label=label)
    ax.set_xlabel("Inference step")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.set_title("30 Hz carrier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    manifest = {
        "panel": "5b",
        "carrier_hz": INFERENCE_CARRIER_30HZ,
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": {"pass": args.smoke, "smoke_override": args.smoke},
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
