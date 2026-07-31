#!/usr/bin/env python3
"""Ravivarapu Fig 5b — inference @ 30 Hz carrier (post-train eval).

Paper panel: 10 stimulation steps; SEA-DBS below Baseline; weaker than 50 Hz.
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

from controllers.sea_dbs.config import (
    ABLATION_EVAL_STEPS,
    INFERENCE_CARRIER_30HZ,
    INFERENCE_CARRIER_50HZ,
    SEADBSConfig,
)
from controllers.sea_dbs.eval import evaluate

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/5b")
FIGURES_DIR = Path("figures/ravivarapu/images/5b")
FIG5A_SERIES = Path("artifacts/figures/papers/ravivarapu/5a/series.json")
OUT_STEM = "inference_30hz"
VARIANTS = ("baseline", "paper")


def _ckpt(variant: str, seed: int) -> Path:
    path = Path("artifacts/figures/papers/ravivarapu/4") / f"{variant}_train{seed}.pt"
    if path.is_file():
        return path
    return Path("artifacts/sea_dbs") / f"{variant}_train{seed}.pt"


def evaluate_gates(
    traces: dict[str, list[float]],
    *,
    traces_50: dict[str, list[float]] | None,
) -> dict:
    base = np.asarray(traces["baseline"], dtype=float)
    paper = np.asarray(traces["paper"], dtype=float)
    n = min(base.size, paper.size)
    if n < 3:
        return {"pass": False, "reason": "too_few_steps", "n_steps": n}
    base_tail = float(np.mean(base[max(1, n // 2) : n]))
    paper_tail = float(np.mean(paper[max(1, n // 2) : n]))
    gates: dict = {
        "n_steps": n,
        "carrier_hz": INFERENCE_CARRIER_30HZ,
        "paper_below_baseline_tail": paper_tail < base_tail,
        "paper_end_below_baseline": float(paper[n - 1]) < float(base[n - 1]),
        "baseline_tail_mean": base_tail,
        "paper_tail_mean": paper_tail,
    }
    if traces_50 and "paper" in traces_50:
        paper_50 = np.asarray(traces_50["paper"], dtype=float)
        m = min(n, paper_50.size)
        # 30 Hz should be weaker suppression → higher PSD than 50 Hz (paper claim).
        weaker_than_50 = float(np.mean(paper[max(1, m // 2) : m])) > float(
            np.mean(paper_50[max(1, m // 2) : m])
        )
        gates["weaker_than_50hz"] = weaker_than_50
        gates["paper_50_tail_mean"] = float(np.mean(paper_50[max(1, m // 2) : m]))
    else:
        gates["weaker_than_50hz"] = None
    ordering_ok = bool(
        gates["paper_below_baseline_tail"] and gates["paper_end_below_baseline"]
    )
    cross_ok = gates["weaker_than_50hz"] in (True, None)
    gates["pass"] = ordering_ok and cross_ok
    return gates


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
            carrier_hz=INFERENCE_CARRIER_30HZ,
        )
        traces[variant] = payload["p_beta_trajectories"][0]

    traces_50 = None
    if FIG5A_SERIES.is_file():
        traces_50 = json.loads(FIG5A_SERIES.read_text())["traces"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, label in (("baseline", "Baseline 30Hz"), ("paper", "SEA-DBS 30Hz")):
        ax.plot(traces[variant], label=label)
    ax.set_xlabel("Steps")
    ax.set_ylabel("PSD (norm)")
    ax.set_title("Beta stimulation freq. 30 Hz")
    ax.legend()
    ax.grid(True, alpha=0.3)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    if args.smoke:
        gates = {"pass": True, "smoke_override": True}
    else:
        gates = evaluate_gates(traces, traces_50=traces_50)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series_path = CACHE_DIR / "series.json"
    series_path.write_text(json.dumps({"traces": traces, "steps": steps}, indent=2) + "\n")
    manifest = {
        "panel": "5b",
        "carrier_hz": INFERENCE_CARRIER_30HZ,
        "cross_check_carrier_hz": INFERENCE_CARRIER_50HZ,
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
