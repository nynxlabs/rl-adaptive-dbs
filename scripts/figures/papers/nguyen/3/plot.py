#!/usr/bin/env python3
"""Nguyen et al. (paper 2) Figure 3 — GPi α–β power distribution (PD vs healthy).

Collects GPi α–β oscillation power (7–35 Hz) over short plant segments with no
DBS for healthy (`pd=0`) and Parkinsonian (`pd=1`) conditions. Qualitative
gates: PD mass above healthy; θ=150 near the PD lower quartile when sample
size allows.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --seeds 0,1,2 --duration-s 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from controllers.snn.reward import alpha_beta_power
from envs.plant import DbsSpec, PlantConfig, PythonPlant

FIGURE_DIR = Path("artifacts/figures/papers/nguyen/3")
DEFAULT_SAMPLES = FIGURE_DIR / "samples.json"
DEFAULT_OUT = Path("figures/nguyen/images/3/alpha_beta_dist.png")
DEFAULT_MANIFEST = FIGURE_DIR / "manifest.json"
DEFAULT_DURATION_S = 1.0
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))
THRESHOLD = 150.0

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#cccccc",
    "font.size": 10,
}


def parse_seeds(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds must list at least one integer")
    return tuple(int(p) for p in parts)


def collect_samples(
    *,
    seeds: tuple[int, ...],
    duration_s: float,
) -> dict[str, Any]:
    healthy: list[float] = []
    pd_vals: list[float] = []
    for seed in seeds:
        for pd, bucket in ((0, healthy), (1, pd_vals)):
            plant = PythonPlant(config=PlantConfig(pd=pd))
            try:
                plant.reset(seed=seed)
                result = plant.integrate(
                    duration_s,
                    DbsSpec.none(),
                    record_spikes=True,
                )
                value = alpha_beta_power(
                    result.gpi_spikes,
                    duration_s=result.duration_s,
                    dt_ms=result.dt_ms,
                )
                bucket.append(float(value))
            finally:
                plant.close()
    return {
        "seeds": list(seeds),
        "duration_s": duration_s,
        "threshold": THRESHOLD,
        "healthy": healthy,
        "pd": pd_vals,
        "healthy_median": float(np.median(healthy)) if healthy else None,
        "pd_median": float(np.median(pd_vals)) if pd_vals else None,
        "pd_q1": float(np.percentile(pd_vals, 25)) if pd_vals else None,
    }


def evaluate_gates(samples: dict[str, Any]) -> dict[str, Any]:
    healthy = np.asarray(samples["healthy"], dtype=float)
    pd_vals = np.asarray(samples["pd"], dtype=float)
    ordering_ok = bool(np.median(pd_vals) > np.median(healthy))
    pd_q1 = float(np.percentile(pd_vals, 25))
    # Soft gate: θ within a broad band around PD Q1 (paper chooses 150).
    threshold_plausible = bool(abs(pd_q1 - THRESHOLD) / max(THRESHOLD, 1.0) < 0.75)
    return {
        "ordering_pd_above_healthy": ordering_ok,
        "threshold_near_pd_q1": threshold_plausible,
        "pd_q1": pd_q1,
        "pass": ordering_ok,
    }


def plot_samples(samples: dict[str, Any], out_path: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)

    healthy = samples["healthy"]
    pd_vals = samples["pd"]
    rng = np.random.default_rng(0)

    ax0 = axes[0]
    ax0.scatter(
        rng.normal(0, 0.06, size=len(healthy)),
        healthy,
        c="#2ca02c",
        alpha=0.75,
        label="Healthy",
    )
    ax0.scatter(
        rng.normal(1, 0.06, size=len(pd_vals)),
        pd_vals,
        c="#d62728",
        alpha=0.75,
        label="PD",
    )
    ax0.axhline(THRESHOLD, color="#555555", ls="--", lw=1, label=f"θ={THRESHOLD:g}")
    ax0.set_xticks([0, 1], ["Healthy", "PD"])
    ax0.set_ylabel("GPi α–β power")
    ax0.set_title("(a) samples")
    ax0.legend(frameon=False, fontsize=8)

    ax1 = axes[1]
    ax1.boxplot(
        [healthy, pd_vals],
        tick_labels=["Healthy", "PD"],
        patch_artist=True,
        boxprops={"facecolor": "#eeeeee"},
    )
    ax1.axhline(THRESHOLD, color="#555555", ls="--", lw=1)
    ax1.set_ylabel("GPi α–β power")
    ax1.set_title("(b) distribution")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    t0 = time.perf_counter()
    if args.plot_only:
        if not args.samples.is_file():
            print(f"missing samples cache: {args.samples}", file=sys.stderr)
            return 2
        samples = json.loads(args.samples.read_text(encoding="utf-8"))
    else:
        print(f"collecting α–β samples seeds={seeds} duration_s={args.duration_s}", flush=True)
        samples = collect_samples(seeds=seeds, duration_s=float(args.duration_s))
        write_json(args.samples, samples)

    gates = evaluate_gates(samples)
    plot_samples(samples, args.out)
    manifest = {
        "panel": "2/3",
        "out": args.out.as_posix(),
        "samples": args.samples.as_posix(),
        "gates": gates,
        "elapsed_s": time.perf_counter() - t0,
        "caption": (
            f"GPi α–β (7–35 Hz); PD median={samples.get('pd_median')}, "
            f"healthy median={samples.get('healthy_median')}, "
            f"PD Q1={gates['pd_q1']:.1f}; ordering_pass={gates['pass']}"
        ),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
