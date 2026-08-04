#!/usr/bin/env python3
"""Nguyen et al.  Figure 3 — GPi α–β power distribution (PD Off vs PD On).

Collects GPi α–β oscillation power (7–35 Hz) over many short (100 ms) plant
segments with no DBS for healthy (`pd=0`, PD Off) and Parkinsonian (`pd=1`,
PD On) conditions. Matches the paper layout: scatter by simulation iteration,
mean reference lines, and boxplot summary.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --plot-only

Plant sampling only — no RL training; checkpoint resume is not applicable.
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/3/plot.py --n-iterations 50

Each run writes ``figures/nguyen/images/3/alpha_beta_dist_vN.png`` (N
auto-increments) and updates the replication image link in
``figures/nguyen/replications.md``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

import matplotlib.pyplot as plt
import numpy as np

from controllers.snn.reward import alpha_beta_power
from envs.plant import DbsSpec, PlantConfig, PythonPlant

FIGURES_DIR = Path("figures/nguyen/images/3")
FIGURE_DIR = Path("artifacts/figures/papers/nguyen/3")
DEFAULT_SAMPLES = FIGURE_DIR / "samples.json"
DEFAULT_MANIFEST = FIGURE_DIR / "manifest.json"
OUT_STEM = "alpha_beta_dist"
DEFAULT_N_ITERATIONS = 500
DEFAULT_DURATION_S = 0.1  # Nguyen RL step (§IV)
SCATTER_SIZE = 22
THRESHOLD = 150.0  # reward threshold (§IV); not drawn on Fig 3 panel

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#cccccc",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.size": 10,
}

COLOR_PD_OFF = "#1f77b4"
COLOR_PD_ON = "#ff7f0e"
COLOR_MEAN_OFF = "#d62728"
COLOR_MEAN_ON = "#000000"


def _vault_backed_png(path: Path) -> Path:
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    vault_dir = paper.resolve().parent
    vault_target = vault_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    if not vault_target.exists():
        vault_target.touch()
    path.symlink_to(vault_target)
    return path


def collect_condition(
    plant: PythonPlant,
    *,
    n_iterations: int,
    duration_s: float,
    seed_offset: int = 0,
    log_every: int = 50,
) -> list[float]:
    values: list[float] = []
    for iteration in range(n_iterations):
        seed = seed_offset + iteration
        plant.reset(seed=seed)
        result = plant.integrate(
            duration_s,
            DbsSpec.none(),
            record_spikes=True,
        )
        values.append(
            float(
                alpha_beta_power(
                    result.gpi_spikes,
                    duration_s=result.duration_s,
                    dt_ms=result.dt_ms,
                )
            )
        )
        if log_every > 0 and (iteration + 1) % log_every == 0:
            print(
                f"  {iteration + 1}/{n_iterations} samples "
                f"(last={values[-1]:.1f})",
                flush=True,
            )
    return values


def collect_samples(
    *,
    n_iterations: int,
    duration_s: float,
    seed_offset: int,
    log_every: int,
) -> dict[str, Any]:
    pd_off: list[float] = []
    pd_on: list[float] = []
    plant_off = PythonPlant(config=PlantConfig(pd=0))
    plant_on = PythonPlant(config=PlantConfig(pd=1))
    try:
        print("collecting PD Off (healthy) samples", flush=True)
        pd_off = collect_condition(
            plant_off,
            n_iterations=n_iterations,
            duration_s=duration_s,
            seed_offset=seed_offset,
            log_every=log_every,
        )
        print("collecting PD On samples", flush=True)
        pd_on = collect_condition(
            plant_on,
            n_iterations=n_iterations,
            duration_s=duration_s,
            seed_offset=seed_offset,
            log_every=log_every,
        )
    finally:
        plant_off.close()
        plant_on.close()

    pd_off_arr = np.asarray(pd_off, dtype=float)
    pd_on_arr = np.asarray(pd_on, dtype=float)
    return {
        "n_iterations": n_iterations,
        "duration_s": duration_s,
        "seed_offset": seed_offset,
        "threshold": THRESHOLD,
        "pd_off": pd_off,
        "pd_on": pd_on,
        "pd_off_mean": float(np.mean(pd_off_arr)),
        "pd_on_mean": float(np.mean(pd_on_arr)),
        "pd_off_median": float(np.median(pd_off_arr)),
        "pd_on_median": float(np.median(pd_on_arr)),
        "pd_on_q1": float(np.percentile(pd_on_arr, 25)),
    }


def evaluate_gates(samples: dict[str, Any]) -> dict[str, Any]:
    pd_off = np.asarray(samples["pd_off"], dtype=float)
    pd_on = np.asarray(samples["pd_on"], dtype=float)
    ordering_ok = bool(np.median(pd_on) > np.median(pd_off))
    pd_q1 = float(np.percentile(pd_on, 25))
    threshold_plausible = bool(abs(pd_q1 - THRESHOLD) / max(THRESHOLD, 1.0) < 0.75)
    return {
        "ordering_pd_on_above_pd_off": ordering_ok,
        "threshold_near_pd_on_q1": threshold_plausible,
        "pd_on_q1": pd_q1,
        "pass": ordering_ok,
    }


def plot_samples(samples: dict[str, Any], out_path: Path) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)

    pd_off = np.asarray(samples["pd_off"], dtype=float)
    pd_on = np.asarray(samples["pd_on"], dtype=float)
    iterations = np.arange(len(pd_off))

    mean_off = float(np.mean(pd_off))
    mean_on = float(np.mean(pd_on))

    ax0 = axes[0]
    ax0.scatter(
        iterations,
        pd_off,
        c=COLOR_PD_OFF,
        s=SCATTER_SIZE,
        alpha=0.65,
        label="PD Off",
        edgecolors="none",
    )
    ax0.scatter(
        iterations,
        pd_on,
        c=COLOR_PD_ON,
        s=SCATTER_SIZE,
        alpha=0.65,
        label="PD On",
        edgecolors="none",
    )
    ax0.axhline(
        mean_off,
        color=COLOR_MEAN_OFF,
        lw=1.5,
        label="Mean PD Off",
    )
    ax0.axhline(
        mean_on,
        color=COLOR_MEAN_ON,
        lw=1.5,
        label="Mean PD On",
    )
    ax0.set_xlabel("Simulation Iteration")
    ax0.set_ylabel("GPi α–β Oscillation Power")
    ax0.set_title("(a)")
    ax0.legend(frameon=True, framealpha=0.75, fontsize=8, loc="lower right")

    ax1 = axes[1]
    ax1.boxplot(
        [pd_off, pd_on],
        tick_labels=["PD Off", "PD On"],
        patch_artist=True,
        boxprops={"facecolor": "#eeeeee"},
        medianprops={"color": COLOR_PD_ON, "linewidth": 1.5},
    )
    ax1.set_ylabel("GPi α–β Oscillation Power")
    ax1.set_title("(b)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=DEFAULT_N_ITERATIONS,
        help=f"simulation iterations per condition (default {DEFAULT_N_ITERATIONS})",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"integration window per sample in seconds (default {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="base RNG seed; iteration i uses seed_offset + i",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="progress log interval (0 disables)",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: auto-increment "
            f"{FIGURES_DIR}/{OUT_STEM}_vN.png"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-update-docs", action="store_true")
    args = parser.parse_args(argv)

    if args.n_iterations < 1:
        print("--n-iterations must be >= 1", file=sys.stderr)
        return 2

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(
            FIGURES_DIR,
            OUT_STEM,
        )
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    t0 = time.perf_counter()
    if args.plot_only:
        if not args.samples.is_file():
            print(f"missing samples cache: {args.samples}", file=sys.stderr)
            return 2
        samples = json.loads(args.samples.read_text(encoding="utf-8"))
    else:
        print(
            "collecting α–β samples "
            f"n_iterations={args.n_iterations} duration_s={args.duration_s}",
            flush=True,
        )
        samples = collect_samples(
            n_iterations=args.n_iterations,
            duration_s=float(args.duration_s),
            seed_offset=int(args.seed_offset),
            log_every=int(args.log_every),
        )
        write_json(args.samples, samples)

    gates = evaluate_gates(samples)
    plot_samples(samples, args.out)
    caption = (
        f"GPi α–β (7–35 Hz), {samples['n_iterations']} iters × {samples['duration_s']} s; "
        f"PD On mean={samples['pd_on_mean']:.1f}, "
        f"PD Off mean={samples['pd_off_mean']:.1f}, "
        f"PD On Q1={gates['pd_on_q1']:.1f}; ordering_pass={gates['pass']}"
    )
    manifest = {
        "panel": "2/3",
        "out": args.out.as_posix(),
        "samples": args.samples.as_posix(),
        "gates": gates,
        "elapsed_s": time.perf_counter() - t0,
        "png_version": png_version,
        "caption": caption,
    }
    write_json(args.manifest, manifest)

    if not args.no_update_docs:
        updated = _figure_promote.promote_nguyen_2_3(
            manifest=manifest,
            png_path=args.out,
        )
        print(f"updated comparison doc: {updated['doc']}", flush=True)

    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    if png_version is not None:
        print(f"output PNG version={png_version}", flush=True)
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
