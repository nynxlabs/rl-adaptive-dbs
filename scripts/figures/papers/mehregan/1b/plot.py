#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 1b — mean GPi power spectral density.

Plots multitaper PSD (Kumaravelu / Chronux-style, 1–100 Hz) averaged across the
10 GPi neurons for three plant conditions:

  - Healthy (`pd = 0`, no DBS)
  - Parkinsonian (`pd = 1`, no DBS)
  - Parkinsonian + 130 Hz conventional STN DBS

By default, PSD curves are averaged over eval seeds 0–9 to smooth single-draw spikes.

Run:
  uv run python scripts/figures/papers/mehregan/1b/plot.py

Plant PSD only — no RL training; checkpoint resume is not applicable.
  uv run python scripts/figures/papers/mehregan/1b/plot.py --plot-only --y-max 90
  uv run python scripts/figures/papers/mehregan/1b/plot.py --seeds 0,1,7,42,99
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.biomarkers import SpectrumParams, multitaper_psd_point_process, p_beta

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig1b_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

FIGURE_DIR = Path("artifacts/figures/papers/mehregan/1b")
DEFAULT_CURVES = FIGURE_DIR / "curves.json"
DEFAULT_OUT = FIGURE_DIR / "gpi_psd.png"
DEFAULT_MANIFEST = FIGURE_DIR / "manifest.json"
DEFAULT_DURATION_S = 10.0
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))

# Paper-style light figure (Fig 1b is a qualitative PSD comparison panel).
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
SERIES = {
    "healthy": {"label": "Healthy Control", "color": "#2ca02c"},
    "pd": {"label": "PD no Treatment", "color": "#d62728"},
    "pd_130hz": {"label": "PD 130 Hz Treatment", "color": "#1f77b4"},
}


@dataclass(frozen=True)
class Condition:
    key: str
    config: PlantConfig
    dbs: DbsSpec


CONDITIONS: tuple[Condition, ...] = (
    Condition("healthy", PlantConfig(pd=0), DbsSpec.none()),
    Condition("pd", PlantConfig(pd=1), DbsSpec.none()),
    Condition("pd_130hz", PlantConfig(pd=1), DbsSpec.from_frequency_hz(130.0)),
)


def parse_seeds(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        msg = "--seeds must list at least one integer"
        raise ValueError(msg)
    return tuple(int(p) for p in parts)


def mean_gpi_psd(
    gpi_spikes: list[np.ndarray],
    *,
    dt_ms: float,
    segment_duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Average multitaper PSD across GPi neurons (Mehregan Fig 1b / Eq. (1) setup)."""
    params = SpectrumParams.from_dt_ms(dt_ms)
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for spikes in gpi_spikes:
        psd, f = multitaper_psd_point_process(
            spikes,
            params,
            segment_duration_s=segment_duration_s,
        )
        psds.append(psd)
        freqs = f
    if freqs is None:
        msg = "gpi_spikes must be non-empty"
        raise ValueError(msg)
    return freqs, np.mean(np.vstack(psds), axis=0)


def simulate_condition(
    plant: PythonPlant,
    condition: Condition,
    *,
    seeds: tuple[int, ...],
    duration_s: float,
) -> dict[str, Any]:
    plant.config = condition.config
    psd_stack: list[np.ndarray] = []
    p_beta_by_seed: dict[str, float] = {}
    freqs: np.ndarray | None = None
    t_cond0 = time.monotonic()
    dbs_note = f"{condition.dbs.frequency_hz:.0f} Hz" if condition.dbs.frequency_hz > 0 else "none"
    print(
        f"  [{condition.key}] pd={condition.config.pd} dbs={dbs_note} "
        f"({len(seeds)} seeds, {duration_s:.1f}s each)",
        file=sys.stderr,
        flush=True,
    )

    for seed in seeds:
        t_seed0 = time.monotonic()
        print(f"    seed {seed} integrate {duration_s:.1f}s...", file=sys.stderr, flush=True)
        result = plant.reset(seed=seed).integrate(duration_s, condition.dbs)
        seed_elapsed = time.monotonic() - t_seed0
        spike_count = sum(int(np.asarray(s).size) for s in result.gpi_spikes)
        f, psd = mean_gpi_psd(
            result.gpi_spikes,
            dt_ms=plant.config.dt_ms,
            segment_duration_s=duration_s,
        )
        freqs = f
        psd_stack.append(psd)
        pb = float(
            p_beta(
                result.gpi_spikes,
                dt_ms=plant.config.dt_ms,
                segment_duration_s=duration_s,
            ),
        )
        p_beta_by_seed[str(seed)] = pb
        print(
            f"    seed {seed} done in {seed_elapsed:.1f}s "
            f"({spike_count:,} GPi spikes, P_beta={pb:.1f})",
            file=sys.stderr,
            flush=True,
        )

    cond_elapsed = time.monotonic() - t_cond0
    print(
        f"  [{condition.key}] done in {cond_elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    if freqs is None:
        msg = "no seeds simulated"
        raise ValueError(msg)

    mean_psd = np.mean(np.vstack(psd_stack), axis=0)
    p_beta_values = list(p_beta_by_seed.values())
    return {
        "key": condition.key,
        "pd": condition.config.pd,
        "dbs_hz": condition.dbs.frequency_hz,
        "freqs_hz": freqs.tolist(),
        "psd": mean_psd.tolist(),
        "p_beta_mean": float(np.mean(p_beta_values)),
        "p_beta_std": float(np.std(p_beta_values)),
        "p_beta_by_seed": p_beta_by_seed,
        "n_seeds": len(seeds),
    }


def save_curves(
    curves: list[dict[str, Any]],
    path: Path,
    *,
    duration_s: float,
    seeds: tuple[int, ...],
) -> None:
    payload = {
        "figure": "mehregan_fig1b",
        "duration_s": duration_s,
        "seeds": list(seeds),
        "curves": curves,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_curves(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    curves = payload.get("curves")
    if not isinstance(curves, list) or not curves:
        msg = f"no curves in {path}"
        raise ValueError(msg)
    return curves, payload


def plot_fig1b(
    curves: list[dict[str, Any]],
    *,
    out_path: Path,
    title: str,
    y_max: float | None = None,
    y_headroom: float = 1.1,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)

    peak_by_series: dict[str, float] = {}
    for curve in curves:
        meta = SERIES[curve["key"]]
        freqs = np.asarray(curve["freqs_hz"], dtype=float)
        psd = np.asarray(curve["psd"], dtype=float)
        in_band = freqs <= 50.0
        peak_by_series[curve["key"]] = float(np.max(psd[in_band]))
        ax.plot(freqs, psd, color=meta["color"], linewidth=1.8, label=meta["label"])

    paper_y = _paper_overlay.overlay_mehregan_fig1b(ax)
    paper_peaks = [arr[0] for arr in paper_y.values() if arr[0].size]

    data_max = max(peak_by_series.values(), default=0.0)
    if paper_peaks:
        data_max = max(data_max, max(float(np.max(y)) for y in paper_peaks))
    ymax = y_max if y_max is not None else float(np.ceil(data_max * y_headroom))

    ax.set_xlim(1.0, 50.0)
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    _paper_overlay.place_legend(ax, loc="upper right", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "out": str(out_path),
        "series": [c["key"] for c in curves],
        "psd_peak_hz_0_50": peak_by_series,
        "y_max": ymax,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated eval seeds to average (default: 0–9)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--curves",
        type=Path,
        default=DEFAULT_CURVES,
        help="Cached PSD curves JSON (written on simulate; read with --plot-only)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from --curves cache (seconds; no plant simulation)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Fixed y-axis max (default: auto from data + 10%% headroom)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    if args.duration_s <= 0:
        print("--duration-s must be positive", file=sys.stderr)
        return 2

    if args.plot_only:
        if not args.curves.exists():
            print(f"missing curves cache: {args.curves}", file=sys.stderr)
            print("Run without --plot-only once to build the cache.", file=sys.stderr)
            return 2
        curves, cache_meta = load_curves(args.curves)
        duration_s = float(cache_meta.get("duration_s", args.duration_s))
        seeds = tuple(cache_meta.get("seeds", args.seeds))
        print(f"loaded {len(curves)} curves from {args.curves}", file=sys.stderr)
    else:
        duration_s = args.duration_s
        seeds = args.seeds
        curves = []
        with PythonPlant() as plant:
            t_total0 = time.monotonic()
            for condition in CONDITIONS:
                print(
                    f"simulating {condition.key} ({duration_s:.1f} s, "
                    f"{len(seeds)} seeds)...",
                    file=sys.stderr,
                    flush=True,
                )
                curves.append(
                    simulate_condition(
                        plant,
                        condition,
                        seeds=seeds,
                        duration_s=duration_s,
                    ),
                )
            total_elapsed = time.monotonic() - t_total0
            print(
                f"all {len(CONDITIONS)} conditions done in {total_elapsed:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        save_curves(curves, args.curves, duration_s=duration_s, seeds=seeds)
        print(f"wrote curves cache {args.curves}", file=sys.stderr)

    title = "Oscillatory activity of model neurons in the GPi"
    panel = plot_fig1b(curves, out_path=args.out, title=title, y_max=args.y_max)

    replication = {
        c["key"]: (
            np.asarray(c["freqs_hz"], dtype=float),
            np.asarray(c["psd"], dtype=float),
        )
        for c in curves
    }
    paper_gates = fig1b_gates(replication)
    panel["gates"] = paper_gates["gates"]
    panel["gates_pass"] = paper_gates["pass"]
    panel["paper_gate_metrics"] = paper_gates["metrics"]

    manifest = {
        "figure": "mehregan_fig1b",
        "duration_s": duration_s,
        "seeds": list(seeds),
        "curves_cache": str(args.curves),
        "plot_only": args.plot_only,
        "panel": panel,
        "gates": paper_gates["gates"],
        "gates_pass": paper_gates["pass"],
        "paper_ref": paper_gates["paper_ref"],
        "conditions": [
            {
                "key": c["key"],
                "pd": c["pd"],
                "dbs_hz": c["dbs_hz"],
                "p_beta_mean": c["p_beta_mean"],
                "p_beta_std": c["p_beta_std"],
                "p_beta_by_seed": c["p_beta_by_seed"],
                "n_seeds": c["n_seeds"],
            }
            for c in curves
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"gates_pass={paper_gates['pass']} gates={paper_gates['gates']}",
        file=sys.stderr,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if paper_gates["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
