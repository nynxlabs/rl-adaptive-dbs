#!/usr/bin/env python3
"""Fig 2b EI diagnostic: misfire breakdown + BoC SMC amplitude sweep (Python plant).

Usage:
  uv run python scripts/probes/fig2b_ei_smc_amplitude_probe.py --smc-site cortical
  uv run python scripts/probes/fig2b_ei_smc_amplitude_probe.py --smc-site thalamic --amplitudes 0.5,1,2,3.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.biomarkers import (
    DEFAULT_EI_WINDOW_S,
    error_index,
    thalamic_misfire_breakdown,
)
from envs.plant.dbs import create_dbs_current

WARMUP_S = 2.0
DISPLAY_S = 12.0
INTEGRATE_S = WARMUP_S + DISPLAY_S
DBS_ONSET_DISPLAY_S = 2.0
DBS_ONSET_SIM_S = WARMUP_S + DBS_ONSET_DISPLAY_S
WINDOW_S = DEFAULT_EI_WINDOW_S
SPIKE_HEADROOM_HZ = 60.0
SPIKE_BUFFER_MARGIN = 64
DEFAULT_AMPLITUDES_THALAMIC = (0.5, 1.0, 2.0, 3.5)
DEFAULT_AMPLITUDES_CORTICAL = (10.0, 50.0, 100.0, 350.0)
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_smc_amplitude_probe.json")


def fig2_spike_buffer_size(*, integrate_s: float = INTEGRATE_S) -> int:
    return max(
        512,
        int(np.ceil(integrate_s * SPIKE_HEADROOM_HZ)) + SPIKE_BUFFER_MARGIN,
    )


def trailing_window_sim(display_t: float) -> tuple[float, float]:
    end = min(WARMUP_S + display_t, INTEGRATE_S)
    return max(0.0, end - WINDOW_S), end


def dbs_spec_130hz(*, duration_s: float, dt_ms: float) -> DbsSpec:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    stim = create_dbs_current(
        130.0,
        tmax_ms=(duration_s - DBS_ONSET_SIM_S) * 1000.0,
        dt_ms=dt_ms,
    )
    onset_idx = int(round(DBS_ONSET_SIM_S * 1000.0 / dt_ms))
    end = min(onset_idx + stim.size, n_steps)
    idbs[onset_idx:end] = stim[: end - onset_idx]
    return DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(130.0).pick_dbs_freq,
        idbs=idbs,
        mean_hz=130.0,
    )


def run_condition(
    *,
    seed: int,
    smc_amplitude: float,
    smc_site: str,
    stim_hz: float,
    label: str,
) -> dict[str, Any]:
    use_cortical = smc_site == "cortical"
    with PythonPlant() as plant:
        kwargs: dict[str, Any] = dict(
            pd=1,
            corstim=0 if use_cortical else 1,
            smc_schedule="boc",
            smc_site=smc_site,
        )
        if use_cortical:
            kwargs["smc_cortical_amplitude"] = smc_amplitude
        else:
            kwargs["smc_amplitude"] = smc_amplitude
        plant.config = PlantConfig(**kwargs)
        plant.reset(seed=seed)
        dt_ms = plant.config.dt_ms
        spec = dbs_spec_130hz(duration_s=INTEGRATE_S, dt_ms=dt_ms) if stim_hz > 0 else DbsSpec.none()
        t0 = time.monotonic()
        result = plant.integrate(
            INTEGRATE_S,
            spec,
            record_spikes=False,
            record_th_spikes=True,
            th_spike_buffer_size=fig2_spike_buffer_size(),
        )
        elapsed = time.monotonic() - t0
        th_spikes = result.info.get("th_spikes")
        smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
        n_neurons = plant.config.neurons_per_region
    if not th_spikes:
        msg = f"{label}: integrate did not return th_spikes"
        raise RuntimeError(msg)

    samples: dict[str, float] = {}
    breakdowns: dict[str, dict[str, int]] = {}
    for display_t in (0.0, 2.0, 12.0):
        t_start, t_end = trailing_window_sim(display_t)
        inclusive = display_t == 12.0
        key = f"t{display_t:g}"
        samples[key] = error_index(
            th_spikes,
            smc_times,
            t_start=t_start,
            t_end=t_end,
            inclusive_pulse_end=inclusive,
            n_neurons=n_neurons,
        )
        if display_t == 12.0:
            breakdowns[key] = thalamic_misfire_breakdown(
                th_spikes,
                smc_times,
                t_start=t_start,
                t_end=t_end,
                inclusive_pulse_end=inclusive,
            )

    b12 = breakdowns["t12"]
    return {
        "label": label,
        "stim_hz": stim_hz,
        "integrate_s": INTEGRATE_S,
        "elapsed_s": round(elapsed, 2),
        "th_spikes": int(sum(np.asarray(s).size for s in th_spikes)),
        "smc_pulses_total": int(smc_times.size),
        "ei": samples,
        "breakdown_t12": b12,
    }


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    no_tx = row["no_treatment"]
    hz130 = row["hz130"]
    ei0_gap = abs(no_tx["ei"]["t0"] - hz130["ei"]["t0"])
    ei12_no = no_tx["ei"]["t12"]
    ei12_hz = hz130["ei"]["t12"]
    row["overlap_t0"] = ei0_gap < 1e-9
    row["ordering_ok"] = ei12_hz < ei12_no
    row["baseline_band_ok"] = 0.20 <= ei12_no <= 0.45
    row["gate_ok"] = row["ordering_ok"] and row["baseline_band_ok"]
    return row


def print_table(results: list[dict[str, Any]], *, smc_site: str) -> None:
    print(f"\n=== SMC amplitude sweep (Python, BoC {smc_site}, seed 0) ===")
    print(
        f"{'amp':>5} | {'EI t0':>6} {'EI t2':>6} | "
        f"{'no_tx@12':>9} {'130Hz@12':>9} | {'order':>5} {'base':>5}"
    )
    print("-" * 72)
    for row in results:
        amp = row["smc_amplitude"]
        nt = row["no_treatment"]
        hz = row["hz130"]
        print(
            f"{amp:5.1f} | {nt['ei']['t0']:6.3f} {nt['ei']['t2']:6.3f} | "
            f"{nt['ei']['t12']:9.3f} {hz['ei']['t12']:9.3f} | "
            f"{'OK' if row['ordering_ok'] else 'INV':>5} "
            f"{'OK' if row['baseline_band_ok'] else 'low':>5}"
        )


def print_breakdown(row: dict[str, Any], *, smc_amplitude: float) -> None:
    print(f"\n=== Misfire breakdown @ display t=12 s (amp={smc_amplitude} µA/cm²) ===")
    for cond in ("no_treatment", "hz130"):
        block = row[cond]
        b = block["breakdown_t12"]
        ei = block["ei"]["t12"]
        trials = b["trials"] or 1
        print(f"\n{cond}:")
        print(f"  EI={ei:.4f}  pulses={b['n_pulses']}  trials={trials}")
        print(
            f"  misses={b['misses']} ({100*b['misses']/trials:.1f}%)  "
            f"doubles={b['doubles']} ({100*b['doubles']/trials:.1f}%)  "
            f"correct={b['correct']} ({100*b['correct']/trials:.1f}%)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smc-site",
        choices=("thalamic", "cortical"),
        default="cortical",
    )
    parser.add_argument(
        "--amplitudes",
        default=None,
        help="Comma-separated amplitudes (µA/cm²); defaults depend on --smc-site",
    )
    parser.add_argument(
        "--breakdown-only",
        action="store_true",
        help="Only run detailed breakdown at --amplitude (skip sweep)",
    )
    parser.add_argument("--amplitude", type=float, default=3.5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.amplitudes is not None:
        amplitudes = tuple(float(p.strip()) for p in args.amplitudes.split(",") if p.strip())
    elif args.smc_site == "cortical":
        amplitudes = DEFAULT_AMPLITUDES_CORTICAL
    else:
        amplitudes = DEFAULT_AMPLITUDES_THALAMIC

    if args.breakdown_only:
        amplitudes = (args.amplitude,)

    results: list[dict[str, Any]] = []
    t_all = time.monotonic()
    for amp in amplitudes:
        print(f"\n--- smc_site={args.smc_site} amplitude={amp} ---", file=sys.stderr, flush=True)
        no_tx = run_condition(
            seed=args.seed,
            smc_amplitude=amp,
            smc_site=args.smc_site,
            stim_hz=0.0,
            label="no_treatment",
        )
        hz130 = run_condition(
            seed=args.seed,
            smc_amplitude=amp,
            smc_site=args.smc_site,
            stim_hz=130.0,
            label="hz130",
        )
        row: dict[str, Any] = {
            "smc_site": args.smc_site,
            "smc_amplitude": amp,
            "seed": args.seed,
            "no_treatment": no_tx,
            "hz130": hz130,
        }
        evaluate_row(row)
        results.append(row)
        if args.breakdown_only or amp == amplitudes[-1]:
            print_breakdown(row, smc_amplitude=amp)

    if not args.breakdown_only:
        print_table(results, smc_site=args.smc_site)

    payload = {
        "probe": "fig2b_ei_smc_amplitude",
        "backend": "python",
        "smc_schedule": "boc",
        "smc_site": args.smc_site,
        "corstim": 0 if args.smc_site == "cortical" else 1,
        "seed": args.seed,
        "elapsed_s": round(time.monotonic() - t_all, 2),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {args.out}", file=sys.stderr)

    winners = [r for r in results if r.get("gate_ok")]
    if winners:
        best = min(winners, key=lambda r: r["hz130"]["ei"]["t12"])
        print(
            f"\nQualitative gate passed at amp={best['smc_amplitude']}: "
            f"no_tx@12={best['no_treatment']['ei']['t12']:.3f}, "
            f"130Hz@12={best['hz130']['ei']['t12']:.3f}",
            file=sys.stderr,
        )
    else:
        print("\nNo amplitude passed ordering + baseline band gate.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
