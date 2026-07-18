#!/usr/bin/env python3
"""Fig 2b EI diagnostic: STN DBS frequency × amplitude landscape (Python plant).

Holds SMC fixed (cortical BoC by default) and sweeps DBS carrier frequency and
amplitude after display t=2 s. Scores ordering (EI_130 < EI_no_tx at t=12) and
miss delta from misfire breakdown.

Usage:
  uv run python scripts/probes/fig2b_ei_dbs_sweep_probe.py
  uv run python scripts/probes/fig2b_ei_dbs_sweep_probe.py --quick
  uv run python scripts/probes/fig2b_ei_dbs_sweep_probe.py --frequencies 45,130 --amplitudes 100,300
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
from envs.plant.dbs import DBS_AMPLITUDE_NA_PER_CM2, DBS_PULSE_WIDTH_MS, create_dbs_current

WARMUP_S = 2.0
DISPLAY_S = 12.0
INTEGRATE_S = WARMUP_S + DISPLAY_S
DBS_ONSET_DISPLAY_S = 2.0
DBS_ONSET_SIM_S = WARMUP_S + DBS_ONSET_DISPLAY_S
WINDOW_S = DEFAULT_EI_WINDOW_S
SPIKE_HEADROOM_HZ = 60.0
SPIKE_BUFFER_MARGIN = 64
DEFAULT_FREQUENCIES_HZ = (45.0, 80.0, 100.0, 130.0, 160.0, 200.0)
DEFAULT_AMPLITUDES_NA = (75.0, 150.0, 300.0, 450.0)
QUICK_FREQUENCIES_HZ = (45.0, 130.0)
QUICK_AMPLITUDES_NA = (150.0, 300.0)
DEFAULT_SMC_CORTICAL_AMPLITUDE = 100.0
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_dbs_sweep_probe.json")


def fig2_spike_buffer_size(*, integrate_s: float = INTEGRATE_S) -> int:
    return max(
        512,
        int(np.ceil(integrate_s * SPIKE_HEADROOM_HZ)) + SPIKE_BUFFER_MARGIN,
    )


def trailing_window_sim(display_t: float) -> tuple[float, float]:
    end = min(WARMUP_S + display_t, INTEGRATE_S)
    return max(0.0, end - WINDOW_S), end


def dbs_spec_with_onset(
    *,
    onset_s: float,
    frequency_hz: float,
    duration_s: float,
    dt_ms: float,
    amplitude_na_per_cm2: float = DBS_AMPLITUDE_NA_PER_CM2,
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS,
) -> DbsSpec:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    if frequency_hz > 0.0 and onset_s < duration_s:
        stim = create_dbs_current(
            frequency_hz,
            tmax_ms=(duration_s - onset_s) * 1000.0,
            dt_ms=dt_ms,
            amplitude=amplitude_na_per_cm2,
            pulse_width_ms=pulse_width_ms,
        )
        onset_idx = int(round(onset_s * 1000.0 / dt_ms))
        end = min(onset_idx + stim.size, n_steps)
        idbs[onset_idx:end] = stim[: end - onset_idx]
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(frequency_hz).pick_dbs_freq,
            idbs=idbs,
            mean_hz=frequency_hz,
        )
    return DbsSpec.none()


def run_integrate(
    *,
    seed: int,
    smc_cortical_amplitude: float,
    dbs_spec: DbsSpec,
) -> dict[str, Any]:
    with PythonPlant() as plant:
        plant.config = PlantConfig(
            pd=1,
            corstim=0,
            smc_schedule="boc",
            smc_site="cortical",
            smc_cortical_amplitude=smc_cortical_amplitude,
        )
        plant.reset(seed=seed)
        dt_ms = plant.config.dt_ms
        t0 = time.monotonic()
        result = plant.integrate(
            INTEGRATE_S,
            dbs_spec,
            record_spikes=False,
            record_th_spikes=True,
            th_spike_buffer_size=fig2_spike_buffer_size(),
        )
        elapsed = time.monotonic() - t0
        th_spikes = result.info.get("th_spikes")
        smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
        n_neurons = plant.config.neurons_per_region
    if not th_spikes:
        msg = "integrate did not return th_spikes"
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
        "integrate_s": INTEGRATE_S,
        "elapsed_s": round(elapsed, 2),
        "th_spikes": int(sum(np.asarray(s).size for s in th_spikes)),
        "smc_pulses_total": int(smc_times.size),
        "ei": samples,
        "breakdown_t12": b12,
    }


def evaluate_point(
    *,
    no_tx: dict[str, Any],
    dbs: dict[str, Any],
    frequency_hz: float,
    amplitude_na_per_cm2: float,
    pulse_width_ms: float,
) -> dict[str, Any]:
    ei12_no = no_tx["ei"]["t12"]
    ei12_dbs = dbs["ei"]["t12"]
    misses_no = no_tx["breakdown_t12"]["misses"]
    misses_dbs = dbs["breakdown_t12"]["misses"]
    delta_misses = misses_dbs - misses_no
    delta_ei = ei12_dbs - ei12_no
    ordering_ok = ei12_dbs < ei12_no
    baseline_band_ok = 0.20 <= ei12_no <= 0.45
    gate_ok = ordering_ok and baseline_band_ok
    return {
        "frequency_hz": frequency_hz,
        "amplitude_na_per_cm2": amplitude_na_per_cm2,
        "pulse_width_ms": pulse_width_ms,
        "dbs": dbs,
        "ei12_no_tx": ei12_no,
        "ei12_dbs": ei12_dbs,
        "delta_ei": round(delta_ei, 6),
        "delta_misses": delta_misses,
        "ordering_ok": ordering_ok,
        "baseline_band_ok": baseline_band_ok,
        "gate_ok": gate_ok,
    }


def parse_float_list(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        msg = "expected comma-separated floats"
        raise ValueError(msg)
    return tuple(float(p) for p in parts)


def print_table(
    points: list[dict[str, Any]],
    *,
    frequencies: tuple[float, ...],
    amplitudes: tuple[float, ...],
) -> None:
    print("\n=== DBS sweep @ display t=12 (ordering: DBS EI < no_tx EI) ===")
    header = f"{'freq':>6} |" + "".join(f" {a:>7.0f}" for a in amplitudes)
    print(header)
    print("-" * len(header))
    for freq in frequencies:
        cells: list[str] = []
        for amp in amplitudes:
            row = next(
                (
                    p
                    for p in points
                    if p.get("frequency_hz") == freq
                    and p.get("amplitude_na_per_cm2") == amp
                ),
                None,
            )
            if row is None or "error" in row:
                cells.append("     ERR")
            elif row["ordering_ok"]:
                cells.append(f" {row['ei12_dbs']:7.3f}*")
            else:
                cells.append(f" {row['ei12_dbs']:7.3f}")
        print(f"{freq:6.0f} |" + "".join(cells))
    print("(* = ordering_ok; columns are nA/cm² amplitude, values are DBS EI@12)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smc-cortical-amplitude",
        type=float,
        default=DEFAULT_SMC_CORTICAL_AMPLITUDE,
    )
    parser.add_argument("--frequencies", default=None, help="Comma-separated Hz")
    parser.add_argument("--amplitudes", default=None, help="Comma-separated nA/cm²")
    parser.add_argument(
        "--pulse-width-ms",
        type=float,
        default=DBS_PULSE_WIDTH_MS,
        help=f"DBS pulse width (default {DBS_PULSE_WIDTH_MS})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small 2×2 grid (45/130 Hz × 150/300 nA/cm²)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.quick:
        frequencies = QUICK_FREQUENCIES_HZ
        amplitudes = QUICK_AMPLITUDES_NA
    else:
        frequencies = (
            parse_float_list(args.frequencies)
            if args.frequencies
            else DEFAULT_FREQUENCIES_HZ
        )
        amplitudes = (
            parse_float_list(args.amplitudes)
            if args.amplitudes
            else DEFAULT_AMPLITUDES_NA
        )

    t_all = time.monotonic()
    print("running no_treatment baseline (once)...", file=sys.stderr, flush=True)
    no_tx = run_integrate(
        seed=args.seed,
        smc_cortical_amplitude=args.smc_cortical_amplitude,
        dbs_spec=DbsSpec.none(),
    )
    print(
        f"  no_tx EI@12={no_tx['ei']['t12']:.3f} "
        f"misses={no_tx['breakdown_t12']['misses']}",
        file=sys.stderr,
        flush=True,
    )

    points: list[dict[str, Any]] = []
    total = len(frequencies) * len(amplitudes)
    done = 0
    for freq in frequencies:
        for amp in amplitudes:
            done += 1
            print(
                f"[{done}/{total}] DBS {freq:g} Hz @ {amp:g} nA/cm² ...",
                file=sys.stderr,
                flush=True,
            )
            spec = dbs_spec_with_onset(
                onset_s=DBS_ONSET_SIM_S,
                frequency_hz=freq,
                duration_s=INTEGRATE_S,
                dt_ms=0.01,
                amplitude_na_per_cm2=amp,
                pulse_width_ms=args.pulse_width_ms,
            )
            try:
                dbs = run_integrate(
                    seed=args.seed,
                    smc_cortical_amplitude=args.smc_cortical_amplitude,
                    dbs_spec=spec,
                )
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
                points.append(
                    {
                        "frequency_hz": freq,
                        "amplitude_na_per_cm2": amp,
                        "pulse_width_ms": args.pulse_width_ms,
                        "error": str(exc),
                        "ordering_ok": False,
                        "gate_ok": False,
                    }
                )
                continue
            point = evaluate_point(
                no_tx=no_tx,
                dbs=dbs,
                frequency_hz=freq,
                amplitude_na_per_cm2=amp,
                pulse_width_ms=args.pulse_width_ms,
            )
            points.append(point)
            print(
                f"  EI@12={point['ei12_dbs']:.3f} "
                f"Δmiss={point['delta_misses']:+d} "
                f"{'OK' if point['ordering_ok'] else 'INV'}",
                file=sys.stderr,
                flush=True,
            )

    winners = [p for p in points if p.get("gate_ok")]
    best_ordering = sorted(
        [p for p in points if p.get("ordering_ok")],
        key=lambda p: (p["ei12_dbs"], abs(p["delta_ei"])),
    )
    failed = [p for p in points if "error" in p]

    payload: dict[str, Any] = {
        "probe": "fig2b_ei_dbs_sweep",
        "backend": "python",
        "smc_schedule": "boc",
        "smc_site": "cortical",
        "smc_cortical_amplitude": args.smc_cortical_amplitude,
        "pulse_width_ms": args.pulse_width_ms,
        "seed": args.seed,
        "frequencies_hz": list(frequencies),
        "amplitudes_na_per_cm2": list(amplitudes),
        "no_treatment": no_tx,
        "points": points,
        "ordering_winners": [
            {
                "frequency_hz": p["frequency_hz"],
                "amplitude_na_per_cm2": p["amplitude_na_per_cm2"],
                "ei12_dbs": p["ei12_dbs"],
                "ei12_no_tx": p["ei12_no_tx"],
                "delta_misses": p["delta_misses"],
            }
            for p in best_ordering
        ],
        "gate_winners": [
            {
                "frequency_hz": p["frequency_hz"],
                "amplitude_na_per_cm2": p["amplitude_na_per_cm2"],
                "ei12_dbs": p["ei12_dbs"],
                "ei12_no_tx": p["ei12_no_tx"],
            }
            for p in winners
        ],
        "failed_points": failed,
        "elapsed_s": round(time.monotonic() - t_all, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print_table(points, frequencies=frequencies, amplitudes=amplitudes)
    print(f"\nWrote {args.out}", file=sys.stderr)
    if winners:
        best = min(winners, key=lambda p: p["ei12_dbs"])
        print(
            f"\nGate passed: {best['frequency_hz']:g} Hz @ "
            f"{best['amplitude_na_per_cm2']:g} nA/cm² — "
            f"no_tx@12={best['ei12_no_tx']:.3f}, "
            f"dbs@12={best['ei12_dbs']:.3f}, "
            f"Δmiss={best['delta_misses']:+d}",
            file=sys.stderr,
        )
    elif best_ordering:
        best = best_ordering[0]
        print(
            f"\nOrdering OK (baseline band may fail): {best['frequency_hz']:g} Hz @ "
            f"{best['amplitude_na_per_cm2']:g} nA/cm² — "
            f"dbs@12={best['ei12_dbs']:.3f} < no_tx@12={best['ei12_no_tx']:.3f}",
            file=sys.stderr,
        )
    else:
        print(
            "\nNo (frequency, amplitude) point flipped ordering at t=12.",
            file=sys.stderr,
        )
    print(json.dumps(payload["ordering_winners"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
