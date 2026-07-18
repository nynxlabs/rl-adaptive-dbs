#!/usr/bin/env python3
"""Fig 2b EI diagnostic: compare drive vs Cor-spike SMCτ (Python plant).

Usage:
  uv run python scripts/probes/fig2b_ei_cor_spike_probe.py
  uv run python scripts/probes/fig2b_ei_cor_spike_probe.py --smc-cortical-amplitude 100
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
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_cor_spike_probe.json")


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
    smc_pulse_source: str,
    smc_cortical_amplitude: float,
    stim_hz: float,
    label: str,
) -> dict[str, Any]:
    with PythonPlant() as plant:
        plant.config = PlantConfig(
            pd=1,
            corstim=0,
            smc_schedule="boc",
            smc_site="cortical",
            smc_pulse_source=smc_pulse_source,  # type: ignore[arg-type]
            smc_cortical_amplitude=smc_cortical_amplitude,
        )
        plant.reset(seed=seed)
        dt_ms = plant.config.dt_ms
        spec = dbs_spec_130hz(duration_s=INTEGRATE_S, dt_ms=dt_ms) if stim_hz > 0 else DbsSpec.none()
        buf = fig2_spike_buffer_size()
        t0 = time.monotonic()
        result = plant.integrate(
            INTEGRATE_S,
            spec,
            record_spikes=False,
            record_th_spikes=True,
            th_spike_buffer_size=buf,
            cor_spike_buffer_size=buf if smc_pulse_source == "cor_spikes" else None,
        )
        elapsed = time.monotonic() - t0
        th_spikes = result.info.get("th_spikes")
        smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
        drive_times = np.asarray(result.info.get("smc_drive_times_s", []), dtype=float)
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
        breakdowns[key] = thalamic_misfire_breakdown(
            th_spikes,
            smc_times,
            t_start=t_start,
            t_end=t_end,
            inclusive_pulse_end=inclusive,
        )

    return {
        "label": label,
        "stim_hz": stim_hz,
        "smc_pulse_source": smc_pulse_source,
        "integrate_s": INTEGRATE_S,
        "elapsed_s": elapsed,
        "n_smc_pulses": int(smc_times.size),
        "n_drive_pulses": int(drive_times.size),
        "n_th_spikes": int(sum(int(np.asarray(s).size) for s in th_spikes)),
        "ei": samples,
        "misfire_breakdown": breakdowns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smc-cortical-amplitude",
        type=float,
        default=100.0,
        help="BoC Iappco amplitude (default 100)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for pulse_source in ("drive", "cor_spikes"):
        for stim_hz, cond in ((0.0, "no_tx"), (130.0, "hz130")):
            label = f"{pulse_source}_{cond}"
            print(f"running {label}...", file=sys.stderr, flush=True)
            rows.append(
                run_condition(
                    seed=args.seed,
                    smc_pulse_source=pulse_source,
                    smc_cortical_amplitude=args.smc_cortical_amplitude,
                    stim_hz=stim_hz,
                    label=label,
                )
            )

    payload = {
        "probe": "fig2b_ei_cor_spike",
        "seed": args.seed,
        "smc_schedule": "boc",
        "smc_site": "cortical",
        "smc_cortical_amplitude": args.smc_cortical_amplitude,
        "conditions": rows,
        "ordering_at_t12": {
            "drive": _ordering(rows, "drive"),
            "cor_spikes": _ordering(rows, "cor_spikes"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def _ordering(rows: list[dict[str, Any]], pulse_source: str) -> str:
    no_tx = next(r for r in rows if r["smc_pulse_source"] == pulse_source and r["stim_hz"] == 0.0)
    hz130 = next(r for r in rows if r["smc_pulse_source"] == pulse_source and r["stim_hz"] == 130.0)
    ei_no = no_tx["ei"]["t12"]
    ei_hz = hz130["ei"]["t12"]
    if ei_hz < ei_no:
        return "blue_below_red"
    if ei_hz > ei_no:
        return "blue_above_red"
    return "equal"


if __name__ == "__main__":
    sys.exit(main())
