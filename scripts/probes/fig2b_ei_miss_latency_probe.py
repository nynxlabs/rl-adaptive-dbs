#!/usr/bin/env python3
"""Fig 2b EI audit: miss/double breakdown + TH response latency vs Gao protocol.

Gao ICCPS 2020 Eq. (4): correct TH response = exactly one spike in (τ, τ+25 ms)
after SMCτ. SMC is embedded in TH activations (Iappth). Windowed EI uses Tw=2 s
(Gao Eq. 6). Mehregan Fig 2b uses 130 Hz cDBS; Gao demos often use 180 Hz.

This probe measures, at display t=12 (sim window [12, 14]):
  - misses / doubles / correct counts
  - latency of first TH spike after each SMCτ (hits and late arrivals)
  - EI under response windows {15, 25, 40, 60} ms
  - cortical Iappco vs thalamic Iappth (Gao default)
  - 0 / 130 / 180 Hz STN DBS

Usage:
  uv run python scripts/probes/fig2b_ei_miss_latency_probe.py
  uv run python scripts/probes/fig2b_ei_miss_latency_probe.py --quick
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
    DEFAULT_EI_RESPONSE_WINDOW_S,
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
DISPLAY_T = 12.0
RESPONSE_WINDOWS_S = (0.015, 0.025, 0.040, 0.060)
LATENCY_CAP_S = 0.100  # collect first spike up to 100 ms for late-arrival stats
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_miss_latency_probe.json")


def fig2_spike_buffer_size(*, integrate_s: float = INTEGRATE_S) -> int:
    return max(
        512,
        int(np.ceil(integrate_s * SPIKE_HEADROOM_HZ)) + SPIKE_BUFFER_MARGIN,
    )


def trailing_window_sim(display_t: float) -> tuple[float, float]:
    end = min(WARMUP_S + display_t, INTEGRATE_S)
    return max(0.0, end - WINDOW_S), end


def dbs_spec_hz(*, frequency_hz: float, duration_s: float, dt_ms: float) -> DbsSpec:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    if frequency_hz <= 0.0:
        return DbsSpec.none()
    stim = create_dbs_current(
        frequency_hz,
        tmax_ms=(duration_s - DBS_ONSET_SIM_S) * 1000.0,
        dt_ms=dt_ms,
    )
    onset_idx = int(round(DBS_ONSET_SIM_S * 1000.0 / dt_ms))
    end = min(onset_idx + stim.size, n_steps)
    idbs[onset_idx:end] = stim[: end - onset_idx]
    return DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(frequency_hz).pick_dbs_freq,
        idbs=idbs,
        mean_hz=frequency_hz,
    )


def first_spike_latencies(
    th_spikes: list[np.ndarray],
    smc_pulse_times_s: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    latency_cap_s: float = LATENCY_CAP_S,
    inclusive_pulse_end: bool = True,
) -> dict[str, Any]:
    """Per-(pulse, neuron) latency of earliest TH spike in (τ, τ+cap]."""
    pulses = np.asarray(smc_pulse_times_s, dtype=np.float64).reshape(-1)
    if inclusive_pulse_end:
        pulse_mask = (pulses >= t_start) & (pulses <= t_end)
    else:
        pulse_mask = (pulses >= t_start) & (pulses < t_end)
    window_pulses = pulses[pulse_mask]

    latencies: list[float] = []
    no_spike = 0
    for tau in window_pulses:
        t_lo = float(tau)
        t_hi = t_lo + latency_cap_s
        for spikes in th_spikes:
            arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                no_spike += 1
                continue
            in_window = (arr > t_lo) & (arr <= t_hi)
            if not np.any(in_window):
                no_spike += 1
                continue
            latencies.append(float(np.min(arr[in_window]) - t_lo))

    lat = np.asarray(latencies, dtype=np.float64)
    bins_ms = [0, 5, 10, 15, 25, 40, 60, 100]
    hist: dict[str, int] = {}
    if lat.size:
        counts, _ = np.histogram(lat * 1000.0, bins=bins_ms)
        for i, c in enumerate(counts):
            hist[f"{bins_ms[i]}-{bins_ms[i + 1]}ms"] = int(c)
    return {
        "n_trials": int(window_pulses.size * len(th_spikes)),
        "n_with_spike_within_cap": int(lat.size),
        "n_no_spike_within_cap": int(no_spike),
        "latency_ms_mean": float(lat.mean() * 1000.0) if lat.size else None,
        "latency_ms_median": float(np.median(lat) * 1000.0) if lat.size else None,
        "latency_ms_p90": float(np.percentile(lat, 90) * 1000.0) if lat.size else None,
        "frac_first_spike_gt_25ms": (
            float(np.mean(lat > DEFAULT_EI_RESPONSE_WINDOW_S)) if lat.size else None
        ),
        "hist_ms": hist,
    }


def run_condition(
    *,
    seed: int,
    smc_site: str,
    smc_pulse_source: str,
    smc_cortical_amplitude: float,
    smc_amplitude: float,
    stim_hz: float,
    label: str,
) -> dict[str, Any]:
    with PythonPlant() as plant:
        plant.config = PlantConfig(
            pd=1,
            corstim=0,
            smc_schedule="boc",
            smc_site=smc_site,  # type: ignore[arg-type]
            smc_pulse_source=smc_pulse_source,  # type: ignore[arg-type]
            smc_cortical_amplitude=smc_cortical_amplitude,
            smc_amplitude=smc_amplitude,
        )
        plant.reset(seed=seed)
        dt_ms = plant.config.dt_ms
        spec = dbs_spec_hz(frequency_hz=stim_hz, duration_s=INTEGRATE_S, dt_ms=dt_ms)
        buf = fig2_spike_buffer_size()
        need_cor = smc_pulse_source == "cor_spikes"
        t0 = time.monotonic()
        result = plant.integrate(
            INTEGRATE_S,
            spec,
            record_spikes=False,
            record_th_spikes=True,
            th_spike_buffer_size=buf,
            cor_spike_buffer_size=buf if need_cor else None,
        )
        elapsed = time.monotonic() - t0
        th_spikes = result.info.get("th_spikes")
        smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
        n_neurons = plant.config.neurons_per_region

    if not th_spikes:
        msg = f"{label}: integrate did not return th_spikes"
        raise RuntimeError(msg)

    t_start, t_end = trailing_window_sim(DISPLAY_T)
    breakdown = thalamic_misfire_breakdown(
        th_spikes,
        smc_times,
        t_start=t_start,
        t_end=t_end,
        inclusive_pulse_end=True,
    )
    ei_by_window: dict[str, float] = {}
    for rw in RESPONSE_WINDOWS_S:
        ei_by_window[f"{int(rw * 1000)}ms"] = error_index(
            th_spikes,
            smc_times,
            t_start=t_start,
            t_end=t_end,
            response_window_s=rw,
            inclusive_pulse_end=True,
            n_neurons=n_neurons,
        )
    latency = first_spike_latencies(
        th_spikes,
        smc_times,
        t_start=t_start,
        t_end=t_end,
        inclusive_pulse_end=True,
    )
    trials = max(1, breakdown["trials"])
    return {
        "label": label,
        "stim_hz": stim_hz,
        "smc_site": smc_site,
        "smc_pulse_source": smc_pulse_source,
        "elapsed_s": elapsed,
        "n_smc_pulses_total": int(smc_times.size),
        "n_th_spikes_total": int(sum(int(np.asarray(s).size) for s in th_spikes)),
        "window_sim": [t_start, t_end],
        "breakdown_t12": breakdown,
        "rates_t12": {
            "miss_rate": breakdown["misses"] / trials,
            "double_rate": breakdown["doubles"] / trials,
            "correct_rate": breakdown["correct"] / trials,
            "ei_25ms": ei_by_window["25ms"],
        },
        "ei_by_response_window_t12": ei_by_window,
        "latency_t12": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smc-cortical-amplitude", type=float, default=100.0)
    parser.add_argument("--smc-amplitude", type=float, default=3.5, help="Iappth BoC amp")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only cortical cor_spikes × {0,130} Hz (2 runs)",
    )
    args = parser.parse_args()

    if args.quick:
        grid = [
            ("cortical", "cor_spikes", 0.0),
            ("cortical", "cor_spikes", 130.0),
        ]
    else:
        grid = [
            ("cortical", "cor_spikes", 0.0),
            ("cortical", "cor_spikes", 130.0),
            ("cortical", "cor_spikes", 180.0),
            ("thalamic", "drive", 0.0),
            ("thalamic", "drive", 130.0),
            ("thalamic", "drive", 180.0),
        ]

    rows: list[dict[str, Any]] = []
    for smc_site, pulse_source, stim_hz in grid:
        label = f"{smc_site}_{pulse_source}_hz{stim_hz:g}"
        print(f"running {label}...", file=sys.stderr, flush=True)
        rows.append(
            run_condition(
                seed=args.seed,
                smc_site=smc_site,
                smc_pulse_source=pulse_source,
                smc_cortical_amplitude=args.smc_cortical_amplitude,
                smc_amplitude=args.smc_amplitude,
                stim_hz=stim_hz,
                label=label,
            )
        )

    # Pairwise ordering: DBS EI < no_tx EI at 25 ms
    ordering: dict[str, Any] = {}
    by_key = {(r["smc_site"], r["smc_pulse_source"], r["stim_hz"]): r for r in rows}
    for site, src in {("cortical", "cor_spikes"), ("thalamic", "drive")}:
        base = by_key.get((site, src, 0.0))
        if base is None:
            continue
        site_ord: dict[str, Any] = {"no_tx_ei": base["rates_t12"]["ei_25ms"]}
        for hz in (130.0, 180.0):
            dbs = by_key.get((site, src, hz))
            if dbs is None:
                continue
            ei_dbs = dbs["rates_t12"]["ei_25ms"]
            ei_base = base["rates_t12"]["ei_25ms"]
            site_ord[f"hz{hz:g}"] = {
                "ei": ei_dbs,
                "delta_ei": ei_dbs - ei_base,
                "delta_misses": dbs["breakdown_t12"]["misses"] - base["breakdown_t12"]["misses"],
                "delta_doubles": dbs["breakdown_t12"]["doubles"] - base["breakdown_t12"]["doubles"],
                "delta_correct": dbs["breakdown_t12"]["correct"] - base["breakdown_t12"]["correct"],
                "ordering_ok": ei_dbs < ei_base,
            }
        ordering[f"{site}_{src}"] = site_ord

    payload = {
        "probe": "fig2b_ei_miss_latency",
        "gao_protocol_notes": {
            "correct_response": "exactly one TH spike in (SMCτ, SMCτ+25ms)",
            "smc_embedding": "Iappth (thalamic) in Gao ICCPS 2020; Mehregan fig uses SMC for EI",
            "window_s": WINDOW_S,
            "dbs_demo_hz_gao": 180,
            "dbs_demo_hz_mehregan_fig2b": 130,
        },
        "seed": args.seed,
        "smc_cortical_amplitude": args.smc_cortical_amplitude,
        "smc_amplitude_thalamic": args.smc_amplitude,
        "conditions": rows,
        "ordering_at_t12": ordering,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(args.out), "ordering_at_t12": ordering}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
