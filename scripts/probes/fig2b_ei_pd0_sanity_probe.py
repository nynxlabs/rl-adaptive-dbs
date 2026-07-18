#!/usr/bin/env python3
"""Fig 2b EI sanity: healthy (pd=0) vs PD (pd=1) miss rates.

Gao Fig. 4 / narrative: healthy TH follows SMC with rare errors; PD without DBS
has many errors; DBS should move PD toward healthy. If pd=0 EI is already high
(or DBS worsens pd=0), SMC wiring / plant gap is implicated.

Usage:
  uv run python scripts/probes/fig2b_ei_pd0_sanity_probe.py
  uv run python scripts/probes/fig2b_ei_pd0_sanity_probe.py --sites cortical,thalamic
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
DBS_ONSET_SIM_S = WARMUP_S + 2.0
WINDOW_S = DEFAULT_EI_WINDOW_S
SPIKE_HEADROOM_HZ = 60.0
SPIKE_BUFFER_MARGIN = 64
DISPLAY_T = 12.0
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_pd0_sanity_probe.json")


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


def run_condition(
    *,
    seed: int,
    pd: int,
    smc_site: str,
    smc_pulse_source: str,
    smc_cortical_amplitude: float,
    smc_amplitude: float,
    stim_hz: float,
    label: str,
) -> dict[str, Any]:
    with PythonPlant() as plant:
        plant.config = PlantConfig(
            pd=pd,
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
        raise RuntimeError(f"{label}: no th_spikes")

    t_start, t_end = trailing_window_sim(DISPLAY_T)
    breakdown = thalamic_misfire_breakdown(
        th_spikes,
        smc_times,
        t_start=t_start,
        t_end=t_end,
        inclusive_pulse_end=True,
    )
    ei = error_index(
        th_spikes,
        smc_times,
        t_start=t_start,
        t_end=t_end,
        inclusive_pulse_end=True,
        n_neurons=n_neurons,
    )
    trials = max(1, breakdown["trials"])
    return {
        "label": label,
        "pd": pd,
        "stim_hz": stim_hz,
        "smc_site": smc_site,
        "smc_pulse_source": smc_pulse_source,
        "elapsed_s": elapsed,
        "n_smc_pulses_total": int(smc_times.size),
        "n_th_spikes_total": int(sum(int(np.asarray(s).size) for s in th_spikes)),
        "ei_t12": ei,
        "breakdown_t12": breakdown,
        "rates_t12": {
            "miss_rate": breakdown["misses"] / trials,
            "double_rate": breakdown["doubles"] / trials,
            "correct_rate": breakdown["correct"] / trials,
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-site gates inspired by Gao: healthy << PD no_tx; DBS should lower PD EI."""
    out: dict[str, Any] = {}
    for site_key in ("cortical_cor_spikes", "thalamic_drive"):
        site_rows = [r for r in rows if f"{r['smc_site']}_{r['smc_pulse_source']}" == site_key]
        if not site_rows:
            continue
        by = {(r["pd"], r["stim_hz"]): r for r in site_rows}
        h0 = by.get((0, 0.0))
        p0 = by.get((1, 0.0))
        p130 = by.get((1, 130.0))
        h130 = by.get((0, 130.0))
        if not (h0 and p0 and p130):
            continue
        gate = {
            "healthy_ei": h0["ei_t12"],
            "pd_no_tx_ei": p0["ei_t12"],
            "pd_130_ei": p130["ei_t12"],
            "healthy_lt_pd_no_tx": h0["ei_t12"] < p0["ei_t12"],
            "dbs_lowers_pd": p130["ei_t12"] < p0["ei_t12"],
            "pd_130_near_healthy": abs(p130["ei_t12"] - h0["ei_t12"])
            <= 0.05 + 0.5 * abs(p0["ei_t12"] - h0["ei_t12"]),
            "delta_pd_misses_130": (
                p130["breakdown_t12"]["misses"] - p0["breakdown_t12"]["misses"]
            ),
        }
        if h130 is not None:
            gate["healthy_130_ei"] = h130["ei_t12"]
            gate["dbs_raises_healthy"] = h130["ei_t12"] > h0["ei_t12"]
        out[site_key] = gate
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smc-cortical-amplitude", type=float, default=100.0)
    parser.add_argument("--smc-amplitude", type=float, default=3.5)
    parser.add_argument(
        "--sites",
        type=str,
        default="cortical,thalamic",
        help="Comma list: cortical and/or thalamic",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    grid: list[tuple[int, str, str, float]] = []
    for site in sites:
        if site == "cortical":
            src = "cor_spikes"
        elif site == "thalamic":
            src = "drive"
        else:
            raise SystemExit(f"unknown site {site!r}")
        for pd in (0, 1):
            for hz in (0.0, 130.0):
                grid.append((pd, site, src, hz))

    rows: list[dict[str, Any]] = []
    for pd, site, src, hz in grid:
        label = f"pd{pd}_{site}_{src}_hz{hz:g}"
        print(f"running {label}...", file=sys.stderr, flush=True)
        rows.append(
            run_condition(
                seed=args.seed,
                pd=pd,
                smc_site=site,
                smc_pulse_source=src,
                smc_cortical_amplitude=args.smc_cortical_amplitude,
                smc_amplitude=args.smc_amplitude,
                stim_hz=hz,
                label=label,
            )
        )

    gates = summarize(rows)
    payload = {
        "probe": "fig2b_ei_pd0_sanity",
        "seed": args.seed,
        "expectation": {
            "healthy_ei_low": "Gao: rare errors in healthy",
            "pd_no_tx_high": "Gao: frequent errors in PD without DBS",
            "dbs_moves_pd_toward_healthy": "Gao/Mehregan Fig 2b direction",
        },
        "conditions": rows,
        "gates": gates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(args.out), "gates": gates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
