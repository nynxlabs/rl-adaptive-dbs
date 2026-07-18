#!/usr/bin/env python3
"""Path A sweep: So-style TH SMC (baseline 0) × ggith for Fig 2b EI gates.

So et al. 2012: SMC pulses (3.5 µA/cm², 5 ms) into TH; TH not spontaneously
active. Kumaravelu replaced those pulses with constant Iappth=1.2 and kept
ggith=0.112. This probe restores So-like drive (iappth_baseline=0 + BoC pulses)
and scales GPi→TH conductance looking for:

  healthy_ei < pd_no_tx_ei   (Gao gate)
  pd_130_ei  < pd_no_tx_ei   (Fig 2b blue-below-red)

Usage:
  uv run python scripts/probes/fig2b_ei_so_path_a_sweep.py
  uv run python scripts/probes/fig2b_ei_so_path_a_sweep.py --quick
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
from envs.plant.biomarkers import DEFAULT_EI_WINDOW_S, error_index, thalamic_misfire_breakdown
from envs.plant.dbs import create_dbs_current

WARMUP_S = 2.0
DISPLAY_S = 12.0
INTEGRATE_S = WARMUP_S + DISPLAY_S
DBS_ONSET_SIM_S = WARMUP_S + 2.0
WINDOW_S = DEFAULT_EI_WINDOW_S
SPIKE_HEADROOM_HZ = 60.0
SPIKE_BUFFER_MARGIN = 64
DISPLAY_T = 12.0
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_so_path_a_sweep.json")
DEFAULT_GGITH = (0.112, 0.224, 0.336, 0.448, 0.672)
QUICK_GGITH = (0.112, 0.336, 0.672)


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


def run_one(
    *,
    seed: int,
    pd: int,
    stim_hz: float,
    iappth_baseline: float,
    ggith: float,
    smc_amplitude: float,
) -> dict[str, Any]:
    with PythonPlant() as plant:
        plant.config = PlantConfig(
            pd=pd,
            corstim=0,
            smc_schedule="boc",
            smc_site="thalamic",
            smc_pulse_source="drive",
            smc_amplitude=smc_amplitude,
            iappth_baseline=iappth_baseline,
            ggith=ggith,
        )
        plant.reset(seed=seed)
        dt_ms = plant.config.dt_ms
        spec = dbs_spec_hz(frequency_hz=stim_hz, duration_s=INTEGRATE_S, dt_ms=dt_ms)
        buf = fig2_spike_buffer_size()
        t0 = time.monotonic()
        result = plant.integrate(
            INTEGRATE_S,
            spec,
            record_spikes=False,
            record_th_spikes=True,
            th_spike_buffer_size=buf,
        )
        elapsed = time.monotonic() - t0
        th_spikes = result.info.get("th_spikes")
        smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
        n_neurons = plant.config.neurons_per_region
    if not th_spikes:
        raise RuntimeError("no th_spikes")
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
    return {
        "pd": pd,
        "stim_hz": stim_hz,
        "iappth_baseline": iappth_baseline,
        "ggith": ggith,
        "smc_amplitude": smc_amplitude,
        "elapsed_s": elapsed,
        "ei_t12": ei,
        "breakdown_t12": breakdown,
        "n_th_spikes_total": int(sum(int(np.asarray(s).size) for s in th_spikes)),
    }


def score_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["pd"], r["stim_hz"]): r for r in rows}
    h0 = by[(0, 0.0)]
    p0 = by[(1, 0.0)]
    p130 = by[(1, 130.0)]
    healthy_lt_pd = h0["ei_t12"] < p0["ei_t12"]
    dbs_helps = p130["ei_t12"] < p0["ei_t12"]
    return {
        "healthy_ei": h0["ei_t12"],
        "pd_no_tx_ei": p0["ei_t12"],
        "pd_130_ei": p130["ei_t12"],
        "healthy_lt_pd": healthy_lt_pd,
        "dbs_lowers_pd": dbs_helps,
        "fig2b_gates_ok": healthy_lt_pd and dbs_helps,
        "delta_pd_ei_130": p130["ei_t12"] - p0["ei_t12"],
        "delta_misses_130": (
            p130["breakdown_t12"]["misses"] - p0["breakdown_t12"]["misses"]
        ),
        "delta_doubles_130": (
            p130["breakdown_t12"]["doubles"] - p0["breakdown_t12"]["doubles"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smc-amplitude", type=float, default=3.5)
    parser.add_argument(
        "--baselines",
        type=str,
        default="0",
        help="Comma list of iappth_baseline values (So-style default: 0)",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    baselines = [float(x) for x in args.baselines.split(",") if x.strip()]
    ggiths = QUICK_GGITH if args.quick else DEFAULT_GGITH
    conditions = ((0, 0.0), (1, 0.0), (1, 130.0))  # healthy, PD, PD+130

    groups: list[dict[str, Any]] = []
    for baseline in baselines:
        for ggith in ggiths:
            rows: list[dict[str, Any]] = []
            for pd, hz in conditions:
                label = f"base{baseline:g}_ggith{ggith:g}_pd{pd}_hz{hz:g}"
                print(f"running {label}...", file=sys.stderr, flush=True)
                rows.append(
                    run_one(
                        seed=args.seed,
                        pd=pd,
                        stim_hz=hz,
                        iappth_baseline=baseline,
                        ggith=ggith,
                        smc_amplitude=args.smc_amplitude,
                    )
                )
            score = score_group(rows)
            print(
                f"  → healthy={score['healthy_ei']:.3f} pd={score['pd_no_tx_ei']:.3f} "
                f"pd130={score['pd_130_ei']:.3f} "
                f"healthy<pd={score['healthy_lt_pd']} dbs_helps={score['dbs_lowers_pd']}",
                file=sys.stderr,
                flush=True,
            )
            groups.append(
                {
                    "iappth_baseline": baseline,
                    "ggith": ggith,
                    "smc_amplitude": args.smc_amplitude,
                    "conditions": rows,
                    "score": score,
                }
            )

    winners = [g for g in groups if g["score"]["fig2b_gates_ok"]]
    almost = [
        g
        for g in groups
        if g["score"]["healthy_lt_pd"] or g["score"]["dbs_lowers_pd"]
    ]
    payload = {
        "probe": "fig2b_ei_so_path_a_sweep",
        "seed": args.seed,
        "notes": {
            "so_smc": "3.5 uA/cm2, 5 ms pulses into TH; TH not spontaneously active",
            "kumaravelu_default": "iappth_baseline=1.2, ggith=0.112",
            "gates": "healthy_ei < pd_no_tx and pd_130 < pd_no_tx",
        },
        "groups": groups,
        "n_winners": len(winners),
        "winners": [
            {
                "iappth_baseline": w["iappth_baseline"],
                "ggith": w["ggith"],
                "score": w["score"],
            }
            for w in winners
        ],
        "partial_hits": [
            {
                "iappth_baseline": g["iappth_baseline"],
                "ggith": g["ggith"],
                "score": g["score"],
            }
            for g in almost
            if not g["score"]["fig2b_gates_ok"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "n_winners": len(winners),
                "winners": payload["winners"],
                "partial_hits": payload["partial_hits"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
