#!/usr/bin/env python3
"""Fig 2b deep-dive A: why cortical healthy EI > PD EI on Kumaravelu.

Kumaravelu TH has no Cor→TH synapse — only GPi inhibition (Igith) + Iappth.
Cortical SMC (Iappco) can affect TH only via Cor → BG → GPi → TH. This probe
checks whether EI "hits" are time-locked responses or rate/coincidence effects.

For pd in {0,1}, cortical BoC Cor-spike SMCτ, no DBS:
  - region spike rates (TH, GPi, Cor)
  - TH PSTH in 0–60 ms after SMCτ (1 ms bins)
  - EI vs chance EI (SMC times circularly shifted)
  - fraction of "correct" trials with first spike <10 ms (plausible evoked)

Usage:
  uv run python scripts/probes/fig2b_ei_pd_pathway_probe.py
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

WARMUP_S = 2.0
DISPLAY_S = 12.0
INTEGRATE_S = WARMUP_S + DISPLAY_S
WINDOW_S = DEFAULT_EI_WINDOW_S
SPIKE_HEADROOM_HZ = 60.0
SPIKE_BUFFER_MARGIN = 64
DISPLAY_T = 12.0
PSTH_EDGE_MS = np.arange(0.0, 61.0, 1.0)
N_SHUFFLES = 20
DEFAULT_OUT = Path("artifacts/probes/fig2b_ei_pd_pathway_probe.json")


def fig2_spike_buffer_size(*, integrate_s: float = INTEGRATE_S) -> int:
    return max(
        512,
        int(np.ceil(integrate_s * SPIKE_HEADROOM_HZ)) + SPIKE_BUFFER_MARGIN,
    )


def trailing_window_sim(display_t: float) -> tuple[float, float]:
    end = min(WARMUP_S + display_t, INTEGRATE_S)
    return max(0.0, end - WINDOW_S), end


def spike_count(spikes: list[np.ndarray], *, t0: float, t1: float) -> int:
    total = 0
    for s in spikes:
        arr = np.asarray(s, dtype=np.float64).reshape(-1)
        total += int(np.sum((arr >= t0) & (arr < t1)))
    return total


def th_psth(
    th_spikes: list[np.ndarray],
    smc_times: np.ndarray,
    *,
    t_start: float,
    t_end: float,
) -> dict[str, Any]:
    pulses = np.asarray(smc_times, dtype=np.float64)
    pulses = pulses[(pulses >= t_start) & (pulses <= t_end)]
    counts = np.zeros(len(PSTH_EDGE_MS) - 1, dtype=np.float64)
    n_trials = 0
    for tau in pulses:
        for spikes in th_spikes:
            arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
            rel = (arr - float(tau)) * 1000.0
            rel = rel[(rel >= 0.0) & (rel < PSTH_EDGE_MS[-1])]
            if rel.size:
                hist, _ = np.histogram(rel, bins=PSTH_EDGE_MS)
                counts += hist
            n_trials += 1
    rate_per_trial = counts / max(1, n_trials)
    # early evoked band vs late
    early = float(rate_per_trial[:10].sum())  # 0–10 ms
    mid = float(rate_per_trial[10:25].sum())  # 10–25 ms
    late = float(rate_per_trial[25:60].sum())  # 25–60 ms
    return {
        "n_pulse_neuron_trials": int(n_trials),
        "bin_edges_ms": PSTH_EDGE_MS.tolist(),
        "count_per_trial": rate_per_trial.tolist(),
        "mass_0_10ms": early,
        "mass_10_25ms": mid,
        "mass_25_60ms": late,
        "early_over_late_density": (
            (early / 10.0) / max(1e-12, late / 35.0)
        ),
    }


def chance_ei(
    th_spikes: list[np.ndarray],
    smc_times: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    n_neurons: int,
    rng: np.random.Generator,
    n_shuffles: int = N_SHUFFLES,
) -> dict[str, float]:
    pulses = np.asarray(smc_times, dtype=np.float64)
    window_pulses = pulses[(pulses >= t_start) & (pulses <= t_end)]
    if window_pulses.size == 0:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    span = max(t_end - t_start, 1e-9)
    values: list[float] = []
    for _ in range(n_shuffles):
        shift = float(rng.uniform(0.0, span))
        shuffled = t_start + np.mod(window_pulses - t_start + shift, span)
        values.append(
            error_index(
                th_spikes,
                shuffled,
                t_start=t_start,
                t_end=t_end,
                inclusive_pulse_end=True,
                n_neurons=n_neurons,
            )
        )
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def correct_latency_stats(
    th_spikes: list[np.ndarray],
    smc_times: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    response_window_s: float = DEFAULT_EI_RESPONSE_WINDOW_S,
) -> dict[str, Any]:
    pulses = np.asarray(smc_times, dtype=np.float64)
    pulses = pulses[(pulses >= t_start) & (pulses <= t_end)]
    hit_lat_ms: list[float] = []
    n_correct = 0
    n_correct_lt_10ms = 0
    for tau in pulses:
        for spikes in th_spikes:
            arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
            in_win = (arr > float(tau)) & (arr < float(tau) + response_window_s)
            count = int(in_win.sum())
            if count != 1:
                continue
            n_correct += 1
            lat = float(arr[in_win][0] - tau) * 1000.0
            hit_lat_ms.append(lat)
            if lat < 10.0:
                n_correct_lt_10ms += 1
    return {
        "n_correct": n_correct,
        "n_correct_lt_10ms": n_correct_lt_10ms,
        "frac_correct_lt_10ms": (
            n_correct_lt_10ms / n_correct if n_correct else None
        ),
        "hit_latency_ms_median": (
            float(np.median(hit_lat_ms)) if hit_lat_ms else None
        ),
    }


def run_condition(
    *,
    seed: int,
    pd: int,
    smc_site: str,
    smc_pulse_source: str,
    smc_cortical_amplitude: float,
    smc_amplitude: float,
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
        buf = fig2_spike_buffer_size()
        need_cor = smc_site == "cortical" or smc_pulse_source == "cor_spikes"
        t0 = time.monotonic()
        result = plant.integrate(
            INTEGRATE_S,
            DbsSpec.none(),
            record_spikes=True,  # GPi
            record_th_spikes=True,
            th_spike_buffer_size=buf,
            cor_spike_buffer_size=buf if need_cor else None,
            gpi_spike_buffer_size=buf,
        )
        elapsed = time.monotonic() - t0
        th_spikes = result.info.get("th_spikes")
        gpi_spikes = result.gpi_spikes
        cor_spikes = result.info.get("cor_spikes")
        smc_times = np.asarray(result.info.get("smc_pulse_times_s", []), dtype=float)
        n_neurons = plant.config.neurons_per_region

    if not th_spikes:
        raise RuntimeError(f"{label}: no th_spikes")

    t_start, t_end = trailing_window_sim(DISPLAY_T)
    dur = t_end - t_start
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
    rng = np.random.default_rng(seed + 17 + pd)
    chance = chance_ei(
        th_spikes,
        smc_times,
        t_start=t_start,
        t_end=t_end,
        n_neurons=n_neurons,
        rng=rng,
    )
    rates = {
        "th_hz_per_neuron": spike_count(th_spikes, t0=t_start, t1=t_end)
        / (n_neurons * dur),
        "gpi_hz_per_neuron": spike_count(list(gpi_spikes), t0=t_start, t1=t_end)
        / (n_neurons * dur),
    }
    if cor_spikes:
        rates["cor_hz_per_neuron"] = spike_count(list(cor_spikes), t0=t_start, t1=t_end) / (
            n_neurons * dur
        )

    return {
        "label": label,
        "pd": pd,
        "smc_site": smc_site,
        "smc_pulse_source": smc_pulse_source,
        "elapsed_s": elapsed,
        "ei_t12": ei,
        "chance_ei_t12": chance,
        "ei_minus_chance": ei - chance["mean"],
        "breakdown_t12": breakdown,
        "rates_t12": rates,
        "psth_t12": th_psth(th_spikes, smc_times, t_start=t_start, t_end=t_end),
        "correct_latency_t12": correct_latency_stats(
            th_spikes, smc_times, t_start=t_start, t_end=t_end
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smc-cortical-amplitude", type=float, default=100.0)
    parser.add_argument("--smc-amplitude", type=float, default=3.5)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    grid = [
        (0, "cortical", "cor_spikes"),
        (1, "cortical", "cor_spikes"),
        (0, "thalamic", "drive"),
        (1, "thalamic", "drive"),
    ]
    rows: list[dict[str, Any]] = []
    for pd, site, src in grid:
        label = f"pd{pd}_{site}_{src}"
        print(f"running {label}...", file=sys.stderr, flush=True)
        rows.append(
            run_condition(
                seed=args.seed,
                pd=pd,
                smc_site=site,
                smc_pulse_source=src,
                smc_cortical_amplitude=args.smc_cortical_amplitude,
                smc_amplitude=args.smc_amplitude,
                label=label,
            )
        )

    by = {r["label"]: r for r in rows}
    contrast = {
        "cortical": {
            "healthy_ei": by["pd0_cortical_cor_spikes"]["ei_t12"],
            "pd_ei": by["pd1_cortical_cor_spikes"]["ei_t12"],
            "healthy_chance": by["pd0_cortical_cor_spikes"]["chance_ei_t12"]["mean"],
            "pd_chance": by["pd1_cortical_cor_spikes"]["chance_ei_t12"]["mean"],
            "healthy_ei_minus_chance": by["pd0_cortical_cor_spikes"]["ei_minus_chance"],
            "pd_ei_minus_chance": by["pd1_cortical_cor_spikes"]["ei_minus_chance"],
            "healthy_th_hz": by["pd0_cortical_cor_spikes"]["rates_t12"]["th_hz_per_neuron"],
            "pd_th_hz": by["pd1_cortical_cor_spikes"]["rates_t12"]["th_hz_per_neuron"],
            "healthy_early_over_late": by["pd0_cortical_cor_spikes"]["psth_t12"][
                "early_over_late_density"
            ],
            "pd_early_over_late": by["pd1_cortical_cor_spikes"]["psth_t12"][
                "early_over_late_density"
            ],
            "interpretation_hint": (
                "If |EI-chance|~0 and early_over_late~1, cortical EI is coincidence, "
                "not evoked Cor→TH following (Kumaravelu has no Cor→TH)."
            ),
        },
        "thalamic": {
            "healthy_ei": by["pd0_thalamic_drive"]["ei_t12"],
            "pd_ei": by["pd1_thalamic_drive"]["ei_t12"],
            "healthy_chance": by["pd0_thalamic_drive"]["chance_ei_t12"]["mean"],
            "pd_chance": by["pd1_thalamic_drive"]["chance_ei_t12"]["mean"],
            "healthy_ei_minus_chance": by["pd0_thalamic_drive"]["ei_minus_chance"],
            "pd_ei_minus_chance": by["pd1_thalamic_drive"]["ei_minus_chance"],
            "healthy_early_over_late": by["pd0_thalamic_drive"]["psth_t12"][
                "early_over_late_density"
            ],
            "pd_early_over_late": by["pd1_thalamic_drive"]["psth_t12"][
                "early_over_late_density"
            ],
        },
        "architecture": {
            "kumaravelu_th_inputs": "Igith (from GPi) + Iappth only — no Cor→TH synapse",
            "pd_params": [
                "gcordrstr: 0.07 (healthy) → 0.026 (PD)",
                "g_m factor: 2.6 (healthy) → 1.5 (PD)",
                "ggege scale: 0.25 (healthy) → 1.0 (PD)",
            ],
        },
    }

    payload = {
        "probe": "fig2b_ei_pd_pathway",
        "seed": args.seed,
        "conditions": rows,
        "contrast": contrast,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(args.out), "contrast": contrast}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
