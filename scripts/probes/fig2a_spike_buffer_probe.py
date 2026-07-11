#!/usr/bin/env python3
"""Probe GPi spike-buffer saturation on long Fig 2a integrates (seed 0 default).

The Numba integrator caps GPI spike recording at 512 events per neuron
(``integrator.py`` → ``numba_gpi_buf``). On 14 s no-DBS runs, several neurons
hit that cap around sim 11–12 s and stop recording — this drives the artificial
Fig 2a trailing-window cliff, not PD dynamics or integrate duration.

Usage:
  uv run python scripts/probes/fig2a_spike_buffer_probe.py
  uv run python scripts/probes/fig2a_spike_buffer_probe.py --seeds 0,1,2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.network.numba_loop import numba_loop_available

GPI_SPIKE_BUF_CAP = 512


def parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(p.strip()) for p in raw.split(",") if p.strip())


def probe_seed(seed: int, *, duration_s: float) -> dict[str, object]:
    with PythonPlant() as plant:
        plant.config = PlantConfig(pd=1)
        plant.reset(seed=seed)
        result = plant.integrate(duration_s, DbsSpec.none())

    per_neuron = [len(s) for s in result.gpi_spikes]
    last_t = [
        float(np.max(s)) if len(s) else float("nan") for s in result.gpi_spikes
    ]
    after_12 = [int((np.asarray(s) >= 12.0).sum()) for s in result.gpi_spikes]

    def bin_total(t0: float, t1: float) -> int:
        return sum(
            int(((np.asarray(s) >= t0) & (np.asarray(s) < t1)).sum())
            for s in result.gpi_spikes
        )

    return {
        "seed": seed,
        "duration_s": duration_s,
        "total_spikes": sum(per_neuron),
        "per_neuron": per_neuron,
        "capped_neurons": sum(c >= GPI_SPIKE_BUF_CAP for c in per_neuron),
        "last_spike_s": last_t,
        "spikes_after_12s": sum(after_12),
        "bin_10_12": bin_total(10.0, 12.0),
        "bin_12_14": bin_total(12.0, 14.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, default=(0,))
    parser.add_argument("--duration-s", type=float, default=14.0)
    args = parser.parse_args()

    print(f"numba_loop_available: {numba_loop_available()}")
    print(f"GPI spike buffer cap (Numba): {GPI_SPIKE_BUF_CAP} per neuron")
    print(f"integrate duration: {args.duration_s:.0f} s\n")

    for seed in args.seeds:
        row = probe_seed(seed, duration_s=args.duration_s)
        print(f"seed {seed}:")
        print(f"  total GPi spikes: {row['total_spikes']}")
        print(f"  per-neuron counts: {row['per_neuron']}")
        print(
            f"  capped at {GPI_SPIKE_BUF_CAP}: {row['capped_neurons']}/10 neurons"
        )
        print(f"  last spike times (s): {[f'{t:.2f}' for t in row['last_spike_s']]}")
        print(f"  spikes recorded after sim 12 s: {row['spikes_after_12s']}")
        print(f"  bin [10,12): {row['bin_10_12']} spikes")
        print(f"  bin [12,14): {row['bin_12_14']} spikes  ← cliff driver if low")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
