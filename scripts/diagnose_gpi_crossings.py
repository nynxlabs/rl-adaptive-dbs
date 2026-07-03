#!/usr/bin/env python3
"""Report GPi -20 mV crossings and post-spike voltage for integrator debugging."""

from __future__ import annotations

import argparse

import numpy as np

from envs.plant import DbsSpec, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-ms", type=float, default=50.0)
    parser.add_argument("--neuron", type=int, default=0)
    parser.add_argument(
        "--fixture",
        type=str,
        default="tests/fixtures/plant_init_seed42.npz",
    )
    args = parser.parse_args()

    draws = NetworkInitDraws.from_npz(args.fixture)
    cfg = PlantConfig()
    duration_s = args.duration_ms / 1000.0
    rng = np.random.default_rng(args.seed)

    result = integrate_network(
        config=cfg,
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=rng,
        iteration=1,
        seed=args.seed,
        init_draws=draws,
        return_traces=("vgi",),
    )

    v = result.info["traces"]["vgi"][args.neuron]
    dt_ms = cfg.dt_ms
    t_s = np.arange(v.size) * dt_ms / 1000.0
    cross_idx = np.flatnonzero((v[:-1] <= -20.0) & (v[1:] > -20.0))

    print(f"GPi neuron {args.neuron}: {len(cross_idx)} upward -20 mV crossings")
    for k, idx in enumerate(cross_idx[:8]):
        t_cross = t_s[idx]
        post = v[idx : idx + 200]
        below = np.flatnonzero(post < -20.0)
        first_repol = below[0] if below.size else None
        repol_ms = (first_repol * dt_ms) if first_repol is not None else None
        print(
            f"  crossing {k}: t={t_cross:.5f}s step={idx} "
            f"v@cross={v[idx]:.2f} min_200steps={post.min():.2f} "
            f"repol_below_-20_in={repol_ms} ms"
        )


if __name__ == "__main__":
    main()
