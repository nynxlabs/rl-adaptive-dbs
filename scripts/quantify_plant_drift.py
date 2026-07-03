#!/usr/bin/env python3
"""Report MATLAB vs PythonPlant GPi spike-train drift (fixed ICs and cross-backend)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from envs.plant import DbsSpec, MatlabPlant, PlantConfig, PythonPlant
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from envs.plant.network.matlab_rng import load_cached_init_draws
from tests.envs.plant_backends import spike_count_vector


def _first_spike_mismatch(
    matlab_times: np.ndarray,
    python_times: np.ndarray,
    *,
    atol_s: float = 1e-5,
) -> dict[str, object] | None:
    for index, (ref, cand) in enumerate(zip(matlab_times, python_times, strict=False)):
        if not np.isclose(ref, cand, rtol=0.0, atol=atol_s):
            return {
                "spike_index": index,
                "matlab_s": float(ref),
                "python_s": float(cand),
                "delta_ms": float((cand - ref) * 1000.0),
            }
    if matlab_times.size != python_times.size:
        return {
            "spike_index": min(matlab_times.size, python_times.size),
            "matlab_count": int(matlab_times.size),
            "python_count": int(python_times.size),
        }
    return None


def fixed_ic_report(
    *,
    seed: int,
    duration_s: float,
    fixture: Path | None,
) -> dict[str, object]:
    draws_path = fixture
    if draws_path is None:
        cached = load_cached_init_draws(seed)
        if cached is None:
            msg = f"no init fixture for seed={seed}"
            raise FileNotFoundError(msg)
        draws = cached
    else:
        draws = NetworkInitDraws.from_npz(draws_path)

    cfg = PlantConfig()
    with MatlabPlant(cfg) as matlab_plant:
        matlab = matlab_plant.reset(seed=seed).integrate(duration_s, DbsSpec.none())

    python = integrate_network(
        config=cfg,
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=np.random.default_rng(seed),
        iteration=1,
        seed=seed,
        init_draws=draws,
    )

    neuron = 0
    mismatch = _first_spike_mismatch(matlab.gpi_spikes[neuron], python.gpi_spikes[neuron])
    return {
        "mode": "fixed_ic",
        "seed": seed,
        "duration_s": duration_s,
        "neuron": neuron,
        "matlab_spike_counts": spike_count_vector(matlab.gpi_spikes),
        "python_spike_counts": spike_count_vector(python.gpi_spikes),
        "matlab_p_beta": matlab.p_beta,
        "python_p_beta": python.p_beta,
        "first_mismatch_neuron0": mismatch,
    }


def cross_backend_report(*, seed: int, duration_s: float) -> dict[str, object]:
    cfg = PlantConfig()
    with MatlabPlant(cfg) as matlab_plant, PythonPlant(cfg) as python_plant:
        matlab = matlab_plant.reset(seed=seed).integrate(duration_s, DbsSpec.none())
        python = python_plant.reset(seed=seed).integrate(duration_s, DbsSpec.none())

    mismatch = _first_spike_mismatch(matlab.gpi_spikes[0], python.gpi_spikes[0])
    rel_p_beta = None
    if matlab.p_beta and matlab.p_beta != 0.0 and python.p_beta is not None:
        rel_p_beta = abs(python.p_beta - matlab.p_beta) / abs(matlab.p_beta)

    return {
        "mode": "cross_backend",
        "seed": seed,
        "duration_s": duration_s,
        "matlab_spike_counts": spike_count_vector(matlab.gpi_spikes),
        "python_spike_counts": spike_count_vector(python.gpi_spikes),
        "matlab_p_beta": matlab.p_beta,
        "python_p_beta": python.p_beta,
        "p_beta_rel_err": rel_p_beta,
        "first_mismatch_neuron0": mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/plant_init_seed42.npz"),
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    reports = [
        fixed_ic_report(seed=args.seed, duration_s=0.06907, fixture=args.fixture),
        fixed_ic_report(seed=args.seed, duration_s=0.06908, fixture=args.fixture),
        fixed_ic_report(seed=args.seed, duration_s=args.duration_s, fixture=args.fixture),
        cross_backend_report(seed=args.seed, duration_s=args.duration_s),
    ]

    if args.json:
        print(json.dumps(reports, indent=2))
        return

    for report in reports:
        print(f"=== {report['mode']} duration={report['duration_s']}s seed={report['seed']} ===")
        print(f"  matlab counts: {report['matlab_spike_counts']}")
        print(f"  python counts: {report['python_spike_counts']}")
        print(f"  p_beta matlab={report['matlab_p_beta']} python={report['python_p_beta']}")
        print(f"  first mismatch (GPi0): {report['first_mismatch_neuron0']}")
        print()


if __name__ == "__main__":
    main()
