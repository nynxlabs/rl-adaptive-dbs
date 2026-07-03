#!/usr/bin/env python3
"""Compare MATLAB vs Python GPe pre-update state at integrator step 5185.

Python ``step`` maps to MATLAB ``i = step + 1`` (see integrator loop comment).
Snapshots are taken immediately before the GPe voltage update, using synaptic
state from the prior convolver step. At step 5185 neuron 0, ``Igege`` drift is
driven by ``S31c[0] = S3c[1]`` (peer GPe self-inhibition), not STN ``S2a``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from envs.plant import DbsSpec, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network
from scripts.compare_vgi_trace import _repo_model_dir

DEBUG_STEP = 5185
MATLAB_I = DEBUG_STEP + 1


def matlab_snapshot(*, seed: int, duration_ms: float) -> dict:
    import matlab.engine

    eng = matlab.engine.start_matlab()
    try:
        eng.cd(str(_repo_model_dir()), nargout=0)
        raw = eng.simulate_network_model(
            1.0,
            1.0,
            0.0,
            1.0,
            True,
            float(seed),
            float(duration_ms),
            nargout=11,
        )
        snap = raw[10]
        stn_times_raw = snap["stn_times"]
        gpe_times_raw = snap["gpe_times"]
        if hasattr(stn_times_raw, "size"):
            n_stn = int(stn_times_raw.size)
            stn_lists = [stn_times_raw[j] for j in range(n_stn)]
            gpe_lists = [gpe_times_raw[j] for j in range(n_stn)]
        else:
            stn_lists = list(stn_times_raw)
            gpe_lists = list(gpe_times_raw)
        return {
            "step": int(snap["step"]),
            "V2": np.array(snap["V2"], dtype=np.float64).reshape(-1),
            "V3": np.array(snap["V3"], dtype=np.float64).reshape(-1),
            "S2a": np.array(snap["S2a"], dtype=np.float64).reshape(-1),
            "S21a": np.array(snap["S21a"], dtype=np.float64).reshape(-1),
            "S3c": np.array(snap["S3c"], dtype=np.float64).reshape(-1),
            "S31c": np.array(snap["S31c"], dtype=np.float64).reshape(-1),
            "S32c": np.array(snap["S32c"], dtype=np.float64).reshape(-1),
            "N3": np.array(snap["N3"], dtype=np.float64).reshape(-1),
            "H3": np.array(snap["H3"], dtype=np.float64).reshape(-1),
            "R3": np.array(snap["R3"], dtype=np.float64).reshape(-1),
            "CA3": np.array(snap["CA3"], dtype=np.float64).reshape(-1),
            "isngeampa": np.array(snap["Isngeampa"], dtype=np.float64).reshape(-1),
            "isngenmda": np.array(snap["Isngenmda"], dtype=np.float64).reshape(-1),
            "igege": np.array(snap["Igege"], dtype=np.float64).reshape(-1),
            "istrgpe": np.array(snap["Istrgpe"], dtype=np.float64).reshape(-1),
            "ik3": np.array(snap["Ik3"], dtype=np.float64).reshape(-1),
            "ina3": np.array(snap["Ina3"], dtype=np.float64).reshape(-1),
            "stn_spike_times": [
                np.array(stn_lists[j], dtype=np.int64).reshape(-1)
                for j in range(len(stn_lists))
            ],
            "gpe_spike_times": [
                np.array(gpe_lists[j], dtype=np.int64).reshape(-1)
                for j in range(len(gpe_lists))
            ],
        }
    finally:
        eng.quit()


def python_snapshot(*, seed: int, duration_s: float, draws: NetworkInitDraws) -> dict:
    result = integrate_network(
        config=PlantConfig(),
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=False,
        rng=np.random.default_rng(seed),
        iteration=1,
        seed=seed,
        init_draws=draws,
        debug_steps=(DEBUG_STEP,),
    )
    snaps = result.info.get("debug_snapshots", {})
    if DEBUG_STEP not in snaps:
        msg = f"debug snapshot missing for step {DEBUG_STEP}"
        raise KeyError(msg)
    return snaps[DEBUG_STEP]


def _compare_field(name: str, mat: np.ndarray, py: np.ndarray, neuron: int) -> None:
    d = float(py[neuron] - mat[neuron])
    print(f"  {name:12s} mat={mat[neuron]:.10g} py={py[neuron]:.10g} delta={d:+.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--neuron", type=int, default=0)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/plant_init_seed42.npz"),
    )
    args = parser.parse_args()

    draws = NetworkInitDraws.from_npz(args.fixture)
    duration_ms = 52.0
    mat = matlab_snapshot(seed=args.seed, duration_ms=duration_ms)
    py = python_snapshot(seed=args.seed, duration_s=duration_ms / 1000.0, draws=draws)

    n = args.neuron
    print(f"GPe pre-update snapshot (py step {DEBUG_STEP}, mat i {MATLAB_I}), neuron {n}")
    print(f"mat step field: {mat['step']}")

    for key in (
        "V2",
        "V3",
        "S2a",
        "S21a",
        "S3c",
        "S31c",
        "S32c",
        "N3",
        "H3",
        "R3",
        "CA3",
        "isngeampa",
        "isngenmda",
        "igege",
        "istrgpe",
        "ik3",
        "ina3",
    ):
        _compare_field(key, mat[key], py[key], n)

    print("  stn_spike_times mat:", mat["stn_spike_times"][n].tolist())
    print("  stn_spike_times py :", py["stn_spike_times"][n])
    print("  gpe_spike_times mat:", mat["gpe_spike_times"][n].tolist())
    print("  gpe_spike_times py :", py["gpe_spike_times"][n])


if __name__ == "__main__":
    main()
