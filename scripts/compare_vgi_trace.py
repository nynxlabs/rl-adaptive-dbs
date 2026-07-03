#!/usr/bin/env python3
"""Compare MATLAB vs Python GPi voltage traces (fixed ICs) for drift localization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from envs.plant import DbsSpec, PlantConfig
from envs.plant.network.integrator import NetworkInitDraws, integrate_network


def _repo_model_dir() -> Path:
    env = os.environ.get("RL_ADAPTIVE_DBS_MATLAB_MODEL")
    if env:
        return Path(env).resolve()
    root = os.environ.get("RL_ADAPTIVE_DBS_ROOT")
    if root:
        return Path(root).resolve() / "reference-material" / "KumaraveluEtAl2016"
    return Path(__file__).resolve().parents[1] / "reference-material" / "KumaraveluEtAl2016"


def matlab_traces(
    *,
    seed: int,
    duration_ms: float,
    pd: int = 1,
    corstim: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import matlab.engine

    eng = matlab.engine.start_matlab()
    try:
        eng.cd(str(_repo_model_dir()), nargout=0)
        raw = eng.simulate_network_model(
            1.0,
            float(pd),
            float(corstim),
            1.0,
            True,
            float(seed),
            float(duration_ms),
            nargout=9,
        )
        vgi = np.array(raw[6], dtype=np.float64)
        vsn = np.array(raw[7], dtype=np.float64)
        vge = np.array(raw[8], dtype=np.float64)
        return vgi, vsn, vge
    finally:
        eng.quit()


def matlab_vgi(
    *,
    seed: int,
    duration_ms: float,
    pd: int = 1,
    corstim: int = 0,
) -> np.ndarray:
    return matlab_traces(seed=seed, duration_ms=duration_ms, pd=pd, corstim=corstim)[0]


def python_vgi(
    *,
    seed: int,
    duration_s: float,
    draws: NetworkInitDraws,
) -> np.ndarray:
    result = integrate_network(
        config=PlantConfig(),
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=np.random.default_rng(seed),
        iteration=1,
        seed=seed,
        init_draws=draws,
        return_traces=("vgi",),
    )
    traces = result.info.get("traces", {})
    return traces["vgi"]


def python_traces(
    *,
    seed: int,
    duration_s: float,
    draws: NetworkInitDraws,
) -> dict[str, np.ndarray]:
    result = integrate_network(
        config=PlantConfig(),
        duration_s=duration_s,
        dbs_spec=DbsSpec.none(),
        record_spikes=True,
        rng=np.random.default_rng(seed),
        iteration=1,
        seed=seed,
        init_draws=draws,
        return_traces=("vgi", "vsn", "vge"),
    )
    return result.info["traces"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration-ms", type=float, default=69.1)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/plant_init_seed42.npz"),
    )
    parser.add_argument("--neuron", type=int, default=0)
    parser.add_argument("--window", type=int, default=20, help="Steps around first diff")
    args = parser.parse_args()

    draws = NetworkInitDraws.from_npz(args.fixture)
    duration_s = args.duration_ms / 1000.0

    mat = matlab_vgi(seed=args.seed, duration_ms=args.duration_ms)
    py = python_vgi(seed=args.seed, duration_s=duration_s, draws=draws)

    if mat.shape != py.shape:
        print(f"shape mismatch mat={mat.shape} py={py.shape}")
        n_cols = min(mat.shape[1], py.shape[1])
        mat = mat[:, :n_cols]
        py = py[:, :n_cols]

    neuron = args.neuron
    diff = np.abs(mat[neuron] - py[neuron])
    first = int(np.argmax(diff > 1e-9)) if np.any(diff > 1e-9) else -1
    max_diff = float(diff.max())
    max_step = int(diff.argmax())

    print(f"seed={args.seed} duration={args.duration_ms}ms neuron={neuron}")
    print(f"first_step_abs_diff_gt_1e-9: {first}")
    print(f"max_abs_diff={max_diff:.6g} at step={max_step}")

    if first >= 0:
        lo = max(0, first - 3)
        hi = min(mat.shape[1], first + args.window)
        print(f"\nstep   t_ms    mat_vgi      py_vgi       delta")
        dt = PlantConfig().dt_ms
        for step in range(lo, hi):
            t_ms = step * dt
            d = py[neuron, step] - mat[neuron, step]
            print(
                f"{step:5d} {t_ms:7.3f} "
                f"{mat[neuron, step]:12.6f} {py[neuron, step]:12.6f} {d:+.6f}"
            )


if __name__ == "__main__":
    main()
