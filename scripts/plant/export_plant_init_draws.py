#!/usr/bin/env python3
"""Export MATLAB CTX_BG_TH_network initialization draws for equivalence fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from envs.plant.matlab_backend import model_dir


def _matlab_vec(raw: object) -> np.ndarray:
    return np.array(raw, dtype=np.float64).reshape(-1)


def _matlab_perm(raw: object) -> np.ndarray:
    return np.array(raw, dtype=np.int64).reshape(-1) - 1


def export_matlab_init_draws(*, seed: int, n: int = 10, pd: int = 1) -> dict[str, np.ndarray]:
    """Read init draws from a short MATLAB run (``plant_init_export`` struct)."""
    path = model_dir()
    import matlab.engine

    eng = matlab.engine.start_matlab()
    try:
        eng.cd(str(path), nargout=0)
        raw = eng.simulate_network_model(
            1.0,
            float(pd),
            0.0,
            1.0,
            True,
            float(seed),
            0.01,
            2.0,
            nargout=12,
        )
        init = raw[11]
        perm_names = (
            "all",
            "bll",
            "cll",
            "dll",
            "ell",
            "fll",
            "gll",
            "hll",
            "ill",
            "jll",
            "kll",
            "lll",
            "mll",
            "nll",
            "oll",
        )
        payload: dict[str, np.ndarray] = {
            "seed": np.array([seed], dtype=np.int64),
            "v1": _matlab_vec(init["v1"]),
            "v2": _matlab_vec(init["v2"]),
            "v3": _matlab_vec(init["v3"]),
            "v4": _matlab_vec(init["v4"]),
            "v5": _matlab_vec(init["v5"]),
            "v6": _matlab_vec(init["v6"]),
            "gcorsna": _matlab_vec(init["gcorsna"]),
            "gcorsnn": _matlab_vec(init["gcorsnn"]),
            "gcordrstr": _matlab_vec(init["gcordrstr"]),
            "ggege": _matlab_vec(init["ggege"]),
            "gsngen": _matlab_vec(init["gsngen"]),
            "gsngea": _matlab_vec(init["gsngea"]),
            "gsngi": _matlab_vec(init["gsngi"]),
        }
        for index, name in enumerate(perm_names):
            payload[f"perm_{index}"] = _matlab_perm(init[name])
        if n != 10:
            msg = "only n=10 Kumaravelu network is supported"
            raise ValueError(msg)
        return payload
    finally:
        eng.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pd", type=int, choices=(0, 1), default=1, help="1 = PD (6-OHDA), 0 = healthy")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "default: <init_draws_filename(seed, pd)> in the private kumaravelu_plant package's data "
            "directory (model-generated data stays private; LICENSING.md), else the user cache"
        ),
    )
    args = parser.parse_args()
    if args.output is None:
        from envs.plant._kumaravelu import data_dir
        from envs.plant.network.matlab_rng import init_draws_cache_dir, init_draws_filename

        args.output = (data_dir() or init_draws_cache_dir()) / init_draws_filename(args.seed, args.pd)

    payload = export_matlab_init_draws(seed=args.seed, pd=args.pd)
    payload["pd"] = np.array([args.pd], dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Wrote {args.output} (seed={args.seed}, pd={args.pd})")


if __name__ == "__main__":
    main()
