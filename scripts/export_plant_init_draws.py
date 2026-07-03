#!/usr/bin/env python3
"""Export MATLAB CTX_BG_TH_network initialization draws for equivalence fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def export_matlab_init_draws(*, seed: int, n: int = 10, pd: int = 1) -> dict[str, np.ndarray]:
    import matlab.engine

    eng = matlab.engine.start_matlab()
    try:
        eng.rng(float(seed), nargout=0)

        def randn_vec() -> np.ndarray:
            return np.array(eng.randn(float(n), 1.0), dtype=np.float64).reshape(-1)

        def rand_vec() -> np.ndarray:
            return np.array(eng.rand(float(n), 1.0), dtype=np.float64).reshape(-1)

        def randperm_vec() -> np.ndarray:
            return np.array(eng.randperm(float(n)), dtype=np.int64).reshape(-1) - 1

        v1 = -62.0 + randn_vec() * 5.0
        v2 = -62.0 + randn_vec() * 5.0
        v3 = -62.0 + randn_vec() * 5.0
        v4 = -62.0 + randn_vec() * 5.0
        v5 = -63.8 + randn_vec() * 5.0
        v6 = -63.8 + randn_vec() * 5.0

        perms = [randperm_vec() for _ in range(15)]

        gcorsna = 0.3 * rand_vec()
        gcorsnn = 0.003 * rand_vec()
        gcordrstr = (0.07 - 0.044 * pd) + 0.001 * rand_vec()
        ggege = rand_vec()

        gsngen = np.zeros(n, dtype=np.float64)
        gsngen_idx = randperm_vec()[:2]
        gsngen[gsngen_idx] = 0.002 * np.array(
            eng.rand(2.0, 1.0), dtype=np.float64
        ).reshape(-1)

        gsngea = np.zeros(n, dtype=np.float64)
        gsngea_idx = randperm_vec()[:2]
        gsngea[gsngea_idx] = 0.3 * np.array(eng.rand(2.0, 1.0), dtype=np.float64).reshape(-1)

        gsngi = np.zeros(n, dtype=np.float64)
        gsngi_idx = randperm_vec()[:5]
        gsngi[gsngi_idx] = 0.15
    finally:
        eng.quit()

    payload: dict[str, np.ndarray] = {
        "seed": np.array([seed], dtype=np.int64),
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "v6": v6,
        "gcorsna": gcorsna,
        "gcorsnn": gcorsnn,
        "gcordrstr": gcordrstr,
        "ggege": ggege,
        "gsngen": gsngen,
        "gsngea": gsngea,
        "gsngi": gsngi,
    }
    for index, perm in enumerate(perms):
        payload[f"perm_{index}"] = perm
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/plant_init_seed42.npz"),
    )
    args = parser.parse_args()

    payload = export_matlab_init_draws(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Wrote {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
