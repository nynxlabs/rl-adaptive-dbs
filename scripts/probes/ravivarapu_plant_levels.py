#!/usr/bin/env python3
"""Ravivarapu plant probe: untreated vs always-stim shot P_beta per episode seed.

Each SEA-DBS adapter step integrates an independent shot from the episode IC,
so within one episode the plant returns one P_beta for a=0 and one for a=1.
This probe tabulates those two values (raw) for several carrier / window /
burst conventions across plant seeds, so a plant convention can be chosen
from numbers instead of training runs.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_plant_levels.py \
      --seeds 0 1 2 3 4 5 6 7 --out artifacts/figures/papers/ravivarapu/probes/plant_levels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig

CONFIGS: dict[str, dict[str, float]] = {
    "130hz_w100_b62": {"carrier_hz": 130.0, "biomarker_window_s": 0.10, "dbs_burst_ms": 62.0},
    "130hz_w100_b100": {"carrier_hz": 130.0, "biomarker_window_s": 0.10, "dbs_burst_ms": 100.0},
    "50hz_w100_b100": {"carrier_hz": 50.0, "biomarker_window_s": 0.10, "dbs_burst_ms": 100.0},
    "50hz_w150_b150": {"carrier_hz": 50.0, "biomarker_window_s": 0.15, "dbs_burst_ms": 150.0},
    "50hz_w200_b200": {"carrier_hz": 50.0, "biomarker_window_s": 0.20, "dbs_burst_ms": 200.0},
    "30hz_w100_b100": {"carrier_hz": 30.0, "biomarker_window_s": 0.10, "dbs_burst_ms": 100.0},
    "30hz_w150_b150": {"carrier_hz": 30.0, "biomarker_window_s": 0.15, "dbs_burst_ms": 150.0},
    "30hz_w200_b200": {"carrier_hz": 30.0, "biomarker_window_s": 0.20, "dbs_burst_ms": 200.0},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out: dict[str, dict[str, list[float]]] = {}
    for name in args.configs:
        knobs = CONFIGS[name]
        cfg = SEADBSConfig(**knobs)
        env = SEA_DBSEnvAdapter(config=cfg)
        env.set_carrier_hz(knobs["carrier_hz"])
        untreated: list[float] = []
        stim: list[float] = []
        try:
            for seed in args.seeds:
                _obs, info = env.reset(seed=seed)
                untreated.append(float(info["p_beta_raw"]))
                _o, _r, _t, _tr, step = env.step(1)
                stim.append(float(step["p_beta_raw"]))
        finally:
            env.close()
        u = np.asarray(untreated)
        s = np.asarray(stim)
        ratio = s / u
        out[name] = {"untreated": untreated, "stim": stim, "ratio": ratio.tolist()}
        print(
            f"{name:18s} untreated {u.mean():7.1f}±{u.std():5.1f}  "
            f"stim {s.mean():7.1f}±{s.std():5.1f}  ratio {ratio.mean():.3f}±{ratio.std():.3f}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seeds, "configs": out}, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
