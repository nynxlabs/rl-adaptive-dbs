#!/usr/bin/env python3
"""Ravivarapu open-loop probe: Figs 5–7-style traces for fixed action sequences.

Rolls the SEA-DBS adapter (training config) through t=0 untreated + N fixed
actions (always-on, always-off) at a given carrier, in ``disconnected`` or
``continuous`` plant mode, for several plant seeds. Prints the Eq. (4)–(5)
window mean (what Figs 5–7 plot) and the raw per-step P_beta.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_openloop_traces.py \
      --modes disconnected continuous --carriers 50 30 --seeds 0 1 2 \
      --out artifacts/figures/papers/ravivarapu/probes/openloop_traces.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig


def rollout(cfg: SEADBSConfig, seed: int, actions: list[int]) -> tuple[list[float], list[float]]:
    env = SEA_DBSEnvAdapter(config=cfg)
    env.set_carrier_hz(cfg.carrier_hz)
    try:
        _obs, info = env.reset(seed=seed)
        mean = [float(info["p_beta_norm"])]
        raw = [float(info["p_beta_raw"]) / cfg.observation_scale]
        for a in actions:
            _o, _r, _t, _tr, step = env.step(a)
            mean.append(float(step["p_beta_norm"]))
            raw.append(float(step["p_beta_raw"]) / cfg.observation_scale)
    finally:
        env.close()
    return mean, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["disconnected", "continuous"])
    parser.add_argument("--carriers", type=float, nargs="+", default=[50.0, 30.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    patterns = {"on": [1] * args.steps, "off": [0] * args.steps}
    out: dict[str, dict[str, list[float]]] = {}
    for mode in args.modes:
        for hz in args.carriers:
            for name, acts in patterns.items():
                if name == "off" and hz != args.carriers[0]:
                    continue
                means, raws = [], []
                for seed in args.seeds:
                    cfg = replace(SEADBSConfig(), plant_integration_mode=mode, carrier_hz=hz)
                    m, r = rollout(cfg, seed, acts)
                    means.append(m)
                    raws.append(r)
                key = f"{mode}_{int(hz)}hz_{name}"
                mm = np.mean(means, axis=0)
                rm = np.mean(raws, axis=0)
                out[key] = {"mean_trace": mm.tolist(), "raw_trace": rm.tolist(), "per_seed_mean": means}
                print(f"{key:28s} mean {' '.join(f'{v:.3f}' for v in mm)}", flush=True)
                print(f"{'':28s} raw  {' '.join(f'{v:.3f}' for v in rm)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seeds, "traces": out}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
