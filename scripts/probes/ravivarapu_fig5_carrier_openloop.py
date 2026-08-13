#!/usr/bin/env python3
"""Open-loop Fig 5 carrier diagnostic (always-stim vs no-stim).

Compares 30 / 50 / 130 Hz and no-stim on the Fig 4a burst plant (62 ms of
carrier in a 100 ms biomarker window). Cheap lineage check before policy
sweeps: does 50 Hz suppress vs no-stim, and more than 30 Hz, on Kumaravelu?

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig5_carrier_openloop.py
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig5_carrier_openloop.py --steps 3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import fig4_ravivarapu_config

OUT = Path("artifacts/figures/papers/ravivarapu/5a/openloop_carrier_probe.json")
CARRIERS = (0.0, 30.0, 50.0, 130.0)


def _rollout(*, carrier_hz: float, action: int, steps: int, seed: int) -> dict:
    cfg = replace(fig4_ravivarapu_config(seed=seed, variant="paper"), carrier_hz=carrier_hz)
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        env.set_carrier_hz(carrier_hz)
        _obs, info = env.reset(seed=seed)
        psd = [float(info["p_beta_norm"])]
        raw = [float(info["p_beta_raw"])]
        reported = [float(info["carrier_hz"])]
        for _ in range(steps):
            _obs, _r, _t, _tr, step = env.step(action)
            psd.append(float(step["p_beta_norm"]))
            raw.append(float(step["p_beta_raw"]))
            reported.append(float(step["carrier_hz"]))
    finally:
        env.close()
    return {
        "carrier_hz": carrier_hz,
        "action": action,
        "reported_carrier_hz": reported,
        "p_beta_norm": psd,
        "p_beta_raw": raw,
        "start": psd[0],
        "end": psd[-1],
        "drop": psd[0] - psd[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = []
    rows.append(_rollout(carrier_hz=130.0, action=0, steps=args.steps, seed=args.seed))
    rows[-1]["carrier_hz"] = 0.0
    rows[-1]["label"] = "no-stim"
    for hz in (30.0, 50.0, 130.0):
        rows.append(_rollout(carrier_hz=hz, action=1, steps=args.steps, seed=args.seed))
    payload = {"steps": args.steps, "seed": args.seed, "dbs_burst_ms": 62.0, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    no_stim = rows[0]
    by_hz = {r["carrier_hz"]: r for r in rows[1:]}
    print(
        "summary:",
        f"no-stim drop={no_stim['drop']:.4f}",
        f"30 drop={by_hz[30.0]['drop']:.4f}",
        f"50 drop={by_hz[50.0]['drop']:.4f}",
        f"130 drop={by_hz[130.0]['drop']:.4f}",
    )


if __name__ == "__main__":
    main()
