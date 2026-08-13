#!/usr/bin/env python3
"""Fig 5 early-prefix probe: open-loop action pairs vs digitized steps 0–2.

Paper Fig 5 x-axis is steps, not training episodes. This matches untreated
t=0 plus the first two stim decisions (x=0,1,2) before the 50 Hz floor binds.

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig5_early_steps.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import fig4_ravivarapu_config

OUT = Path("artifacts/figures/papers/ravivarapu/5a/early_steps_probe.json")
SEQS = ((0, 0), (0, 1), (1, 0), (1, 1))
PAPER_5A = {
    "baseline": (0.4604, 0.4538, 0.4420),
    "sea": (0.4578, 0.4427, 0.4295),
}


def _rollout(*, carrier_hz: float, actions: tuple[int, ...], seed: int) -> dict:
    cfg = replace(fig4_ravivarapu_config(seed=seed, variant="paper"), carrier_hz=carrier_hz)
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        env.set_carrier_hz(carrier_hz)
        _obs, info = env.reset(seed=seed)
        mean = [float(info["p_beta_norm"])]
        last = [float(info["p_beta_raw"]) / cfg.observation_scale]
        for action in actions:
            _obs, _r, _t, _tr, step = env.step(action)
            mean.append(float(step["p_beta_norm"]))
            last.append(float(step["p_beta_raw"]) / cfg.observation_scale)
    finally:
        env.close()
    return {
        "actions": list(actions),
        "mean": mean,
        "last": last,
    }


def _mae(got: list[float], ref: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(got, ref, strict=True)) / len(ref)


def main() -> None:
    rows = []
    for seq in SEQS:
        row = _rollout(carrier_hz=50.0, actions=seq, seed=0)
        row["mae_mean_vs_paper_sea"] = _mae(row["mean"], PAPER_5A["sea"])
        row["mae_mean_vs_paper_baseline"] = _mae(row["mean"], PAPER_5A["baseline"])
        row["mae_last_vs_paper_sea"] = _mae(row["last"], PAPER_5A["sea"])
        row["mae_last_vs_paper_baseline"] = _mae(row["last"], PAPER_5A["baseline"])
        rows.append(row)
        print(
            f"a={seq} mean={[round(x, 4) for x in row['mean']]} "
            f"last={[round(x, 4) for x in row['last']]} "
            f"mae_mean B={row['mae_mean_vs_paper_baseline']:.4f} "
            f"SEA={row['mae_mean_vs_paper_sea']:.4f}"
        )
    payload = {
        "carrier_hz": 50.0,
        "seed": 0,
        "paper_5a": PAPER_5A,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
