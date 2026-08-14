#!/usr/bin/env python3
"""Fig 5 steps 0–5: always-on burst-length vs digitized paper.

Paper SEA @ 50 Hz is already ~0.34 by step 5; Fig 4a 62 ms / 50 Hz floors
near 0.43 after n_obs fills. This probe asks whether a longer eval burst
(paper-silent; Fig 4a train stays 62 ms @ 130 Hz) can close that window.

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig5_burst_steps.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import fig4_ravivarapu_config

OUT = Path("artifacts/figures/papers/ravivarapu/5a/burst_steps_probe.json")
N_STIM = 5
BURSTS_MS = (62.0, 80.0, 90.0, 100.0)
CARRIERS = (50.0, 30.0)
PAPER_5A = {
    "baseline": (0.4604, 0.4531, 0.4420, 0.4305, 0.4177, 0.4037),
    "sea": (0.4578, 0.4427, 0.4295, 0.3960, 0.3627, 0.3381),
}
PAPER_5B = {
    "baseline": (0.4622, 0.4687, 0.4662, 0.4513, 0.4337, 0.4299),
    "sea": (0.4610, 0.4576, 0.4543, 0.4449, 0.4361, 0.4278),
}


def _mae(got: list[float], ref: tuple[float, ...]) -> float:
    n = min(len(got), len(ref))
    return sum(abs(got[i] - ref[i]) for i in range(n)) / n


def _rollout(*, carrier_hz: float, burst_ms: float, seed: int = 0) -> dict:
    cfg = replace(
        fig4_ravivarapu_config(seed=seed, variant="paper"),
        carrier_hz=carrier_hz,
        dbs_burst_ms=burst_ms,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        env.set_carrier_hz(carrier_hz)
        _obs, info = env.reset(seed=seed)
        mean = [float(info["p_beta_norm"])]
        last = [float(info["p_beta_raw"]) / cfg.observation_scale]
        for _ in range(N_STIM):
            _obs, _r, _t, _tr, step = env.step(1)
            mean.append(float(step["p_beta_norm"]))
            last.append(float(step["p_beta_raw"]) / cfg.observation_scale)
    finally:
        env.close()
    paper = PAPER_5A if carrier_hz == 50.0 else PAPER_5B
    return {
        "carrier_hz": carrier_hz,
        "dbs_burst_ms": burst_ms,
        "mean": mean,
        "last": last,
        "drop_0_5_mean": mean[0] - mean[-1],
        "mae_mean_vs_paper_sea": _mae(mean, paper["sea"]),
        "mae_mean_vs_paper_baseline": _mae(mean, paper["baseline"]),
        "mae_last_vs_paper_sea": _mae(last, paper["sea"]),
    }


def main() -> None:
    rows = []
    for hz in CARRIERS:
        for burst in BURSTS_MS:
            row = _rollout(carrier_hz=hz, burst_ms=burst)
            rows.append(row)
            m = [round(x, 4) for x in row["mean"]]
            print(
                f"{hz:.0f} Hz burst={burst:.0f} ms mean={m} "
                f"drop={row['drop_0_5_mean']:.4f} "
                f"MAE_SEA={row['mae_mean_vs_paper_sea']:.4f} "
                f"MAE_B={row['mae_mean_vs_paper_baseline']:.4f}"
            )
    payload = {"n_stim": N_STIM, "paper_5a": PAPER_5A, "paper_5b": PAPER_5B, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
