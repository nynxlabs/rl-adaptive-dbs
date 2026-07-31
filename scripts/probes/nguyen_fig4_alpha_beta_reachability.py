#!/usr/bin/env python3
"""Open-loop probe: can fixed DBS triples drive GPi α–β under θ=150 for t_u=3?

Reports per-triple mean α–β, max sub-threshold streak, and whether early-stop
would fire within 25 steps. Cheap diagnostic before Fig 4 reward/train knobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.config import fig4_nguyen_config
from controllers.snn.dbs_params import DBSParameterState

HOLD = np.array([1, 1, 1], dtype=np.int64)  # MultiDiscrete indices → ternary 0,0,0

TRIPLES = {
    "init": DBSParameterState(amplitude=300.0, frequency_hz=40.0, pulse_width_ms=0.3),
    "paper_anchor": DBSParameterState(amplitude=262.0, frequency_hz=78.65, pulse_width_ms=1.0),
    "open_loop_130": DBSParameterState(amplitude=300.0, frequency_hz=130.0, pulse_width_ms=0.3),
    "strong_130": DBSParameterState(amplitude=500.0, frequency_hz=130.0, pulse_width_ms=1.0),
    "min_energy": DBSParameterState(amplitude=50.0, frequency_hz=10.0, pulse_width_ms=0.05),
}


def run_fixed(*, name: str, params: DBSParameterState, seeds: list[int], max_steps: int) -> dict:
    rows = []
    for seed in seeds:
        cfg = fig4_nguyen_config(seed=seed, num_episodes=1)
        env = NguyenEnvAdapter(config=cfg)
        try:
            env.reset(seed=seed)
            alphas: list[float] = []
            streak = 0
            max_streak = 0
            terminated = False
            truncated = False
            for _ in range(max_steps):
                # Freeze open-loop triple (hold then overwrite any residual drift).
                env._dbs = DBSParameterState(
                    amplitude=params.amplitude,
                    frequency_hz=params.frequency_hz,
                    pulse_width_ms=params.pulse_width_ms,
                )
                _obs, _reward, terminated, truncated, info = env.step(HOLD)
                env._dbs = DBSParameterState(
                    amplitude=params.amplitude,
                    frequency_hz=params.frequency_hz,
                    pulse_width_ms=params.pulse_width_ms,
                )
                ab = float(info["alpha_beta"])
                alphas.append(ab)
                if ab < cfg.alpha_beta_threshold:
                    streak += 1
                else:
                    streak = 0
                max_streak = max(max_streak, streak)
                if terminated or truncated:
                    break
            rows.append(
                {
                    "seed": seed,
                    "steps": len(alphas),
                    "alpha_mean": float(np.mean(alphas)),
                    "alpha_min": float(np.min(alphas)),
                    "max_streak": int(max_streak),
                    "terminated": bool(terminated),
                }
            )
        finally:
            env.close()
    max_streaks = [r["max_streak"] for r in rows]
    return {
        "name": name,
        "params": {
            "A": params.amplitude,
            "f": params.frequency_hz,
            "pw": params.pulse_width_ms,
        },
        "n_seeds": len(seeds),
        "mean_alpha": float(np.mean([r["alpha_mean"] for r in rows])),
        "best_max_streak": int(max(max_streaks)),
        "seeds_with_streak_ge_3": int(sum(s >= 3 for s in max_streaks)),
        "seeds_terminated": int(sum(r["terminated"] for r in rows)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/probes/nguyen_fig4_reachability.json"),
    )
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    for name, params in TRIPLES.items():
        print(f"=== {name} A={params.amplitude} f={params.frequency_hz} pw={params.pulse_width_ms} ===", flush=True)
        result = run_fixed(name=name, params=params, seeds=seeds, max_steps=args.max_steps)
        print(
            f"  mean_alpha={result['mean_alpha']:.1f} best_streak={result['best_max_streak']} "
            f"streak>=3 on {result['seeds_with_streak_ge_3']}/{result['n_seeds']} "
            f"terminated={result['seeds_terminated']}",
            flush=True,
        )
        results.append(result)
    payload = {"seeds": seeds, "max_steps": args.max_steps, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
