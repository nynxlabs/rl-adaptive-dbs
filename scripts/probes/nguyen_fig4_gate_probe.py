#!/usr/bin/env python3
"""Quick config probe for Nguyen Fig 4 heuristic + digitization gates."""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "digitization"))

from controllers.snn.adapter import NguyenEnvAdapter  # noqa: E402
from controllers.snn.buffer import ReplayBuffer  # noqa: E402
from controllers.snn.config import fig4_nguyen_config  # noqa: E402
from controllers.snn.networks import DSQN  # noqa: E402
from controllers.snn.trainer import DSQNTrainer  # noqa: E402
from nguyen_gates import (  # noqa: E402
    attach_digitization,
    fig4_length_gates,
    fig4_reward_gates,
)


REWARD_KEYS = (
    "reward_scale_paper",
    "late_reward_above_early",
    "late_reward_near_zero",
)
LENGTH_KEYS = (
    "length_decreases",
    "late_length_paper_band",
    "early_near_max_length",
)


def eval_all(
    rewards: list[float],
    lengths: list[int],
    *,
    max_steps: int = 25,
    late_start: int | None = None,
) -> dict:
    rewards_arr = np.asarray(rewards, dtype=float)
    lengths_arr = np.asarray(lengths, dtype=float)
    n = int(rewards_arr.size)
    late_start = late_start if late_start is not None else min(350, max(51, n - 30))
    reward_heur = {
        "reward_scale_paper": abs(float(np.mean(rewards_arr[:50]))) >= 5e4,
        "late_reward_above_early": float(np.mean(rewards_arr[late_start:])) > float(np.mean(rewards_arr[:50])),
        "late_reward_near_zero": float(np.mean(rewards_arr[late_start:])) > -2e5,
    }
    length_heur = {
        "length_decreases": float(np.mean(lengths_arr[late_start:])) < float(np.mean(lengths_arr[:75])) - 1,
        "late_length_paper_band": float(np.mean(lengths_arr[late_start:])) <= 12,
        "early_near_max_length": float(np.median(lengths_arr[:50])) >= max_steps - 2,
    }
    reward_heur["pass"] = all(reward_heur[k] for k in REWARD_KEYS)
    length_heur["pass"] = all(length_heur[k] for k in LENGTH_KEYS)
    dig_reward = fig4_reward_gates(rewards_arr, late_lo=float(late_start))
    dig_length = fig4_length_gates(lengths_arr, max_episode_steps=max_steps, late_lo=float(late_start))
    reward = attach_digitization(reward_heur, dig_reward)
    length = attach_digitization(length_heur, dig_length)
    return {
        "pass": bool(reward["pass"] and length["pass"]),
        "reward": reward,
        "length": length,
    }


def train_cfg(cfg) -> tuple[list[float], list[int]]:
    env = NguyenEnvAdapter(config=cfg)
    try:
        dsqn = DSQN(cfg)
        buffer = ReplayBuffer(cfg, seed=cfg.seed)
        trainer = DSQNTrainer(dsqn, buffer, cfg)
        result = trainer.train_episodes(env)
        return result.episode_rewards, result.episode_lengths
    finally:
        env.close()


def main() -> int:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    variants = [
        (
            "tau300_noshape",
            replace(
                fig4_nguyen_config(seed=0, num_episodes=episodes),
                alpha_beta_progress_coef=0.0,
                warm_zone_bonus_coef=0.0,
                warm_zone_upper=0.0,
                threshold_reward=300.0,
                energy_penalty=0.0,
                truncation_penalty=600_000.0,
            ),
        ),
        (
            "eq7_prog300",
            replace(
                fig4_nguyen_config(seed=0, num_episodes=episodes),
                alpha_beta_progress_coef=300.0,
                warm_zone_bonus_coef=0.0,
                warm_zone_upper=0.0,
                threshold_reward=1.0,
                energy_penalty=0.01,
                truncation_penalty=600_000.0,
            ),
        ),
        (
            "eq7_trunc",
            replace(
                fig4_nguyen_config(seed=0, num_episodes=episodes),
                alpha_beta_progress_coef=0.0,
                warm_zone_bonus_coef=0.0,
                warm_zone_upper=0.0,
                threshold_reward=1.0,
                energy_penalty=0.01,
                truncation_penalty=600_000.0,
            ),
        ),
        (
            "eq7_pure",
            replace(
                fig4_nguyen_config(seed=0, num_episodes=episodes),
                alpha_beta_progress_coef=0.0,
                warm_zone_bonus_coef=0.0,
                warm_zone_upper=0.0,
                threshold_reward=1.0,
                energy_penalty=0.01,
                truncation_penalty=0.0,
            ),
        ),
        (
            "v9_current",
            fig4_nguyen_config(seed=0, num_episodes=episodes),
        ),
    ]

    out: list[dict] = []
    for name, cfg in variants:
        t0 = time.perf_counter()
        print(f"\n=== {name} ({episodes} ep) ===", flush=True)
        rewards, lengths = train_cfg(cfg)
        late_start = min(350, max(51, len(rewards) - 30))
        merged = eval_all(rewards, lengths, late_start=late_start)
        heur_fail = [k for k in REWARD_KEYS if not merged["reward"].get(k, True)]
        heur_fail += [k for k in LENGTH_KEYS if not merged["length"].get(k, True)]
        dig_fail = [
            k
            for k, v in merged["reward"].items()
            if k.startswith("paper_") and v is False
        ] + [
            k
            for k, v in merged["length"].items()
            if k.startswith("paper_") and v is False
        ]
        row = {
            "name": name,
            "first50": float(np.mean(rewards[:50])),
            "late": float(np.mean(rewards[late_start:])),
            "late_len": float(np.mean(lengths[late_start:])),
            "early_stops": int(np.sum(np.asarray(lengths) < 25)),
            "pass": merged["pass"],
            "heur_fail": heur_fail,
            "dig_fail": dig_fail,
            "time_s": time.perf_counter() - t0,
        }
        out.append(row)
        print(row, flush=True)

    out_path = _ROOT / "logs" / "nguyen-fig4-probe.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
