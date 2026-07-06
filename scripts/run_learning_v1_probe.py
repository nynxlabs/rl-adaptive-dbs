#!/usr/bin/env python3
"""Short learning_v1 probe (phase4-results.md §10.4) — TASK-67 follow-up."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from controllers.ddpg import train
from controllers.ddpg.config import DDPGConfig
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.state_length_sweep import _analyze_policy, _rollout_actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ddpg/learning_v1_probe.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/ddpg/paper_train0_learning_v1_probe.pt"),
    )
    args = parser.parse_args()

    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=15)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        config = DDPGConfig(
            variant="paper",
            seed=0,
            num_episodes=args.episodes,
            exploration_mode="softmax",
            exploration_temperature_start=3.0,
            exploration_temperature_end=1.0,
            conv_channels=32,
            shrink_dim=8,
            init_bias_scale=0.5,
        )
        result = train(env, config, checkpoint_path=args.checkpoint)
        offline = _analyze_policy(result.actor, 15)
        rollout = _rollout_actions(env, result.actor, seed=1000)
        out = {
            "profile": "learning_v1",
            "state_length": 15,
            "episodes": args.episodes,
            "exploration_mode": "softmax",
            "exploration_temperature": "3.0->1.0",
            "init_bias_scale": 0.5,
            "conv_channels": 32,
            "shrink_dim": 8,
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": int(offline["dominant_action"]),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        summary = {k: v for k, v in out.items() if k != "rollout_actions"}
        print(json.dumps(summary, indent=2))
        print(
            f"rollout_unique={out['unique_actions_rollout']} "
            f"offline_unique={out['unique_actions_offline']}",
            flush=True,
        )
        return 0 if out["unique_actions_rollout"] > 1 and out["unique_actions_offline"] > 1 else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
