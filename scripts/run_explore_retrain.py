#!/usr/bin/env python3
"""Retrain DDPG with exploration and report adaptivity metrics (TASK-67)."""

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
    parser.add_argument("--state-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--exploration-mode", choices=("epsilon", "softmax"), default="softmax")
    parser.add_argument(
        "--learning-v1",
        action="store_true",
        help="phase4-results.md §10.4 profile (softmax 3→1, init_bias_scale=0.5, wider CNN)",
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/ddpg/explore_retrain.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/ddpg/paper_train0_explore_retrain.pt"),
    )
    args = parser.parse_args()

    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=args.state_length)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        config = DDPGConfig(
            variant="paper",
            seed=args.seed,
            num_episodes=args.episodes,
            exploration_mode=args.exploration_mode,
        )
        if args.learning_v1:
            config = DDPGConfig(
                variant="paper",
                seed=args.seed,
                num_episodes=args.episodes,
                exploration_mode="softmax",
                exploration_temperature_start=3.0,
                exploration_temperature_end=1.0,
                conv_channels=32,
                shrink_dim=8,
                init_bias_scale=0.5,
                critic_action_input="one_hot",
            )
        result = train(env, config, checkpoint_path=args.checkpoint)
        offline = _analyze_policy(result.actor, args.state_length)
        rollout = _rollout_actions(env, result.actor, seed=args.seed + 1000)
        out = {
            "state_length": args.state_length,
            "episodes": args.episodes,
            "exploration_mode": config.exploration_mode,
            "critic_action_input": config.critic_action_input,
            "learning_v1": args.learning_v1,
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
