#!/usr/bin/env python3
"""Paper-exact baseline: MSE, no warmup, no normalization, no noise, logits critic, greedy argmax.

This is the control run to compare against TASK-75's extensions.
If this also produces constant policy, the root cause is architectural (not missing exploration).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from controllers.ddpg import train
from controllers.ddpg.config import DDPGConfig
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.state_length_sweep import _analyze_policy, _rollout_actions


def main() -> int:
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=15)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        config = DDPGConfig(
            variant="paper",
            seed=0,
            num_episodes=30,
            # Paper-exact: greedy argmax, no exploration
            exploration_mode="softmax",
            exploration_temperature_start=1.0,
            exploration_temperature_end=1.0,
            # Paper-exact: logits critic
            critic_action_input="logits",
            # NO extensions:
            critic_warmup_steps=0,
            reward_normalize=False,
            critic_loss_fn="mse",
            logit_noise_std=0.0,
            log_episodes=True,
        )

        print(f"=== Paper-exact baseline (control) ===", flush=True)
        print(f"  state_length=15", flush=True)
        print(f"  episodes=30", flush=True)
        print(f"  critic_action_input=logits", flush=True)
        print(f"  exploration: greedy argmax (temp=1.0)", flush=True)
        print(f"  NO warmup, NO normalization, NO Huber, NO noise", flush=True)
        print(flush=True)

        t0 = time.time()
        result = train(env, config, checkpoint_path=Path("artifacts/ddpg/paper_train0_paper_exact.pt"))
        elapsed = time.time() - t0

        offline = _analyze_policy(result.actor, 15)
        rollout = _rollout_actions(env, result.actor, seed=1000)

        out = {
            "task": "paper-exact-baseline",
            "state_length": 15,
            "episodes": 30,
            "exploration": "greedy_argmax_temp1",
            "critic_action_input": "logits",
            "extensions": "none (paper-exact)",
            "seed": 0,
            "elapsed_s": round(elapsed, 1),
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": int(offline["dominant_action"]),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
            "acceptance_pass": len(set(rollout)) > 1 and int(offline["unique_actions_offline"]) > 1,
        }
        out_path = Path("artifacts/ddpg/paper_exact_baseline.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        summary = dict(out)
        summary.pop("rollout_actions", None)
        print(json.dumps(summary, indent=2))
        print(
            f"\nrollout_unique={out['unique_actions_rollout']} "
            f"offline_unique={out['unique_actions_offline']}",
            flush=True,
        )
        return 0 if out["acceptance_pass"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
