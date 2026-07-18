#!/usr/bin/env python3
"""TASK-104: sl=15 pattern-mode smoke retrain (10 ep, paper settings, obs_normalize).

Writes:
  artifacts/ddpg/task104_sl15_pattern_smoke.pt
  artifacts/ddpg/task104_sl15_pattern_smoke.json
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
from scripts.lib.state_length_sweep import _analyze_policy, _rollout_actions

ARTIFACTS = Path("artifacts/ddpg")
CKPT = ARTIFACTS / "task104_sl15_pattern_smoke.pt"
OUT = ARTIFACTS / "task104_sl15_pattern_smoke.json"


def main() -> int:
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(
        state_length=15,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=45.0,
    )
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        config = DDPGConfig(
            variant="paper",
            seed=0,
            num_episodes=10,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
            exploration_mode="softmax",
            exploration_temperature_start=1.0,
            exploration_temperature_end=1.0,
            critic_action_input="one_hot",
            critic_warmup_steps=0,
            reward_normalize=False,
            critic_loss_fn="mse",
            logit_noise_std=0.0,
            obs_normalize=True,
            log_episodes=True,
        )

        print("=== TASK-104 sl=15 pattern-mode smoke retrain ===", flush=True)
        print(f"  state_length={env_cfg.state_length}", flush=True)
        print(f"  obs_normalize={config.obs_normalize}", flush=True)
        print(f"  action_space_mode={env_cfg.action_space_mode}", flush=True)
        print(f"  episodes={config.num_episodes}", flush=True)
        print(flush=True)

        t0 = time.time()
        result = train(env, config, checkpoint_path=CKPT)
        elapsed = time.time() - t0

        offline = _analyze_policy(result.actor, env_cfg.state_length)
        rollout = _rollout_actions(env, result.actor, seed=1000)

        out = {
            "task": "TASK-104",
            "action_space_mode": env_cfg.action_space_mode,
            "pattern_mean_hz": env_cfg.pattern_mean_hz,
            "state_length": env_cfg.state_length,
            "obs_normalize": config.obs_normalize,
            "episodes": config.num_episodes,
            "exploration": "greedy_argmax_temp1",
            "critic_action_input": config.critic_action_input,
            "variant": config.variant,
            "seed": config.seed,
            "elapsed_s": round(elapsed, 1),
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": int(offline["dominant_action"]),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
            "adaptivity_pass": len(set(rollout)) > 1 and int(offline["unique_actions_offline"]) > 1,
        }
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2) + "\n")

        summary = dict(out)
        summary.pop("rollout_actions", None)
        print(json.dumps(summary, indent=2), flush=True)
        print(
            f"\nrollout_unique={out['unique_actions_rollout']} "
            f"offline_unique={out['unique_actions_offline']} "
            f"adaptivity_pass={out['adaptivity_pass']}",
            flush=True,
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    from scripts.lib.train_runtime_guard import run_main

    sys.exit(run_main(main, label="task104-sl15-smoke"))
