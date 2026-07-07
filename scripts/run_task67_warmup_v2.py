#!/usr/bin/env python3
"""TASK-67: Train DDPG with random warmup + exploration to break constant-policy collapse.

Root cause: replay buffer fills with biased transitions from the collapsing policy,
so the critic never sees diverse state-action-reward pairs. Fix: random warmup phase
fills buffer before any policy training begins.

Settings:
  - random_warmup_steps=200 (diverse buffer)
  - obs_normalize=True (amplify signal for CNN)
  - init_bias_scale=0.0 (no initialization bias)
  - exploration_mode=softmax, temp 3.0→1.0 (high exploration during training)
  - state_length=15, seed=0, 20 episodes
"""
import os, sys, json, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
os.environ["PYTHONUNBUFFERED"] = "1"

from controllers.ddpg import train
from controllers.ddpg.config import DDPGConfig
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.state_length_sweep import _analyze_policy, _rollout_actions

def main():
    t0 = time.time()
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=15)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)

    config = DDPGConfig(
        variant="paper",
        seed=0,
        num_episodes=20,
        exploration_mode="softmax",
        exploration_temperature_start=3.0,
        exploration_temperature_end=1.0,
        init_bias_scale=0.0,
        obs_normalize=True,
        random_warmup_steps=200,
        min_buffer_size=200,
        critic_action_input="one_hot",
        log_episodes=True,
    )

    try:
        result = train(env, config, checkpoint_path="artifacts/ddpg/task67_warmup_v2.pt")
        offline = _analyze_policy(result.actor, 15)
        rollout = _rollout_actions(env, result.actor, seed=1000)
        elapsed = time.time() - t0

        out = {
            "task": "TASK-67",
            "approach": "random_warmup_200 + obs_normalize + no_init_bias + softmax_3to1",
            "state_length": 15,
            "episodes": 20,
            "random_warmup_steps": 200,
            "obs_normalize": True,
            "init_bias_scale": 0.0,
            "exploration_mode": "softmax",
            "seed": 0,
            "elapsed_s": elapsed,
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": int(offline["dominant_action"]),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
        }

        os.makedirs("artifacts/ddpg", exist_ok=True)
        with open("artifacts/ddpg/task67_warmup_v2.json", "w") as f:
            json.dump(out, f, indent=2)

        # Print summary
        summary = {k: v for k, v in out.items() if k != "rollout_actions"}
        print(json.dumps(summary, indent=2))
        print(f"\nrollout_unique={out['unique_actions_rollout']} offline_unique={out['unique_actions_offline']}")
        print(f"ACCEPTANCE: {'PASS' if out['unique_actions_rollout'] > 1 and out['unique_actions_offline'] > 1 else 'FAIL'}")

    finally:
        env.close()

if __name__ == "__main__":
    main()
