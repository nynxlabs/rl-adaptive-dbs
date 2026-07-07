#!/usr/bin/env python3
"""Quick test: state_length=1 (paper's original) — should show adaptive behavior."""
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
    resolved = resolve_config()
    env_cfg = MehreganEnvConfig(state_length=1)
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg)

    config = DDPGConfig(
        variant="paper",
        seed=0,
        num_episodes=10,
        exploration_mode="softmax",
        exploration_temperature_start=1.0,
        exploration_temperature_end=1.0,
        critic_action_input="logits",
        init_bias_scale=2.0,
        log_episodes=True,
    )

    print("=== state_length=1 test (paper original) ===", flush=True)
    print(f"  episodes: 10", flush=True)
    print(f"  exploration: greedy argmax (temp=1.0)", flush=True)
    print(flush=True)

    t0 = time.time()
    result = train(env, config, checkpoint_path="artifacts/ddpg/paper_train0_sl1_test.pt")
    elapsed = time.time() - t0

    offline = _analyze_policy(result.actor, 1)
    rollout = _rollout_actions(env, result.actor, seed=1000)

    out = {
        "test": "state_length=1 paper original",
        "state_length": 1,
        "episodes": 10,
        "elapsed_s": round(elapsed, 1),
        "unique_actions_offline": int(offline["unique_actions_offline"]),
        "dominant_action": int(offline["dominant_action"]),
        "dominant_fraction": float(offline["dominant_fraction"]),
        "unique_actions_rollout": len(set(rollout)),
        "rollout_actions": rollout,
        "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
        "mean_reward": float(sum(result.metrics.episode_rewards) / len(result.metrics.episode_rewards)),
    }

    with open("artifacts/ddpg/task67_sl1_test.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nunique_actions_rollout: {out['unique_actions_rollout']}", flush=True)
    print(f"unique_actions_offline: {out['unique_actions_offline']}", flush=True)
    print(f"dominant_action: {out['dominant_action']} ({out['dominant_fraction']:.2%})", flush=True)
    print(f"mean_reward: {out['mean_reward']:.2f}", flush=True)
    print(f"elapsed: {elapsed:.1f}s", flush=True)
    print(f"rollout: {rollout}", flush=True)

if __name__ == "__main__":
    main()
