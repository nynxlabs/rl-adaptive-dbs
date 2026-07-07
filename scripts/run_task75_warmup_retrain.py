#!/usr/bin/env python3
"""TASK-75: Retrain DDPG with critic warmup + Q-scale fix + Huber loss + logit noise.

Applies all4 fixes from COO directive (TASK-74) and runs 30 episodes with logits
critic to verify adaptive policy. Fixes:
  1. Critic warmup (100 steps) — stabilize Q-values before actor exploits
  2. Reward normalization — match Q-scale to reward magnitude
  3. Huber loss — more robust than MSE for noisy rewards
  4. Logit noise (0.15) — prevent logit margin collapse
  5. Save critic in checkpoint — for post-hoc Q probes
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logit-noise", type=float, default=0.15)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ddpg/task75_warmup_retrain.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/ddpg/paper_train0_task75_warmup.pt"),
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
            exploration_mode="softmax",
            exploration_temperature_start=2.0,
            exploration_temperature_end=0.5,
            critic_action_input="logits",
            # The 4 fixes:
            critic_warmup_steps=args.warmup_steps,
            reward_normalize=True,
            critic_loss_fn="huber",
            logit_noise_std=args.logit_noise,
            log_episodes=True,
        )

        print(f"=== TASK-75 warmup retrain ===", flush=True)
        print(f"  state_length={args.state_length}", flush=True)
        print(f"  episodes={args.episodes}", flush=True)
        print(f"  critic_action_input=logits", flush=True)
        print(f"  critic_warmup_steps={args.warmup_steps}", flush=True)
        print(f"  reward_normalize=True", flush=True)
        print(f"  critic_loss_fn=huber", flush=True)
        print(f"  logit_noise_std={args.logit_noise}", flush=True)
        print(flush=True)

        t0 = time.time()
        result = train(env, config, checkpoint_path=args.checkpoint)
        elapsed = time.time() - t0

        offline = _analyze_policy(result.actor, args.state_length)
        rollout = _rollout_actions(env, result.actor, seed=args.seed + 1000)

        # Q-discrimination probe on final critic
        import torch
        import torch.nn.functional as F
        q_stds = []
        with torch.no_grad():
            for i in range(min(5, len(result.metrics.episode_rewards))):
                state, _ = env.reset(seed=args.seed + 2000 + i)
                state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                logits = result.actor(state_t)
                batch = state_t.expand(config.critic_action_input == "logits" and 1 or 1, -1)
                # Get Q-values for all actions via one_hot
                n_actions = int(env.action_space.n)
                eye = torch.eye(n_actions)
                q_vals = []
                for a in range(n_actions):
                    af = eye[a:a+1]
                    q = result.critic(state_t, af).item()
                    q_vals.append(q)
                q_stds.append(float(torch.tensor(q_vals).std()))
        q_std_mean = sum(q_stds) / len(q_stds) if q_stds else 0.0

        out = {
            "task": "TASK-75",
            "state_length": args.state_length,
            "episodes": args.episodes,
            "exploration_mode": "softmax",
            "critic_action_input": "logits",
            "critic_warmup_steps": args.warmup_steps,
            "reward_normalize": True,
            "critic_loss_fn": "huber",
            "logit_noise_std": args.logit_noise,
            "seed": args.seed,
            "elapsed_s": round(elapsed, 1),
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": int(offline["dominant_action"]),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
            "q_discrimination_std_mean": q_std_mean,
            "acceptance_pass": len(set(rollout)) > 1 and int(offline["unique_actions_offline"]) > 1,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        summary = {k: v for k, v in out.items() if k != "rollout_actions"}
        print(json.dumps(summary, indent=2))
        print(
            f"\nrollout_unique={out['unique_actions_rollout']} "
            f"offline_unique={out['unique_actions_offline']} "
            f"q_std={q_std_mean:.4f}",
            flush=True,
        )
        return 0 if out["acceptance_pass"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
