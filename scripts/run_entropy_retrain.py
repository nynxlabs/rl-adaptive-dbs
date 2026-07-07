#!/usr/bin/env python3
"""TASK-67: DDPG with entropy regularization to prevent logit collapse.

Root cause: actor logit margins collapse to ~0, argmax picks one action.
Fix: entropy regularization on actor loss + one_hot critic + epsilon-greedy.

Config:
  - critic_action_input=one_hot (correct for discrete actions)
  - exploration_mode=epsilon (default: 0.5→0.1)
  - entropy_coeff=0.05 (prevents logit collapse)
  - critic_warmup_steps=100
  - reward_normalize=True
  - critic_loss_fn=huber
  - logit_noise_std=0.1
  - 50 episodes
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
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--entropy-coeff", type=float, default=0.05)
    parser.add_argument("--logit-noise", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ddpg/entropy_retrain.json"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/ddpg/paper_train0_entropy.pt"),
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
            # Correct discrete-action critic
            critic_action_input="one_hot",
            # Epsilon-greedy exploration (default schedule)
            exploration_mode="epsilon",
            exploration_epsilon_start=0.5,
            exploration_epsilon_end=0.1,
            # Entropy regularization to prevent logit collapse
            entropy_coeff=args.entropy_coeff,
            # Other fixes
            critic_warmup_steps=args.warmup_steps,
            reward_normalize=True,
            critic_loss_fn="huber",
            logit_noise_std=args.logit_noise,
            log_episodes=True,
        )

        print(f"=== Entropy-regularized DDPG retrain ===", flush=True)
        print(f"  state_length={args.state_length}", flush=True)
        print(f"  episodes={args.episodes}", flush=True)
        print(f"  critic_action_input=one_hot", flush=True)
        print(f"  exploration_mode=epsilon (0.5→0.1)", flush=True)
        print(f"  entropy_coeff={args.entropy_coeff}", flush=True)
        print(f"  logit_noise_std={args.logit_noise}", flush=True)
        print(f"  critic_warmup_steps={args.warmup_steps}", flush=True)
        print(f"  reward_normalize=True", flush=True)
        print(f"  critic_loss_fn=huber", flush=True)
        print(flush=True)

        t0 = time.time()
        result = train(env, config, checkpoint_path=args.checkpoint)
        elapsed = time.time() - t0

        offline = _analyze_policy(result.actor, args.state_length)
        rollout = _rollout_actions(env, result.actor, seed=args.seed + 1000)

        # Q-discrimination probe
        import torch
        import torch.nn.functional as F
        q_stds = []
        with torch.no_grad():
            for i in range(min(5, len(result.metrics.episode_rewards))):
                state, _ = env.reset(seed=args.seed + 2000 + i)
                state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                n_actions = int(env.action_space.n)
                eye = torch.eye(n_actions)
                q_vals = []
                for a in range(n_actions):
                    af = eye[a:a+1]
                    q = result.critic(state_t, af).item()
                    q_vals.append(q)
                q_stds.append(float(torch.tensor(q_vals).std()))
        q_std_mean = sum(q_stds) / len(q_stds) if q_stds else 0.0

        # Logit statistics from last episode
        import numpy as np
        state, _ = env.reset(seed=args.seed + 3000)
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            final_logits = result.actor(state_t).squeeze(0).cpu().numpy()
        logit_std = float(np.std(final_logits))
        logit_max_min = float(np.max(final_logits) - np.min(final_logits))

        out = {
            "task": "TASK-67-entropy",
            "state_length": args.state_length,
            "episodes": args.episodes,
            "exploration_mode": "epsilon",
            "critic_action_input": "one_hot",
            "entropy_coeff": args.entropy_coeff,
            "logit_noise_std": args.logit_noise,
            "critic_warmup_steps": args.warmup_steps,
            "reward_normalize": True,
            "critic_loss_fn": "huber",
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
            "logit_std": logit_std,
            "logit_max_min": logit_max_min,
            "acceptance_pass": len(set(rollout)) > 1 and int(offline["unique_actions_offline"]) > 1,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        summary = {k: v for k, v in out.items() if k != "rollout_actions"}
        print(json.dumps(summary, indent=2))
        print(
            f"\nrollout_unique={out['unique_actions_rollout']} "
            f"offline_unique={out['unique_actions_offline']} "
            f"q_std={q_std_mean:.4f} "
            f"logit_std={logit_std:.4f} "
            f"logit_max_min={logit_max_min:.4f}",
            flush=True,
        )
        return 0 if out["acceptance_pass"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
