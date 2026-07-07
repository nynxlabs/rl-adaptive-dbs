#!/usr/bin/env python3
"""Pattern-mode DDPG training (TASK-83 Option C).

Train with the fixed-mean-frequency pattern action space matching
Mehregan et al.'s formulation. All patterns share the same mean rate
(45 Hz default); the agent learns to shape temporal irregularity.

Run: uv run python scripts/run_pattern_train.py
     tmux new-session -d -s train-pattern 'uv run python scripts/run_pattern_train.py'
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
from controllers.ddpg.networks import Actor
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config


def _analyze_policy(actor: Actor, state_length: int, *, n_samples: int = 500) -> dict:
    """Offline policy analysis: what actions does the actor prefer across states?"""
    import numpy as np
    import torch

    actions = []
    for _ in range(n_samples):
        state = np.random.randn(state_length).astype(np.float32) * 0.1 + 0.4
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = actor(state_t)
            action = int(torch.argmax(logits, dim=-1).item())
            actions.append(action)

    from collections import Counter
    counts = Counter(actions)
    return {
        "unique_actions_offline": len(counts),
        "dominant_action": counts.most_common(1)[0][0],
        "dominant_fraction": counts.most_common(1)[0][1] / n_samples,
        "action_distribution": dict(sorted(counts.items())),
    }


def _rollout_actions(env: MehreganEnv, actor: Actor, seed: int) -> list[int]:
    """Run one episode and record actions."""
    import numpy as np
    import torch

    state, _ = env.reset(seed=seed)
    actions = []
    done = False
    while not done:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = actor(state_t)
            action = int(torch.argmax(logits, dim=-1).item())
        actions.append(action)
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return actions


def main() -> int:
    resolved = resolve_config()
    mean_hz = 45.0

    alphabet = FixedMeanPatternAlphabet(mean_hz=mean_hz)
    env_cfg = MehreganEnvConfig(state_length=1)  # paper original
    plant = PythonPlant(config=resolved.plant)
    env = MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)
    try:
        config = DDPGConfig(
            variant="paper",
            seed=0,
            num_episodes=30,
            # Paper-exact: greedy argmax, no exploration noise
            exploration_mode="softmax",
            exploration_temperature_start=1.0,
            exploration_temperature_end=1.0,
            # Logits critic (paper architecture)
            critic_action_input="logits",
            # Extensions OFF for clean baseline
            critic_warmup_steps=0,
            reward_normalize=False,
            critic_loss_fn="mse",
            logit_noise_std=0.0,
            log_episodes=True,
        )

        print(f"=== Pattern-mode DDPG training (Option C) ===", flush=True)
        print(f"  mean_hz={mean_hz}, n_patterns={alphabet.n_actions}", flush=True)
        print(f"  state_length={env_cfg.state_length}", flush=True)
        print(f"  episodes={config.num_episodes}", flush=True)
        print(f"  exploration: greedy argmax (temp=1.0)", flush=True)
        print(flush=True)

        t0 = time.time()
        result = train(env, config, checkpoint_path=Path("artifacts/ddpg/pattern_train0.pt"))
        elapsed = time.time() - t0

        offline = _analyze_policy(result.actor, env_cfg.state_length)
        rollout = _rollout_actions(env, result.actor, seed=1000)

        out = {
            "task": "pattern-action-space-train",
            "option": "C",
            "mean_hz": mean_hz,
            "n_patterns": alphabet.n_actions,
            "state_length": env_cfg.state_length,
            "episodes": config.num_episodes,
            "seed": config.seed,
            "elapsed_s": round(elapsed, 1),
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "unique_actions_offline": int(offline["unique_actions_offline"]),
            "dominant_action": int(offline["dominant_action"]),
            "dominant_fraction": float(offline["dominant_fraction"]),
            "action_distribution_offline": offline["action_distribution"],
            "unique_actions_rollout": len(set(rollout)),
            "rollout_actions": rollout,
            "episode_rewards": [float(r) for r in result.metrics.episode_rewards],
            "acceptance_pass": len(set(rollout)) > 1 and int(offline["unique_actions_offline"]) > 1,
        }

        out_path = Path("artifacts/ddpg/pattern_train0.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")

        summary = dict(out)
        summary.pop("rollout_actions", None)
        summary.pop("episode_rewards", None)
        print(json.dumps(summary, indent=2), flush=True)
        print(
            f"\nrollout_unique={out['unique_actions_rollout']} "
            f"offline_unique={out['unique_actions_offline']}",
            flush=True,
        )

        if out["acceptance_pass"]:
            print("\n✓ ACCEPTANCE PASS: policy uses multiple actions (adaptive)", flush=True)
        else:
            print("\n✗ ACCEPTANCE FAIL: policy collapsed to constant action", flush=True)

        return 0 if out["acceptance_pass"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main())
