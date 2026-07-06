#!/usr/bin/env python3
"""Sweep state_length values and report policy adaptivity after short DDPG training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from controllers.ddpg import train
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config


@dataclass
class SweepResult:
    state_length: int
    final_reward: float
    unique_actions_offline: int
    dominant_action: int
    dominant_fraction: float
    unique_actions_rollout: int
    rollout_actions: list[int]
    episode_rewards: list[float]


def _analyze_policy(actor: Actor, state_length: int, *, n_samples: int = 500) -> dict[str, float | int]:
    """Probe actor on synthetic biomarker windows; count argmax diversity."""
    actor.eval()
    actions: list[int] = []
    with torch.no_grad():
        for _ in range(n_samples):
            # Random temporal window in normalized operating range (0.3–0.6).
            window = np.random.uniform(0.3, 0.6, size=state_length).astype(np.float32)
            state_t = torch.as_tensor(window).unsqueeze(0)
            logits = actor(state_t)
            action, _ = Actor.select_action(logits)
            actions.append(int(action.item()))
    unique = sorted(set(actions))
    counts = np.bincount(actions, minlength=actor.head.out_features)
    dominant = int(counts.argmax())
    dominant_fraction = float(counts[dominant] / len(actions))
    return {
        "unique_actions_offline": len(unique),
        "dominant_action": dominant,
        "dominant_fraction": dominant_fraction,
    }


def _rollout_actions(env: MehreganEnv, actor: Actor, seed: int) -> list[int]:
    actor.eval()
    state, _ = env.reset(seed=seed)
    actions: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            logits = actor(state_t)
            action, _ = Actor.select_action(logits)
        actions.append(int(action.item()))
        state, _r, terminated, truncated, _info = env.step(int(action.item()))
    return actions


def run_sweep(
    state_lengths: list[int],
    *,
    seed: int,
    episodes: int,
    out_path: Path,
) -> list[SweepResult]:
    resolved = resolve_config()
    results: list[SweepResult] = []

    for sl in state_lengths:
        print(f"=== state_length={sl} ===", flush=True)
        env_cfg = MehreganEnvConfig(state_length=sl)
        plant = PythonPlant(config=resolved.plant)
        env = MehreganEnv(plant=plant, config=env_cfg)
        try:
            config = DDPGConfig(variant="paper", seed=seed, num_episodes=episodes)
            result = train(env, config)
            offline = _analyze_policy(result.actor, sl)
            rollout = _rollout_actions(env, result.actor, seed=seed + 1000)
            sweep = SweepResult(
                state_length=sl,
                final_reward=float(result.metrics.episode_rewards[-1]),
                unique_actions_offline=int(offline["unique_actions_offline"]),
                dominant_action=int(offline["dominant_action"]),
                dominant_fraction=float(offline["dominant_fraction"]),
                unique_actions_rollout=len(set(rollout)),
                rollout_actions=rollout,
                episode_rewards=[float(r) for r in result.metrics.episode_rewards],
            )
            results.append(sweep)
            print(
                f"  final_reward={sweep.final_reward:.2f} "
                f"offline_unique={sweep.unique_actions_offline} "
                f"dominant={sweep.dominant_action}@{sweep.dominant_fraction:.1%} "
                f"rollout_unique={sweep.unique_actions_rollout}",
                flush=True,
            )
        finally:
            env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-lengths",
        default="5,10,15,30",
        help="Comma-separated state_length values",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/ddpg/state_length_sweep.json"),
    )
    args = parser.parse_args()
    lengths = [int(x.strip()) for x in args.state_lengths.split(",") if x.strip()]
    run_sweep(lengths, seed=args.seed, episodes=args.episodes, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
