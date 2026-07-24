"""Nguyen DSQN evaluation (50 episodes × 25 steps by default)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer
from controllers.snn.config import EVAL_EPISODES, SNNConfig
from controllers.snn.networks import DSQN
from controllers.snn.trainer import DSQNTrainer, load_checkpoint


def evaluate(
    checkpoint: str | Path,
    *,
    config: SNNConfig | None = None,
    plant: Any | None = None,
    episodes: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Roll out a trained DSQN; returns summary metrics + per-episode traces."""
    cfg = (config or SNNConfig()).with_variant_defaults()
    if max_steps is not None:
        cfg = replace(cfg, max_episode_steps=int(max_steps))

    payload = load_checkpoint(checkpoint, map_location=cfg.device)
    dsqn = DSQN(cfg)
    dsqn.load_state_dict(payload["dsqn_state_dict"])
    dsqn.to(torch.device(cfg.device))
    dsqn.eval()

    buffer = ReplayBuffer(cfg, seed=cfg.seed)
    trainer = DSQNTrainer(dsqn, buffer, cfg)
    env = NguyenEnvAdapter(plant=plant, config=cfg)
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    alpha_betas: list[float] = []
    n_eps = int(episodes) if episodes is not None else EVAL_EPISODES
    try:
        for ep in range(n_eps):
            obs, info = env.reset(seed=cfg.seed + 10_000 + ep)
            ep_reward = 0.0
            steps = 0
            last_ab = float(info.get("alpha_beta", 0.0))
            for _ in range(cfg.max_episode_steps):
                _action_index, indices = trainer.act(obs, explore=False)
                obs, reward, terminated, truncated, step_info = env.step(indices)
                ep_reward += float(reward)
                steps += 1
                last_ab = float(step_info.get("alpha_beta", last_ab))
                if terminated or truncated:
                    break
            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)
            alpha_betas.append(last_ab)
    finally:
        env.close()

    return {
        "protocol": "nguyen_eval",
        "controller": "snn",
        "variant": cfg.variant,
        "seed": cfg.seed,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "alpha_beta_final": alpha_betas,
        "reward_sum": float(np.sum(episode_rewards)),
        "reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "alpha_beta_mean": float(np.mean(alpha_betas)) if alpha_betas else 0.0,
        "p_beta_mean": float(np.mean(alpha_betas)) if alpha_betas else 0.0,
        "n_episodes": len(episode_rewards),
    }
