"""SEA-DBS evaluation (inference rollouts, carrier-frequency knob)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.checkpoint import load_actor_from_payload, load_checkpoint
from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, SEADBSConfig
from controllers.sea_dbs.networks import Actor
from controllers.sea_dbs.quantization import FP16ActorWrapper, apply_fp16_ptq, is_ptq_variant


def evaluate(
    checkpoint: str | Path,
    *,
    config: SEADBSConfig | None = None,
    plant: Any | None = None,
    episodes: int = 1,
    max_steps: int | None = None,
    carrier_hz: float | None = None,
    use_fp16_ptq: bool = False,
) -> dict[str, Any]:
    """Roll out a trained actor; returns summary metrics and per-step traces."""
    cfg = (config or SEADBSConfig()).with_variant_defaults()
    if max_steps is not None:
        cfg = replace(cfg, max_episode_steps=int(max_steps))

    payload = load_checkpoint(checkpoint, device=cfg.device)
    actor, ckpt_cfg = load_actor_from_payload(payload, device=cfg.device)
    if config is not None:
        cfg = replace(cfg, variant=config.variant, seed=config.seed)

    policy: Actor | FP16ActorWrapper = actor
    if use_fp16_ptq or is_ptq_variant(cfg.variant):
        policy = FP16ActorWrapper(apply_fp16_ptq(actor))

    env = SEA_DBSEnvAdapter(plant=plant, config=cfg)
    hz = float(carrier_hz if carrier_hz is not None else cfg.carrier_hz)
    env.set_carrier_hz(hz)

    episode_rewards: list[float] = []
    p_beta_trajectories: list[list[float]] = []
    action_trajectories: list[list[int]] = []

    try:
        for ep in range(int(episodes)):
            state, info = env.reset(seed=cfg.seed + 10_000 + ep)
            ep_reward = 0.0
            ep_p_beta = [float(info.get("p_beta_norm", 0.0))]
            ep_actions: list[int] = []
            for _ in range(cfg.max_episode_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                with torch.no_grad():
                    logits = policy(state_t)
                    if isinstance(policy, FP16ActorWrapper):
                        action_t, _ = FP16ActorWrapper.select_action(logits)
                    else:
                        action_t, _ = Actor.select_action(logits)
                action = int(action_t.item())
                state, reward, _term, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                ep_p_beta.append(float(step_info.get("p_beta_norm", ep_p_beta[-1])))
                ep_actions.append(action)
                if truncated:
                    break
            episode_rewards.append(ep_reward)
            p_beta_trajectories.append(ep_p_beta)
            action_trajectories.append(ep_actions)
    finally:
        env.close()

    p_beta_final = [traj[-1] for traj in p_beta_trajectories] if p_beta_trajectories else []
    return {
        "protocol": "sea_dbs_eval",
        "controller": "sea_dbs",
        "variant": cfg.variant,
        "seed": cfg.seed,
        "carrier_hz": hz,
        "checkpoint_variant": ckpt_cfg.variant,
        "episode_rewards": episode_rewards,
        "p_beta_trajectories": p_beta_trajectories,
        "action_trajectories": action_trajectories,
        "p_beta_final": p_beta_final,
        "reward_sum": float(np.sum(episode_rewards)),
        "reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "p_beta_mean": float(np.mean(p_beta_final)) if p_beta_final else 0.0,
        "n_episodes": len(episode_rewards),
        "max_steps": cfg.max_episode_steps,
        "fp16_ptq": bool(use_fp16_ptq or is_ptq_variant(cfg.variant)),
    }


def evaluate_ablation_steps(
    checkpoint: str | Path,
    *,
    config: SEADBSConfig | None = None,
    plant: Any | None = None,
    n_steps: int = ABLATION_EVAL_STEPS,
    carrier_hz: float | None = None,
) -> dict[str, Any]:
    """Short PSD eval trace (Fig 7 / Fig 6 — 10 stimulation steps)."""
    return evaluate(
        checkpoint,
        config=config,
        plant=plant,
        episodes=1,
        max_steps=n_steps,
        carrier_hz=carrier_hz,
    )
