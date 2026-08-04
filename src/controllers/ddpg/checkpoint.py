"""Checkpoint save/load for trained DDPG actors."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from controllers.common.resume import (
    DDPG_MATERIAL_FIELDS,
    config_to_dict,
    infer_completed_episodes,
    validate_resume_config_fields,
)
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor, Critic
from controllers.ddpg.quantization import QATActor


def _config_from_checkpoint_payload(raw: dict[str, Any]) -> DDPGConfig:
    """Drop unknown keys so pre–TASK-146 checkpoints still deserialize config."""
    valid = {field.name for field in fields(DDPGConfig)}
    return DDPGConfig(**{key: value for key, value in raw.items() if key in valid})


def fp_actor_state_dict(actor: Actor) -> dict[str, torch.Tensor]:
    """Strip QAT fake-quant buffers from an actor that was trained with ``prepare_qat``."""
    return {
        key: value
        for key, value in actor.state_dict().items()
        if ".activation_post_process" not in key and ".weight_fake_quant" not in key
    }


def save_checkpoint(
    path: str | Path,
    *,
    actor: Actor,
    config: DDPGConfig,
    state_length: int,
    n_actions: int,
    policy: nn.Module | None = None,
    critic: nn.Module | None = None,
    trainer: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist actor weights and training config for ``evaluate`` / resume."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "controller": "ddpg",
        "variant": config.variant,
        "actor_state_dict": fp_actor_state_dict(actor),
        "ddpg_config": asdict(config),
        "state_length": int(state_length),
        "n_actions": int(n_actions),
    }
    if isinstance(policy, QATActor):
        payload["qat_state_dict"] = policy.state_dict()
    if critic is not None:
        payload["critic_state_dict"] = critic.state_dict()
    if trainer is not None:
        payload["actor_target_state_dict"] = trainer.actor_target.state_dict()
        payload["critic_target_state_dict"] = trainer.critic_target.state_dict()
        payload["buffer_state_dict"] = trainer.buffer.state_dict()
        if trainer.actor_optimizer is not None:
            payload["actor_optimizer_state_dict"] = trainer.actor_optimizer.state_dict()
        if trainer.critic_optimizer is not None:
            payload["critic_optimizer_state_dict"] = trainer.critic_optimizer.state_dict()
        payload["trainer_state"] = {
            "env_step": int(trainer._env_step),
            "warmup_steps_done": int(trainer._warmup_steps_done),
            "reward_running_mean": float(trainer._reward_running_mean),
            "reward_running_var": float(trainer._reward_running_var),
            "reward_count": int(trainer._reward_count),
            "obs_count": int(trainer._obs_count),
            "obs_mean": trainer._obs_mean.copy(),
            "obs_m2": trainer._obs_m2.copy(),
            "episode_rewards": list(trainer.metrics.episode_rewards),
            "episode_steps": list(trainer.metrics.episode_steps),
        }
    if extra:
        payload["extra"] = extra
    torch.save(payload, out)
    return out


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict from disk."""
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_actor(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Actor, DDPGConfig]:
    """Restore ``Actor`` and ``DDPGConfig`` from a checkpoint file."""
    payload = load_checkpoint(path, device=device)
    config = _config_from_checkpoint_payload(payload["ddpg_config"])
    actor = Actor(
        state_length=int(payload["state_length"]),
        n_actions=int(payload["n_actions"]),
        conv1_out=config.conv1_out,
        conv2_out=config.conv2_out,
        pool_kernel=config.pool_kernel,
        fc_hidden=config.fc_hidden,
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device)
    actor.eval()
    return actor, config


def qat_state_dict_from_checkpoint(payload: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """Return saved QAT observer/fake-quant state when present."""
    state = payload.get("qat_state_dict")
    if state is None:
        return None
    return state


def validate_resume_config(
    saved_config: DDPGConfig,
    active_config: DDPGConfig,
    *,
    resume_start: int = 0,
) -> None:
    validate_resume_config_fields(
        config_to_dict(saved_config),
        config_to_dict(active_config),
        DDPG_MATERIAL_FIELDS,
        label="DDPGConfig",
        resume_start=resume_start,
    )


def infer_ddpg_start_episode(
    payload: dict[str, Any],
    *,
    metrics_path: Path | None = None,
    start_episode: int | None = None,
) -> int:
    trainer_state = payload.get("trainer_state")
    if isinstance(trainer_state, dict) and "episode_rewards" in trainer_state:
        wrapped = {"extra": {"episode_rewards": trainer_state["episode_rewards"]}}
        return infer_completed_episodes(
            wrapped,
            metrics_path=metrics_path,
            start_episode=start_episode,
        )
    extra = payload.get("extra")
    if isinstance(extra, dict) and "episode_rewards" in extra:
        return infer_completed_episodes(
            payload,
            metrics_path=metrics_path,
            start_episode=start_episode,
        )
    return infer_completed_episodes(
        payload,
        metrics_path=metrics_path,
        start_episode=start_episode,
    )
