"""Checkpoint save/load for SEA-DBS."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from controllers.common.resume import (
    SEA_DBS_MATERIAL_FIELDS,
    config_to_dict,
    infer_completed_episodes,
    validate_resume_config_fields,
)
from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.networks import Actor, Critic, PredictiveModel


def save_checkpoint(
    path: str | Path,
    *,
    actor: Actor,
    critic: Critic,
    config: SEADBSConfig,
    predictive_model: PredictiveModel | None = None,
    trainer: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "controller": "sea_dbs",
        "variant": config.variant,
        "sea_dbs_config": asdict(config),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "state_dim": int(config.state_dim),
        "n_actions": int(config.n_actions),
    }
    if predictive_model is not None:
        payload["predictive_state_dict"] = predictive_model.state_dict()
    if trainer is not None:
        payload["actor_target_state_dict"] = trainer.actor_target.state_dict()
        payload["critic_target_state_dict"] = trainer.critic_target.state_dict()
        payload["buffer_state_dict"] = trainer.buffer.state_dict()
        payload["actor_optimizer_state_dict"] = trainer.actor_optimizer.state_dict()
        payload["critic_optimizer_state_dict"] = trainer.critic_optimizer.state_dict()
        if trainer.pred_optimizer is not None:
            payload["pred_optimizer_state_dict"] = trainer.pred_optimizer.state_dict()
        payload["trainer_state"] = {
            "total_steps": int(trainer._total_steps),
            "update_count": int(trainer._update_count),
            "rng_state": trainer._rng.bit_generator.state,
            "episode_rewards": list(trainer.metrics.episode_rewards),
            "episode_psd": list(trainer.metrics.episode_psd),
        }
    if extra:
        payload["extra"] = extra
    torch.save(payload, out)
    return out


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_actor_from_payload(
    payload: dict[str, Any],
    *,
    device: str = "cpu",
) -> tuple[Actor, SEADBSConfig]:
    raw = payload["sea_dbs_config"]
    config = SEADBSConfig(**raw)
    actor = Actor(
        state_dim=int(payload.get("state_dim", config.state_dim)),
        n_actions=int(payload.get("n_actions", config.n_actions)),
        hidden_size=config.hidden_size,
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device)
    actor.eval()
    return actor, config


def validate_resume_config(
    saved_config: SEADBSConfig,
    active_config: SEADBSConfig,
    *,
    resume_start: int = 0,
) -> None:
    validate_resume_config_fields(
        config_to_dict(saved_config),
        config_to_dict(active_config),
        SEA_DBS_MATERIAL_FIELDS,
        label="SEADBSConfig",
        resume_start=resume_start,
    )


def infer_sea_dbs_start_episode(
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
    return infer_completed_episodes(
        payload,
        metrics_path=metrics_path,
        start_episode=start_episode,
    )
