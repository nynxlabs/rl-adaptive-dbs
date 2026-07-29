"""Checkpoint save/load for SEA-DBS."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from controllers.sea_dbs.config import SEADBSConfig
from controllers.sea_dbs.networks import Actor, Critic, PredictiveModel


def save_checkpoint(
    path: str | Path,
    *,
    actor: Actor,
    critic: Critic,
    config: SEADBSConfig,
    predictive_model: PredictiveModel | None = None,
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
