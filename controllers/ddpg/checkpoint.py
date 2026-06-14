"""Checkpoint save/load for trained DDPG actors."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor


def save_checkpoint(
    path: str | Path,
    *,
    actor: Actor,
    config: DDPGConfig,
    state_length: int,
    n_actions: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist actor weights and training config for ``evaluate`` / resume."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "controller": "ddpg",
        "variant": config.variant,
        "actor_state_dict": actor.state_dict(),
        "ddpg_config": asdict(config),
        "state_length": int(state_length),
        "n_actions": int(n_actions),
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
    config = DDPGConfig(**payload["ddpg_config"])
    actor = Actor(
        state_length=int(payload["state_length"]),
        n_actions=int(payload["n_actions"]),
        conv_channels=config.conv_channels,
        shrink_dim=config.shrink_dim,
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device)
    actor.eval()
    return actor, config
