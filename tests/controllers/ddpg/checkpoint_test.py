"""DDPG checkpoint unit tests."""

from __future__ import annotations

from pathlib import Path

import torch

from controllers.ddpg.checkpoint import load_actor, save_checkpoint
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor


def test_save_and_load_actor(tmp_path: Path) -> None:
    config = DDPGConfig(variant="init-30hz", seed=9)
    actor = Actor(state_length=4, n_actions=41)
    path = save_checkpoint(
        tmp_path / "model.pt",
        actor=actor,
        config=config,
        state_length=4,
        n_actions=41,
        extra={"episode": 1},
    )
    loaded, loaded_config = load_actor(path)
    assert loaded_config.variant == "init-30hz"
    assert loaded_config.seed == 9
    state_in = torch.zeros(1, 4)
    assert torch.equal(actor(state_in), loaded(state_in))
