"""Checkpoint resume smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from controllers.common.resume import ConfigMismatchError
from controllers.snn.config import SNNConfig
from controllers.snn.trainer import (
    load_checkpoint,
    resume_dsqn_trainer,
    train_dsqn,
)


@pytest.mark.slow
def test_snn_train_resume_episode_count(tmp_path: Path) -> None:
    cfg = SNNConfig(seed=0).for_smoke(episodes=3, max_steps=5)
    ckpt = tmp_path / "resume.pt"
    first = train_dsqn(config=cfg, checkpoint_path=ckpt, checkpoint_interval=3)
    assert len(first.episode_rewards) == 3

    resume_cfg = SNNConfig(seed=0).for_smoke(episodes=5, max_steps=5)
    second = train_dsqn(
        config=resume_cfg,
        checkpoint_path=ckpt,
        resume_path=ckpt,
        checkpoint_interval=2,
    )
    assert len(second.episode_rewards) == 5
    payload = load_checkpoint(ckpt)
    extra = payload.get("extra", {})
    assert int(extra.get("completed_episodes", len(second.episode_rewards))) == 5


@pytest.mark.slow
def test_snn_resume_config_mismatch(tmp_path: Path) -> None:
    cfg = SNNConfig(seed=0).for_smoke(episodes=2, max_steps=5)
    ckpt = tmp_path / "mismatch.pt"
    train_dsqn(config=cfg, checkpoint_path=ckpt)

    payload = load_checkpoint(ckpt)
    bad_cfg = SNNConfig(seed=0).for_smoke(episodes=2, max_steps=8)
    with pytest.raises(ConfigMismatchError):
        resume_dsqn_trainer(payload, config=bad_cfg)
