"""DDPG checkpoint resume smoke tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from controllers.common.resume import ConfigMismatchError
from controllers.ddpg.checkpoint import load_checkpoint, validate_resume_config
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.trainer import train_ddpg


def _pattern_train_env(cfg: DDPGConfig):
    from controllers.ddpg import default_train_env

    return default_train_env(cfg)


@pytest.mark.slow
def test_ddpg_train_resume_episode_count(tmp_path: Path) -> None:
    cfg = DDPGConfig(
        seed=0,
        num_episodes=2,
        max_episode_steps=5,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=45.0,
        log_episodes=False,
    )
    env = _pattern_train_env(cfg)
    ckpt = tmp_path / "ddpg_resume.pt"
    try:
        first = train_ddpg(env, cfg, checkpoint_path=str(ckpt), checkpoint_interval=2)
        assert len(first.metrics.episode_rewards) == 2

        resume_cfg = DDPGConfig(
            seed=0,
            num_episodes=4,
            max_episode_steps=5,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
            log_episodes=False,
        )
        second = train_ddpg(
            env,
            resume_cfg,
            checkpoint_path=str(ckpt),
            resume_path=str(ckpt),
            checkpoint_interval=2,
        )
        assert len(second.metrics.episode_rewards) == 4
    finally:
        env.close()


@pytest.mark.slow
def test_ddpg_resume_config_mismatch(tmp_path: Path) -> None:
    cfg = DDPGConfig(
        seed=0,
        num_episodes=2,
        max_episode_steps=5,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=45.0,
        log_episodes=False,
    )
    env = _pattern_train_env(cfg)
    ckpt = tmp_path / "ddpg_mismatch.pt"
    try:
        train_ddpg(env, cfg, checkpoint_path=str(ckpt))
        payload = load_checkpoint(ckpt)
        bad = DDPGConfig(
            seed=1,
            num_episodes=4,
            max_episode_steps=5,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=45.0,
        )
        with pytest.raises(ConfigMismatchError):
            validate_resume_config(
                DDPGConfig(**payload["ddpg_config"]),
                bad,
                resume_start=2,
            )
    finally:
        env.close()
