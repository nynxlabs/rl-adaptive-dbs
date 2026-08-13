"""Tests for Mehregan Fig 4a DDPG training profile."""

from __future__ import annotations

from controllers.ddpg.config import fig4a_ddpg_config


def test_fig4a_profile_uses_one_hot_critic_and_softmax() -> None:
    cfg = fig4a_ddpg_config()
    assert cfg.action_space_mode == "fixed_mean_pattern"
    assert cfg.pattern_mean_hz == 45.0
    assert cfg.exploration_mode == "softmax"
    assert cfg.critic_action_input == "one_hot"
    assert cfg.init_bias_scale == 0.5
    assert cfg.exploration_temperature_end == 1.4
    assert cfg.critic_warmup_steps == 100
    assert cfg.random_warmup_steps == 100
