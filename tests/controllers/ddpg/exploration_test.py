"""DDPG training exploration unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def trainer() -> DDPGTrainer:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=30, state_length=2),
    )
    config = DDPGConfig(
        num_episodes=10,
        max_episode_steps=30,
        exploration_epsilon_start=0.5,
        exploration_epsilon_end=0.1,
        exploration_temperature_start=2.0,
        exploration_temperature_end=0.5,
        seed=0,
    )
    t = DDPGTrainer(env, config)
    yield t
    env.close()


def test_exploration_fraction_bounds(trainer: DDPGTrainer) -> None:
    assert trainer._exploration_fraction(0) == 0.0
    total = trainer.config.num_episodes * trainer.env.config.max_episode_steps
    assert trainer._exploration_fraction(total) == 1.0
    assert trainer._exploration_fraction(total + 100) == 1.0


def test_exploration_epsilon_linear_decay(trainer: DDPGTrainer) -> None:
    assert trainer._exploration_epsilon(0) == pytest.approx(0.5)
    total = trainer.config.num_episodes * trainer.env.config.max_episode_steps
    assert trainer._exploration_epsilon(total) == pytest.approx(0.1)
    mid = total // 2
    assert trainer._exploration_epsilon(mid) == pytest.approx(0.3)


def test_exploration_temperature_linear_decay(trainer: DDPGTrainer) -> None:
    assert trainer._exploration_temperature(0) == pytest.approx(2.0)
    total = trainer.config.num_episodes * trainer.env.config.max_episode_steps
    assert trainer._exploration_temperature(total) == pytest.approx(0.5)


def test_select_action_epsilon_greedy_explores(trainer: DDPGTrainer) -> None:
    state = np.zeros(trainer.env.observation_space.shape, dtype=np.float32)
    trainer.env.action_space.sample = MagicMock(return_value=7)  # type: ignore[method-assign]

    with patch("numpy.random.random", return_value=0.0):
        action, logits = trainer._select_action(state, env_step=0)

    assert action == 7
    assert logits.shape == (trainer.env.action_space.n,)


def test_select_action_epsilon_greedy_exploits(trainer: DDPGTrainer) -> None:
    state = np.zeros(trainer.env.observation_space.shape, dtype=np.float32)
    trainer.actor.init_toward_action(3)

    with patch("numpy.random.random", return_value=1.0):
        action, logits = trainer._select_action(state, env_step=0)

    assert action == 3
    assert logits.shape == (trainer.env.action_space.n,)


def test_select_action_softmax_samples() -> None:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=30, state_length=2),
    )
    config = DDPGConfig(exploration_mode="softmax", seed=0)
    trainer = DDPGTrainer(env, config)
    state = np.zeros(env.observation_space.shape, dtype=np.float32)

    try:
        with patch("torch.multinomial", return_value=torch.tensor([5])):
            action, logits = trainer._select_action(state, env_step=0)

        assert action == 5
        assert logits.shape == (env.action_space.n,)
    finally:
        env.close()
