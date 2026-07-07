"""Critic action encoding unit tests."""

from __future__ import annotations

import pytest
import torch

from controllers.ddpg import DDPGConfig, train_ddpg
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def train_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=5, state_length=4),
    )
    yield env
    env.close()


def test_q_all_actions_shape(train_env: MehreganEnv) -> None:
    trainer = DDPGTrainer(
        train_env,
        DDPGConfig(batch_size=4, min_buffer_size=4, critic_action_input="one_hot"),
    )
    states = torch.zeros(3, 4)
    q_all = trainer._q_all_actions(trainer.critic, states)
    assert q_all.shape == (3, int(train_env.action_space.n))


def test_action_features_one_hot_vs_logits(train_env: MehreganEnv) -> None:
    trainer = DDPGTrainer(
        train_env,
        DDPGConfig(batch_size=4, min_buffer_size=4, critic_action_input="one_hot"),
    )
    states = torch.zeros(2, 4)
    actions = torch.tensor([3, 17], dtype=torch.long)
    logits = torch.randn(2, int(train_env.action_space.n))
    one_hot = trainer._action_features(states=states, actions=actions, stored_logits=logits)
    assert one_hot.shape == logits.shape
    assert int(one_hot[0].argmax()) == 3
    assert int(one_hot[1].argmax()) == 17

    trainer_logits = DDPGTrainer(
        train_env,
        DDPGConfig(batch_size=4, min_buffer_size=4, critic_action_input="logits"),
    )
    stored = trainer_logits._action_features(states=states, actions=actions, stored_logits=logits)
    assert torch.equal(stored, logits)


def test_actor_q_expectation_softmax_weights(train_env: MehreganEnv) -> None:
    trainer = DDPGTrainer(
        train_env,
        DDPGConfig(batch_size=4, min_buffer_size=4, critic_action_input="one_hot"),
    )
    states = torch.zeros(2, 4)
    logits = torch.zeros(2, int(train_env.action_space.n))
    logits[0, 5] = 10.0
    logits[1, 12] = 10.0
    with torch.no_grad():
        q_all = trainer._q_all_actions(trainer.critic, states)
        expected = (torch.softmax(logits, dim=-1) * q_all).sum(dim=-1)
    got = trainer._actor_q_expectation(states, logits)
    assert torch.allclose(got, expected)


@pytest.mark.slow
def test_one_hot_training_smoke(train_env: MehreganEnv) -> None:
    config = DDPGConfig(
        num_episodes=4,
        max_episode_steps=5,
        batch_size=8,
        min_buffer_size=8,
        buffer_capacity=256,
        update_frequency=2,
        seed=11,
        critic_action_input="one_hot",
        exploration_mode="epsilon",
        exploration_epsilon_start=0.9,
        exploration_epsilon_end=0.3,
    )
    result = train_ddpg(train_env, config)
    assert len(result.metrics.actor_losses) > 0
    assert len(result.metrics.critic_losses) > 0
def test_logits_critic_smoke(train_env: MehreganEnv) -> None:
    config = DDPGConfig(
        num_episodes=2,
        max_episode_steps=3,
        batch_size=4,
        min_buffer_size=4,
        critic_action_input="logits",
        seed=3,
    )
    result = train_ddpg(train_env, config)
    assert len(result.metrics.critic_losses) > 0
