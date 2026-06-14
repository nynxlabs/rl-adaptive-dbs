"""DDPG trainer smoke tests (mock plant)."""

from __future__ import annotations

import pytest

from controllers.ddpg import DDPGConfig, train_ddpg
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def train_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=3, state_length=2),
    )
    yield env
    env.close()


@pytest.mark.slow
def test_train_ddpg_smoke(train_env: MehreganEnv) -> None:
    config = DDPGConfig(
        num_episodes=2,
        max_episode_steps=3,
        batch_size=4,
        min_buffer_size=4,
        buffer_capacity=64,
        update_frequency=1,
        seed=7,
    )
    result = train_ddpg(train_env, config)
    assert len(result.metrics.episode_rewards) == 2
    assert len(result.metrics.critic_losses) > 0
    assert len(result.metrics.actor_losses) > 0


def test_init_30hz_variant_baseline() -> None:
    config = DDPGConfig(variant="init-30hz")
    assert config.init_baseline == "periodic-30hz"
