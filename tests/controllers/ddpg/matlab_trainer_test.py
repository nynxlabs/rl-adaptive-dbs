"""DDPG train + eval on the Kumaravelu MATLAB plant (@pytest.mark.matlab)."""

from __future__ import annotations

from pathlib import Path

import pytest

from controllers.ddpg import DDPGConfig, EvalConfig, evaluate, train
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from envs.plant import MatlabPlant


@pytest.fixture(scope="module")
def matlab_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MatlabPlant(),
        config=MehreganEnvConfig(max_episode_steps=2),
    )
    yield env
    env.close()


@pytest.mark.matlab
@pytest.mark.slow
def test_train_and_eval_on_matlab_plant(matlab_env: MehreganEnv, tmp_path: Path) -> None:
    """Short DDPG loop on the real plant — Phase 3 end-to-end smoke."""
    config = DDPGConfig(
        num_episodes=1,
        batch_size=2,
        min_buffer_size=2,
        buffer_capacity=64,
        seed=5,
    )
    checkpoint = tmp_path / "actor.pt"
    result = train(matlab_env, config, checkpoint_path=checkpoint)
    assert len(result.metrics.episode_rewards) == 1
    assert checkpoint.is_file()

    metrics = evaluate(
        matlab_env,
        checkpoint,
        config=EvalConfig(seed=7, eval_steps=1),
    )
    assert metrics["protocol"] == "mehregan_eval"
    assert metrics["steps"] == 1
    assert len(metrics["p_beta"]) == 2  # reset + one policy step
    assert all(p > 0.0 for p in metrics["p_beta"])
