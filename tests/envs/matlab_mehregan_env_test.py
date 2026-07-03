"""MehreganEnv integration — parametrized plant backends."""

from __future__ import annotations

import pytest

from envs.mehregan import MehreganEnv, MehreganEnvConfig, run_baseline_rollout
from envs.plant import MatlabPlant, PythonPlant
from tests.envs.plant_backends import PlantBackendName, plant_session


@pytest.fixture(scope="module")
def mehregan_env(plant_backend: PlantBackendName) -> MehreganEnv:
    with plant_session(plant_backend) as plant:
        env = MehreganEnv(
            plant=plant,
            config=MehreganEnvConfig(max_episode_steps=2),
        )
        yield env
        env.close()


@pytest.mark.slow
def test_reset_and_one_step(mehregan_env: MehreganEnv) -> None:
    obs, info = mehregan_env.reset(seed=42)
    assert obs.shape == (1,)
    assert info["p_beta_raw"] > 0.0
    obs2, reward, terminated, truncated, step_info = mehregan_env.step(0)
    assert obs2.shape == (1,)
    assert step_info["p_beta_raw"] > 0.0
    assert not terminated
    assert not truncated


@pytest.mark.slow
@pytest.mark.parametrize("name", ["none", "cdbs-130hz", "periodic-45hz"])
def test_baseline_rollout(mehregan_env: MehreganEnv, name: str) -> None:
    result = run_baseline_rollout(mehregan_env, name, seed=11)
    assert result["steps"] == 2
    assert all(p > 0 for p in result["p_beta"])
