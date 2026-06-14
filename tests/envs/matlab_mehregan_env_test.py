"""MehreganEnv integration (@pytest.mark.matlab)."""

from __future__ import annotations

import pytest

from envs.mehregan import MehreganEnv, MehreganEnvConfig, run_baseline_rollout
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
def test_reset_and_one_step(matlab_env: MehreganEnv) -> None:
    obs, info = matlab_env.reset(seed=42)
    assert obs.shape == (1,)
    assert info["p_beta_raw"] > 0.0
    obs2, reward, terminated, truncated, step_info = matlab_env.step(0)
    assert obs2.shape == (1,)
    assert step_info["p_beta_raw"] > 0.0
    assert not terminated
    assert not truncated


@pytest.mark.matlab
@pytest.mark.slow
@pytest.mark.parametrize("name", ["none", "cdbs-130hz", "periodic-45hz"])
def test_baseline_rollout(matlab_env: MehreganEnv, name: str) -> None:
    result = run_baseline_rollout(matlab_env, name, seed=11)
    assert result["steps"] == 2
    assert all(p > 0 for p in result["p_beta"])
