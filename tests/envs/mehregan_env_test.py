"""MehreganEnv unit tests (mock plant)."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs.mehregan import MehreganEnv, MehreganEnvConfig, baseline_action, run_baseline_rollout
from envs.mehregan.patterns import PatternAlphabet
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def mock_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=3, state_length=2),
    )
    yield env
    env.close()


def test_gymnasium_api_check(mock_env: MehreganEnv) -> None:
    check_env(mock_env, skip_render_check=True)


def test_reset_returns_normalized_observation(mock_env: MehreganEnv) -> None:
    obs, info = mock_env.reset(seed=1)
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert info["p_beta_raw"] == pytest.approx(450.0)
    assert info["p_beta_norm"] == pytest.approx(0.45)
    assert info["dw"] == 0.0
    assert "reward" in info


def test_step_truncates_at_horizon(mock_env: MehreganEnv) -> None:
    mock_env.reset(seed=0)
    action = baseline_action("cdbs-130hz")
    for step in range(2):
        _, _, terminated, truncated, _ = mock_env.step(action)
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, info = mock_env.step(action)
    assert not terminated
    assert truncated
    assert info["dw"] == 1.0
    assert info["episode_step"] == 3


def test_higher_frequency_tends_to_lower_p_beta(mock_env: MehreganEnv) -> None:
    mock_env.reset(seed=0)
    _, _, _, _, low = mock_env.step(baseline_action("none"))
    mock_env.reset(seed=0)
    _, _, _, _, high = mock_env.step(baseline_action("cdbs-130hz"))
    assert high["p_beta_raw"] < low["p_beta_raw"]


def test_pattern_alphabet_has_41_actions() -> None:
    assert PatternAlphabet().n_actions == 41


def test_baseline_rollout_mock(mock_env: MehreganEnv) -> None:
    result = run_baseline_rollout(mock_env, "periodic-45hz", seed=5)
    assert result["baseline"] == "periodic-45hz"
    assert result["steps"] == 3
    assert len(result["p_beta"]) == 4  # reset + 3 steps


def test_continuous_rejects_matlab_plant() -> None:
    from envs.plant.matlab_backend import MatlabPlant

    with pytest.raises(ValueError, match="PythonPlant"):
        MehreganEnv(
            plant=MatlabPlant(),
            config=MehreganEnvConfig(plant_integration_mode="continuous"),
        )


def test_unknown_plant_integration_mode_rejected() -> None:
    with pytest.raises(ValueError, match="plant_integration_mode"):
        MehreganEnv(
            plant=MockPlant(),
            config=MehreganEnvConfig(plant_integration_mode="not-a-mode"),
        )


@pytest.mark.plant
def test_continuous_mode_keeps_one_network_per_episode() -> None:
    """Continuous episodes carry state on one network (no wiring redraw per 2 s step)."""
    from envs.plant import PythonPlant

    env = MehreganEnv(
        plant=PythonPlant(),
        config=MehreganEnvConfig(
            step_duration_s=0.01,
            max_episode_steps=3,
            plant_integration_mode="continuous",
        ),
    )
    try:
        env.reset(seed=123_456)  # no cached MATLAB draws for this seed
        first = env._plant._init_draws
        assert first is not None
        env.step(baseline_action("periodic-45hz"))
        env.step(baseline_action("periodic-45hz"))
        assert env._plant._init_draws is first
    finally:
        env.close()
