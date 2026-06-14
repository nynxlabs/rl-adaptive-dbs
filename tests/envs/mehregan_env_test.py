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
