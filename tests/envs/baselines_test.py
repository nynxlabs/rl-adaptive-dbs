"""Baseline rollout tests (mock plant)."""

from __future__ import annotations

import pytest

from envs.mehregan import MehreganEnv, MehreganEnvConfig, default_baselines, run_baseline_rollout
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def mock_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=2),
    )
    yield env
    env.close()


@pytest.mark.parametrize("name", list(default_baselines().keys()))
def test_baseline_rollout_runs(mock_env: MehreganEnv, name: str) -> None:
    result = run_baseline_rollout(mock_env, name, seed=0)
    assert result["baseline"] == name
    assert result["steps"] == 2
    assert len(result["p_beta"]) == 3


def test_cdbs_lowers_p_beta_vs_none(mock_env: MehreganEnv) -> None:
    none = run_baseline_rollout(mock_env, "none", seed=1)
    cdbs = run_baseline_rollout(mock_env, "cdbs-130hz", seed=1)
    assert cdbs["p_beta"][-1] < none["p_beta"][-1]
