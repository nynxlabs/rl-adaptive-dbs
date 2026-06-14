"""DDPG replication workflow tests (mock plant)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from controllers.ddpg import DDPGConfig
from controllers.ddpg.replication import (
    ReplicationConfig,
    baseline_names_for_variant,
    run_replication,
    write_replication_summary,
)
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def repl_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=30),
    )
    yield env
    env.close()


def test_baseline_names_for_init_30hz() -> None:
    names = baseline_names_for_variant("init-30hz")
    assert "periodic-30hz" in names
    assert "periodic-45hz" not in names


def test_run_replication_short(repl_env: MehreganEnv, tmp_path: Path) -> None:
    config = ReplicationConfig(
        variant="paper",
        train_seed=1,
        eval_seed=2,
        ddpg=DDPGConfig(
            variant="paper",
            num_episodes=2,
            batch_size=8,
            min_buffer_size=8,
            seed=1,
        ),
        checkpoint_dir=tmp_path,
    )
    result = run_replication(repl_env, config)
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.is_file()
    assert result.eval_metrics["protocol"] == "mehregan_eval"
    assert set(result.baseline_metrics) == {"none", "cdbs-130hz", "periodic-45hz"}
    assert all(m["p_beta_mean"] > 0 for m in result.baseline_metrics.values())


def test_init_30hz_replication(repl_env: MehreganEnv, tmp_path: Path) -> None:
    config = ReplicationConfig(
        variant="init-30hz",
        ddpg=DDPGConfig(variant="init-30hz", num_episodes=1, batch_size=4, min_buffer_size=4),
        checkpoint_dir=tmp_path,
    )
    result = run_replication(repl_env, config)
    assert result.variant == "init-30hz"
    assert "periodic-30hz" in result.baseline_metrics


def test_write_replication_summary(repl_env: MehreganEnv, tmp_path: Path) -> None:
    config = ReplicationConfig(
        ddpg=DDPGConfig(num_episodes=1, batch_size=4, min_buffer_size=4),
        checkpoint_dir=tmp_path,
    )
    result = run_replication(repl_env, config)
    out = write_replication_summary(result, tmp_path / "summary.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["variant"] == "paper"
    assert "ddpg_eval" in payload
    assert "baselines" in payload


@pytest.mark.slow
def test_paper_episode_horizon_on_mock(repl_env: MehreganEnv, tmp_path: Path) -> None:
    """Full 30-step episode horizon with paper episode count (mock plant)."""
    config = ReplicationConfig(
        ddpg=DDPGConfig(
            num_episodes=10,
            batch_size=32,
            min_buffer_size=32,
            seed=0,
        ),
        checkpoint_dir=tmp_path,
    )
    result = run_replication(repl_env, config)
    assert len(result.train.metrics.episode_rewards) == 10
    assert all(steps == 30 for steps in result.train.metrics.episode_steps)
