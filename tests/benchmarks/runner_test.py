"""Benchmark runner integration tests (mock plant)."""

from __future__ import annotations

from pathlib import Path

from dataclasses import replace

import pytest

from benchmarks import load_run_metrics, load_suite_manifest, list_suite_runs, run_suite
from benchmarks.runner import BenchmarkOptions, execute_planned_run
from benchmarks.suite import expand_planned_runs, load_suite
from controllers.ddpg import DDPGConfig, EvalConfig, evaluate, save_checkpoint, train
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def mock_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=5, state_length=1),
    )
    yield env
    env.close()


def test_run_smoke_suite(mock_env: MehreganEnv, tmp_path: Path) -> None:
    result = run_suite(
        "mehregan_eval_smoke",
        mock_env,
        options=BenchmarkOptions(
            results_dir=tmp_path / "results",
            repo_root=REPO_ROOT,
        ),
    )
    assert len(result.records) == 4
    assert result.manifest_path.is_file()
    manifest = load_suite_manifest(result.suite_dir)
    assert manifest["completed_runs"] == 4
    assert manifest["protocol"] == "mehregan"
    runs = list_suite_runs(result.suite_dir)
    assert len(runs) == 4
    metrics = load_run_metrics(runs[0])
    assert "p_beta_mean" in metrics
    assert metrics["controller"] == "baseline"


def test_dry_run_writes_manifest_only(tmp_path: Path) -> None:
    result = run_suite(
        "mehregan_eval_smoke",
        options=BenchmarkOptions(
            results_dir=tmp_path / "results",
            dry_run=True,
            repo_root=REPO_ROOT,
        ),
    )
    assert result.records == []
    assert result.manifest_path.is_file()
    assert not (result.suite_dir / "runs").exists()


def test_ddpg_entry_with_checkpoint(mock_env: MehreganEnv, tmp_path: Path) -> None:
    config = DDPGConfig(
        num_episodes=1,
        batch_size=4,
        min_buffer_size=4,
        buffer_capacity=32,
        seed=0,
    )
    train_result = train(mock_env, config)
    ckpt_dir = tmp_path / "artifacts" / "ddpg"
    ckpt = save_checkpoint(
        ckpt_dir / "paper_train0.pt",
        actor=train_result.actor,
        config=config,
        state_length=1,
        n_actions=mock_env.action_space.n,
    )

    suite = load_suite("mehregan_eval_smoke", repo_root=REPO_ROOT)
    planned = expand_planned_runs(
        suite,
        seeds=(0,),
        controller_filter={("ddpg", "paper")},
        repo_root=REPO_ROOT,
    )
    assert len(planned) == 0  # smoke suite has no ddpg entry

    planned_run = expand_planned_runs(
        load_suite("mehregan_eval", repo_root=REPO_ROOT),
        seeds=(0,),
        controller_filter={("ddpg", "paper")},
        repo_root=REPO_ROOT,
    )[0]
    planned_run = replace(planned_run, checkpoint=ckpt)

    record = execute_planned_run(mock_env, planned_run, suite)
    assert record.metrics["controller"] == "ddpg"
    assert record.metrics["episode_length"] == suite.eval_steps

    eval_payload = evaluate(
        mock_env,
        ckpt,
        config=EvalConfig(seed=0, eval_steps=suite.eval_steps),
        protocol="mehregan_eval",
    )
    assert eval_payload["protocol"] == "mehregan_eval"
