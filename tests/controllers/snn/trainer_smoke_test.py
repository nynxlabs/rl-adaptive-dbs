"""DSQN trainer smoke on the Python plant (short rollouts)."""

from __future__ import annotations

from pathlib import Path

import pytest

from controllers.snn.config import SNNConfig
from controllers.snn.trainer import train_dsqn, write_train_metrics


@pytest.mark.slow
def test_train_dsqn_python_plant_smoke(tmp_path: Path) -> None:
    cfg = SNNConfig(seed=0).for_smoke(episodes=2, max_steps=8)
    ckpt = tmp_path / "paper_train0.pt"
    result = train_dsqn(config=cfg, checkpoint_path=ckpt)
    assert len(result.episode_rewards) == 2
    assert len(result.metrics) == 2
    assert all(m.episode_length > 0 for m in result.metrics)
    assert ckpt.is_file()
    metrics_path = ckpt.with_suffix(".metrics.json")
    assert metrics_path.is_file()
    write_train_metrics(result, metrics_path)


@pytest.mark.slow
def test_eval_returns_trajectories(tmp_path: Path) -> None:
    from controllers.snn.eval import evaluate

    cfg = SNNConfig(seed=1).for_smoke(episodes=2, max_steps=5)
    ckpt = tmp_path / "paper_train1.pt"
    train_dsqn(config=cfg, checkpoint_path=ckpt)
    payload = evaluate(ckpt, config=cfg, episodes=2, max_steps=5)
    assert len(payload["alpha_beta_trajectories"]) == 2
    assert len(payload["dbs_trajectories"]) == 2
    assert all(len(traj) >= 2 for traj in payload["alpha_beta_trajectories"])
    assert all("amplitude" in snap for traj in payload["dbs_trajectories"] for snap in traj)
