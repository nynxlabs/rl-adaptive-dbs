"""Nguyen DSQN evaluation: real plant values only, full horizon, checkpoint config."""

from __future__ import annotations

from pathlib import Path

from controllers.snn.config import SNNConfig
from controllers.snn.eval import evaluate
from controllers.snn.networks import DSQN
from controllers.snn.trainer import save_checkpoint
from tests.controllers.snn.shape_contracts_test import _SpikeMockPlant


class _AlphaRecordingPlant(_SpikeMockPlant):
    def __init__(self) -> None:
        super().__init__()
        self.alphas: list[float] = []

    def integrate(self, duration_s, dbs_spec=None, **kwargs):  # type: ignore[no-untyped-def]
        from controllers.snn.reward import alpha_beta_power

        result = super().integrate(duration_s, dbs_spec, **kwargs)
        self.alphas.append(alpha_beta_power(result.gpi_spikes, duration_s=duration_s, dt_ms=result.dt_ms))
        return result


def test_eval_uses_only_plant_values_and_runs_full_horizon(tmp_path: Path) -> None:
    cfg = SNNConfig(sequence_steps=4, neurons_per_region=10, hidden_size=16, max_episode_steps=6, seed=3)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, dsqn=DSQN(cfg), config=cfg)
    plant = _AlphaRecordingPlant()
    out = evaluate(ckpt, plant=plant, episodes=2)
    # Config comes from the checkpoint: 6-step horizon, every episode runs it in full.
    assert out["episode_lengths"] == [6, 6]
    flat = [a for traj in out["alpha_beta_trajectories"] for a in traj]
    assert len(flat) == 2 * 7  # reset window + 6 steps per episode
    assert flat == plant.alphas  # nothing injected: every point is a plant reading
