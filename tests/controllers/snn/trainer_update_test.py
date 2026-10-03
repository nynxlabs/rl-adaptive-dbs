"""Replay-flush update rule: gradient steps per flush and the no-target-network mode."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from controllers.snn.buffer import ReplayBuffer, Transition
from controllers.snn.config import SNNConfig
from controllers.snn.networks import DSQN
from controllers.snn.trainer import DSQNTrainer, saved_config, training_budget


def _trainer(**overrides: object) -> DSQNTrainer:
    cfg = replace(SNNConfig(seed=0).for_smoke(episodes=2, max_steps=8), **overrides)
    return DSQNTrainer(DSQN(cfg), ReplayBuffer(cfg, seed=0), cfg)


def _fill(trainer: DSQNTrainer, n: int) -> None:
    rng = np.random.default_rng(0)
    dim = trainer.config.flat_observation_dim
    for _ in range(n):
        trainer.buffer.add(
            Transition(
                state=rng.integers(0, 2, dim).astype(np.float32),
                action=int(rng.integers(0, 27)),
                reward=float(rng.normal()),
                next_state=rng.integers(0, 2, dim).astype(np.float32),
                done=False,
            )
        )


def test_flush_runs_replay_update_steps_gradient_steps() -> None:
    trainer = _trainer(replay_update_steps=3)
    _fill(trainer, trainer.config.replay_update_cadence)
    assert trainer.maybe_update()
    assert trainer.update_count == 3
    assert not trainer.maybe_update()


def test_no_target_network_bootstraps_from_online_weights() -> None:
    trainer = _trainer(target_update_period=0, replay_update_steps=2)
    frozen = {k: v.clone() for k, v in trainer.target_dsqn.state_dict().items()}
    _fill(trainer, trainer.config.replay_update_cadence)
    assert trainer.maybe_update()
    # The unused target copy never changes; the online net does.
    for key, value in trainer.target_dsqn.state_dict().items():
        assert torch.equal(value, frozen[key])
    assert any(
        not torch.equal(value, frozen[key]) for key, value in trainer.dsqn.state_dict().items()
    )


def test_target_network_syncs_inside_a_flush() -> None:
    trainer = _trainer(target_update_period=2, replay_update_steps=2)
    _fill(trainer, trainer.config.replay_update_cadence)
    assert trainer.maybe_update()
    for key, value in trainer.target_dsqn.state_dict().items():
        assert torch.equal(value, trainer.dsqn.state_dict()[key])


def test_training_budget_counts_steps_per_flush_and_skips_sync_warning() -> None:
    cfg = SNNConfig(seed=0, num_episodes=10, max_episode_steps=25)
    budget = training_budget(cfg)
    flushes = (10 * 25 - cfg.batch_size) // cfg.replay_update_cadence + 1
    assert budget["sgd_updates_range"][1] == flushes * cfg.replay_update_steps
    assert budget["target_network"] is False
    assert budget["warnings"] == []


def test_saved_config_fills_fields_older_checkpoints_predate() -> None:
    record = {k: v for k, v in vars(SNNConfig(seed=3)).items() if k != "replay_update_steps"}
    assert saved_config(record).replay_update_steps == 1
    assert saved_config(record).seed == 3
    assert saved_config(SNNConfig(replay_update_steps=4)).replay_update_steps == 4
