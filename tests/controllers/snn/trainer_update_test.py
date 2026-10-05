"""Replay-flush update rule: gradient steps per flush and the no-target-network mode."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
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


def test_soft_target_tracks_online_weights_by_polyak_average() -> None:
    trainer = _trainer(
        target_soft_update_tau=0.5,
        replay_update_steps=1,
        optimizer="adamw",
        grad_clip_mode="value",
        grad_clip=100.0,
    )
    assert trainer.uses_target_network
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    before = {k: v.clone() for k, v in trainer.target_dsqn.state_dict().items()}
    _fill(trainer, trainer.config.replay_update_cadence)
    assert trainer.maybe_update()
    online = trainer.dsqn.state_dict()
    for key, value in trainer.target_dsqn.state_dict().items():
        assert torch.allclose(value, 0.5 * before[key] + 0.5 * online[key])
    assert training_budget(trainer.config)["target_network"] is True


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


def test_exponential_epsilon_follows_dqn_tutorial_formula() -> None:
    cfg = replace(
        SNNConfig().for_smoke(),
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay_steps=2000,
        epsilon_schedule="exp",
    )
    trainer = DSQNTrainer(DSQN(cfg), ReplayBuffer(cfg, seed=0), cfg)
    assert trainer.current_epsilon() == pytest.approx(0.9)
    for _ in range(2000):
        trainer.note_step()
    assert trainer.current_epsilon() == pytest.approx(0.05 + 0.85 * math.exp(-1.0))


def test_replay_warmup_holds_updates_until_buffer_fills() -> None:
    cfg = replace(SNNConfig().for_smoke(), batch_size=8, replay_update_cadence=8, replay_warmup_transitions=32)
    trainer = DSQNTrainer(DSQN(cfg), ReplayBuffer(cfg, seed=0), cfg)
    obs = np.zeros(cfg.flat_observation_dim, dtype=np.float32)
    updates = []
    for _ in range(40):
        trainer.buffer.add(Transition(obs, 0, -1.0, obs, False))
        updates.append(trainer.maybe_update())
    assert not any(updates[:31])
    assert any(updates[31:])
