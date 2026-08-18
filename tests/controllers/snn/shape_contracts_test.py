"""SNN controller shape-contract tests (scaffold — no training)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from controllers.snn.actions import decode_factored_action, select_action
from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer, Transition
from controllers.snn.config import SNNConfig
from controllers.snn.dbs_params import DBSParameterState
from controllers.snn.encoder import SpikeObservationEncoder
from controllers.snn.energy import dbs_energy_index
from controllers.snn.networks import DSQN
from controllers.snn.reward import nguyen_reward
from controllers.snn.trainer import DSQNTrainer
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult
from tests.envs.mock_plant import MockPlant


def _synthetic_spikes(n_neurons: int = 10) -> list[np.ndarray]:
    return [np.array([0.01 + 0.002 * idx], dtype=float) for idx in range(n_neurons)]


class _SpikeMockPlant(MockPlant):
    """Mock plant that returns deterministic GPi spike trains for encoder tests."""

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
    ) -> IntegrateResult:
        result = super().integrate(duration_s, dbs_spec, record_spikes=record_spikes)
        return IntegrateResult(
            gpi_spikes=_synthetic_spikes(),
            duration_s=result.duration_s,
            dt_ms=result.dt_ms,
            pd=result.pd,
            dbs_spec=result.dbs_spec,
            seed=result.seed,
            p_beta=result.p_beta,
            info=result.info,
        )


def test_encoder_output_shape() -> None:
    cfg = SNNConfig(sequence_steps=8, neurons_per_region=10, n_regions=1)
    encoder = SpikeObservationEncoder(cfg)
    obs = encoder.encode(_synthetic_spikes(10), duration_s=cfg.step_duration_s)
    assert obs.shape == (8, 10)
    assert obs.dtype == np.float32
    assert np.all((obs == 0.0) | (obs == 1.0))
    flat = encoder.flatten(obs)
    assert flat.shape == (80,)


def test_dbs_parameter_state_apply_delta_clamps() -> None:
    cfg = SNNConfig(
        amplitude_sensitivity=50.0,
        frequency_sensitivity=20.0,
        pulse_width_sensitivity=0.2,
    )
    state = DBSParameterState()
    state.apply_delta([1, 1, 1], cfg)
    assert state.amplitude > DBSParameterState().amplitude
    assert state.frequency_hz > DBSParameterState().frequency_hz
    assert state.pulse_width_ms > DBSParameterState().pulse_width_ms
    spec = state.to_dbs_spec(duration_s=0.1)
    assert spec.idbs is not None
    assert spec.idbs.shape[0] == int(round(100.0 / 0.01)) + 1


def test_frequency_sensitivity_schedules_with_epsilon() -> None:
    cfg = SNNConfig(
        frequency_sensitivity=10.0,
        frequency_sensitivity_explore=20.0,
        epsilon_start=1.0,
        epsilon_end=0.2,
        frequency_sensitivity_explore_epsilon_max=0.7,
    )
    assert cfg.frequency_sensitivity_at_epsilon(1.0) == 10.0
    assert cfg.frequency_sensitivity_at_epsilon(0.75) == 10.0
    assert cfg.frequency_sensitivity_at_epsilon(0.2) == 10.0
    mid = cfg.frequency_sensitivity_at_epsilon(0.5)
    assert 10.0 < mid < 20.0

    state_hi = DBSParameterState()
    state_mid = DBSParameterState()
    state_lo = DBSParameterState()
    state_hi.apply_delta([0, 1, 0], cfg, epsilon=1.0)
    state_mid.apply_delta([0, 1, 0], cfg, epsilon=0.5)
    state_lo.apply_delta([0, 1, 0], cfg, epsilon=0.2)
    assert state_hi.frequency_hz == 40.0 + 10.0
    assert state_mid.frequency_hz == 40.0 + mid
    assert state_lo.frequency_hz == 40.0 + 10.0


def test_frequency_sensitivity_episode_curriculum() -> None:
    cfg = SNNConfig(
        frequency_sensitivity=20.0,
        frequency_sensitivity_early=10.0,
        frequency_sensitivity_early_episodes=35,
    )
    assert cfg.frequency_sensitivity_at_epsilon(1.0, episode=0) == 10.0
    assert cfg.frequency_sensitivity_at_epsilon(1.0, episode=34) == 10.0
    assert cfg.frequency_sensitivity_at_epsilon(1.0, episode=35) == 20.0

    early = DBSParameterState()
    late = DBSParameterState()
    early.apply_delta([0, 1, 0], cfg, epsilon=1.0, episode=10)
    late.apply_delta([0, 1, 0], cfg, epsilon=0.05, episode=50)
    assert early.frequency_hz == 50.0
    assert late.frequency_hz == 60.0


def test_fig4_init_floors_block_frequency_collapse() -> None:
    from controllers.snn.config import fig4_nguyen_config

    cfg = fig4_nguyen_config()
    state = DBSParameterState()
    for _ in range(8):
        state.apply_delta([0, -1, 0], cfg, epsilon=0.05, episode=60)
    assert state.frequency_hz == cfg.frequency_min == 40.0
    assert state.amplitude >= cfg.amplitude_min


def test_fig4_v76_ship_config() -> None:
    from controllers.snn.config import fig4_nguyen_config

    cfg = fig4_nguyen_config()
    assert cfg.epsilon_end == 0.05
    assert cfg.frequency_sensitivity_early == 5.0
    assert cfg.target_update_period == 100


def test_dsqn_forward_shapes() -> None:
    cfg = SNNConfig(sequence_steps=5, neurons_per_region=4, n_regions=1)
    model = DSQN(cfg)
    batch = 3
    x = torch.zeros(batch, cfg.flat_observation_dim)
    out = model(x)
    assert out.spike_counts.shape == (batch, cfg.n_action_outputs)
    assert out.membrane.shape == (batch, cfg.n_action_outputs)


def test_select_action_factored_shape() -> None:
    cfg = SNNConfig(action_scheme="factored")
    counts = np.arange(9, dtype=float)
    action_index, ternary = select_action(counts, config=cfg, epsilon=0.0)
    assert 0 <= action_index < 27
    assert ternary.shape == (3,)
    assert set(ternary.tolist()).issubset({-1, 0, 1})


def test_decode_factored_action_mapping() -> None:
    assert decode_factored_action([0, 1, 2]).tolist() == [-1, 0, 1]


def test_replay_buffer_shape_contracts() -> None:
    cfg = SNNConfig(sequence_steps=2, neurons_per_region=3, n_regions=1)
    buf = ReplayBuffer(cfg, seed=0)
    dim = cfg.flat_observation_dim
    for idx in range(cfg.replay_update_cadence + 2):
        state = np.full(dim, float(idx), dtype=np.float32)
        buf.add(
            Transition(
                state=state,
                action=idx % 9,
                reward=float(idx),
                next_state=state + 1.0,
                done=False,
            )
        )
    assert len(buf) == cfg.replay_capacity or len(buf) == cfg.replay_update_cadence + 2
    assert buf.ready_for_update()
    batch = buf.sample(min(4, len(buf)))
    assert batch.state.shape[1] == dim
    assert batch.action.ndim == 1


def test_energy_index_positive() -> None:
    energy = dbs_energy_index(
        frequency_hz=40.0,
        amplitude=300.0,
        pulse_width_ms=0.3,
        step_duration_s=0.1,
        stimulated_neurons=10,
    )
    assert energy > 0.0


def test_nguyen_reward_terminated_bonus() -> None:
    reward = nguyen_reward(
        alpha_beta=100.0,
        energy=1.0,
        terminated=True,
        remaining_steps=5,
    )
    assert isinstance(reward, float)


def test_nguyen_reward_progress_cap() -> None:
    cfg = SNNConfig(
        alpha_beta_threshold=150.0,
        alpha_beta_progress_coef=2000.0,
        alpha_beta_progress_cap_per_step=10_000.0,
    )
    capped = nguyen_reward(
        alpha_beta=200.0,
        energy=0.0,
        terminated=False,
        remaining_steps=10,
        prev_alpha_beta=220.0,
        config=cfg,
    )
    uncapped = nguyen_reward(
        alpha_beta=200.0,
        energy=0.0,
        terminated=False,
        remaining_steps=10,
        prev_alpha_beta=220.0,
        config=replace(cfg, alpha_beta_progress_cap_per_step=0.0),
    )
    assert capped < uncapped


def test_adapter_spaces_and_step_shapes() -> None:
    cfg = SNNConfig(sequence_steps=4, neurons_per_region=10, max_episode_steps=3)
    env = NguyenEnvAdapter(plant=_SpikeMockPlant(), config=cfg)
    try:
        assert env.observation_space.shape == (4, 10)
        assert env.action_space.nvec.tolist() == [3, 3, 3]
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 10)
        assert "alpha_beta" in info
        assert info["adapter"] is True
        obs2, reward, terminated, truncated, step_info = env.step(np.array([1, 1, 1]))
        assert obs2.shape == obs.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["ternary_action"].shape == (3,)
    finally:
        env.close()


def test_trainer_train_step_and_mock_episodes() -> None:
    cfg = SNNConfig(
        sequence_steps=2,
        neurons_per_region=2,
        n_regions=1,
        num_episodes=2,
        max_episode_steps=4,
        batch_size=8,
        replay_update_cadence=8,
        replay_capacity=64,
        target_update_period=2,
        log_episodes=False,
        seed=0,
    )
    env = NguyenEnvAdapter(plant=_SpikeMockPlant(), config=cfg)
    try:
        from controllers.snn.trainer import train_dsqn

        result = train_dsqn(env, cfg)
        assert len(result.episode_rewards) == 2
        assert result.update_count >= 0
        # Force a train_step once the buffer is warm.
        trainer = DSQNTrainer(result.dsqn, ReplayBuffer(cfg, seed=1), cfg)
        dim = cfg.flat_observation_dim
        for idx in range(cfg.batch_size + 1):
            state = np.zeros(dim, dtype=np.float32)
            trainer.buffer.add(
                Transition(
                    state=state,
                    action=idx % 27,
                    reward=0.1,
                    next_state=state,
                    done=False,
                )
            )
        loss = trainer.train_step()
        assert isinstance(loss, float)
    finally:
        env.close()
