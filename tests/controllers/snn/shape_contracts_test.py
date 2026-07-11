"""SNN controller shape-contract tests (scaffold — no training)."""

from __future__ import annotations

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


def test_trainer_scaffold_raises_on_train_step() -> None:
    cfg = SNNConfig(sequence_steps=2, neurons_per_region=2)
    trainer = DSQNTrainer(DSQN(cfg), ReplayBuffer(cfg), cfg)
    trainer.note_step()
    assert trainer.current_epsilon() <= cfg.epsilon_start
    with pytest.raises(NotImplementedError):
        trainer.train_step()
