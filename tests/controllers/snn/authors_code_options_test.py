"""Off-by-default options that follow the Nguyen group's published environment code."""

from __future__ import annotations

import pytest
from typing import Any

import numpy as np

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer
from controllers.snn.config import SNNConfig, parse_config_overrides
from controllers.snn.networks import DSQN
from controllers.snn.trainer import DSQNTrainer
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult
from envs.plant._kumaravelu import require
from envs.plant import PythonPlant
from tests.controllers.snn.shape_contracts_test import _SpikeMockPlant


class _KwargsPlant(_SpikeMockPlant):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[dict[str, Any], np.ndarray | None]] = []
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> Any:
        self.reset_seeds.append(seed)
        return super().reset(seed=seed)

    def integrate(self, duration_s: float, dbs_spec: DbsSpec | None = None, **kwargs: Any) -> IntegrateResult:
        self.calls.append((dict(kwargs), None if dbs_spec is None else dbs_spec.idbs))
        return super().integrate(duration_s, dbs_spec, **kwargs)


def _onsets(idbs: np.ndarray) -> np.ndarray:
    on = idbs > 0
    return np.flatnonzero(on & ~np.concatenate(([False], on[:-1])))


def _carry_cfg(**kw: Any) -> SNNConfig:
    return SNNConfig(sequence_steps=4, neurons_per_region=10, max_episode_steps=3, plant_carry=True, **kw)


@pytest.mark.plant
def test_reset_window_runs_at_its_own_frequency_then_initial_setting() -> None:
    plant = _KwargsPlant()
    env = NguyenEnvAdapter(plant=plant, config=_carry_cfg(reset_window_frequency_hz=100.0))
    try:
        _, info = env.reset(seed=0)
        assert info["dbs"].frequency_hz == 40.0
        env.step(np.array([1, 1, 1]))  # "maintain" on every head
    finally:
        env.close()
    (_, window), (_, step1) = plant.calls
    assert np.all(np.diff(_onsets(window)) == 1000)  # 100 Hz on the 0.01 ms grid
    assert np.all(np.diff(_onsets(step1)) == 2500)  # back to 40 Hz for the first action


@pytest.mark.plant
def test_reset_window_off_keeps_initial_frequency() -> None:
    plant = _KwargsPlant()
    env = NguyenEnvAdapter(plant=plant, config=_carry_cfg())
    try:
        env.reset(seed=0)
    finally:
        env.close()
    assert np.all(np.diff(_onsets(plant.calls[0][1])) == 2500)


@pytest.mark.plant
def test_pulse_restart_opens_every_segment_with_a_pulse() -> None:
    plant = _KwargsPlant()
    env = NguyenEnvAdapter(plant=plant, config=_carry_cfg(plant_pulse_restart=True))
    try:
        env.reset(seed=0)
        env.step(np.array([1, 1, 1]))
    finally:
        env.close()
    (_, first), (_, second) = plant.calls
    np.testing.assert_array_equal(second, first)
    assert _onsets(second)[0] == 0


@pytest.mark.plant
def test_restart_synapses_flag_reaches_the_plant_only_when_on() -> None:
    for on in (False, True):
        plant = _KwargsPlant()
        env = NguyenEnvAdapter(plant=plant, config=_carry_cfg(plant_restart_synapses=on))
        try:
            env.reset(seed=0)
        finally:
            env.close()
        assert ("restart_synapses" in plant.calls[0][0]) is on


@pytest.mark.plant
def test_carried_state_zeroes_only_synapses() -> None:
    plant = PythonPlant()
    plant._dyn = {"vsn": np.full(3, -60.0), "S2a": np.full(3, 0.4), "spike_n": np.ones(3, dtype=np.int32)}
    restarted = plant._carried_state(True)
    assert restarted is not None
    np.testing.assert_array_equal(restarted["vsn"], -60.0)
    assert not restarted["S2a"].any() and not restarted["spike_n"].any()
    assert plant._carried_state(False) is plant._dyn
    SYNAPTIC_DYN_KEYS = require("network.integrator").SYNAPTIC_DYN_KEYS
    assert {"S2a", "Z1a", "spike_idx", "spike_n"} <= SYNAPTIC_DYN_KEYS
    assert not {"vsn", "CAsn2", "ca3_state", "H1"} & SYNAPTIC_DYN_KEYS


def _train(cfg: SNNConfig) -> tuple[_KwargsPlant, DSQNTrainer]:
    plant = _KwargsPlant()
    env = NguyenEnvAdapter(plant=plant, config=cfg)
    trainer = DSQNTrainer(DSQN(cfg), ReplayBuffer(cfg, seed=cfg.seed), cfg)
    try:
        trainer.train_episodes(env)
    finally:
        env.close()
    return plant, trainer


def _tiny(**kw: Any) -> SNNConfig:
    return SNNConfig(
        sequence_steps=2,
        neurons_per_region=10,
        num_episodes=3,
        max_episode_steps=2,
        batch_size=64,
        replay_capacity=64,
        terminate_on_subthreshold=False,
        seed=7,
        **kw,
    )


@pytest.mark.plant
def test_fixed_episode_seed_reuses_the_run_seed() -> None:
    plant, _ = _train(_tiny())
    assert plant.reset_seeds == [7, 8, 9]
    plant, _ = _train(_tiny(train_fixed_episode_seed=True))
    assert plant.reset_seeds == [7, 7, 7]


@pytest.mark.plant
def test_truncation_bootstraps_when_not_terminal() -> None:
    _, trainer = _train(_tiny())
    assert trainer.buffer._done[: len(trainer.buffer)].tolist() == [False, True] * 3
    _, trainer = _train(_tiny(replay_truncation_terminal=False))
    assert not trainer.buffer._done[: len(trainer.buffer)].any()


def test_set_parser_types_new_fields() -> None:
    assert parse_config_overrides(
        ["reset_window_frequency_hz=100", "train_fixed_episode_seed=true", "replay_truncation_terminal=0"]
    ) == {"reset_window_frequency_hz": 100.0, "train_fixed_episode_seed": True, "replay_truncation_terminal": False}
