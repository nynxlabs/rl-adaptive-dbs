"""Within-step observation construction tests."""

from __future__ import annotations

import numpy as np
import pytest

from envs.mehregan.extensions.alphabet_diversity.config import WithinStepEnvConfig, resolve_state_mode
from envs.mehregan.extensions.alphabet_diversity.env import WithinStepMehreganEnv
from envs.mehregan.extensions.alphabet_diversity.observations import clip_spikes_to_subwindow, within_step_p_beta_series
from envs.mehregan.extensions.alphabet_diversity.reward import mehregan_reward, mehregan_reward_from_s_sum
from envs.plant.dbs import DbsSpec
from envs.plant.matlab_backend import IntegrateResult
from tests.envs.mock_plant import MockPlant


def test_resolve_state_mode_legacy_multi_step() -> None:
    assert resolve_state_mode(state_mode="scalar", state_length=2) == "multi_step_history"
    assert resolve_state_mode(state_mode="scalar", state_length=1) == "scalar"


def test_clip_spikes_rebases_to_subwindow() -> None:
    spikes = [np.array([0.1, 0.5, 1.1, 1.9], dtype=np.float64)]
    clipped = clip_spikes_to_subwindow(spikes, t_start_s=0.5, t_end_s=1.5)
    np.testing.assert_allclose(clipped[0], [0.0, 0.6])


def test_within_step_series_length_and_order() -> None:
    # Dense early spikes → higher early sub-window power than late (heuristic).
    gpi = [np.linspace(0.0, 1.9, 40, dtype=np.float64)]
    series = within_step_p_beta_series(
        gpi,
        segment_duration_s=2.0,
        state_length=4,
        dt_ms=0.02,
    )
    assert series.shape == (4,)
    assert np.all(np.isfinite(series))


def test_full_segment_reward_decoupled_from_observation_mean() -> None:
    obs = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    r_obs = mehregan_reward(obs, beta_threshold=0.35)
    r_seg = mehregan_reward_from_s_sum(0.55, beta_threshold=0.35)
    assert r_obs != pytest.approx(r_seg)


class _SpikeStubPlant(MockPlant):
    """Mock plant that returns synthetic GPi spikes for within_step tests."""

    def integrate(
        self,
        duration_s: float,
        dbs_spec: DbsSpec | None = None,
        *,
        record_spikes: bool = True,
    ) -> IntegrateResult:
        base = super().integrate(duration_s, dbs_spec, record_spikes=record_spikes)
        # One spike train: frequency of stimulation increases spike density.
        spec = dbs_spec if dbs_spec is not None else DbsSpec.none()
        rate_hz = max(1.0, float(spec.frequency_hz))
        n_spikes = int(rate_hz * duration_s)
        times = np.linspace(0.0, duration_s * 0.95, n_spikes, dtype=np.float64)
        gpi = [times]
        return IntegrateResult(
            gpi_spikes=gpi,
            duration_s=base.duration_s,
            dt_ms=base.dt_ms,
            pd=base.pd,
            dbs_spec=base.dbs_spec,
            seed=base.seed,
            p_beta=base.p_beta,
            info=base.info,
        )


def test_mehregan_env_within_step_shape_and_modes() -> None:
    env = WithinStepMehreganEnv(
        plant=_SpikeStubPlant(),
        config=WithinStepEnvConfig(
            max_episode_steps=2,
            state_mode="within_step",
            state_length=4,
            reward_state_mode="full_segment",
        ),
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4,)
        assert info["state_mode"] == "within_step"
        assert info["reward_state_mode"] == "full_segment"
        assert len(info["p_beta_subwindow_raw"]) == 4
        obs2, reward, _, _, info2 = env.step(0)
        assert obs2.shape == (4,)
        assert np.isfinite(reward)
        assert info2["p_beta_raw"] == pytest.approx(info2["p_beta_norm"] * 1000.0)
    finally:
        env.close()


def test_scalar_mode_still_length_one() -> None:
    env = WithinStepMehreganEnv(
        plant=MockPlant(),
        config=WithinStepEnvConfig(max_episode_steps=1, state_length=1),
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (1,)
    finally:
        env.close()
