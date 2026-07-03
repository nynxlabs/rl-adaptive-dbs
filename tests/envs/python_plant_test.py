"""Python plant backend — Phase A scaffold and DBS waveform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.plant import PythonPlant, create_dbs_current
from envs.plant.dbs import DBS_AMPLITUDE_NA_PER_CM2, DBS_PULSE_WIDTH_MS, DbsSpec
from envs.plant.matlab_backend import IntegrateResult
from rl_adaptive_dbs.env_factory import build_mehregan_env


def test_create_dbs_current_zero_frequency() -> None:
    trace = create_dbs_current(0.0, tmax_ms=10.0, dt_ms=0.01)
    assert trace.shape == (1001,)
    assert np.all(trace == 0.0)


def test_create_dbs_current_pulse_shape() -> None:
    dt_ms = 0.01
    pulse_len = int(round(DBS_PULSE_WIDTH_MS / dt_ms))
    trace = create_dbs_current(130.0, tmax_ms=50.0, dt_ms=dt_ms)
    assert trace[0] == pytest.approx(DBS_AMPLITUDE_NA_PER_CM2)
    assert np.all(trace[1:pulse_len] == DBS_AMPLITUDE_NA_PER_CM2)
    assert trace[pulse_len] == pytest.approx(0.0)


def test_create_dbs_current_130hz_isi_spacing() -> None:
    dt_ms = 0.01
    pulse_len = int(round(DBS_PULSE_WIDTH_MS / dt_ms))
    isi_steps = int(round((1000.0 / 130.0) / dt_ms))
    trace = create_dbs_current(130.0, tmax_ms=200.0, dt_ms=dt_ms)
    assert trace[isi_steps] == pytest.approx(DBS_AMPLITUDE_NA_PER_CM2)
    assert np.all(trace[isi_steps + 1 : isi_steps + pulse_len] == DBS_AMPLITUDE_NA_PER_CM2)


def test_create_dbs_current_matches_matlab_fixture() -> None:
    """Bit-exact on the integration grid (MATLAB tail past ``tmax`` is unused)."""
    fixture = np.load(
        Path(__file__).resolve().parents[1] / "fixtures" / "creatdbs_130hz.npz"
    )
    trace = create_dbs_current(
        float(fixture["pattern_hz"]),
        tmax_ms=float(fixture["tmax_ms"]),
        dt_ms=float(fixture["dt_ms"]),
    )
    np.testing.assert_array_equal(trace, fixture["idbs_matlab"])


def test_create_dbs_current_45hz_pulse_grid() -> None:
    dt_ms = 0.01
    tmax_ms = 100.0
    trace = create_dbs_current(45.0, tmax_ms=tmax_ms, dt_ms=dt_ms)
    pulse_len = int(round(DBS_PULSE_WIDTH_MS / dt_ms))
    isi_steps = int(round((1000.0 / 45.0) / dt_ms))
    n_steps = int(round(tmax_ms / dt_ms)) + 1
    assert trace.shape == (n_steps,)
    pulse_starts = list(range(0, n_steps, isi_steps))
    for start in pulse_starts:
        end = min(start + pulse_len, n_steps)
        assert np.all(trace[start:end] == DBS_AMPLITUDE_NA_PER_CM2)
        if end < n_steps:
            assert trace[end] == pytest.approx(0.0)


def test_python_plant_reset_and_close() -> None:
    plant = PythonPlant()
    assert plant.reset(seed=42) is plant
    plant.close()


def test_python_plant_integrate_short_segment() -> None:
    plant = PythonPlant().reset(seed=1)
    result = plant.integrate(0.01, DbsSpec.none())
    assert isinstance(result, IntegrateResult)
    assert len(result.gpi_spikes) == 10
    assert all(isinstance(times, np.ndarray) for times in result.gpi_spikes)
    assert isinstance(result.p_beta, float)


def test_python_plant_integrate_reproducible_with_seed() -> None:
    first = PythonPlant().reset(seed=99).integrate(0.01, DbsSpec.none())
    second = PythonPlant().reset(seed=99).integrate(0.01, DbsSpec.none())
    assert first.p_beta == pytest.approx(second.p_beta)
    for a, b in zip(first.gpi_spikes, second.gpi_spikes):
        np.testing.assert_array_equal(a, b)


def test_build_mehregan_env_python_backend() -> None:
    env = build_mehregan_env(plant_backend="python")
    try:
        assert isinstance(env._plant, PythonPlant)
    finally:
        env.close()
