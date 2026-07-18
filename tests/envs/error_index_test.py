"""Unit tests for Mehregan / Gao Error Index (EI) biomarker."""

from __future__ import annotations

import numpy as np

from envs.plant.biomarkers import (
    error_index,
    resolve_smc_pulse_times,
    smc_pulse_times_from_cor_spikes,
    smc_pulse_times_from_iappco,
    thalamic_misfire_count,
)


def test_smc_pulse_times_from_iappco_detects_rising_edges() -> None:
    iappco = np.array([0.0, 350.0, 350.0, 0.0, 0.0, 350.0], dtype=np.float64)
    times = smc_pulse_times_from_iappco(iappco, dt_ms=1.0)
    np.testing.assert_allclose(times, [0.001, 0.005])


def test_error_index_perfect_responses() -> None:
    smc = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    th = [np.array([0.01, 1.01, 2.01]) for _ in range(2)]
    ei = error_index(th, smc, t_start=0.0, t_end=3.0, n_neurons=2)
    assert ei == 0.0


def test_error_index_counts_miss_and_double() -> None:
    smc = np.array([0.0], dtype=np.float64)
    th = [
        np.array([]),  # miss
        np.array([0.005, 0.010]),  # double
        np.array([0.008]),  # hit
    ]
    ei = error_index(th, smc, t_start=0.0, t_end=0.0, inclusive_pulse_end=True, n_neurons=3)
    assert ei == 2.0 / 3.0


def test_thalamic_misfire_count_respects_window() -> None:
    smc = np.array([0.0, 5.0], dtype=np.float64)
    th = [np.array([0.01]) for _ in range(2)]
    misfires, n_pulses = thalamic_misfire_count(
        th,
        smc,
        t_start=0.0,
        t_end=1.0,
    )
    assert n_pulses == 1
    assert misfires == 0


def test_smc_pulse_times_from_cor_spikes_picks_earliest_in_window() -> None:
    drives = np.array([1.0, 2.0], dtype=np.float64)
    cor = [
        np.array([1.002, 1.008]),
        np.array([2.001]),
    ]
    times = smc_pulse_times_from_cor_spikes(cor, drives, pulse_width_s=0.005)
    np.testing.assert_allclose(times, [1.002, 2.001])


def test_smc_pulse_times_from_cor_spikes_falls_back_to_drive() -> None:
    drives = np.array([0.5], dtype=np.float64)
    cor = [np.array([])]
    times = smc_pulse_times_from_cor_spikes(cor, drives, pulse_width_s=0.005)
    np.testing.assert_allclose(times, [0.5])


def test_resolve_smc_pulse_times_cor_spikes_branch() -> None:
    drives = np.array([1.0], dtype=np.float64)
    cor = [np.array([1.003])]
    resolved = resolve_smc_pulse_times(
        drive_pulse_times_s=drives,
        cor_spikes=cor,
        pulse_source="cor_spikes",
        pulse_width_ms=5.0,
    )
    np.testing.assert_allclose(resolved, [1.003])
    drive_only = resolve_smc_pulse_times(
        drive_pulse_times_s=drives,
        cor_spikes=cor,
        pulse_source="drive",
        pulse_width_ms=5.0,
    )
    np.testing.assert_allclose(drive_only, [1.0])
