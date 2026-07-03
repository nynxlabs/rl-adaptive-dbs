"""Tests for GPi spike time extraction (Kumaravelu find_spike_times port)."""

from __future__ import annotations

import numpy as np

from envs.plant.spikes import find_spike_times


def test_find_spike_times_upward_crossing_only() -> None:
    """Bool np.diff treats repolarization as +1; must match MATLAB diff on 0/1."""
    dt_ms = 0.01
    t_ms = np.arange(6, dtype=np.float64) * dt_ms
    # One AP: below, below, cross up, stay up, cross down, below
    v = np.array([[-30.0, -30.0, -25.0, 10.0, -25.0, -30.0]])
    spikes = find_spike_times(v, t_ms, n_neurons=1)
    np.testing.assert_allclose(spikes[0], [0.00002], rtol=0.0, atol=0.0)


def test_find_spike_times_no_spurious_downcross() -> None:
    """Repolarization True→False must not register as a spike."""
    dt_ms = 0.01
    n = 8
    t_ms = np.arange(n, dtype=np.float64) * dt_ms
    trace = np.array([-30.0, -30.0, -25.0, 10.0, 20.0, 15.0, -25.0, -30.0])
    v = np.stack([trace])
    spikes = find_spike_times(v, t_ms, n_neurons=1)
    assert len(spikes[0]) == 1
