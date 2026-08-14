"""Guards for long single-shot PythonPlant integrates (Fig 4a continuous stitch)."""

from __future__ import annotations

import numpy as np

from envs.plant.network.integrator import (
    DEFAULT_GPI_SPIKE_BUFFER,
    INTRACELL_CA_MIN,
    default_gpi_spike_buffer_size,
)


def test_two_second_spike_buffer_stays_at_default() -> None:
    assert default_gpi_spike_buffer_size(2.0) == DEFAULT_GPI_SPIKE_BUFFER


def test_long_stitch_spike_buffer_grows() -> None:
    buf_22 = default_gpi_spike_buffer_size(22.0)
    buf_62 = default_gpi_spike_buffer_size(62.0)
    assert buf_22 > DEFAULT_GPI_SPIKE_BUFFER
    assert buf_62 >= buf_22


def test_ca_floor_keeps_nernst_finite() -> None:
    assert INTRACELL_CA_MIN > 0.0
    cao = 2000.0
    ecasn = np.log(cao / INTRACELL_CA_MIN)
    assert np.isfinite(ecasn)
    zeros = np.zeros(3)
    ecasn_vec = np.log(cao / np.maximum(zeros, INTRACELL_CA_MIN))
    assert np.all(np.isfinite(ecasn_vec))
