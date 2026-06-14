"""GPi biomarker ($P_\\beta$) unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from envs.plant.biomarkers import (
    MEHREGAN_BETA_BAND_HZ,
    REFERENCE_ALPHA_BETA_BAND_HZ,
    SpectrumParams,
    band_power,
    neuron_band_power,
    p_beta,
)


def _periodic_spikes(rate_hz: float, duration_s: float) -> np.ndarray:
    period = 1.0 / rate_hz
    n = int(duration_s * rate_hz)
    return np.arange(n, dtype=float) * period


def test_empty_spike_train_returns_zero_band_power() -> None:
    params = SpectrumParams.from_dt_ms(0.01)
    assert neuron_band_power(
        np.array([]),
        params,
        *MEHREGAN_BETA_BAND_HZ,
        segment_duration_s=2.0,
    ) == pytest.approx(0.0)
    assert p_beta([np.array([])] * 10, segment_duration_s=2.0) == pytest.approx(0.0)


def test_p_beta_reproducible() -> None:
    duration = 2.0
    spikes = [_periodic_spikes(22.0 + idx, duration) for idx in range(10)]
    first = p_beta(spikes, segment_duration_s=duration)
    second = p_beta(spikes, segment_duration_s=duration)
    assert first == pytest.approx(second)


def test_higher_beta_rate_yields_higher_p_beta() -> None:
    duration = 2.0
    params = SpectrumParams.from_dt_ms(0.01)
    low = neuron_band_power(
        _periodic_spikes(18.0, duration),
        params,
        *MEHREGAN_BETA_BAND_HZ,
        segment_duration_s=duration,
    )
    high = neuron_band_power(
        _periodic_spikes(25.0, duration),
        params,
        *MEHREGAN_BETA_BAND_HZ,
        segment_duration_s=duration,
    )
    assert high > low > 0.0


def test_mehregan_band_is_subset_of_reference_band() -> None:
    duration = 2.0
    spikes = _periodic_spikes(20.0, duration)
    params = SpectrumParams.from_dt_ms(0.01)
    mehregan = neuron_band_power(
        spikes, params, *MEHREGAN_BETA_BAND_HZ, segment_duration_s=duration
    )
    reference = neuron_band_power(
        spikes, params, *REFERENCE_ALPHA_BETA_BAND_HZ, segment_duration_s=duration
    )
    assert 0.0 < mehregan <= reference


def test_band_power_trapezoid_on_simple_peak() -> None:
    freqs = np.linspace(0.0, 100.0, 1001)
    psd = np.exp(-0.5 * ((freqs - 20.0) / 2.0) ** 2)
    low, high = MEHREGAN_BETA_BAND_HZ
    integral = band_power(psd, freqs, low, high)
    assert integral > band_power(psd, freqs, 40.0, 60.0)
