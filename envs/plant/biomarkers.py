"""GPi biomarkers from spike trains (Mehregan $P_\\beta$, plant.md §6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import dpss

MEHREGAN_BETA_BAND_HZ: tuple[float, float] = (13.0, 35.0)
REFERENCE_ALPHA_BETA_BAND_HZ: tuple[float, float] = (7.0, 35.0)


@dataclass(frozen=True)
class SpectrumParams:
    """Chronux-style multitaper settings (Kumaravelu reference defaults)."""

    dt_ms: float = 0.01
    fpass: tuple[float, float] = (1.0, 100.0)
    time_bandwidth: float = 3.0
    n_tapers: int = 5
    pad: int = 0

    @classmethod
    def from_dt_ms(cls, dt_ms: float) -> SpectrumParams:
        return cls(dt_ms=dt_ms)

    @property
    def fs(self) -> float:
        return 1.0 / (self.dt_ms * 1e-3)


def _nfft(n_samples: int, pad: int) -> int:
    exponent = max(0, n_samples - 1).bit_length()
    return max(2 ** (exponent + pad), n_samples)


def _time_grid(
    spike_times: np.ndarray,
    fs: float,
    segment_duration_s: float | None,
) -> np.ndarray:
    dt = 1.0 / fs
    if spike_times.size == 0:
        if segment_duration_s is None:
            msg = "segment_duration_s required when spike train is empty"
            raise ValueError(msg)
        return np.arange(-dt, segment_duration_s + dt, dt)

    mintime = float(np.min(spike_times))
    maxtime = float(np.max(spike_times))
    if segment_duration_s is not None:
        maxtime = max(maxtime, segment_duration_s)
    return np.arange(mintime - dt, maxtime + dt + dt / 2.0, dt)


def multitaper_psd_point_process(
    spike_times: np.ndarray,
    params: SpectrumParams,
    segment_duration_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-process multitaper PSD (Chronux ``mtspectrumpt``-style, single neuron)."""
    spike_times = np.asarray(spike_times, dtype=float).reshape(-1)
    fs = params.fs
    t = _time_grid(spike_times, fs, segment_duration_s)
    n_samples = len(t)
    nfft = _nfft(n_samples, params.pad)

    freqs_full = np.arange(nfft, dtype=float) * (fs / nfft)
    band = params.fpass
    findx = np.where((freqs_full >= band[0]) & (freqs_full <= band[1]))[0]
    freqs = freqs_full[findx]

    # MATLAB: dpss(N, TW, K) * sqrt(Fs), shape N×K
    tapers = dpss(
        n_samples,
        params.time_bandwidth,
        params.n_tapers,
        sym=False,
    ).T * np.sqrt(fs)
    n_tapers = tapers.shape[1]

    taper_fft = np.fft.fft(tapers, n=nfft, axis=0)[findx, :]
    angular = 2.0 * np.pi * freqs

    if spike_times.size > 0:
        in_window = (spike_times >= t[0]) & (spike_times <= t[-1])
        times = spike_times[in_window]
    else:
        times = np.array([], dtype=float)

    mean_rate = len(times) / n_samples if n_samples else 0.0
    if mean_rate == 0.0:
        return np.zeros_like(freqs), freqs

    projections = np.column_stack(
        [
            interp1d(t, tapers[:, idx], bounds_error=False, fill_value=0.0)(times)
            for idx in range(n_tapers)
        ]
    )
    exponential = np.exp(-1j * angular[:, None] * (times[None, :] - t[0]))
    spectrum_coeffs = exponential @ projections - taper_fft * mean_rate
    psd = np.mean(np.abs(spectrum_coeffs) ** 2, axis=1)
    return psd, freqs


def band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def neuron_band_power(
    spike_times: np.ndarray,
    params: SpectrumParams,
    f_low: float,
    f_high: float,
    segment_duration_s: float | None = None,
) -> float:
    psd, freqs = multitaper_psd_point_process(
        spike_times,
        params,
        segment_duration_s=segment_duration_s,
    )
    return band_power(psd, freqs, f_low, f_high)


def p_beta(
    gpi_spikes: list[np.ndarray],
    *,
    dt_ms: float = 0.01,
    segment_duration_s: float | None = None,
    f_low: float = MEHREGAN_BETA_BAND_HZ[0],
    f_high: float = MEHREGAN_BETA_BAND_HZ[1],
    params: SpectrumParams | None = None,
) -> float:
    """Mehregan Eq. (1): mean over GPi neurons of 13–35 Hz PSD integral."""
    spectrum = params if params is not None else SpectrumParams.from_dt_ms(dt_ms)
    if not gpi_spikes:
        return 0.0
    per_neuron = [
        neuron_band_power(
            spikes,
            spectrum,
            f_low,
            f_high,
            segment_duration_s=segment_duration_s,
        )
        for spikes in gpi_spikes
    ]
    return float(np.mean(per_neuron))
