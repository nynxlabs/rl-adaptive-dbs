"""GPi biomarkers from spike trains (Mehregan $P_\\beta$, plant.md §6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import dpss

MEHREGAN_BETA_BAND_HZ: tuple[float, float] = (13.0, 35.0)
REFERENCE_ALPHA_BETA_BAND_HZ: tuple[float, float] = (7.0, 35.0)
DEFAULT_EI_RESPONSE_WINDOW_S: float = 0.025  # Gao / Mehregan: one TH spike in (SMCτ, SMCτ+25 ms)
DEFAULT_EI_WINDOW_S: float = 2.0


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
        mintime = min(mintime, 0.0)  # ensure grid covers full segment
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


def smc_pulse_times_from_trace(trace: np.ndarray, *, dt_ms: float) -> np.ndarray:
    """Rising edges of a pulsed drive trace (SMCτ events) in seconds."""
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    if trace.size < 2:
        if trace.size == 1 and trace[0] > 0.0:
            return np.array([0.0], dtype=np.float64)
        return np.array([], dtype=np.float64)
    rising = (trace[:-1] <= 0.0) & (trace[1:] > 0.0)
    step_idx = np.flatnonzero(rising) + 1
    times = step_idx.astype(np.float64) * dt_ms / 1000.0
    if trace[0] > 0.0:
        times = np.concatenate(([0.0], times))
    return times


def smc_pulse_times_from_iappco(iappco: np.ndarray, *, dt_ms: float) -> np.ndarray:
    """Backward-compatible alias for :func:`smc_pulse_times_from_trace`."""
    return smc_pulse_times_from_trace(iappco, dt_ms=dt_ms)


def smc_pulse_times_from_cor_spikes(
    cor_spikes: list[np.ndarray],
    drive_pulse_times_s: np.ndarray,
    *,
    pulse_width_s: float = 0.005,
    fallback_to_drive: bool = True,
) -> np.ndarray:
    """Map each SMC drive pulse to earliest Cor spike in (τ, τ+pulse_width] (Gao SMCτ)."""
    drives = np.asarray(drive_pulse_times_s, dtype=np.float64).reshape(-1)
    if drives.size == 0:
        return np.array([], dtype=np.float64)
    events: list[float] = []
    for tau in drives:
        t_lo = float(tau)
        t_hi = t_lo + pulse_width_s
        best: float | None = None
        for spikes in cor_spikes:
            arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                continue
            in_window = (arr > t_lo) & (arr <= t_hi)
            if not np.any(in_window):
                continue
            candidate = float(np.min(arr[in_window]))
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            events.append(best)
        elif fallback_to_drive:
            events.append(t_lo)
    return np.asarray(events, dtype=np.float64)


def resolve_smc_pulse_times(
    *,
    drive_pulse_times_s: np.ndarray,
    cor_spikes: list[np.ndarray] | None,
    pulse_source: str,
    pulse_width_ms: float,
) -> np.ndarray:
    """Select SMCτ times for EI from scheduled drive or Cor spike alignment."""
    drives = np.asarray(drive_pulse_times_s, dtype=np.float64).reshape(-1)
    if pulse_source == "cor_spikes" and cor_spikes:
        return smc_pulse_times_from_cor_spikes(
            cor_spikes,
            drives,
            pulse_width_s=pulse_width_ms / 1000.0,
        )
    return drives.copy()


def thalamic_misfire_count(
    th_spikes: list[np.ndarray],
    smc_pulse_times_s: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    response_window_s: float = DEFAULT_EI_RESPONSE_WINDOW_S,
    inclusive_pulse_end: bool = False,
) -> tuple[int, int]:
    """Count TH misfires and SMC pulses in ``[t_start, t_end]``.

    Correct response: exactly one TH spike in ``(SMCτ, SMCτ + response_window)``
    (Gao ICCPS 2020 Eq. 4; Mehregan Eq. 2 misfire definition).
    """
    pulses = np.asarray(smc_pulse_times_s, dtype=np.float64).reshape(-1)
    if inclusive_pulse_end:
        pulse_mask = (pulses >= t_start) & (pulses <= t_end)
    else:
        pulse_mask = (pulses >= t_start) & (pulses < t_end)
    window_pulses = pulses[pulse_mask]
    n_pulses = int(window_pulses.size)
    if n_pulses == 0:
        return 0, 0

    misfires = 0
    for tau in window_pulses:
        t_lo = float(tau)
        t_hi = t_lo + response_window_s
        for spikes in th_spikes:
            arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                misfires += 1
                continue
            in_window = (arr > t_lo) & (arr < t_hi)
            count = int(in_window.sum())
            if count != 1:
                misfires += 1
    return misfires, n_pulses


def thalamic_misfire_breakdown(
    th_spikes: list[np.ndarray],
    smc_pulse_times_s: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    response_window_s: float = DEFAULT_EI_RESPONSE_WINDOW_S,
    inclusive_pulse_end: bool = False,
) -> dict[str, int]:
    """Classify TH misfires as misses (0 spikes) vs doubles (>1 spike) in ``[t_start, t_end]``."""
    pulses = np.asarray(smc_pulse_times_s, dtype=np.float64).reshape(-1)
    if inclusive_pulse_end:
        pulse_mask = (pulses >= t_start) & (pulses <= t_end)
    else:
        pulse_mask = (pulses >= t_start) & (pulses < t_end)
    window_pulses = pulses[pulse_mask]
    misses = 0
    doubles = 0
    correct = 0
    for tau in window_pulses:
        t_lo = float(tau)
        t_hi = t_lo + response_window_s
        for spikes in th_spikes:
            arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
            if arr.size == 0:
                misses += 1
                continue
            count = int(((arr > t_lo) & (arr < t_hi)).sum())
            if count == 0:
                misses += 1
            elif count == 1:
                correct += 1
            else:
                doubles += 1
    n_pulses = int(window_pulses.size)
    n_neurons = len(th_spikes)
    return {
        "n_pulses": n_pulses,
        "n_neurons": n_neurons,
        "misses": misses,
        "doubles": doubles,
        "correct": correct,
        "misfires": misses + doubles,
        "trials": n_pulses * n_neurons,
    }


def error_index(
    th_spikes: list[np.ndarray],
    smc_pulse_times_s: np.ndarray,
    *,
    t_start: float,
    t_end: float,
    response_window_s: float = DEFAULT_EI_RESPONSE_WINDOW_S,
    inclusive_pulse_end: bool = False,
    n_neurons: int | None = None,
) -> float:
    """Windowed Error Index EI_Tω (Mehregan Eq. 2 / Gao Eq. 6).

    Metric definition follows Gao et al. (ICCPS 2020); Fig 2b plant drive for this
    metric uses So-style SMC pulses into TH (see ``PlantConfig.iappth_baseline``).
    """
    n = n_neurons if n_neurons is not None else len(th_spikes)
    if n <= 0:
        msg = "error_index requires at least one thalamic neuron"
        raise ValueError(msg)
    misfires, n_pulses = thalamic_misfire_count(
        th_spikes,
        smc_pulse_times_s,
        t_start=t_start,
        t_end=t_end,
        response_window_s=response_window_s,
        inclusive_pulse_end=inclusive_pulse_end,
    )
    if n_pulses == 0:
        return 0.0
    return float(misfires) / float(n * n_pulses)
