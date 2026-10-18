"""Plant biomarkers: Mehregan $P_\\beta$ (plant.md §6) and the Gao Error Index.

The spectrum estimate behind $P_\\beta$ (``SpectrumParams``,
``multitaper_psd_point_process``, ``band_power``, ``neuron_band_power``,
``p_beta``) follows the Kumaravelu model's ``make_Spectrum`` / Chronux
``mtspectrumpt`` code, so it lives in the private ``kumaravelu_plant`` package
(LICENSING.md); the names below resolve to it on first use. The Error Index
functions are ours.
"""

from __future__ import annotations

import numpy as np

from envs.plant._kumaravelu import deferred

MEHREGAN_BETA_BAND_HZ: tuple[float, float] = (13.0, 35.0)
REFERENCE_ALPHA_BETA_BAND_HZ: tuple[float, float] = (7.0, 35.0)
DEFAULT_EI_RESPONSE_WINDOW_S: float = 0.025  # Gao / Mehregan: one TH spike in (SMCτ, SMCτ+25 ms)
DEFAULT_EI_WINDOW_S: float = 2.0

SpectrumParams = deferred("biomarkers", "SpectrumParams")
multitaper_psd_point_process = deferred("biomarkers", "multitaper_psd_point_process")
band_power = deferred("biomarkers", "band_power")
neuron_band_power = deferred("biomarkers", "neuron_band_power")
p_beta = deferred("biomarkers", "p_beta")


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
