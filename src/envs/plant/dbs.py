"""STN DBS drive specification for the Kumaravelu reference script."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Kumaravelu simulate_network_model.m defaults (lines 21–22).
DBS_PULSE_WIDTH_MS: float = 0.3
DBS_AMPLITUDE_NA_PER_CM2: float = 300.0


def create_dbs_current(
    frequency_hz: float,
    *,
    tmax_ms: float,
    dt_ms: float = 0.01,
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS,
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2,
) -> np.ndarray:
    """STN DBS current trace — port of ``creatdbs`` in simulate_network_model.m.

    Returns a 1-D array of length ``int(tmax_ms / dt_ms) + 1`` (time grid
    ``0:dt_ms:tmax_ms``), matching MATLAB column-major indexing converted to
    0-based Python slices.
    """
    if frequency_hz <= 0:
        n_steps = int(round(tmax_ms / dt_ms)) + 1
        return np.zeros(n_steps, dtype=np.float64)

    n_steps = int(round(tmax_ms / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    pulse_len = int(round(pulse_width_ms / dt_ms))
    if pulse_len <= 0:
        msg = "pulse_width_ms must be positive"
        raise ValueError(msg)

    pulse = np.full(pulse_len, amplitude, dtype=np.float64)
    isi_steps = int(round((1000.0 / frequency_hz) / dt_ms))
    if isi_steps <= 0:
        msg = "frequency_hz too high for dt_ms grid"
        raise ValueError(msg)

    i = 0
    while i < n_steps:
        end = min(i + pulse_len, n_steps)
        idbs[i:end] = pulse[: end - i]
        i += isi_steps
    return idbs


@dataclass(frozen=True)
class DbsSpec:
    """Index into freqs = 0:5:200 Hz in simulate_network_model.m.

    pick_dbs_freq == 1 forces zero DBS current (reference convention).

    Option C (fixed-mean pattern action space, TASK-84): ``idbs`` optionally
    carries a **precomputed** STN drive trace on the plant time grid. When set,
    the Python integrator applies it directly instead of synthesizing a regular
    train from ``frequency_hz`` (see envs/plant/network/integrator.py and
    envs/mehregan/fixed_mean_patterns.py). ``mean_hz`` records the constant mean
    stimulation rate for logging/metrics; ``frequency_hz`` reports it. ``idbs``
    is excluded from equality/hash so specs stay hashable and comparable by
    ``(pick_dbs_freq, mean_hz)``. The MATLAB backend ignores ``idbs`` — pattern
    mode is Python-plant only.
    """

    pick_dbs_freq: int = 1
    idbs: np.ndarray | None = field(default=None, compare=False)
    mean_hz: float | None = None

    @classmethod
    def none(cls) -> DbsSpec:
        return cls(pick_dbs_freq=1)

    @classmethod
    def from_frequency_hz(cls, hz: float) -> DbsSpec:
        """Map carrier frequency (Hz) to reference script index (1-based)."""
        if hz <= 0:
            return cls.none()
        index = int(round(hz / 5.0)) + 1
        return cls(pick_dbs_freq=index)

    @property
    def frequency_hz(self) -> float:
        if self.mean_hz is not None:
            return float(self.mean_hz)
        if self.pick_dbs_freq <= 1:
            return 0.0
        return float((self.pick_dbs_freq - 1) * 5)
