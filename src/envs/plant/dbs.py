"""STN DBS drive specification for the Kumaravelu reference script."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Kumaravelu simulate_network_model.m defaults (lines 21–22).
DBS_PULSE_WIDTH_MS: float = 0.3
DBS_AMPLITUDE_NA_PER_CM2: float = 300.0


def stn_dbs_gains(n: int, spread: float) -> np.ndarray:
    """Share of the DBS current each STN neuron receives, by distance from the contact.

    Neurons sit uniformly through a sphere's volume between r0 and ``spread`` * r0
    (r_i = r0 * (1 + (spread^3 - 1) * i / (n - 1))^(1/3)) and take (r0 / r_i)^2 of the
    current, so the nearest neuron gets all of it. ``spread`` = 1 gives every neuron the
    full current (Kumaravelu).
    """
    if spread < 1.0:
        raise ValueError(f"stn_dbs_spread must be >= 1, got {spread}")
    if spread == 1.0 or n == 1:
        return np.ones(n, dtype=np.float64)
    r = (1.0 + (spread**3 - 1.0) * np.arange(n) / (n - 1)) ** (1.0 / 3.0)
    return 1.0 / r**2


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


class ContinuousPulseTrain:
    """Phase-continuous STN pulse train across sequential carried segments.

    ``create_dbs_current`` starts every segment with a pulse at t=0, which is
    right for one-shot integrates but restarts the pulse clock at each segment
    boundary when plant state is carried. This keeps one global sample clock:
    the next onset follows the last one by the current ISI, and a pulse that
    crosses a boundary finishes in the next segment at its own amplitude.
    Segment grids are ``0:dt_ms:tmax_ms``; the integrator applies samples
    ``1..N``, so sample 0 of a segment is the previous segment's sample N.
    """

    def __init__(self, dt_ms: float = 0.01) -> None:
        self.dt_ms = float(dt_ms)
        self.reset()

    def reset(self) -> None:
        self._origin = 0  # global index of this segment's local sample 0
        self._last_onset: int | None = None
        self._last_len = 0
        self._last_amp = 0.0

    def segment(
        self,
        *,
        frequency_hz: float,
        pulse_width_ms: float,
        amplitude: float,
        tmax_ms: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return the segment trace and the state to commit after a successful integrate."""
        n = int(round(tmax_ms / self.dt_ms))
        idbs = np.zeros(n + 1, dtype=np.float64)
        g0 = self._origin
        last_onset, last_len, last_amp = self._last_onset, self._last_len, self._last_amp
        if last_onset is not None:
            lo, hi = max(last_onset, g0) - g0, min(last_onset + last_len, g0 + n + 1) - g0
            if hi > lo:
                idbs[lo:hi] = last_amp
        if frequency_hz > 0.0 and amplitude > 0.0:
            pulse_len = int(round(pulse_width_ms / self.dt_ms))
            isi = int(round((1000.0 / frequency_hz) / self.dt_ms))
            if pulse_len <= 0 or isi <= 0:
                msg = "pulse_width_ms and frequency_hz must map to positive sample counts"
                raise ValueError(msg)
            onset = g0 if last_onset is None else max(last_onset + isi, g0 + 1)
            while onset <= g0 + n:
                idbs[onset - g0 : min(onset + pulse_len, g0 + n + 1) - g0] = amplitude
                last_onset, last_len, last_amp = onset, pulse_len, amplitude
                onset += isi
        else:
            # Stimulator off: no pending onset; a later re-enable starts a fresh train.
            if last_onset is not None and last_onset + last_len <= g0 + n:
                last_onset = None
        state = {"origin": g0 + n, "last_onset": last_onset, "last_len": last_len, "last_amp": last_amp}
        return idbs, state

    def commit(self, state: dict[str, object]) -> None:
        self._origin = int(state["origin"])  # type: ignore[arg-type]
        self._last_onset = state["last_onset"]  # type: ignore[assignment]
        self._last_len = int(state["last_len"])  # type: ignore[arg-type]
        self._last_amp = float(state["last_amp"])  # type: ignore[arg-type]
