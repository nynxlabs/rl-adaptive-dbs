"""Probe-only pattern alphabet constructions (TASK-177).

These are **not** integrated into ``make_alphabet`` / training — they exist for
landscape sweeps comparing alternative constructions against no-stim at 30 Hz.

Each class mirrors :class:`FixedMeanPatternAlphabet` (``n_actions``,
``idbs_for_pattern``, ``to_dbs_spec``) so ``MehreganEnv`` can accept them via
the ``alphabet=`` constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np

from envs.mehregan.fixed_mean_patterns import _grid_steps
from envs.plant.dbs import (
    DBS_AMPLITUDE_NA_PER_CM2,
    DBS_PULSE_WIDTH_MS,
    DbsSpec,
    create_dbs_current,
)


class PatternAlphabetLike(Protocol):
    mean_hz: float

    @property
    def n_actions(self) -> int: ...

    def idbs_for_pattern(self, index: int) -> np.ndarray: ...

    def to_dbs_spec(self, action: int) -> DbsSpec: ...


def _trace_from_onsets(
    onsets: np.ndarray,
    *,
    n_steps: int,
    pulse_len: int,
    amplitude: float,
) -> np.ndarray:
    trace = np.zeros(n_steps, dtype=np.float64)
    for onset in onsets:
        start = int(onset)
        if start >= n_steps:
            break
        end = min(start + pulse_len, n_steps)
        trace[start:end] = amplitude
    trace.setflags(write=False)
    return trace


def _regular_onsets(*, n_steps: int, isi_steps: int) -> np.ndarray:
    return np.arange(0, n_steps, isi_steps, dtype=np.int64)


def _regular_trace(
    *,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
) -> np.ndarray:
    regular = create_dbs_current(
        mean_hz,
        tmax_ms=step_duration_s * 1000.0,
        dt_ms=dt_ms,
        pulse_width_ms=pulse_width_ms,
        amplitude=amplitude,
    )
    regular.setflags(write=False)
    return regular


@lru_cache(maxsize=None)
def _build_burst_idbs(
    *,
    index: int,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
) -> np.ndarray:
    n_steps = _grid_steps(step_duration_s=step_duration_s, dt_ms=dt_ms)
    if mean_hz <= 0:
        trace = np.zeros(n_steps, dtype=np.float64)
        trace.setflags(write=False)
        return trace

    pulse_len = int(round(pulse_width_ms / dt_ms))
    isi_steps = int(round((1000.0 / mean_hz) / dt_ms))
    regular_onsets = _regular_onsets(n_steps=n_steps, isi_steps=isi_steps)
    n_pulses = int(regular_onsets.shape[0])

    if index == 0 or n_pulses <= 2:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )

    burst_sizes = (2, 3, 4, 5, 6)
    burst_hz_list = (60.0, 80.0, 100.0, 120.0)
    variant = index - 1
    burst_size = burst_sizes[variant % len(burst_sizes)]
    burst_hz = burst_hz_list[(variant // len(burst_sizes)) % len(burst_hz_list)]

    intra_isi = max(pulse_len + 1, int(round((1000.0 / burst_hz) / dt_ms)))
    n_bursts = max(1, n_pulses // burst_size)
    onsets: list[int] = []
    t = 0
    pulses_placed = 0
    span_steps = n_steps - pulse_len
    for burst_i in range(n_bursts):
        if pulses_placed + burst_size > n_pulses or t >= span_steps:
            break
        for _ in range(burst_size):
            if t >= span_steps:
                break
            onsets.append(t)
            pulses_placed += 1
            t += intra_isi
        if pulses_placed >= n_pulses:
            break
        remaining_bursts = n_bursts - (burst_i + 1)
        remaining_pulses = n_pulses - pulses_placed
        if remaining_bursts > 0 and remaining_pulses > 0:
            gap = max(
                intra_isi,
                (span_steps - t)
                // max(1, remaining_bursts + remaining_pulses // burst_size),
            )
            t += gap

    if len(onsets) < n_pulses:
        t = onsets[-1] + isi_steps if onsets else 0
        while len(onsets) < n_pulses and t < span_steps:
            onsets.append(t)
            t += isi_steps

    onsets_arr = np.array(sorted(onsets[:n_pulses]), dtype=np.int64)
    return _trace_from_onsets(
        onsets_arr, n_steps=n_steps, pulse_len=pulse_len, amplitude=amplitude
    )


@lru_cache(maxsize=None)
def _build_random_idbs(
    *,
    index: int,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
) -> np.ndarray:
    n_steps = _grid_steps(step_duration_s=step_duration_s, dt_ms=dt_ms)
    if mean_hz <= 0 or index == 0:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )

    pulse_len = int(round(pulse_width_ms / dt_ms))
    isi_steps = int(round((1000.0 / mean_hz) / dt_ms))
    regular_onsets = _regular_onsets(n_steps=n_steps, isi_steps=isi_steps)
    n_pulses = int(regular_onsets.shape[0])
    if n_pulses <= 2:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )

    rng = np.random.default_rng([int(round(mean_hz * 100)), index, 777])
    onsets = regular_onsets.copy()
    max_jitter = max(1, isi_steps)
    jitter = rng.integers(-max_jitter, max_jitter + 1, size=n_pulses - 2)
    onsets[1:-1] = onsets[1:-1] + jitter
    onsets.sort()
    for i in range(1, n_pulses):
        floor = onsets[i - 1] + pulse_len
        if onsets[i] < floor:
            onsets[i] = floor

    return _trace_from_onsets(
        onsets, n_steps=n_steps, pulse_len=pulse_len, amplitude=amplitude
    )


@lru_cache(maxsize=None)
def _build_alternating_idbs(
    *,
    index: int,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
) -> np.ndarray:
    n_steps = _grid_steps(step_duration_s=step_duration_s, dt_ms=dt_ms)
    if mean_hz <= 0 or index == 0:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )

    pulse_len = int(round(pulse_width_ms / dt_ms))
    isi_steps = int(round((1000.0 / mean_hz) / dt_ms))
    regular_onsets = _regular_onsets(n_steps=n_steps, isi_steps=isi_steps)
    n_pulses = int(regular_onsets.shape[0])
    if n_pulses <= 2:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )

    fracs = np.linspace(0.1, 0.45, num=40)
    frac = float(fracs[(index - 1) % len(fracs)])
    short_isi = max(pulse_len + 1, int(round(isi_steps * (1.0 - frac))))
    long_isi = max(short_isi + 1, int(round(isi_steps * (1.0 + frac))))

    onsets: list[int] = [0]
    use_short = True
    while len(onsets) < n_pulses:
        step = short_isi if use_short else long_isi
        next_onset = onsets[-1] + step
        if next_onset + pulse_len >= n_steps:
            break
        onsets.append(next_onset)
        use_short = not use_short

    while len(onsets) < n_pulses:
        next_onset = onsets[-1] + isi_steps
        if next_onset + pulse_len >= n_steps:
            break
        onsets.append(next_onset)

    onsets_arr = np.array(onsets[:n_pulses], dtype=np.int64)
    return _trace_from_onsets(
        onsets_arr, n_steps=n_steps, pulse_len=pulse_len, amplitude=amplitude
    )


@dataclass(frozen=True)
class BurstPatternAlphabet:
    """Burst clusters at fixed mean rate (probe-only, TASK-177)."""

    mean_hz: float
    step_duration_s: float = 2.0
    dt_ms: float = 0.02
    n_patterns: int = 41
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2

    @property
    def n_actions(self) -> int:
        return self.n_patterns

    def idbs_for_pattern(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.n_actions:
            msg = f"pattern index {index} outside [0, {self.n_actions})"
            raise ValueError(msg)
        return _build_burst_idbs(
            index=index,
            mean_hz=self.mean_hz,
            step_duration_s=self.step_duration_s,
            dt_ms=self.dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
        )

    def to_dbs_spec(self, action: int) -> DbsSpec:
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(self.mean_hz).pick_dbs_freq,
            idbs=self.idbs_for_pattern(int(action)),
            mean_hz=self.mean_hz,
        )


@dataclass(frozen=True)
class RandomPulseAlphabet:
    """Fully stochastic inter-pulse intervals at fixed pulse count (TASK-177)."""

    mean_hz: float
    step_duration_s: float = 2.0
    dt_ms: float = 0.02
    n_patterns: int = 41
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2

    @property
    def n_actions(self) -> int:
        return self.n_patterns

    def idbs_for_pattern(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.n_actions:
            msg = f"pattern index {index} outside [0, {self.n_actions})"
            raise ValueError(msg)
        return _build_random_idbs(
            index=index,
            mean_hz=self.mean_hz,
            step_duration_s=self.step_duration_s,
            dt_ms=self.dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
        )

    def to_dbs_spec(self, action: int) -> DbsSpec:
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(self.mean_hz).pick_dbs_freq,
            idbs=self.idbs_for_pattern(int(action)),
            mean_hz=self.mean_hz,
        )


@dataclass(frozen=True)
class AlternatingPatternAlphabet:
    """Deterministic short/long ISI alternation without PRNG jitter (TASK-177)."""

    mean_hz: float
    step_duration_s: float = 2.0
    dt_ms: float = 0.02
    n_patterns: int = 41
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2

    @property
    def n_actions(self) -> int:
        return self.n_patterns

    def idbs_for_pattern(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.n_actions:
            msg = f"pattern index {index} outside [0, {self.n_actions})"
            raise ValueError(msg)
        return _build_alternating_idbs(
            index=index,
            mean_hz=self.mean_hz,
            step_duration_s=self.step_duration_s,
            dt_ms=self.dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
        )

    def to_dbs_spec(self, action: int) -> DbsSpec:
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(self.mean_hz).pick_dbs_freq,
            idbs=self.idbs_for_pattern(int(action)),
            mean_hz=self.mean_hz,
        )


@lru_cache(maxsize=None)
def _build_continuous_hz_idbs(
    *,
    frequency_hz: int,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
) -> np.ndarray:
    n_steps = _grid_steps(step_duration_s=step_duration_s, dt_ms=dt_ms)
    if frequency_hz <= 0:
        trace = np.zeros(n_steps, dtype=np.float64)
        trace.setflags(write=False)
        return trace
    trace = create_dbs_current(
        float(frequency_hz),
        tmax_ms=step_duration_s * 1000.0,
        dt_ms=dt_ms,
        pulse_width_ms=pulse_width_ms,
        amplitude=amplitude,
    )
    trace.setflags(write=False)
    return trace


@dataclass(frozen=True)
class ContinuousFrequencyAlphabet:
    """1 Hz scalar-frequency probe (0..max_hz inclusive, TASK-177 exp 6).

    Unlike :class:`envs.mehregan.patterns.PatternAlphabet` (Kumaravelu 0:5:200 grid),
    each action maps to an exact integer Hz via a precomputed ``idbs`` trace.
    """

    max_hz: int = 200
    step_hz: int = 1
    step_duration_s: float = 2.0
    dt_ms: float = 0.02
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2

    @property
    def n_actions(self) -> int:
        return self.max_hz // self.step_hz + 1

    def frequency_hz_for_action(self, action: int) -> float:
        if action < 0 or action >= self.n_actions:
            msg = f"action {action} outside [0, {self.n_actions})"
            raise ValueError(msg)
        return float(action * self.step_hz)

    def idbs_for_pattern(self, index: int) -> np.ndarray:
        hz = int(self.frequency_hz_for_action(index))
        return _build_continuous_hz_idbs(
            frequency_hz=hz,
            step_duration_s=self.step_duration_s,
            dt_ms=self.dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
        )

    def to_dbs_spec(self, action: int) -> DbsSpec:
        hz = self.frequency_hz_for_action(int(action))
        if hz <= 0:
            return DbsSpec.none()
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(hz).pick_dbs_freq,
            idbs=self.idbs_for_pattern(int(action)),
            mean_hz=hz,
        )
