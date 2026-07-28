"""Near-hub burst alphabet for open-loop diversity probes (not promoted to panels)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from envs.mehregan.fixed_mean_patterns import _grid_steps
from envs.mehregan.pattern_alternatives import (
    _regular_onsets,
    _regular_trace,
    _trace_from_onsets,
)
from envs.plant.dbs import (
    DBS_AMPLITUDE_NA_PER_CM2,
    DBS_PULSE_WIDTH_MS,
    DbsSpec,
)

# Open-loop near-bests @ 45 Hz (burst n=41 + n=256) plus soft-train winner pattern 62.
NEAR_HUB_DEFAULT: tuple[int, ...] = (15, 35, 3, 23, 65, 114, 69, 62)


def _burst_base_params(
    index: int, *, mean_hz: float, isi_steps: int
) -> tuple[int, float, int, float]:
    burst_sizes = (2, 3, 4, 5, 6)
    if index < 41:
        burst_hz_list = (60.0, 80.0, 100.0, 120.0)
        variant = index - 1
        burst_size = burst_sizes[variant % len(burst_sizes)]
        burst_hz = burst_hz_list[(variant // len(burst_sizes)) % len(burst_hz_list)]
        return burst_size, float(burst_hz), 0, 1.0
    rng = np.random.default_rng([int(round(mean_hz * 100)), index, 9001])
    burst_size = int(rng.integers(2, 7))
    burst_hz = float(rng.choice([60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]))
    phase_steps = int(rng.integers(0, max(1, isi_steps)))
    gap_scale = float(rng.uniform(0.4, 2.5))
    return burst_size, burst_hz, phase_steps, gap_scale


def _build_burst_from_params(
    *,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
    burst_size: int,
    burst_hz: float,
    phase_steps: int,
    gap_scale: float,
) -> np.ndarray:
    n_steps = _grid_steps(step_duration_s=step_duration_s, dt_ms=dt_ms)
    if mean_hz <= 0:
        trace = np.zeros(n_steps, dtype=np.float64)
        trace.setflags(write=False)
        return trace

    pulse_len = int(round(pulse_width_ms / dt_ms))
    isi_steps = int(round((1000.0 / mean_hz) / dt_ms))
    n_pulses = int(_regular_onsets(n_steps=n_steps, isi_steps=isi_steps).shape[0])
    if n_pulses <= 2:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )

    intra_isi = max(pulse_len + 1, int(round((1000.0 / burst_hz) / dt_ms)))
    n_bursts = max(1, n_pulses // burst_size)
    onsets: list[int] = []
    t = int(phase_steps)
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
            t += max(intra_isi, int(round(gap * gap_scale)))

    if len(onsets) < n_pulses:
        t = onsets[-1] + isi_steps if onsets else int(phase_steps)
        while len(onsets) < n_pulses and t < span_steps:
            onsets.append(t)
            t += isi_steps

    onsets_arr = np.array(sorted(onsets[:n_pulses]), dtype=np.int64)
    return _trace_from_onsets(
        onsets_arr, n_steps=n_steps, pulse_len=pulse_len, amplitude=amplitude
    )


@lru_cache(maxsize=None)
def _build_near_hub_idbs(
    *,
    index: int,
    hubs: tuple[int, ...],
    n_per_hub: int,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
) -> np.ndarray:
    if mean_hz <= 0 or index == 0:
        return _regular_trace(
            mean_hz=mean_hz,
            step_duration_s=step_duration_s,
            dt_ms=dt_ms,
            pulse_width_ms=pulse_width_ms,
            amplitude=amplitude,
        )
    if n_per_hub < 1 or not hubs:
        raise ValueError("near-hub alphabet needs hubs and n_per_hub >= 1")

    isi_steps = int(round((1000.0 / mean_hz) / dt_ms))
    k = index - 1
    hub = hubs[(k // n_per_hub) % len(hubs)]
    pert = k % n_per_hub
    burst_size, burst_hz, phase0, gap0 = _burst_base_params(
        hub, mean_hz=mean_hz, isi_steps=isi_steps
    )

    n_phase = 8
    n_gap = max(1, n_per_hub // n_phase)
    phase_bin = pert % n_phase
    gap_bin = (pert // n_phase) % n_gap
    phase_delta = max(1, isi_steps // 128)
    phase_steps = int(phase0 + (phase_bin - (n_phase // 2)) * phase_delta) % max(
        1, isi_steps
    )
    gap_mid = (n_gap - 1) / 2.0
    gap_scale = float(gap0 * (1.0 + (gap_bin - gap_mid) * 0.015))
    gap_scale = float(np.clip(gap_scale, 0.35, 2.8))

    return _build_burst_from_params(
        mean_hz=mean_hz,
        step_duration_s=step_duration_s,
        dt_ms=dt_ms,
        pulse_width_ms=pulse_width_ms,
        amplitude=amplitude,
        burst_size=burst_size,
        burst_hz=burst_hz,
        phase_steps=phase_steps,
        gap_scale=gap_scale,
    )


@dataclass(frozen=True)
class NearHubBurstAlphabet:
    """Local phase/gap copies of strong burst hubs (diversity probe only)."""

    mean_hz: float
    hubs: tuple[int, ...] = NEAR_HUB_DEFAULT
    n_per_hub: int = 32
    step_duration_s: float = 2.0
    dt_ms: float = 0.02
    skip_regular: bool = False
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2

    @property
    def n_patterns(self) -> int:
        return 1 + len(self.hubs) * self.n_per_hub

    @property
    def n_actions(self) -> int:
        return self.n_patterns - 1 if self.skip_regular else self.n_patterns

    def _pattern_index(self, action: int) -> int:
        if self.skip_regular:
            return int(action) + 1
        return int(action)

    def idbs_for_pattern(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.n_patterns:
            msg = f"pattern index {index} outside [0, {self.n_patterns})"
            raise ValueError(msg)
        return _build_near_hub_idbs(
            index=index,
            hubs=tuple(self.hubs),
            n_per_hub=int(self.n_per_hub),
            mean_hz=self.mean_hz,
            step_duration_s=self.step_duration_s,
            dt_ms=self.dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
        )

    def idbs_for_action(self, action: int) -> np.ndarray:
        return self.idbs_for_pattern(self._pattern_index(action))

    def to_dbs_spec(self, action: int) -> DbsSpec:
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(self.mean_hz).pick_dbs_freq,
            idbs=self.idbs_for_action(int(action)),
            mean_hz=self.mean_hz,
        )

    def action_for_dbs_spec(self, spec: DbsSpec) -> int:
        if self.skip_regular:
            return 0
        return self.action_for_frequency_hz(spec.frequency_hz)

    def action_for_frequency_hz(self, hz: float) -> int:
        if abs(float(hz) - self.mean_hz) < 1e-6:
            return 0
        msg = (
            f"frequency {hz} Hz has no near-hub pattern (mean_hz={self.mean_hz}); "
            "only the mean-rate periodic baseline maps to a pattern (index 0)"
        )
        raise ValueError(msg)
