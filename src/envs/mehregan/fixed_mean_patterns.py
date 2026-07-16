"""Fixed-mean-frequency pulse pattern action space (Option C, TASK-83/84).

Mehregan et al. frame the action as a **temporal pulse pattern at a fixed mean
frequency** (45 Hz paper / 30 Hz ablation): the mean stimulation rate is held
constant by construction and the agent shapes only the *irregularity* of the
train (§IV.A.2, Fig. 5b). This module implements that action space as a discrete
alphabet of precomputed STN drive traces on the plant time grid.

Paper-silent choices (documented in docs/environment.md §4.2):

- ``n_patterns = 41`` — matches the scalar-frequency alphabet / actor head size,
  so no DDPG topology change is needed.
- **Pattern 0** is the regular periodic train at ``mean_hz`` — byte-identical to
  ``create_dbs_current(mean_hz)`` (the paper init target).
- **Patterns 1–40** are deterministic irregular trains with the **same pulse
  count** as pattern 0 (mean rate preserved exactly). Pulse onsets are the
  regular grid with seeded integer jitter on the interior pulses; the first and
  last onsets are pinned, so the total span (and thus the sum of inter-spike
  intervals) is unchanged — the ISI perturbations sum to zero.
- Irregular jitter uses a PRNG seeded by ``(mean_hz, pattern_index)`` so the
  same pattern is reproducible across training, evaluation, and quantization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from envs.plant.dbs import (
    DBS_AMPLITUDE_NA_PER_CM2,
    DBS_PULSE_WIDTH_MS,
    DbsSpec,
    create_dbs_current,
)

DEFAULT_MEAN_STIM_HZ: float = 45.0

# Default interior-onset jitter, as a fraction of the regular inter-pulse
# interval. 1/3 keeps consecutive onsets well separated (>> pulse width) while
# still producing visibly irregular trains.
DEFAULT_JITTER_FRACTION: float = 1.0 / 3.0


def _grid_steps(*, step_duration_s: float, dt_ms: float) -> int:
    tmax_ms = step_duration_s * 1000.0
    return int(round(tmax_ms / dt_ms)) + 1


@lru_cache(maxsize=None)
def _build_idbs(
    *,
    index: int,
    mean_hz: float,
    step_duration_s: float,
    dt_ms: float,
    pulse_width_ms: float,
    amplitude: float,
    jitter_fraction: float,
) -> np.ndarray:
    """Precompute the STN drive trace for ``index`` (cached, read-only).

    Returns a length ``n_steps`` array on the ``0:dt_ms:tmax_ms`` grid, matching
    what the integrator expects for ``duration_s = step_duration_s``.
    """
    n_steps = _grid_steps(step_duration_s=step_duration_s, dt_ms=dt_ms)

    if mean_hz <= 0:
        trace = np.zeros(n_steps, dtype=np.float64)
        trace.setflags(write=False)
        return trace

    # Pattern 0: regular train — reuse the reference synthesizer for exact parity
    # with the scalar-frequency baseline / paper init target.
    regular = create_dbs_current(
        mean_hz,
        tmax_ms=step_duration_s * 1000.0,
        dt_ms=dt_ms,
        pulse_width_ms=pulse_width_ms,
        amplitude=amplitude,
    )
    if index == 0:
        regular.setflags(write=False)
        return regular

    pulse_len = int(round(pulse_width_ms / dt_ms))
    isi_steps = int(round((1000.0 / mean_hz) / dt_ms))
    onsets = np.arange(0, n_steps, isi_steps, dtype=np.int64)
    n_pulses = int(onsets.shape[0])

    # Deterministic per-(mean, index) irregular onsets. Pin the first and last
    # onset so the span (sum of ISIs) is preserved: mean rate stays exact.
    trace = np.zeros(n_steps, dtype=np.float64)
    if n_pulses <= 2:
        # Degenerate window: nothing to perturb, fall back to the regular train.
        regular.setflags(write=False)
        return regular

    rng = np.random.default_rng([int(round(mean_hz * 100)), index])
    max_jitter = max(1, int(isi_steps * jitter_fraction))
    jitter = rng.integers(-max_jitter, max_jitter + 1, size=n_pulses - 2)
    onsets[1:-1] = onsets[1:-1] + jitter
    onsets.sort()
    # Enforce a minimum separation of one pulse width so pulses never merge
    # (with the default jitter fraction this repair is essentially never hit).
    for i in range(1, n_pulses):
        floor = onsets[i - 1] + pulse_len
        if onsets[i] < floor:
            onsets[i] = floor

    for onset in onsets:
        start = int(onset)
        if start >= n_steps:
            break
        end = min(start + pulse_len, n_steps)
        trace[start:end] = amplitude

    trace.setflags(write=False)
    return trace


@dataclass(frozen=True)
class FixedMeanPatternAlphabet:
    """Discrete pattern action space at a fixed mean stimulation rate.

    Drop-in for :class:`envs.mehregan.patterns.PatternAlphabet`: exposes
    ``n_actions``, ``to_dbs_spec``, ``action_for_dbs_spec``, and
    ``action_for_frequency_hz``. Action ``i`` maps to the precomputed pattern
    trace ``i`` (pattern 0 = regular at ``mean_hz``).
    """

    mean_hz: float = DEFAULT_MEAN_STIM_HZ
    step_duration_s: float = 2.0
    dt_ms: float = 0.01
    n_patterns: int = 41
    skip_regular: bool = False  # True → exclude pattern 0 (regular periodic); agent sees only irregular patterns
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2
    jitter_fraction: float = DEFAULT_JITTER_FRACTION

    @property
    def n_actions(self) -> int:
        return self.n_patterns - 1 if self.skip_regular else self.n_patterns

    def _pattern_index(self, action: int) -> int:
        """Map agent action to internal pattern index.

        When ``skip_regular`` is True, agent action 0 maps to pattern 1
        (the first irregular pattern), skipping pattern 0 (regular periodic).
        """
        if self.skip_regular:
            return action + 1  # action 0 → pattern 1, action 39 → pattern 40
        return action

    def idbs_for_pattern(self, index: int) -> np.ndarray:
        """Return the (cached, read-only) STN drive trace for pattern ``index``.

        ``index`` is the **internal** pattern index (0 = regular, 1–40 = irregular).
        Use :meth:`idbs_for_action` for agent action → trace lookup.
        """
        if index < 0 or index >= self.n_patterns:
            msg = f"pattern index {index} outside [0, {self.n_patterns})"
            raise ValueError(msg)
        return _build_idbs(
            index=index,
            mean_hz=self.mean_hz,
            step_duration_s=self.step_duration_s,
            dt_ms=self.dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
            jitter_fraction=self.jitter_fraction,
        )

    def idbs_for_action(self, action: int) -> np.ndarray:
        """Return the STN drive trace for an agent action (respects ``skip_regular``)."""
        return self.idbs_for_pattern(self._pattern_index(action))

    def pulse_count(self, index: int) -> int:
        """Number of pulses (rising edges) in pattern ``index`` — the mean rate."""
        trace = self.idbs_for_pattern(index)
        active = trace > 0.0
        rising = np.concatenate(([active[0]], active[1:] & ~active[:-1]))
        return int(np.count_nonzero(rising))

    def to_dbs_spec(self, action: int) -> DbsSpec:
        idbs = self.idbs_for_action(int(action))
        return DbsSpec(
            pick_dbs_freq=DbsSpec.from_frequency_hz(self.mean_hz).pick_dbs_freq,
            idbs=idbs,
            mean_hz=self.mean_hz,
        )

    def action_for_dbs_spec(self, spec: DbsSpec) -> int:
        """Map a baseline periodic-at-mean spec to the regular-pattern action.

        When ``skip_regular`` is True, maps to the first irregular pattern (action 0)
        as a fallback for baseline initialization — the regular periodic pattern is
        not in the agent action space, but the trainer needs an init target.
        """
        if self.skip_regular:
            return 0  # first irregular pattern — best available init target
        return self.action_for_frequency_hz(spec.frequency_hz)

    def action_for_frequency_hz(self, hz: float) -> int:
        if abs(float(hz) - self.mean_hz) < 1e-6:
            return 0
        msg = (
            f"frequency {hz} Hz has no fixed-mean pattern (mean_hz={self.mean_hz}); "
            "only the mean-rate periodic baseline maps to a pattern (index 0)"
        )
        raise ValueError(msg)
