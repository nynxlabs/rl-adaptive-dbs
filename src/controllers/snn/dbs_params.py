"""Continuous DBS parameter state with ternary deltas (Nguyen Eq. (5))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from controllers.snn.config import (
    INIT_AMPLITUDE_NA_PER_CM2,
    INIT_FREQUENCY_HZ,
    INIT_PULSE_WIDTH_MS,
    SNNConfig,
)
from envs.plant.dbs import DbsSpec, create_dbs_current


TERNARY_CHOICES: tuple[int, ...] = (-1, 0, 1)


@dataclass
class DBSParameterState:
    """Holds amplitude, frequency, and pulse width; applies ternary parameter deltas."""

    amplitude: float = INIT_AMPLITUDE_NA_PER_CM2
    frequency_hz: float = INIT_FREQUENCY_HZ
    pulse_width_ms: float = INIT_PULSE_WIDTH_MS

    @classmethod
    def from_config(cls, config: SNNConfig | None = None) -> DBSParameterState:
        del config  # episode resets always use the paper §IV triple
        return cls(
            amplitude=INIT_AMPLITUDE_NA_PER_CM2,
            frequency_hz=INIT_FREQUENCY_HZ,
            pulse_width_ms=INIT_PULSE_WIDTH_MS,
        )

    def copy(self) -> DBSParameterState:
        """Shallow copy of the current triple (for plant-guard rollback)."""
        return DBSParameterState(
            amplitude=self.amplitude,
            frequency_hz=self.frequency_hz,
            pulse_width_ms=self.pulse_width_ms,
        )

    def apply_delta(
        self,
        ternary_actions: np.ndarray | list[int],
        config: SNNConfig | None = None,
        *,
        epsilon: float | None = None,
        episode: int | None = None,
    ) -> DBSParameterState:
        """Apply three ternary actions in ``{-1, 0, 1}`` for (A, f, w)."""
        cfg = (config or SNNConfig()).with_variant_defaults()
        actions = np.asarray(ternary_actions, dtype=np.int64).reshape(-1)
        if actions.shape != (3,):
            msg = f"expected 3 ternary actions, got shape {actions.shape}"
            raise ValueError(msg)
        if not np.isin(actions, TERNARY_CHOICES).all():
            msg = f"ternary actions must be in {TERNARY_CHOICES}, got {actions.tolist()}"
            raise ValueError(msg)

        if epsilon is not None or episode is not None:
            eps_val = 0.0 if epsilon is None else epsilon
            freq_sens = cfg.frequency_sensitivity_at_epsilon(
                eps_val,
                episode=episode,
            )
            pw_sens = cfg.pulse_width_sensitivity_at_epsilon(
                eps_val,
                episode=episode,
            )
        else:
            freq_sens = cfg.frequency_sensitivity
            pw_sens = cfg.pulse_width_sensitivity

        self.amplitude = float(
            np.clip(
                self.amplitude + actions[0] * cfg.amplitude_sensitivity,
                cfg.amplitude_min,
                cfg.amplitude_max,
            )
        )
        self.frequency_hz = float(
            np.clip(
                self.frequency_hz + actions[1] * freq_sens,
                cfg.frequency_min,
                cfg.frequency_max,
            )
        )
        pw_min = cfg.pulse_width_min_at_episode(episode)
        self.pulse_width_ms = float(
            np.clip(
                self.pulse_width_ms + actions[2] * pw_sens,
                pw_min,
                cfg.pulse_width_max,
            )
        )
        return self

    def to_dbs_spec(self, *, duration_s: float, dt_ms: float = 0.01) -> DbsSpec:
        """Build a plant ``DbsSpec`` with a precomputed STN drive trace for this triple."""
        tmax_ms = duration_s * 1000.0
        if self.frequency_hz <= 0.0 or self.amplitude <= 0.0:
            return DbsSpec.none()
        idbs = create_dbs_current(
            self.frequency_hz,
            tmax_ms=tmax_ms,
            dt_ms=dt_ms,
            pulse_width_ms=self.pulse_width_ms,
            amplitude=self.amplitude,
        )
        return DbsSpec(
            pick_dbs_freq=2,
            idbs=idbs,
            mean_hz=self.frequency_hz,
        )
