"""DBS stimulation energy index (Nguyen Eq. (6))."""

from __future__ import annotations

import numpy as np

from envs.plant.dbs import create_dbs_current


def dbs_energy_index(
    *,
    frequency_hz: float,
    amplitude: float,
    pulse_width_ms: float,
    step_duration_s: float,
    stimulated_neurons: int,
    dt_ms: float = 0.01,
) -> float:
    """Per-step energy index ``E_t = N * sqrt(mean(I_DBS^2))`` over the RL step."""
    if step_duration_s <= 0:
        msg = "step_duration_s must be positive"
        raise ValueError(msg)
    if stimulated_neurons < 1:
        msg = "stimulated_neurons must be at least 1"
        raise ValueError(msg)

    tmax_ms = step_duration_s * 1000.0
    current = create_dbs_current(
        frequency_hz,
        tmax_ms=tmax_ms,
        dt_ms=dt_ms,
        pulse_width_ms=pulse_width_ms,
        amplitude=amplitude,
    )
    mean_square = float(np.mean(current**2))
    return float(stimulated_neurons * np.sqrt(mean_square))
