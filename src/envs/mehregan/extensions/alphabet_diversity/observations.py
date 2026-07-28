"""Mehregan observation construction (scalar, within-step, multi-step history)."""

from __future__ import annotations

import numpy as np

from envs.plant.biomarkers import p_beta


def clip_spikes_to_subwindow(
    gpi_spikes: list[np.ndarray],
    *,
    t_start_s: float,
    t_end_s: float,
) -> list[np.ndarray]:
    """Keep spikes in [t_start_s, t_end_s) and rebase times to start at 0."""
    clipped: list[np.ndarray] = []
    for spikes in gpi_spikes:
        arr = np.asarray(spikes, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            clipped.append(arr)
            continue
        mask = (arr >= t_start_s) & (arr < t_end_s)
        sub = arr[mask] - t_start_s
        clipped.append(sub)
    return clipped


def within_step_p_beta_series(
    gpi_spikes: list[np.ndarray],
    *,
    segment_duration_s: float,
    state_length: int,
    dt_ms: float,
) -> np.ndarray:
    """Partition one RL step into ``state_length`` equal sub-windows; Pβ each.

    Sub-windows are contiguous, non-overlapping, and cover [0, segment_duration_s).
  Each sub-window is passed to :func:`p_beta` with ``segment_duration_s=sub_dur``
    and spike times rebased to start at 0 (Mehregan §III.B temporal CNN input).
    """
    if state_length < 1:
        msg = f"state_length must be >= 1, got {state_length}"
        raise ValueError(msg)
    if segment_duration_s <= 0:
        msg = f"segment_duration_s must be > 0, got {segment_duration_s}"
        raise ValueError(msg)

    sub_dur = segment_duration_s / state_length
    values = np.empty(state_length, dtype=np.float64)
    for k in range(state_length):
        t0 = k * sub_dur
        t1 = (k + 1) * sub_dur
        window_spikes = clip_spikes_to_subwindow(
            gpi_spikes, t_start_s=t0, t_end_s=t1
        )
        values[k] = p_beta(
            window_spikes,
            dt_ms=dt_ms,
            segment_duration_s=sub_dur,
        )
    return values
