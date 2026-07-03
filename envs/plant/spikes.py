"""Convert MATLAB spike structs to NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np


def spikes_from_matlab_cell(times_cell: Any, *, neurons: int | None = None) -> list[np.ndarray]:
    """Parse Kumaravelu spike times packed as a MATLAB 1×n cell array."""
    if hasattr(times_cell, "size"):
        size = int(times_cell.size)
        iterator = (times_cell[index] for index in range(size))
    else:
        cells = list(times_cell)
        size = len(cells)
        iterator = iter(cells)

    if neurons is not None and size != neurons:
        msg = f"expected {neurons} neurons, got {size}"
        raise ValueError(msg)

    spikes: list[np.ndarray] = []
    for times_raw in iterator:
        times = np.asarray(times_raw, dtype=float).reshape(-1)
        spikes.append(times)
    return spikes


def spikes_from_matlab_struct(
    ap_struct: Any,
    *,
    neurons: int | None = None,
) -> list[np.ndarray]:
    """Parse Kumaravelu find_spike_times output (struct array with field times)."""
    size = int(ap_struct.size)
    if neurons is not None and size != neurons:
        msg = f"expected {neurons} neurons, got {size}"
        raise ValueError(msg)

    spikes: list[np.ndarray] = []
    for index in range(size):
        times = np.asarray(ap_struct[index]["times"], dtype=float).reshape(-1)
        spikes.append(times)
    return spikes


def spike_counts(spikes: list[np.ndarray]) -> np.ndarray:
    return np.array([len(times) for times in spikes], dtype=int)


def find_spike_times(
    v: np.ndarray,
    t_ms: np.ndarray,
    n_neurons: int,
) -> list[np.ndarray]:
    """Port of Kumaravelu ``find_spike_times`` (upward crossing of -20 mV).

    ``v`` is shape ``(n_neurons, n_steps)``; ``t_ms`` is the simulation grid in ms.
    Returned spike times are in **seconds**, matching the MATLAB reference.
    """
    if v.shape[0] < n_neurons:
        msg = f"v has {v.shape[0]} rows, expected at least {n_neurons}"
        raise ValueError(msg)

    t_s = np.asarray(t_ms, dtype=np.float64) / 1000.0
    t_aligned = t_s[:-1]
    spikes: list[np.ndarray] = []
    threshold = -20.0
    for k in range(n_neurons):
        # MATLAB: diff(v>-20)==1 on doubles (logical promotes to 0/1).
        # np.diff(bool) is not equivalent — True→False repolarization yields True.
        below = v[k, :-1] <= threshold
        above = v[k, 1:] > threshold
        crossed = below & above
        spikes.append(t_aligned[crossed].copy())
    return spikes
