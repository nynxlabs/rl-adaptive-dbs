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
