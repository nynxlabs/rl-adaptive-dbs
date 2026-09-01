"""Shared axis helpers for paper figure plot scripts."""
from __future__ import annotations

import numpy as np


def data_ylim(
    *arrays: np.ndarray,
    pad_frac: float | tuple[float, float] = 0.05,
    pad_top: float | None = None,
    integer_snap: bool = False,
    extra_values: tuple[float, ...] | list[float] = (),
) -> tuple[float, float]:
    """Y limits that include every finite sample, optional reference values, plus padding."""
    parts: list[np.ndarray] = []
    for arr in arrays:
        y = np.asarray(arr, dtype=float).ravel()
        if y.size:
            finite = y[np.isfinite(y)]
            if finite.size:
                parts.append(finite)
    for value in extra_values:
        if np.isfinite(value):
            parts.append(np.array([float(value)]))
    if not parts:
        return 0.0, 1.0
    y = np.concatenate(parts)
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    rng = y_max - y_min
    if isinstance(pad_frac, (tuple, list)):
        pad_lo = float(pad_frac[0]) * rng
        pad_hi = float(pad_frac[1]) * rng
    else:
        pad_lo = float(pad_frac) * rng
        pad_hi = float(pad_top) * rng if pad_top is not None else float(pad_frac) * rng
    lo = y_min - pad_lo
    hi = y_max + pad_hi
    if integer_snap:
        lo = float(np.floor(lo))
        hi = float(np.ceil(hi))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi
