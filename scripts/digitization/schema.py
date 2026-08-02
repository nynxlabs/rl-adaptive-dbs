"""Shared series record for the digitization schema (curves.json).

Intentionally thin: keep the full trace and optional display color.
Do **not** add convenience stats here (early/late means, slope, sparse
``at``, etc.) — those mislead gate authors because they are sample-index
based, not x-axis based. Gates should slice ``xy`` by real x (Hz, sec, …).
"""
from __future__ import annotations

import numpy as np


def series_record(
    x: np.ndarray,
    y: np.ndarray,
    *,
    color_rgba: list[int] | None = None,
) -> dict:
    """Build one series entry: ``n``, ``xy``, optional ``color_rgba``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n = int(x.size)
    if n < 1:
        raise ValueError("series is empty")
    out: dict = {
        "n": n,
        "xy": {"x": x.tolist(), "y": y.tolist()},
    }
    if color_rgba is not None:
        out["color_rgba"] = [int(c) for c in color_rgba]
    return out
