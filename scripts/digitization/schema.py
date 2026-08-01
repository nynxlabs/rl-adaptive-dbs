"""Shared series stats for the digitization schema (curves.json).

Computes the fields documented in the rl-paper-replication skill
reference `digitization-schema.md`:

    n, start, end, early_mean, late_mean, drop_early_to_late,
    min, max, mean, at (sparse index -> y), slope

Conventions (documented so all extraction paths agree):
- early mean: first `early_frac` of samples (min 3 samples)
- late mean: last `late_frac` of samples (default: second half)
- slope: linear fit over the whole series
- `at`: sparse map at fixed proportions [0, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0]
  mapped to nearest sample index
"""
from __future__ import annotations

import numpy as np

EARLY_FRAC = 0.25
LATE_FRAC = 0.5
AT_PROPS = (0.0, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0)


def series_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute the digitization schema fields for one (x, y) series."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n = int(x.size)
    if n < 4:
        raise ValueError(f"series too short for stats: n={n}")

    early_n = max(3, int(n * EARLY_FRAC))
    late_n = max(3, int(n * LATE_FRAC))
    early_mean = float(np.mean(y[:early_n]))
    late_mean = float(np.mean(y[-late_n:]))

    # slope: linear fit y ~ a + b*x
    slope = float(np.polyfit(x, y, 1)[0])

    at: dict[str, float] = {}
    for prop in AT_PROPS:
        idx = int(round((n - 1) * prop))
        at[str(idx)] = float(y[idx])

    return {
        "n": n,
        "start": float(y[0]),
        "end": float(y[-1]),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "drop_early_to_late": early_mean - late_mean,
        "min": float(y.min()),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "at": at,
        "slope": slope,
    }
