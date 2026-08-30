"""Tests for scripts/figures/papers/parallel_series.py."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from figures.papers.parallel_series import (  # noqa: E402
    default_parallel_series_count,
    resolve_parallel_series,
)


def test_default_parallel_series_count() -> None:
    cpus = os.cpu_count() or 1
    assert default_parallel_series_count(0) == 1
    assert default_parallel_series_count(4) == min(4, cpus)


def test_resolve_parallel_series_auto() -> None:
    cpus = os.cpu_count() or 1
    assert resolve_parallel_series(0, 4) == min(4, cpus)
    assert resolve_parallel_series(-1, 2) == min(2, cpus)


def test_resolve_parallel_series_explicit() -> None:
    assert resolve_parallel_series(1, 4) == 1
    assert resolve_parallel_series(3, 2) == 2
    assert resolve_parallel_series(8, 4) == 4
