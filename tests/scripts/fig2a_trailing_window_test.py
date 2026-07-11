"""Fig 2a trailing-window edge cases (display axis)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PLOT_PATH = ROOT / "scripts/figures/papers/1/2a/plot.py"
_spec = importlib.util.spec_from_file_location("fig2a_plot", _PLOT_PATH)
assert _spec and _spec.loader
fig2a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig2a)


def test_trailing_window_uses_preroll_at_display_zero() -> None:
    start, end = fig2a.trailing_window_sim(0.0)
    assert (start, end) == (0.0, fig2a.WARMUP_S)


def test_trailing_window_before_dbs_onset_on_display_axis() -> None:
    start, end = fig2a.trailing_window_sim(fig2a.DBS_ONSET_S)
    assert (start, end) == (fig2a.WARMUP_S, fig2a.DBS_ONSET_SIM)


def test_trailing_window_ends_on_sim_fourteen_at_display_twelve() -> None:
    start, end = fig2a.trailing_window_sim(fig2a.DISPLAY_S)
    assert end == fig2a.BIOMARKER_SIM_END_MAX
    assert end == fig2a.WARMUP_S + fig2a.DISPLAY_S
    assert (start, end) == (12.0, 14.0)


def test_dbs_onset_sim_is_warmup_plus_display_onset() -> None:
    assert fig2a.DBS_ONSET_SIM == fig2a.WARMUP_S + fig2a.DBS_ONSET_S


def test_sample_times_cover_display_axis() -> None:
    times = fig2a.sample_times(0.2)
    assert times[0] == 0.0
    assert times[-1] == fig2a.DISPLAY_S
    assert times.size == 61


def test_fig2a_gpi_spike_buffer_exceeds_numba_default() -> None:
    assert fig2a.fig2a_gpi_spike_buffer_size(integrate_s=fig2a.INTEGRATE_S) > 512
