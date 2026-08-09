"""Overlay digitized paper curves on replication training plots.

Loads normalized ``curves_*.json`` under
``artifacts/figures/papers/<paper>/paper_digitization/`` (same sources as
``scripts/digitization/*_gates.py``) and draws them on matplotlib axes next to
replication traces for visual gate iteration.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_DIG = Path(__file__).resolve().parents[2] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import load_refined  # noqa: E402

NGUYEN_DIG = Path("artifacts/figures/papers/nguyen/paper_digitization")
RAVIVARAPU_DIG = Path("artifacts/figures/papers/ravivarapu/paper_digitization")

# Ravivarapu Fig 4a: Baseline black, SEA-DBS light grey (replication solid; paper dashed/dotted).
RAVI_BASELINE_COLOR = "#000000"
RAVI_SEA_COLOR = "#c8c8c8"

PAPER_SMOOTH_STYLE: dict[str, Any] = {
    "color": "#1b4332",
    "linestyle": "--",
    "linewidth": 2.0,
    "alpha": 0.92,
    "zorder": 10,
}
PAPER_RAW_STYLE: dict[str, Any] = {
    "color": "#7f8c8d",
    "linestyle": "--",
    "linewidth": 1.0,
    "alpha": 0.65,
    "zorder": 9,
}


def load_panel_curves(path: Path | str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load ``{series_name: (x, y)}`` from a digitization JSON path."""
    return load_refined(path)


def pick_series(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *preferred: str,
) -> tuple[np.ndarray, np.ndarray]:
    for name in preferred:
        if name in curves:
            return curves[name]
    if not curves:
        msg = "empty digitization curves"
        raise KeyError(msg)
    key = next(iter(curves))
    return curves[key]


def overlay_on_axis(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    raw: bool = False,
    color: str | None = None,
    linestyle: str | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
) -> None:
    style = dict(PAPER_RAW_STYLE if raw else PAPER_SMOOTH_STYLE)
    if color is not None:
        style["color"] = color
    if linestyle is not None:
        style["linestyle"] = linestyle
    if linewidth is not None:
        style["linewidth"] = linewidth
    if alpha is not None:
        style["alpha"] = alpha
    ax.plot(x, y, label=label, **style)


def nguyen_fig4_digitization() -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    """Reward and length digitization for Nguyen Fig. 4 (episode index on x)."""
    reward = load_panel_curves(NGUYEN_DIG / "curves_fig4_reward.json")
    length = load_panel_curves(NGUYEN_DIG / "curves_fig4_length.json")
    return reward, length


def overlay_nguyen_fig4(
    ax_reward,
    ax_length,
    *,
    show_paper_raw: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Draw paper reward/length on the two panel axes; return y arrays for ylim."""
    reward_curves, length_curves = nguyen_fig4_digitization()
    prx, pry = pick_series(reward_curves, "Smoothed", "Raw")
    plx, ply = pick_series(length_curves, "Smoothed", "Raw")
    if show_paper_raw and "Raw" in reward_curves:
        rx, ry = reward_curves["Raw"]
        overlay_on_axis(ax_reward, rx, ry, label="Paper raw (digitized)", raw=True)
    if show_paper_raw and "Raw" in length_curves:
        lx, ly = length_curves["Raw"]
        overlay_on_axis(ax_length, lx, ly, label="Paper raw (digitized)", raw=True)
    overlay_on_axis(ax_reward, prx, pry, label="Paper smoothed (digitized)")
    overlay_on_axis(ax_length, plx, ply, label="Paper smoothed (digitized)")
    r_raw_y = reward_curves["Raw"][1] if "Raw" in reward_curves else pry
    l_raw_y = length_curves["Raw"][1] if "Raw" in length_curves else ply
    return {"reward": (pry, r_raw_y), "length": (ply, l_raw_y)}


def ravivarapu_fig4a_digitization() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Baseline and SEA-DBS PSD vs episode from ``curves_fig4a.json``."""
    return load_panel_curves(RAVIVARAPU_DIG / "curves_fig4a.json")


def overlay_ravivarapu_fig4a(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Draw digitized Baseline / SEA-DBS on the Fig 4a axis (1-based episode x)."""
    curves = ravivarapu_fig4a_digitization()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for series_name, repl_label, paper_color, paper_ls in (
        ("Baseline", "Paper Baseline (digitized)", RAVI_BASELINE_COLOR, "--"),
        ("SEA-DBS", "Paper SEA-DBS (digitized)", RAVI_SEA_COLOR, ":"),
    ):
        px, py = pick_series(curves, series_name)
        # WPD x is 0-based episode index; replication plots use 1..n.
        px_plot = np.asarray(px, dtype=float) + 1.0
        overlay_on_axis(
            ax,
            px_plot,
            py,
            label=repl_label,
            color=paper_color,
            linestyle=paper_ls,
            linewidth=2.4 if series_name == "SEA-DBS" else 2.0,
            alpha=0.95 if series_name == "SEA-DBS" else 0.85,
        )
        out[series_name] = (py, py)
    return out
