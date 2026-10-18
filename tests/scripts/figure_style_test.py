"""Tests for the shared panel style (scripts/figures/papers/figure_style.py)."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402
from PIL import Image  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_PAPERS = _ROOT / "scripts" / "figures" / "papers"
if str(_PAPERS.parent.parent) not in sys.path:
    sys.path.insert(0, str(_PAPERS.parent.parent))

from figures.papers.overlay_import import load_paper_overlay  # noqa: E402

_overlay = load_paper_overlay()
_style = _overlay.figure_style


def _render(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def test_saved_png_has_transparent_background() -> None:
    _overlay.apply_figure_style()
    fig, ax = plt.subplots(figsize=(2, 2), facecolor="white")
    ax.set_facecolor("white")
    ax.plot([0, 1], [0, 1], color="#1f77b4")
    img = np.asarray(_render(fig))
    assert img[0, 0, 3] == 0  # figure margin
    assert img[img.shape[0] // 2, img.shape[1] // 3, 3] == 0  # inside the axes, off the line


def test_finalize_recolors_black_but_keeps_series_colors() -> None:
    _overlay.apply_figure_style()
    fig, ax = plt.subplots()
    black = ax.plot([0, 1], [0, 1], color="k", alpha=0.5)[0]
    blue = ax.plot([0, 1], [1, 0], color="#1f77b4")[0]
    grid_ref = ax.axhline(0.5, color="#cccccc")
    text = ax.text(0.5, 0.5, "x", color="#111111")
    ax.set_xlabel("time", color="black")
    leg = ax.legend(["a", "b"], facecolor="white")
    _style.finalize(fig)
    neutral = to_rgba(_style.NEUTRAL)
    assert to_rgba(black.get_color())[:3] == neutral[:3]
    assert black.get_alpha() == 0.5  # keeps the trace's own alpha
    assert to_hex(blue.get_color()) == "#1f77b4"
    assert to_rgba(grid_ref.get_color())[:3] == neutral[:3]
    assert to_rgba(text.get_color())[:3] == neutral[:3]
    assert to_rgba(ax.xaxis.label.get_color())[:3] == neutral[:3]
    assert to_rgba(leg.get_frame().get_facecolor()) == to_rgba(_style.WASH_RGBA)
    assert to_rgba(fig.patch.get_facecolor())[3] == 0
    plt.close(fig)


def test_paper_ghost_matches_old_white_blend_on_white() -> None:
    """Translucent paper tints look like the old blend-toward-white on a white page."""
    for hex_color in ("#1f77b4", "#d62728", "#2ca02c", "#1f6f6f"):
        ghost = to_rgba(_overlay.paper_color_from_outline(hex_color))
        on_white = [c * ghost[3] + (1.0 - ghost[3]) for c in ghost[:3]]
        old = to_rgba(_overlay.lighten_color(hex_color))
        assert np.allclose(on_white, old[:3], atol=2 / 255)


def test_with_alpha_multiplies_existing_alpha() -> None:
    assert abs(to_rgba(_overlay.with_alpha("#80808080", 0.5))[3] - 0x80 / 255 * 0.5) <= 1 / 255
