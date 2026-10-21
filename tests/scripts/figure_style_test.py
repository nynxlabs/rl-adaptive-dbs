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


def test_finalize_recolors_black_and_maps_series_to_pastel() -> None:
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
    assert to_hex(blue.get_color()) == _style.PASTEL["blue"]
    assert to_rgba(grid_ref.get_color())[:3] == neutral[:3]
    assert to_rgba(text.get_color())[:3] == neutral[:3]
    assert to_rgba(ax.xaxis.label.get_color())[:3] == neutral[:3]
    assert to_rgba(leg.get_frame().get_facecolor())[3] == 0  # unfilled outline
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


def test_finalize_keeps_colored_twin_axis_ticks() -> None:
    _overlay.apply_figure_style()
    fig, ax = plt.subplots()
    twin = ax.twinx()
    ax.plot([0, 1], [0, 1])
    twin.plot([0, 1], [1, 0], color="#d62728")
    twin.tick_params(axis="y", labelcolor="#d62728")
    ax.tick_params(axis="x", labelcolor="black")
    fig.canvas.draw()
    _style.finalize(fig)
    fig.canvas.draw()
    assert to_hex(twin.yaxis.get_major_ticks()[0].label2.get_color()) == _style.PASTEL["red"]
    assert to_hex(ax.xaxis.get_major_ticks()[0].label1.get_color()) == _style.NEUTRAL
    plt.close(fig)


def test_raw_tint_follows_its_series_at_half_alpha() -> None:
    _overlay.apply_figure_style()
    fig, ax = plt.subplots()
    raw = ax.plot([0, 1], [0, 1], color="#9ecae1", alpha=0.8, label="Raw")[0]
    smooth = ax.plot([0, 1], [1, 0], color="#08519c", linewidth=2.0, label="Smoothed")[0]
    odd = ax.plot([0, 1], [0.5, 0.5], color="#e6194b", label="Other red")[0]
    _style.finalize(fig)
    _style.finalize(fig)  # idempotent: no second remap or widening
    blue = to_rgba(_style.PASTEL["blue"])
    assert to_rgba(raw.get_color())[:3] == blue[:3]
    assert abs(raw.get_alpha() - 0.8 * _style.RAW_TINT_ALPHA) < 1e-6
    assert to_hex(smooth.get_color()) == _style.PASTEL["blue"]
    assert abs(smooth.get_linewidth() - 2.0 * _style.LINE_SCALE) < 1e-6
    assert to_hex(odd.get_color()) == _style.PASTEL["red"]  # nearest-hue fallback
    plt.close(fig)


def test_unlisted_colors_get_distinct_pastels() -> None:
    _overlay.apply_figure_style()
    fig, ax = plt.subplots()
    a = ax.plot([0, 1], [0, 1], color="#e8901c", label="paper orange")[0]
    b = ax.plot([0, 1], [1, 0], color="#d9a441", label="paper tan")[0]
    _style.finalize(fig)
    assert to_hex(a.get_color()) != to_hex(b.get_color())
    assert {to_hex(a.get_color()), to_hex(b.get_color())} <= set(_style.PASTEL.values())
    plt.close(fig)
