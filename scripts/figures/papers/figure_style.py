"""One shared look for every paper panel's replication PNG.

Each ``plot.py`` calls :func:`apply` before it builds its figure, and every save goes
through :func:`finalize` (``paper_overlay`` hooks it into ``Figure.savefig``). Style
only: nothing here reads or changes data, gates, or manifests.

The PNGs are shown on light pages (GitHub light, ``#ffffff``) and dark ones (GitHub
dark, ``#0d1117``), so one image has to read on both:

- transparent figure and axes backgrounds;
- text, spines, ticks, and grid in a neutral mid-grey (:data:`NEUTRAL`);
- near-black artists a script drew (``"k"``, ``"#111111"``, paper overlays) become
  :data:`NEUTRAL` too, since black disappears on a dark page; near-white box fills
  become a translucent grey wash;
- legends are a neutral outline with no fill: each panel places its legend over
  empty plot area, so nothing needs hiding behind it;
- the semantic series colors (tab10 hues, paper-overlay tints) are left alone.
"""
from __future__ import annotations

from typing import Any

import matplotlib as mpl
from matplotlib import colors as mcolors

# Mid-grey: about 3.9:1 contrast on #ffffff and 5.3:1 on #0d1117.
NEUTRAL = "#808080"
GRID_ALPHA = 0.35
# Fill for light patches (text boxes, light bar fills): grey at low alpha, so it
# reads as a card on either page without hiding the page color.
WASH_RGBA = "#80808029"  # NEUTRAL at alpha 0.16
LEGEND_EDGE_RGBA = "#80808099"  # NEUTRAL at alpha 0.6
DPI = 150

RC: dict[str, Any] = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "savefig.transparent": True,
    "savefig.dpi": DPI,
    "text.color": NEUTRAL,
    "axes.labelcolor": NEUTRAL,
    "axes.edgecolor": NEUTRAL,
    "axes.titlecolor": NEUTRAL,
    "xtick.color": NEUTRAL,
    "ytick.color": NEUTRAL,
    "xtick.labelcolor": NEUTRAL,
    "ytick.labelcolor": NEUTRAL,
    "grid.color": NEUTRAL,
    "grid.alpha": GRID_ALPHA,
    "grid.linewidth": 0.6,
    "legend.facecolor": "none",
    "legend.edgecolor": LEGEND_EDGE_RGBA,
    "legend.labelcolor": NEUTRAL,
    "legend.framealpha": None,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8,
}


def apply(overrides: dict[str, Any] | None = None) -> None:
    """Set the shared rcParams (call before creating figures)."""
    mpl.rcParams.update(RC)
    if overrides:
        mpl.rcParams.update(overrides)


def _rgba(color: Any) -> tuple[float, float, float, float] | None:
    try:
        return mcolors.to_rgba(color)
    except (TypeError, ValueError):
        return None


def _is_dark_neutral(rgba: tuple[float, float, float, float] | None) -> bool:
    """Black / near-black grey (low chroma, low lightness)."""
    if rgba is None or rgba[3] == 0:
        return False
    r, g, b = rgba[:3]
    return max(r, g, b) <= 0.36 and max(r, g, b) - min(r, g, b) <= 0.08


def _is_light_neutral(rgba: tuple[float, float, float, float] | None) -> bool:
    """White / near-white grey."""
    if rgba is None or rgba[3] == 0:
        return False
    r, g, b = rgba[:3]
    return min(r, g, b) >= 0.85 and max(r, g, b) - min(r, g, b) <= 0.06


def _neutral_keep_alpha(rgba: tuple[float, float, float, float]) -> tuple[float, ...]:
    base = mcolors.to_rgba(NEUTRAL)
    return (*base[:3], rgba[3])


def _recolor_line(line) -> None:
    for get, set_ in (
        ("get_color", "set_color"),
        ("get_markerfacecolor", "set_markerfacecolor"),
        ("get_markeredgecolor", "set_markeredgecolor"),
    ):
        rgba = _rgba(getattr(line, get)())
        if _is_dark_neutral(rgba) or _is_light_grey_line(rgba):
            getattr(line, set_)(_neutral_keep_alpha(rgba))


def _recolor_text(text) -> None:
    rgba = _rgba(text.get_color())
    if _is_dark_neutral(rgba):
        text.set_color(_neutral_keep_alpha(rgba))
    box = text.get_bbox_patch()
    if box is not None:
        _recolor_patch(box)


def _recolor_patch(patch) -> None:
    face = _rgba(patch.get_facecolor())
    if _is_light_neutral(face):
        patch.set_facecolor(WASH_RGBA)
    elif _is_dark_neutral(face):
        patch.set_facecolor(_neutral_keep_alpha(face))
    edge = _rgba(patch.get_edgecolor())
    if _is_dark_neutral(edge):
        patch.set_edgecolor(_neutral_keep_alpha(edge))
    elif _is_light_neutral(edge) and patch.get_linewidth() > 0:
        patch.set_edgecolor(LEGEND_EDGE_RGBA)


def _recolor_collection(coll) -> None:
    for get, set_, light in (
        ("get_facecolor", "set_facecolor", WASH_RGBA),
        ("get_edgecolor", "set_edgecolor", None),
    ):
        try:
            arr = getattr(coll, get)()
        except Exception:
            continue
        if arr is None or len(arr) == 0:
            continue
        changed = False
        out = []
        for c in arr:
            rgba = tuple(float(v) for v in c)
            if _is_dark_neutral(rgba):
                out.append(_neutral_keep_alpha(rgba))
                changed = True
            elif light is not None and _is_light_neutral(rgba):
                out.append(light)
                changed = True
            else:
                out.append(rgba)
        if changed:
            getattr(coll, set_)(out)


def _is_light_grey_line(rgba: tuple[float, float, float, float] | None) -> bool:
    """Light grey reference/grid lines (``#cccccc`` and lighter): glaring on dark pages."""
    if rgba is None or rgba[3] == 0:
        return False
    r, g, b = rgba[:3]
    return min(r, g, b) >= 0.65 and max(r, g, b) - min(r, g, b) <= 0.06


def _style_axes(ax) -> None:
    ax.patch.set_facecolor("none")
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_color(NEUTRAL)
        line.set_alpha(GRID_ALPHA)
    for spine in ax.spines.values():
        if _is_dark_neutral(_rgba(spine.get_edgecolor())):
            spine.set_edgecolor(NEUTRAL)
    # Recolor only black ticks; a twin axis's colored ticks (e.g. red amplitude) stay.
    for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
        for which in ("major", "minor"):
            ticks = axis.get_major_ticks() if which == "major" else axis.get_minor_ticks()
            if not ticks:
                continue
            tick = ticks[0]
            if _is_dark_neutral(_rgba(tick.label1.get_color())) or _is_dark_neutral(
                _rgba(tick.label2.get_color())
            ):
                ax.tick_params(axis=name, which=which, labelcolor=NEUTRAL)
            if _is_dark_neutral(_rgba(tick.tick1line.get_color())):
                ax.tick_params(axis=name, which=which, color=NEUTRAL)
    for label in (ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title):
        rgba = _rgba(label.get_color())
        if _is_dark_neutral(rgba):
            label.set_color(NEUTRAL)
    leg = ax.get_legend()
    if leg is not None:
        style_legend(leg)


def style_legend(leg) -> None:
    """Unfilled legend frame with a neutral outline and text (reads on light and dark pages)."""
    frame = leg.get_frame()
    if frame is not None:
        frame.set_alpha(None)
        frame.set_facecolor("none")
        frame.set_edgecolor(LEGEND_EDGE_RGBA)
    for text in leg.get_texts():
        rgba = _rgba(text.get_color())
        if _is_dark_neutral(rgba):
            text.set_color(NEUTRAL)
    title = leg.get_title()
    if title is not None and _is_dark_neutral(_rgba(title.get_color())):
        title.set_color(NEUTRAL)
    for handle in getattr(leg, "legend_handles", None) or []:
        _recolor_artist(handle)


def _recolor_artist(artist) -> None:
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    if isinstance(artist, Line2D):
        _recolor_line(artist)
    elif isinstance(artist, Text):
        _recolor_text(artist)
    elif isinstance(artist, Patch):
        _recolor_patch(artist)
    elif isinstance(artist, Collection):
        _recolor_collection(artist)


def finalize(fig) -> None:
    """Make an already-drawn figure match the shared style (idempotent)."""
    from matplotlib.legend import Legend

    fig.patch.set_facecolor("none")
    fig.patch.set_edgecolor("none")
    for ax in fig.get_axes():
        for artist in ax.findobj(include_self=False):
            if artist is ax.patch or isinstance(artist, Legend):
                continue
            _recolor_artist(artist)
        _style_axes(ax)
    for leg in fig.legends:
        style_legend(leg)
    for text in fig.texts:
        _recolor_text(text)
    if fig._suptitle is not None:
        _recolor_text(fig._suptitle)
