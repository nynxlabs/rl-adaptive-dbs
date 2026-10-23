"""One shared look for every paper panel's replication PNG.

Each ``plot.py`` calls :func:`apply` before it builds its figure, and every save goes
through :func:`finalize` and :func:`save_themed` (``paper_overlay`` hooks both into
``Figure.savefig``). Style only: nothing here reads or changes data, gates, or
manifests.

Every PNG is saved twice, once per page theme, and the docs show the pair with
``<picture>`` so readers see the one that matches their light or dark theme:

- ``{stem}_vN.png`` — the **light** variant (the fallback everywhere): black ink
  (:data:`INK` ``"light"``) and Catppuccin Latte series colors (:data:`LATTE`);
- ``{stem}_vN_dark.png`` — the **dark** variant (:func:`dark_path`): ink ``#e6e6e6``
  and Catppuccin Frappé pastels (:data:`PASTEL`).

Both variants keep transparent figure and axes backgrounds and the same geometry
(line widths included), so they differ only in color. Drawing happens once, in a
theme-neutral *source* style that :func:`finalize` sets up:

- text, spines, ticks, and grid in :data:`NEUTRAL`; near-black artists a script drew
  (``"k"``, ``"#111111"``, paper overlays) become :data:`NEUTRAL` too, and near-white
  box fills become a translucent :data:`NEUTRAL` wash;
- legends are an unfilled outline: each panel places its legend over empty plot
  area, so nothing needs hiding behind it;
- series colors become one pastel palette (:data:`PASTEL`): each hue a script picks
  (tab10, ColorBrewer, the paper-overlay tints) maps to the pastel of the same role
  (blue stays blue, red stays red), and a pale "raw trace" tint maps to that pastel at
  half alpha so the smoothed line stays the stronger one. Lines get
  :data:`LINE_SCALE` wider.

:func:`apply_theme` then maps that source style to a theme: :data:`NEUTRAL` becomes
the theme's ink (keeping each artist's alpha, so grid, wash, and legend outline stay
as faint as before) and each pastel becomes the same role in the theme's palette.
"""
from __future__ import annotations

import colorsys
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
from cycler import cycler
from matplotlib import colors as mcolors

# Source ink every figure is drawn in; apply_theme swaps it for the theme's INK.
NEUTRAL = "#808080"
GRID_ALPHA = 0.35
# Fill for light patches (text boxes, light bar fills): grey at low alpha, so it
# reads as a card on either page without hiding the page color.
WASH_RGBA = "#80808029"  # NEUTRAL at alpha 0.16
LEGEND_EDGE_RGBA = "#80808099"  # NEUTRAL at alpha 0.6
DPI = 150

# Catppuccin Frappé accents: the source palette and the dark variant's colors.
PASTEL = {
    "blue": "#8caaee",
    "peach": "#ef9f76",
    "green": "#a6d189",
    "red": "#e78284",
    "mauve": "#ca9ee6",
    "teal": "#81c8be",
    "yellow": "#e5c890",
    "pink": "#f4b8e4",
}
# Catppuccin Latte accents (deeper, made for white pages): the light variant's colors,
# same roles as PASTEL.
LATTE = {
    "blue": "#1e66f5",
    "peach": "#fe640b",
    "green": "#40a02b",
    "red": "#d20f39",
    "mauve": "#8839ef",
    "teal": "#179299",
    "yellow": "#df8e1d",
    "pink": "#ea76cb",
}
THEMES = ("light", "dark")
# Text, spines, ticks, grid, and legend ink per theme: the nynxlabs site's own text
# colors (its page backgrounds are #ffffff / #000000).
INK = {"light": "#000000", "dark": "#e6e6e6"}
PALETTE = {"light": LATTE, "dark": PASTEL}
DARK_SUFFIX = "_dark"
# Hues the panel scripts use, by role. Anything else saturated falls back to the
# pastel of nearest hue (see _pastel_for).
SERIES_PASTEL = {
    **dict.fromkeys(
        ("#1f77b4", "#0a64c0", "#08519c", "#377eb8", "#3756e1", "#1f3fd1"), PASTEL["blue"]
    ),
    **dict.fromkeys(("#ff7f0e", "#d95f02", "#e8731a"), PASTEL["peach"]),
    **dict.fromkeys(
        ("#2ca02c", "#16a34a", "#4daf4a", "#2ecc40", "#1b7837"), PASTEL["green"]
    ),
    **dict.fromkeys(("#d62728", "#e41a1c", "#c81219", "#a50f15"), PASTEL["red"]),
    **dict.fromkeys(
        ("#9467bd", "#8336e2", "#4a148c", "#9938b1", "#7b2d8e", "#7b6ba8"), PASTEL["mauve"]
    ),
    "#1f6f6f": PASTEL["teal"],
    "#bcbd22": PASTEL["mauve"],  # yellow is too faint on white for a main series
}
# Pale tints scripts use for raw traces under a smoothed line: the pastel at half alpha.
RAW_TINT_PASTEL = {
    "#9ecae1": PASTEL["blue"],
    **dict.fromkeys(("#fcbba1", "#fb9a99"), PASTEL["red"]),
    **dict.fromkeys(("#a1d99b", "#b2df8a"), PASTEL["green"]),
}
RAW_TINT_ALPHA = 0.5
# Paper overlays are dashed and drawn as pale tints of the series color; a pale tint of
# a pastel vanishes on white, so dashed lines keep at least this alpha instead.
DASHED_MIN_ALPHA = 0.85
LINE_SCALE = 1.2

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
    "axes.prop_cycle": cycler(color=list(PASTEL.values())),
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


_PASTEL_HUES = [(colorsys.rgb_to_hls(*mcolors.to_rgb(c))[0], c) for c in PASTEL.values()]
# Per-figure picks for colors not in the tables (set by _pastelize_figure).
_figure_picks: dict[str, str] = {}


def _hue_gap(a: float, b: float) -> float:
    return min(abs(a - b), 1 - abs(a - b))


def _is_other_series(rgba) -> bool:
    """Saturated color outside the tables (e.g. sampled from a digitized paper figure)."""
    if rgba is None or rgba[3] == 0:
        return False
    key = mcolors.to_hex(rgba[:3])
    if key in SERIES_PASTEL or key in RAW_TINT_PASTEL or key in PASTEL.values():
        return False
    h, l, sat = colorsys.rgb_to_hls(*rgba[:3])
    return sat >= 0.25 and max(rgba[:3]) - min(rgba[:3]) >= 0.12 and l <= 0.72


def _pastel_for(rgba: tuple[float, float, float, float] | None) -> tuple[float, ...] | None:
    """Pastel replacement for a series color, or ``None`` to leave it (neutrals, pastels)."""
    if rgba is None or rgba[3] == 0:
        return None
    key = mcolors.to_hex(rgba[:3])
    if key in SERIES_PASTEL:
        return (*mcolors.to_rgb(SERIES_PASTEL[key]), rgba[3])
    if key in RAW_TINT_PASTEL:
        return (*mcolors.to_rgb(RAW_TINT_PASTEL[key]), rgba[3] * RAW_TINT_ALPHA)
    if key in PASTEL.values():
        return None
    h, l, sat = colorsys.rgb_to_hls(*rgba[:3])
    if sat < 0.25 or max(rgba[:3]) - min(rgba[:3]) < 0.12:
        return None
    hue = _figure_picks.get(key) or min(_PASTEL_HUES, key=lambda hc: _hue_gap(h, hc[0]))[1]
    alpha = rgba[3] * (RAW_TINT_ALPHA if l > 0.72 else 1.0)
    return (*mcolors.to_rgb(hue), alpha)


def _pastelize(artist) -> None:
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    dimmed = False

    def swap(get, set_):
        nonlocal dimmed
        old = _rgba(getattr(artist, get)())
        new = _pastel_for(old)
        if new is None:
            return
        getattr(artist, set_)(new)
        # An artist-level alpha overrides the color's own, so dim it there too (once).
        if new[3] < old[3] - 1e-9 and artist.get_alpha() is not None and not dimmed:
            artist.set_alpha(artist.get_alpha() * RAW_TINT_ALPHA)
            dimmed = True

    if isinstance(artist, Line2D) and artist.get_linestyle() not in ("-", "solid", "None", "none", ""):
        # Paper overlay: same pastel as its series, told apart by the dash, not by fading.
        old = _rgba(artist.get_color())
        new = _pastel_for(old)
        if new is not None:
            artist.set_color((*new[:3], max(old[3], DASHED_MIN_ALPHA)))
            if artist.get_alpha() is not None:
                artist.set_alpha(max(artist.get_alpha(), DASHED_MIN_ALPHA))
        for pair in (("get_markerfacecolor", "set_markerfacecolor"), ("get_markeredgecolor", "set_markeredgecolor")):
            swap(*pair)
    elif isinstance(artist, Line2D):
        for pair in (
            ("get_color", "set_color"),
            ("get_markerfacecolor", "set_markerfacecolor"),
            ("get_markeredgecolor", "set_markeredgecolor"),
        ):
            swap(*pair)
    elif isinstance(artist, Text):
        swap("get_color", "set_color")
    elif isinstance(artist, Patch):
        swap("get_facecolor", "set_facecolor")
        swap("get_edgecolor", "set_edgecolor")
    elif isinstance(artist, Collection):
        for get, set_ in (("get_facecolor", "set_facecolor"), ("get_edgecolor", "set_edgecolor")):
            try:
                arr = getattr(artist, get)()
            except Exception:
                continue
            if arr is None or len(arr) == 0:
                continue
            out = [_pastel_for(tuple(float(v) for v in c)) or tuple(c) for c in arr]
            getattr(artist, set_)(out)


def _pastelize_figure(fig) -> None:
    """Map every series color to :data:`PASTEL` and widen lines (once per figure)."""
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D

    if getattr(fig, "_pastelized", False):
        return
    fig._pastelized = True
    seen: set[int] = set()
    artists = []
    for ax in fig.get_axes():
        artists += ax.findobj(include_self=False)
        for axis in (ax.xaxis, ax.yaxis):
            artists += axis.get_ticklabels() + [axis.label]
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                artists += [tick.label1, tick.label2, tick.tick1line, tick.tick2line]
        artists += list(ax.spines.values())
    for leg in fig.legends:
        artists += leg.findobj()
    # Distinct colors outside the tables get distinct pastels: nearest hue first, skipping
    # pastels the figure already uses, so two paper oranges don't both turn yellow.
    from matplotlib.collections import Collection
    from matplotlib.patches import Patch

    used = set()
    others: dict[str, tuple[float, ...]] = {}
    for artist in artists:
        if isinstance(artist, Line2D):
            colors = [_rgba(artist.get_color())]
        elif isinstance(artist, Patch):
            colors = [_rgba(artist.get_facecolor())]
        elif isinstance(artist, Collection):
            try:
                colors = [tuple(float(v) for v in c) for c in artist.get_facecolor()]
            except Exception:
                colors = []
        else:
            continue
        for rgba in colors:
            if rgba is None or rgba[3] == 0:
                continue
            key = mcolors.to_hex(rgba[:3])
            if key in SERIES_PASTEL:
                used.add(SERIES_PASTEL[key])
            elif _is_other_series(rgba):
                others.setdefault(key, rgba)
    _figure_picks.clear()
    for key, rgba in sorted(others.items(), key=lambda kv: -colorsys.rgb_to_hls(*kv[1][:3])[2]):
        h = colorsys.rgb_to_hls(*rgba[:3])[0]
        ranked = sorted(_PASTEL_HUES, key=lambda hc: _hue_gap(h, hc[0]))
        pick = next((c for _, c in ranked if c not in used), ranked[0][1])
        _figure_picks[key] = pick
        used.add(pick)
    for artist in artists:
        if id(artist) in seen or isinstance(artist, Legend):
            continue
        seen.add(id(artist))
        _pastelize(artist)
        if isinstance(artist, Line2D) and artist.get_label():  # data lines; grid/tick lines have no label
            artist.set_linewidth(artist.get_linewidth() * LINE_SCALE)
    # Colored twin-axis tick labels: set through tick_params so new ticks keep the color.
    for ax in fig.get_axes():
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            ticks = axis.get_major_ticks()
            if not ticks:
                continue
            for label in (ticks[0].label1, ticks[0].label2):
                new = _pastel_for(_rgba(label.get_color()))
                if new is not None:
                    ax.tick_params(axis=name, labelcolor=new)
            new = _pastel_for(_rgba(ticks[0].tick1line.get_color()))
            if new is not None:
                ax.tick_params(axis=name, color=new)


def finalize(fig) -> None:
    """Make an already-drawn figure match the shared style (idempotent)."""
    from matplotlib.legend import Legend

    _pastelize_figure(fig)
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


# Every theme's ink and palette color back to the source color it came from, so a
# figure can switch themes any number of times (light, then dark, then light again).
_ROLE_OF = {hexc: role for role, hexc in PASTEL.items()}
_SOURCE_OF: dict[str, str] = {NEUTRAL: NEUTRAL, **{c: c for c in PASTEL.values()}}
for _theme in THEMES:
    _SOURCE_OF[INK[_theme]] = NEUTRAL
    for _role, _hex in PALETTE[_theme].items():
        _SOURCE_OF[_hex] = PASTEL[_role]


def theme_rgba(rgba: tuple[float, ...] | None, theme: str) -> tuple[float, ...] | None:
    """``rgba`` in ``theme`` (ink or palette swap, alpha kept), or ``None`` to leave it."""
    if rgba is None or rgba[3] == 0:
        return None
    key = mcolors.to_hex(rgba[:3])
    source = _SOURCE_OF.get(key)
    if source is None:
        return None
    target = INK[theme] if source == NEUTRAL else PALETTE[theme][_ROLE_OF[source]]
    if target == key:
        return None
    return (*mcolors.to_rgb(target), rgba[3])


def theme_rc(theme: str) -> dict[str, Any]:
    """rcParams for artists matplotlib creates while saving (e.g. new ticks)."""
    ink = INK[theme]
    return {
        key: ink
        for key, value in RC.items()
        if isinstance(value, str) and value == NEUTRAL
    } | {
        "legend.edgecolor": (*mcolors.to_rgb(ink), mcolors.to_rgba(LEGEND_EDGE_RGBA)[3]),
        "axes.prop_cycle": cycler(color=list(PALETTE[theme].values())),
    }


def _theme_artist(artist, theme: str) -> None:
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    def swap(get, set_):
        new = theme_rgba(_rgba(getattr(artist, get)()), theme)
        if new is not None:
            getattr(artist, set_)(new)

    if isinstance(artist, Line2D):
        swap("get_color", "set_color")
        swap("get_markerfacecolor", "set_markerfacecolor")
        swap("get_markeredgecolor", "set_markeredgecolor")
    elif isinstance(artist, Text):
        swap("get_color", "set_color")
        box = artist.get_bbox_patch()
        if box is not None:
            _theme_artist(box, theme)
    elif isinstance(artist, Patch):
        swap("get_facecolor", "set_facecolor")
        swap("get_edgecolor", "set_edgecolor")
    elif isinstance(artist, Collection):
        if artist.get_array() is not None:  # colormapped: the colormap decides
            return
        for get, set_ in (("get_facecolor", "set_facecolor"), ("get_edgecolor", "set_edgecolor")):
            try:
                arr = getattr(artist, get)()
            except Exception:
                continue
            if arr is None or len(arr) == 0:
                continue
            rows = [tuple(float(v) for v in c) for c in arr]
            new = [theme_rgba(c, theme) for c in rows]
            if any(n is not None for n in new):
                getattr(artist, set_)([n or c for n, c in zip(new, rows)])


def apply_theme(fig, theme: str) -> None:
    """Recolor a finalized figure for ``theme`` (``"light"`` or ``"dark"``); reversible."""
    if theme not in THEMES:
        msg = f"unknown figure theme {theme!r}; expected one of {THEMES}"
        raise ValueError(msg)
    for artist in fig.findobj(include_self=False):
        if artist is fig.patch:
            continue
        _theme_artist(artist, theme)
    # Ticks matplotlib adds later copy the axis's tick settings, so set those too.
    for ax in fig.get_axes():
        for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for which, ticks in (("major", axis.get_major_ticks()), ("minor", axis.get_minor_ticks())):
                if not ticks:
                    continue
                tick = ticks[0]
                label = tick.label2 if tick.label2.get_visible() and not tick.label1.get_visible() else tick.label1
                kw: dict[str, Any] = {}
                for key, color in (
                    ("labelcolor", label.get_color()),
                    ("color", tick.tick1line.get_color()),
                    ("grid_color", tick.gridline.get_color()),
                ):
                    rgba = _rgba(color)
                    if rgba is not None and mcolors.to_hex(rgba[:3]) in _SOURCE_OF:
                        # An opaque color goes in as plain RGB: an RGBA grid color makes
                        # matplotlib skip ``grid.alpha`` on the ticks it re-creates.
                        kw[key] = rgba if rgba[3] < 1 else mcolors.to_hex(rgba[:3])
                if "grid_color" in kw and tick.gridline.get_alpha() is not None:
                    kw["grid_alpha"] = tick.gridline.get_alpha()
                if kw:
                    ax.tick_params(axis=name, which=which, **kw)
    fig._figure_theme = theme


def dark_path(path: str | os.PathLike[str]) -> Path:
    """The dark variant's path for a light PNG: ``x_v3.png`` → ``x_v3_dark.png``."""
    path = Path(path)
    return path.with_name(f"{path.stem}{DARK_SUFFIX}{path.suffix}")


def is_dark_path(path: str | os.PathLike[str]) -> bool:
    return Path(path).stem.endswith(DARK_SUFFIX)


def _dark_destination(path: Path) -> Path:
    """Where the dark variant goes; follows ``path`` into a linked notes folder.

    A light PNG that is a symlink (into the notes folder, ``papers/_linked.py``) gets a
    dark sibling symlink to the same-named file next to its target.
    """
    dark = dark_path(path)
    if path.is_symlink() and not (dark.exists() or dark.is_symlink()):
        target = dark_path(path.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.touch()
        dark.symlink_to(target)
    return dark


def save_themed(fig, fname, *args, save: Callable[..., Any] | None = None, **kwargs) -> Any:
    """Save ``fig`` as its light variant at ``fname`` and, for a ``.png`` path, its
    dark variant at :func:`dark_path`. ``save`` is the unhooked ``Figure.savefig``
    (default: the original behind ``paper_overlay``'s hook)."""
    import matplotlib as _mpl
    from matplotlib.figure import Figure

    if save is None:
        hooked = Figure.savefig
        save = getattr(hooked, "_unhooked", hooked)
    finalize(fig)
    pairs = [("light", fname)]
    if isinstance(fname, (str, os.PathLike)) and str(fname).lower().endswith(".png"):
        if not is_dark_path(fname):
            pairs.append(("dark", None))
    result = None
    for theme, out in pairs:
        if out is None:
            out = _dark_destination(Path(fname))
        apply_theme(fig, theme)
        with _mpl.rc_context(theme_rc(theme)):
            got = save(fig, out, *args, **kwargs)
        if theme == "light":
            result = got
    apply_theme(fig, "light")
    return result
