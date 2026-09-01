"""Overlay digitized paper curves on replication training plots.

Loads normalized ``curves_*.json`` under
``artifacts/figures/papers/<paper>/paper_digitization/`` (same sources as
``scripts/digitization/*_gates.py``) and draws them on matplotlib axes next to
replication traces for visual gate iteration.

Paper overlays use each series' **replication outline color**, lightened and
**dashed**, so they read as ghost references of the matching solid traces.

Draw order: overlays use a high ``zorder`` so they sit **on top of** replication
traces. Panel ``plot.py`` scripts should still call the overlay helpers **after**
plotting replication (shared helper; no per-panel zorder tweaks needed).
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Ravivarapu Figs 5–7 paper crops use PSD ~300–480. Fig 4a is the same
# biomarker labeled ×10⁻³ (~0.3–0.5). Multiply ``p_beta_norm`` by this to
# plot Figs 5–7 in paper y-units; overlay on those panels uses y_scale=1.
RAVI_INFERENCE_PAPER_Y_TO_NORM = 1000.0

_DIG = Path(__file__).resolve().parents[2] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import load_refined  # noqa: E402

MEHREGAN_DIG = Path("artifacts/figures/papers/mehregan")
NGUYEN_DIG = Path("artifacts/figures/papers/nguyen/paper_digitization")
RAVIVARAPU_DIG = Path("artifacts/figures/papers/ravivarapu/paper_digitization")

# Replication outline colors used by panel plot scripts (paper overlays lighten these).
RAVI_REPL_BASELINE = "#1f77b4"
RAVI_REPL_SEA = "#ff7f0e"
# Ravivarapu Fig 4a paper overlays (solid; distinct from replication blue/orange).
RAVI_PAPER_BASELINE_COLOR = "#ff7f0e"
RAVI_PAPER_SEA_COLOR = "#2ca02c"

# Nguyen Fig 4 paper overlays (distinct from replication blues/reds on each panel).
NGUYEN_REWARD = "#ff7f0e"  # panel (a) — paper reward
NGUYEN_LENGTH = "#2ca02c"  # panel (b) — paper length
NGUYEN_SPIKES = "#4a148c"
NGUYEN_ENERGY = "#1b7837"
NGUYEN_POWER = "#08519c"
NGUYEN_AMP = "#e41a1c"
NGUYEN_FREQ = "#377eb8"
NGUYEN_PW = "#4daf4a"

MEH_HEALTHY = "#2ca02c"
MEH_PD = "#d62728"
MEH_PD_130 = "#1f77b4"
MEH_TRAIN_TEAL = "#1f6f6f"
MEH_REWARD_GREEN = "#16a34a"
MEH_NO_STIM = "#111111"
MEH_TRAINED = "#2ca02c"
MEH_PERIODIC = "#ff7f0e"
MEH_CDBS = "#bcbd22"
MEH_PTQ_INT8 = "#1f77b4"
MEH_PTQ_FP16 = "#9467bd"
MEH_QAT = "#ff7f0e"

PAPER_LINESTYLE = "--"
PAPER_LIGHTEN = 0.42  # blend toward white (0 = outline, 1 = white)
PAPER_LINEWIDTH = 1.8
PAPER_RAW_LIGHTEN = 0.55
PAPER_RAW_LINEWIDTH = 1.15
# Above typical replication zorders (usually 1–5) and annotations.
PAPER_ZORDER = 50
PAPER_RAW_ZORDER = 49
# Single legend row for Mehregan (and optional callers): traces stay per-hue,
# but the table only explains that dashed lightened lines are paper digitization.
PAPER_CONDENSED_LEGEND_LABEL = "paper digitalization"
PAPER_CONDENSED_LEGEND_COLOR = "#000000"
# Legend sits above paper overlays (PAPER_ZORDER) and all replication artists.
LEGEND_ZORDER = 1000
LEGEND_FRAMEALPHA = 1.0
LEGEND_FACECOLOR = "white"
LEGEND_EDGECOLOR = "#cccccc"
LEGEND_COMPACT_KW = {
    "shadow": False,
    "fancybox": False,
    "borderpad": 0.22,
    "labelspacing": 0.32,
    "handlelength": 1.35,
    "handletextpad": 0.48,
    "borderaxespad": 0.35,
}
LEGEND_RCPARAMS = {
    "legend.framealpha": LEGEND_FRAMEALPHA,
    "legend.facecolor": LEGEND_FACECOLOR,
    "legend.edgecolor": LEGEND_EDGECOLOR,
}

PAPER_SMOOTH_STYLE: dict[str, Any] = {
    "color": "#000000",
    "linestyle": PAPER_LINESTYLE,
    "linewidth": PAPER_LINEWIDTH,
    "alpha": 0.95,
    "zorder": PAPER_ZORDER,
}
PAPER_RAW_STYLE: dict[str, Any] = {
    "color": "#4a4a4a",
    "linestyle": ":",
    "linewidth": PAPER_RAW_LINEWIDTH,
    "alpha": 0.75,
    "zorder": PAPER_RAW_ZORDER,
}


def lighten_color(color: str, amount: float = PAPER_LIGHTEN) -> str:
    """Blend a hex/CSS color toward white; ``amount`` in [0, 1]."""
    amount = float(np.clip(amount, 0.0, 1.0))
    color = color.strip()
    if color.startswith("#") and len(color) in (4, 7):
        if len(color) == 4:
            r = int(color[1] * 2, 16)
            g = int(color[2] * 2, 16)
            b = int(color[3] * 2, 16)
        else:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
    else:
        return color
    r = int(round(r + (255 - r) * amount))
    g = int(round(g + (255 - g) * amount))
    b = int(round(b + (255 - b) * amount))
    return f"#{r:02x}{g:02x}{b:02x}"


def paper_color_from_outline(outline: str, *, raw: bool = False) -> str:
    return lighten_color(outline, PAPER_RAW_LIGHTEN if raw else PAPER_LIGHTEN)


def _compact_legend_frame(leg) -> None:
    """Tight square frame — no rounded halo or extra interior pad."""
    fr = leg.get_frame()
    if fr is None:
        return
    try:
        fr.set_boxstyle("square", pad=LEGEND_COMPACT_KW["borderpad"])
    except Exception:
        pass
    fr.set_linewidth(0.6)


def _reassert_legend_frame(leg, *, opaque: bool, alpha: float, edgecolor: str) -> None:
    fr = leg.get_frame()
    if fr is None:
        return
    if opaque:
        fr.set_facecolor(LEGEND_FACECOLOR)
        fr.set_alpha(1.0)
    else:
        fr.set_facecolor((1.0, 1.0, 1.0, alpha))
        fr.set_alpha(alpha)
    fr.set_edgecolor(edgecolor)
    fr.set_linewidth(0.6)
    fr.set_zorder(LEGEND_ZORDER)
    leg.set_zorder(LEGEND_ZORDER)
    leg.set_alpha(1.0)
    _compact_legend_frame(leg)


def finalize_figure_legends(fig) -> None:
    """Re-assert opaque legend frames immediately before rasterizing."""
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is None:
            continue
        opaque = bool(getattr(leg, "_overlay_opaque", True))
        alpha = float(getattr(leg, "_overlay_alpha", LEGEND_FRAMEALPHA))
        edgecolor = str(getattr(leg, "_overlay_edgecolor", LEGEND_EDGECOLOR))
        _reassert_legend_frame(leg, opaque=opaque, alpha=alpha, edgecolor=edgecolor)


def save_figure(fig, fname, *args, **kwargs):
    """``fig.savefig`` wrapper that finalizes opaque legend tables first."""
    finalize_figure_legends(fig)
    try:
        fig.canvas.draw()
    except Exception:
        pass
    finalize_figure_legends(fig)
    return fig.savefig(fname, *args, **kwargs)


def _install_savefig_finalize() -> None:
    from matplotlib.figure import Figure

    if getattr(Figure.savefig, "_opaque_legend_patched", False):
        return
    _orig_savefig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        finalize_figure_legends(self)
        try:
            self.canvas.draw()
        except Exception:
            pass
        finalize_figure_legends(self)
        return _orig_savefig(self, fname, *args, **kwargs)

    savefig._opaque_legend_patched = True  # type: ignore[attr-defined]
    Figure.savefig = savefig  # type: ignore[method-assign]


def place_legend(ax, *args, **kwargs):
    """Draw the axes legend in front of traces (opaque frame by default).

    Call this **last** among axes artists (after overlays, grid, vlines). The
    legend patch, handles, and texts share ``LEGEND_ZORDER`` so nothing from the
    plot paints over the table.
    """
    import matplotlib.pyplot as plt

    alpha = float(kwargs.pop("framealpha", LEGEND_FRAMEALPHA))
    opaque = alpha >= 1.0 - 1e-6
    if opaque:
        alpha = 1.0
    for key, value in LEGEND_COMPACT_KW.items():
        kwargs.setdefault(key, value)
    kwargs["fancybox"] = False
    kwargs.setdefault("frameon", True)
    edgecolor = kwargs.setdefault("edgecolor", LEGEND_EDGECOLOR)
    with plt.rc_context(LEGEND_RCPARAMS | {"legend.framealpha": alpha}):
        leg = ax.legend(*args, framealpha=alpha, **kwargs)
    leg._overlay_opaque = opaque  # type: ignore[attr-defined]
    leg._overlay_alpha = alpha  # type: ignore[attr-defined]
    leg._overlay_edgecolor = edgecolor  # type: ignore[attr-defined]
    for artist in list(ax.get_children()):
        if artist is leg:
            continue
        try:
            if float(artist.get_zorder()) >= float(LEGEND_ZORDER):
                artist.set_zorder(LEGEND_ZORDER - 10)
        except Exception:
            pass
    _reassert_legend_frame(leg, opaque=opaque, alpha=alpha, edgecolor=edgecolor)
    for handle in getattr(leg, "legend_handles", None) or []:
        try:
            handle.set_zorder(LEGEND_ZORDER + 1)
        except Exception:
            pass
    for text in leg.get_texts():
        text.set_zorder(LEGEND_ZORDER + 2)
    _compact_legend_frame(leg)
    return leg


def add_condensed_paper_legend(
    ax,
    *,
    label: str = PAPER_CONDENSED_LEGEND_LABEL,
    color: str | None = None,
    alpha: float | None = None,
) -> None:
    """Add one dashed legend handle for paper digitization overlays."""
    line_color = color if color is not None else PAPER_CONDENSED_LEGEND_COLOR
    line_alpha = 0.75 if alpha is None else alpha
    ax.plot(
        [],
        [],
        color=line_color,
        linestyle=(0, (5.0, 2.5)),
        linewidth=PAPER_LINEWIDTH,
        alpha=line_alpha,
        zorder=PAPER_ZORDER,
        label=label,
    )


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
    outline_color: str | None = None,
    linestyle: str | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
    zorder: int | None = None,
    lighten: bool = True,
    mark_endpoints: bool = True,
) -> None:
    """Draw one paper curve. Prefer ``outline_color`` (lightened + dashed by default)."""
    style = dict(PAPER_RAW_STYLE if raw else PAPER_SMOOTH_STYLE)
    if outline_color is not None:
        style["color"] = (
            paper_color_from_outline(outline_color, raw=raw)
            if lighten
            else outline_color
        )
        style["linestyle"] = ":" if raw else (PAPER_LINESTYLE if lighten else "-")
    elif color is not None:
        style["color"] = (
            paper_color_from_outline(color, raw=raw) if lighten else color
        )
        style["linestyle"] = ":" if raw else (PAPER_LINESTYLE if lighten else "-")
    if linestyle is not None:
        style["linestyle"] = linestyle
    if linewidth is not None:
        style["linewidth"] = linewidth
    if alpha is not None:
        style["alpha"] = alpha
    if zorder is not None:
        style["zorder"] = zorder
    color = style["color"]
    zorder = int(style.get("zorder", PAPER_ZORDER))
    # Dash pattern that starts with ink (offset 0); default '--' can look inset.
    if (
        lighten
        and not raw
        and style.get("linestyle") in ("--", PAPER_LINESTYLE)
    ):
        style["linestyle"] = (0, (5.0, 2.5))
    ax.plot(x, y, label=label, **style)
    # Endpoint dots so 0/12 (or series ends) stay visible under dash gaps / legends.
    if mark_endpoints and len(x) > 0:
        ax.plot(
            [float(x[0]), float(x[-1])],
            [float(y[0]), float(y[-1])],
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor=color,
            markeredgecolor=color,
            zorder=zorder + 1,
            label="_nolegend_",
        )


def snap_x_to_limits(
    x: np.ndarray,
    *,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    """Linearly map ``x`` onto ``[x_min, x_max]`` (keeps relative spacing)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return np.full_like(x, 0.5 * (x_min + x_max))
    return x_min + (x - lo) * ((x_max - x_min) / (hi - lo))


def _transform_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_offset: float = 0.0,
    x_scale: float = 1.0,
    y_offset: float = 0.0,
    y_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    px = np.asarray(x, dtype=float) * x_scale + x_offset
    py = np.asarray(y, dtype=float) * y_scale + y_offset
    return px, py


def overlay_named_series(
    ax,
    curves_path: Path | str,
    mapping: Sequence[tuple[str, str]],
    *,
    x_offset: float = 0.0,
    x_scale: float = 1.0,
    y_offset: float = 0.0,
    y_scale: float = 1.0,
    snap_x: tuple[float, float] | None = None,
    label_prefix: str = "Paper ",
    outline_colors: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    condensed_legend: bool = False,
    condensed_legend_label: str = PAPER_CONDENSED_LEGEND_LABEL,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Overlay digitized series listed as ``(digitization_name, legend_label)``.

    ``outline_colors`` / ``colors`` are replication outline hues; overlays are
    drawn lightened and dashed. ``snap_x=(lo, hi)`` linearly maps each series'
    x onto that window (use for panels whose paper axis is fixed, e.g. 0–12 s).
    When ``condensed_legend`` is true, series use ``_nolegend_`` and one proxy
    handle (``condensed_legend_label``) is added instead of per-series Paper rows.
    Returns ``{name: (x_plot, y_plot)}``.
    """
    curves = load_panel_curves(curves_path)
    outlines = outline_colors or colors
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for idx, (series_name, legend_label) in enumerate(mapping):
        if series_name not in curves:
            continue
        px, py = curves[series_name]
        if snap_x is not None:
            px = snap_x_to_limits(px, x_min=snap_x[0], x_max=snap_x[1])
        px_plot, py_plot = _transform_xy(
            px,
            py,
            x_offset=x_offset,
            x_scale=x_scale,
            y_offset=y_offset,
            y_scale=y_scale,
        )
        outline = outlines[idx % len(outlines)] if outlines else "#333333"
        overlay_on_axis(
            ax,
            px_plot,
            py_plot,
            label=(
                "_nolegend_"
                if condensed_legend
                else f"{label_prefix}{legend_label}"
            ),
            outline_color=outline,
        )
        out[series_name] = (px_plot, py_plot)
    if condensed_legend and out:
        add_condensed_paper_legend(ax, label=condensed_legend_label)
    return out


def overlay_smoothed_raw_axis(
    ax,
    curves_path: Path | str,
    *,
    show_raw: bool = True,
    smoothed_names: Sequence[str] = ("Smoothed",),
    raw_names: Sequence[str] = ("Raw",),
    outline_color: str = "#333333",
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay Smoothed (+ optional Raw) from a single digitization file."""
    curves = load_panel_curves(curves_path)
    sx, sy = pick_series(curves, *smoothed_names)
    if show_raw:
        for raw_name in raw_names:
            if raw_name in curves:
                rx, ry = curves[raw_name]
                overlay_on_axis(
                    ax,
                    rx,
                    ry,
                    label="Paper raw (digitized)",
                    outline_color=outline_color,
                    raw=True,
                )
                break
    overlay_on_axis(
        ax,
        sx,
        sy,
        label="Paper smoothed (digitized)",
        outline_color=outline_color,
    )
    raw_y = curves[raw_names[0]][1] if raw_names and raw_names[0] in curves else sy
    return sy, raw_y


# --- Nguyen ---

# Fig 3 is scatter/boxplot — no refined ``curves_fig3`` yet. Overlay the same
# documented paper mean readouts used by ``nguyen_gates.fig3_gates`` (~215 / ~295).
NGUYEN_FIG3_PD_OFF_MEAN = 215.0
NGUYEN_FIG3_PD_ON_MEAN = 295.0
# Match ``nguyen/3/plot.py`` replication mean-line colors (not lightened).
NGUYEN_FIG3_MEAN_OFF = "#d62728"
NGUYEN_FIG3_MEAN_ON = "#000000"


def overlay_nguyen_fig3(
    ax_scatter,
    ax_box=None,
    *,
    mean_off_color: str = NGUYEN_FIG3_MEAN_OFF,
    mean_on_color: str = NGUYEN_FIG3_MEAN_ON,
    box_positions: tuple[float, float] = (1.0, 2.0),
    box_width: float = 0.5,
) -> dict[str, float]:
    """Draw documented paper means: full-width on scatter (a), per-box on box (b)."""
    means = {
        "pd_off": NGUYEN_FIG3_PD_OFF_MEAN,
        "pd_on": NGUYEN_FIG3_PD_ON_MEAN,
    }
    scatter_specs = (
        (means["pd_off"], mean_off_color),
        (means["pd_on"], mean_on_color),
    )
    for y, color in scatter_specs:
        ax_scatter.axhline(
            y,
            color=color,
            linestyle=(0, (5.0, 2.5)),
            linewidth=PAPER_LINEWIDTH,
            alpha=0.95,
            zorder=PAPER_ZORDER,
            label="_nolegend_",
        )
    if ax_box is not None:
        half = box_width / 2.0
        off_x, on_x = box_positions
        box_specs = (
            (means["pd_off"], mean_off_color, off_x),
            (means["pd_on"], mean_on_color, on_x),
        )
        for y, color, x_center in box_specs:
            ax_box.plot(
                [x_center - half, x_center + half],
                [y, y],
                color=color,
                linestyle=(0, (5.0, 2.5)),
                linewidth=PAPER_LINEWIDTH,
                alpha=0.95,
                zorder=PAPER_ZORDER,
                solid_capstyle="butt",
            )
    add_condensed_paper_legend(ax_scatter)
    return means


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
    paper_raw_style = {
        "linestyle": "-",
        "lighten": False,
        "mark_endpoints": False,
    }
    paper_smooth_style = {
        "linestyle": PAPER_LINESTYLE,
        "lighten": False,
        "mark_endpoints": False,
    }
    if show_paper_raw and "Raw" in reward_curves:
        rx, ry = reward_curves["Raw"]
        overlay_on_axis(
            ax_reward,
            rx,
            ry,
            label="Paper raw (digitized)",
            color=NGUYEN_REWARD,
            alpha=0.55,
            raw=True,
            **paper_raw_style,
        )
    if show_paper_raw and "Raw" in length_curves:
        lx, ly = length_curves["Raw"]
        overlay_on_axis(
            ax_length,
            lx,
            ly,
            label="Paper raw (digitized)",
            color=NGUYEN_LENGTH,
            alpha=0.55,
            raw=True,
            **paper_raw_style,
        )
    overlay_on_axis(
        ax_reward,
        prx,
        pry,
        label="Paper smoothed (digitized)",
        color=NGUYEN_REWARD,
        **paper_smooth_style,
    )
    overlay_on_axis(
        ax_length,
        plx,
        ply,
        label="Paper smoothed (digitized)",
        color=NGUYEN_LENGTH,
        **paper_smooth_style,
    )
    r_raw_y = reward_curves["Raw"][1] if "Raw" in reward_curves else pry
    l_raw_y = length_curves["Raw"][1] if "Raw" in length_curves else ply
    return {"reward": (pry, r_raw_y), "length": (ply, l_raw_y)}


def overlay_nguyen_fig5(
    ax_spikes,
    ax_energy,
    *,
    show_paper_raw: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    spike_sy, spike_raw = overlay_smoothed_raw_axis(
        ax_spikes,
        NGUYEN_DIG / "curves_fig5_spikes.json",
        show_raw=False,
        smoothed_names=("Spike Count",),
        raw_names=(),
        outline_color=NGUYEN_SPIKES,
    )
    energy_sy, energy_raw = overlay_smoothed_raw_axis(
        ax_energy,
        NGUYEN_DIG / "curves_fig5_energy.json",
        show_raw=show_paper_raw,
        outline_color=NGUYEN_ENERGY,
    )
    return {"spikes": (spike_sy, spike_raw), "energy": (energy_sy, energy_raw)}


def overlay_nguyen_fig6(
    ax_power,
    ax_params_or_freq,
    ax_amp=None,
    ax_pw=None,
    *,
    show_paper_raw: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    power_sy, power_raw = overlay_smoothed_raw_axis(
        ax_power,
        NGUYEN_DIG / "curves_fig6_power.json",
        show_raw=show_paper_raw,
        outline_color=NGUYEN_POWER,
    )
    if ax_amp is not None and ax_pw is not None:
        ax_map = {
            "curves_fig6_freq.json": (ax_params_or_freq, "Frequency", NGUYEN_FREQ),
            "curves_fig6_amp.json": (ax_amp, "Amplitude", NGUYEN_AMP),
            "curves_fig6_pw.json": (ax_pw, "Pulse width", NGUYEN_PW),
        }
        param_ys: list[np.ndarray] = []
        for stem, (target_ax, label, outline) in ax_map.items():
            curves = load_panel_curves(NGUYEN_DIG / stem)
            px, py = pick_series(curves, "Smoothed", "Raw")
            overlay_on_axis(
                target_ax,
                px,
                py,
                label=f"Paper {label} (digitized)",
                outline_color=outline,
            )
            param_ys.append(py)
    elif isinstance(ax_params_or_freq, dict):
        ax_map = {
            "curves_fig6_freq.json": (ax_params_or_freq.get("freq", ax_params_or_freq), "Frequency", NGUYEN_FREQ),
            "curves_fig6_amp.json": (ax_params_or_freq.get("amp", ax_params_or_freq), "Amplitude", NGUYEN_AMP),
            "curves_fig6_pw.json": (ax_params_or_freq.get("pw", ax_params_or_freq), "Pulse width", NGUYEN_PW),
        }
        param_ys = []
        for stem, (target_ax, label, outline) in ax_map.items():
            curves = load_panel_curves(NGUYEN_DIG / stem)
            px, py = pick_series(curves, "Smoothed", "Raw")
            overlay_on_axis(
                target_ax,
                px,
                py,
                label=f"Paper {label} (digitized)",
                outline_color=outline,
            )
            param_ys.append(py)
    elif isinstance(ax_params_or_freq, (tuple, list)) and len(ax_params_or_freq) == 3:
        ax_f, ax_a, ax_p = ax_params_or_freq
        ax_map = {
            "curves_fig6_freq.json": (ax_f, "Frequency", NGUYEN_FREQ),
            "curves_fig6_amp.json": (ax_a, "Amplitude", NGUYEN_AMP),
            "curves_fig6_pw.json": (ax_p, "Pulse width", NGUYEN_PW),
        }
        param_ys = []
        for stem, (target_ax, label, outline) in ax_map.items():
            curves = load_panel_curves(NGUYEN_DIG / stem)
            px, py = pick_series(curves, "Smoothed", "Raw")
            overlay_on_axis(
                target_ax,
                px,
                py,
                label=f"Paper {label} (digitized)",
                outline_color=outline,
            )
            param_ys.append(py)
    else:
        ax_params = ax_params_or_freq
        param_specs = (
            ("curves_fig6_amp.json", "Amplitude", NGUYEN_AMP, ("Raw", "Smoothed")),
            ("curves_fig6_freq.json", "Frequency", NGUYEN_FREQ, ("Raw", "Smoothed")),
            ("curves_fig6_pw.json", "Pulse width", NGUYEN_PW, ("Raw", "Smoothed")),
        )
        param_ys = []
        for stem, label, outline, names in param_specs:
            curves = load_panel_curves(NGUYEN_DIG / stem)
            px, py = pick_series(curves, *names)
            overlay_on_axis(
                ax_params,
                px,
                py,
                label=f"Paper {label} (digitized)",
                outline_color=outline,
            )
            param_ys.append(py)
    power_curves = load_panel_curves(NGUYEN_DIG / "curves_fig6_power.json")
    if "threshold" in power_curves:
        tx, ty = power_curves["threshold"]
        overlay_on_axis(
            ax_power,
            tx,
            ty,
            label="Paper threshold (digitized)",
            outline_color="#d62728",
            linestyle=":",
            linewidth=1.2,
        )
        power_raw = np.concatenate([power_raw, ty])
    return {"power": (power_sy, power_raw), "params": (np.concatenate(param_ys),)}


def overlay_nguyen_fig7(ax, *, show_confidence: bool = True) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Overlay paper mean, θ threshold, and smooth 95% CI band from digitization."""
    curves = load_panel_curves(NGUYEN_DIG / "curves_fig7.json")
    extra_ys = []
    if show_confidence:
        if "ci_upper" in curves and "ci_lower" in curves:
            ux, uy = curves["ci_upper"]
            lx, ly = curves["ci_lower"]
            x_grid = np.linspace(0.0, 24.0, 100)
            u_grid = np.interp(x_grid, ux, uy)
            l_grid = np.interp(x_grid, lx, ly)
            ax.fill_between(
                x_grid,
                l_grid,
                u_grid,
                color="#ff7f0e",
                alpha=0.15,
                edgecolor="#d95f02",
                linewidth=0.8,
                linestyle="--",
                label="Paper 95% CI (digitized)",
                zorder=1.5,
            )
            extra_ys.extend([u_grid, l_grid])
        elif "95% Confidence" in curves:
            cx, cy = curves["95% Confidence"]
            split_idx = len(cx) // 2 + 1
            ux, uy = cx[:split_idx], cy[:split_idx]
            lx, ly = cx[split_idx - 1:][::-1], cy[split_idx - 1:][::-1]
            u_sort = np.argsort(ux)
            ux, uy = ux[u_sort], uy[u_sort]
            l_sort = np.argsort(lx)
            lx, ly = lx[l_sort], ly[l_sort]

            x_grid = np.linspace(0.0, 24.0, 100)
            u_grid = np.interp(x_grid, ux, uy)
            l_grid = np.interp(x_grid, lx, ly)

            ax.fill_between(
                x_grid,
                l_grid,
                u_grid,
                color="#ff7f0e",
                alpha=0.15,
                edgecolor="#d95f02",
                linewidth=0.8,
                linestyle="--",
                label="Paper 95% CI (digitized)",
                zorder=1.5,
            )
            extra_ys.extend([u_grid, l_grid])

    avg_x, avg_y = pick_series(curves, "average")
    overlay_on_axis(
        ax,
        avg_x,
        avg_y,
        label="Paper mean (digitized)",
        outline_color="#d95f02",
    )
    extra_ys.append(avg_y)
    if "threshold" in curves:
        tx, ty = curves["threshold"]
        overlay_on_axis(
            ax,
            tx,
            ty,
            label="Paper θ (digitized)",
            outline_color="#d62728",
            linestyle=":",
            linewidth=1.4,
        )
        extra_ys.append(ty)
    return {"mean": (avg_y, np.concatenate(extra_ys) if extra_ys else avg_y)}


# --- Mehregan ---


def overlay_mehregan_fig1b(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("healthy", "Healthy"),
        ("pd", "PD no Treatment"),
        ("pd_130hz", "PD 130 Hz Treatment"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "1b/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_HEALTHY, MEH_PD, MEH_PD_130),
        condensed_legend=True,
    )


def overlay_mehregan_fig2(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("pd", "PD no Treatment"),
        ("PD 130Hz Treatment", "PD 130 Hz Treatment"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "2a/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_PD, MEH_PD_130),
        condensed_legend=True,
    )


def overlay_mehregan_fig2b(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("pd", "PD no Treatment"),
        ("PD 130Hz Treatment", "PD 130 Hz Treatment"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "2b/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_PD, MEH_PD_130),
        condensed_legend=True,
    )


def overlay_mehregan_fig4a(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves = load_panel_curves(MEHREGAN_DIG / "4a/paper_digitization/curves_wpd_refined.json")
    px, py = pick_series(curves, "training")
    overlay_on_axis(ax, px, py, label="_nolegend_", outline_color=MEH_TRAIN_TEAL)
    add_condensed_paper_legend(
        ax,
        color=paper_color_from_outline(MEH_TRAIN_TEAL),
        alpha=float(PAPER_SMOOTH_STYLE["alpha"]),
    )
    return {"training": (py, py)}


def overlay_mehregan_fig4b_reward(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "4b/paper_digitization/curves_wpd_refined_reward.json",
        (("Reward", "Reward"),),
        label_prefix="Paper ",
        outline_colors=(MEH_REWARD_GREEN,),
        condensed_legend=True,
    )


def overlay_mehregan_fig4b_psd(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "4b/paper_digitization/curves_wpd_refined_psd.json",
        (("PSD (x10^3)", "PSD"),),
        label_prefix="Paper ",
        outline_colors=(MEH_TRAIN_TEAL,),
        condensed_legend=True,
    )


def overlay_mehregan_fig4b(
    ax_reward,
    ax_psd,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    reward = overlay_mehregan_fig4b_reward(ax_reward)
    psd = overlay_mehregan_fig4b_psd(ax_psd)
    return {
        "reward": next(iter(reward.values()), (np.array([]), np.array([]))),
        "psd": next(iter(psd.values()), (np.array([]), np.array([]))),
    }


def overlay_mehregan_fig5a(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("PD no stim", "PD no stim"),
        ("Fully Trained 45Hz", "Fully Trained 45Hz"),
        ("Periodic 45Hz", "Periodic 45Hz"),
        ("Periodic 130Hz", "Periodic 130Hz"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "5a/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_NO_STIM, MEH_TRAINED, MEH_PERIODIC, MEH_CDBS),
        snap_x=(0.0, 12.0),
        condensed_legend=True,
    )


def overlay_mehregan_fig5b(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("PD no stim", "PD no stim"),
        ("Fully Trained 30Hz", "Fully Trained 30Hz"),
        ("Periodic 30Hz", "Periodic 30Hz"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "5b/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_NO_STIM, MEH_TRAINED, MEH_PERIODIC),
        snap_x=(0.0, 12.0),
        condensed_legend=True,
    )


def overlay_mehregan_fig6a(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("Fully Trained 45Hz", "Fully Trained 45Hz"),
        ("PTQ, INT8", "PTQ INT8"),
        ("PTQ, FP16", "PTQ FP16"),
        ("QAT", "QAT"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "6a/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_TRAINED, MEH_PTQ_INT8, MEH_PTQ_FP16, MEH_QAT),
        snap_x=(0.0, 12.0),
        condensed_legend=True,
    )


def overlay_mehregan_fig6b(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("Fully Trained 30Hz", "Fully Trained 30Hz"),
        ("PTQ, INT8", "PTQ INT8"),
        ("PTQ, FP16", "PTQ FP16"),
        ("QAT", "QAT"),
    )
    return overlay_named_series(
        ax,
        MEHREGAN_DIG / "6b/paper_digitization/curves_wpd_refined.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(MEH_TRAINED, MEH_PTQ_INT8, MEH_PTQ_FP16, MEH_QAT),
        snap_x=(0.0, 12.0),
        condensed_legend=True,
    )


# --- Ravivarapu ---


def ravivarapu_fig4a_digitization() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Baseline and SEA-DBS PSD vs episode from ``curves_fig4a.json``."""
    return load_panel_curves(RAVIVARAPU_DIG / "curves_fig4a.json")


def overlay_ravivarapu_fig4a(ax) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Draw digitized Baseline / SEA-DBS on the Fig 4a axis (1-based episode x)."""
    mapping = (
        ("Baseline", "Baseline (digitized)"),
        ("SEA-DBS", "SEA-DBS (digitized)"),
    )
    curves = ravivarapu_fig4a_digitization()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    paper_colors = (RAVI_PAPER_BASELINE_COLOR, RAVI_PAPER_SEA_COLOR)
    for idx, (series_name, legend_label) in enumerate(mapping):
        px, py = pick_series(curves, series_name)
        # WPD x is 0-based episode index; replication plots use 1..n.
        px_plot = np.asarray(px, dtype=float) + 1.0
        overlay_on_axis(
            ax,
            px_plot,
            py,
            label=f"Paper {legend_label}",
            color=paper_colors[idx],
            linestyle=PAPER_LINESTYLE,
            alpha=0.72,
            lighten=False,
            mark_endpoints=False,
        )
        out[series_name] = (py, py)
    return out


def overlay_ravivarapu_fig4b(
    ax,
    *,
    y_scale: float | None = None,
    replication_early_mean: float | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Episode reward overlay.

    Paper Fig 4b digitization is ~[-1.5, 0] (near per-step Eq. 7); replication
    plots episode **sums** (~[-100, 10]). Scale paper onto the replication axis.
    """
    mapping = (
        ("Baseline Reward", "Baseline"),
        ("SEA-DBS Reward", "SEA-DBS"),
    )
    curves = load_panel_curves(RAVIVARAPU_DIG / "curves_fig4b.json")
    if y_scale is None:
        _bx, by = pick_series(curves, "Baseline Reward")
        paper_early = float(np.mean(by[: max(1, by.size // 10)])) if by.size else -1.3
        target = (
            float(replication_early_mean)
            if replication_early_mean is not None and np.isfinite(replication_early_mean)
            else -95.0
        )
        y_scale = abs(target / paper_early) if abs(paper_early) > 1e-6 else 70.0
    return overlay_named_series(
        ax,
        RAVIVARAPU_DIG / "curves_fig4b.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(RAVI_REPL_BASELINE, RAVI_REPL_SEA),
        y_scale=y_scale,
    )


def overlay_ravivarapu_fig5a(
    ax, *, y_scale: float = 1.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Paper Fig 5a overlay in crop PSD units (~300–480). Pass ``y_scale``
    only if the replication axis is normalized instead."""
    mapping = (
        ("Baseline 50Hz", "Baseline 50Hz"),
        ("SEA-DBS 50Hz", "SEA-DBS 50Hz"),
    )
    return overlay_named_series(
        ax,
        RAVIVARAPU_DIG / "curves_fig5a.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(RAVI_REPL_BASELINE, RAVI_REPL_SEA),
        y_scale=y_scale,
    )


def overlay_ravivarapu_fig5b(
    ax, *, y_scale: float = 1.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mapping = (
        ("Baseline 30Hz", "Baseline 30Hz"),
        ("SEA-DBS 30Hz", "SEA-DBS 30Hz"),
    )
    return overlay_named_series(
        ax,
        RAVIVARAPU_DIG / "curves_fig5b.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=(RAVI_REPL_BASELINE, RAVI_REPL_SEA),
        y_scale=y_scale,
    )


def overlay_ravivarapu_fig6(
    ax, *, y_scale: float = 1.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Paper Fig 6 overlay in crop PSD units (~300–480). Pass ``y_scale``
    only if the replication axis is normalized instead."""
    mapping = (
        ("Baseline", "Baseline"),
        ("Baseline + PTQ(fp16)", "Baseline + PTQ(fp16)"),
        ("SEA-DBS", "SEA-DBS"),
        ("SEA-DBS + PTQ(fp16)", "SEA-DBS + PTQ(fp16)"),
    )
    # Matplotlib C0–C3 defaults used by the panel plot.
    outlines = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    return overlay_named_series(
        ax,
        RAVIVARAPU_DIG / "curves_fig6.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=outlines,
        y_scale=y_scale,
    )


def overlay_ravivarapu_fig7(
    ax, *, y_scale: float = 1.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Paper Fig 7 overlay in crop PSD units (~300–480). Pass ``y_scale``
    only if the replication axis is normalized instead."""
    mapping = (
        ("Baseline", "Baseline"),
        ("Baseline + GS", "Baseline + GS"),
        ("Baseline + PM", "Baseline + PM"),
        ("SEA-DBS", "SEA-DBS"),
    )
    # Digitization series order (GS then PM); colors match common C0/C2/C1/C3.
    outlines = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")
    return overlay_named_series(
        ax,
        RAVIVARAPU_DIG / "curves_fig7.json",
        mapping,
        label_prefix="Paper ",
        outline_colors=outlines,
        y_scale=y_scale,
    )


_install_savefig_finalize()
