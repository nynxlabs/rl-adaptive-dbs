"""Redraw paper-digitized curves as our own figures (no paper image is used).

Each panel's digitized values (``curves_*.json`` under the artifacts root, read off
the published figure) are drawn with matplotlib and saved as
``docs/controllers/<controller>/figures/images/<panel>/digitized_v<N>.png``. The caption says the figure
is ``Digitized from Fig. X of <paper>``, and the footer says it is our redraw. The
figure uses the panels' shared style (``scripts/figures/papers/figure_style.py``), so it
reads on light and dark pages.

    uv run python scripts/digitization/redraw_digitized.py --all
    uv run python scripts/digitization/redraw_digitized.py --panel mehregan/4a --panel nguyen/6

Axis labels come from each panel's ``scripts/figures/papers/<paper>/<panel>/plot.py``
(or the shared ``efficacy_panel.py`` / ``quant_panel.py``). Axis ranges come from
``scripts/digitization/figs/<stem>.json``. The y axis always widens to include every
digitized point. The x axis widens only when a point overshoots its range by more than
2% of the span, so a point a few hundredths outside the frame can sit on the edge.
Re-running with the same ``--version`` overwrites the file, so bump ``--version`` when
a cached image must refresh.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from paper_gates import load_refined

from rl_adaptive_dbs import doc_paths

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = REPO / "artifacts" / "figures" / "papers"
DEFAULT_OUT_ROOT = REPO  # images go to <root>/docs/controllers/<controller>/figures/images/
FIGS_DIR = Path(__file__).resolve().parent / "figs"
FIGURE_STYLE = REPO / "scripts" / "figures" / "papers" / "figure_style.py"

DPI = 150
FOOTER = "Our redraw of values read off the published figure; not the original figure."
CITATIONS = {"mehregan": "Mehregan et al. (2024)", "nguyen": "Nguyen et al. (2025)"}
# Fraction of an axis span that a data overshoot must exceed before the axis widens.
X_TOLERANCE = 0.02
PAD = 0.02


@dataclass(frozen=True)
class Axis:
    """One subplot: the curves JSON it draws and its labels.

    ``source`` is relative to the artifacts root. ``config`` names a
    ``figs/<config>.json`` whose ``panel.x_range`` / ``y_range`` set the limits;
    ``ylim`` is a fixed fallback for y when there is no config.
    """

    source: str
    xlabel: str
    ylabel: str
    config: str | None = None
    ylim: tuple[float, float] | None = None
    bin_near_duplicates: bool = True
    boxplot: bool = False


@dataclass(frozen=True)
class Panel:
    """A redrawn figure. One axis gives a single plot; two or four give a grid."""

    paper: str
    fig: str  # "4a", "3", ... ; the output folder is docs/controllers/<controller>/figures/images/<fig>/
    axes: tuple[Axis, ...]


_M = "mehregan/{p}/paper_digitization/curves_wpd_refined.json"
_N = "nguyen/paper_digitization/curves_{s}.json"

PANELS: dict[str, Panel] = {
    "mehregan/1b": Panel(
        "mehregan",
        "1b",
        (Axis(_M.format(p="1b"), "Frequency (Hz)", "Power Spectral Density", "mehregan_fig1b"),),
    ),
    "mehregan/2a": Panel(
        "mehregan",
        "2a",
        (Axis(_M.format(p="2a"), "Time (sec)", "PSD", "mehregan_fig2a"),),
    ),
    "mehregan/2b": Panel(
        "mehregan",
        "2b",
        (Axis(_M.format(p="2b"), "Time (sec)", "Error Index", "mehregan_fig2b"),),
    ),
    "mehregan/4a": Panel(
        "mehregan",
        "4a",
        (Axis(_M.format(p="4a"), "Steps", "PSD (×10³)", "mehregan_fig4a"),),
    ),
    # Reward left, PSD right: the same order as the 4b figure page's replication pair.
    "mehregan/4b": Panel(
        "mehregan",
        "4b",
        (
            Axis(
                "mehregan/4b/paper_digitization/curves_wpd_refined_reward.json",
                "Episode",
                "Reward",
                "mehregan_fig4b_reward",
            ),
            Axis(
                "mehregan/4b/paper_digitization/curves_wpd_refined_psd.json",
                "Episode",
                "PSD (×10³)",
                "mehregan_fig4b_psd",
            ),
        ),
    ),
    "mehregan/5a": Panel(
        "mehregan",
        "5a",
        (Axis(_M.format(p="5a"), "Time (sec)", "PSD", "mehregan_fig5a"),),
    ),
    "mehregan/5b": Panel(
        "mehregan",
        "5b",
        (Axis(_M.format(p="5b"), "Time (sec)", "PSD", "mehregan_fig5b"),),
    ),
    "mehregan/6a": Panel(
        "mehregan",
        "6a",
        (Axis(_M.format(p="6a"), "Time (sec)", "PSD", "mehregan_fig6a"),),
    ),
    "mehregan/6b": Panel(
        "mehregan",
        "6b",
        (Axis(_M.format(p="6b"), "Time (sec)", "PSD", "mehregan_fig6b"),),
    ),
    "nguyen/3": Panel(
        "nguyen",
        "3",
        (
            Axis(
                _N.format(s="fig3_boxplot"),
                "",
                "GPi α–β Oscillation Power",
                ylim=(50.0, 400.0),  # the eight y tick marks in nguyen_fig3_boxplot.py
                boxplot=True,
            ),
        ),
    ),
    # Reward left, length right (plot.py order).
    "nguyen/4": Panel(
        "nguyen",
        "4",
        (
            Axis(_N.format(s="fig4_reward"), "Episode", "Reward", "nguyen_fig4_reward"),
            Axis(_N.format(s="fig4_length"), "Episode", "Length", "nguyen_fig4_length"),
        ),
    ),
    # Energy left, spikes right. Spikes and energy are raw per-episode traces, so
    # near-duplicate x values are kept (the same rule as nguyen_gates UNBINNED_STEMS).
    "nguyen/5": Panel(
        "nguyen",
        "5",
        (
            Axis(
                _N.format(s="fig5_energy"),
                "Episode",
                "Energy (a.u.)",
                "nguyen_fig5_energy",
                bin_near_duplicates=False,
            ),
            Axis(
                _N.format(s="fig5_spikes"),
                "Episode",
                "Spike count (unit not defined in paper)",
                "nguyen_fig5_spikes",
                bin_near_duplicates=False,
            ),
        ),
    ),
    # 2×2 in the brief's order: amplitude, frequency / power, pulse width.
    "nguyen/6": Panel(
        "nguyen",
        "6",
        (
            Axis(_N.format(s="fig6_amp"), "Episode", "Amplitude (nA/cm²)", "nguyen_fig6_amp"),
            Axis(_N.format(s="fig6_freq"), "Episode", "Frequency (Hz)", "nguyen_fig6_freq"),
            Axis(_N.format(s="fig6_power"), "Episode", "α–β Power", "nguyen_fig6_power"),
            Axis(_N.format(s="fig6_pw"), "Episode", "Pulse width (ms)", "nguyen_fig6_pw"),
        ),
    ),
    "nguyen/7": Panel(
        "nguyen",
        "7",
        (Axis(_N.format(s="fig7"), "Time Step", "α–β Power", "nguyen_fig7"),),
    ),
}

_FIGSIZE = {1: (7.5, 4.8), 2: (12.0, 4.8), 4: (10.0, 8.0)}
_GRID = {1: (1, 1), 2: (1, 2), 4: (2, 2)}


@dataclass(frozen=True)
class _Series:
    name: str
    x: np.ndarray
    y: np.ndarray
    color: tuple[float, float, float] | None


def _rgb(color: list[int] | None) -> tuple[float, float, float] | None:
    if not color:
        return None
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


def _readable(color: tuple[float, float, float] | None) -> tuple[float, float, float] | str | None:
    """A pale, nearly colorless sampled color (e.g. Mehregan Fig 4b's light blue-grey
    curves) vanishes on a light page, so draw it in the shared neutral grey instead."""
    if color is not None and min(color) >= 0.6 and max(color) - min(color) <= 0.12:
        return _figure_style().NEUTRAL
    return color


def _read_curves(path: Path, *, bin_near_duplicates: bool) -> list[_Series]:
    payload = json.loads(path.read_text())
    curves = load_refined(path, bin_near_duplicates=bin_near_duplicates)
    out: list[_Series] = []
    for name, (x, y) in curves.items():
        color = payload["series"][name].get("color_rgba")
        out.append(_Series(name, x, y, _rgb(color)))
    return out


def _read_boxplot(path: Path) -> list[dict[str, float]]:
    groups = json.loads(path.read_text())["groups"]
    stats = []
    for key in ("pd_off", "pd_on"):
        g = groups[key]
        stats.append(
            {
                "med": g["median"],
                "q1": g["q1"],
                "q3": g["q3"],
                "whislo": g["whisker_low"],
                "whishi": g["whisker_high"],
                "mean": g["mean"],
                "fliers": [],
            }
        )
    return stats


def _config_range(stem: str | None, axis: str) -> tuple[float, float] | None:
    if stem is None:
        return None
    payload = json.loads((FIGS_DIR / f"{stem}.json").read_text())
    rng = payload.get("panel", {}).get(f"{axis}_range")
    return (float(rng[0]), float(rng[1])) if rng else None


def _fit(
    rng: tuple[float, float] | None, lo: float, hi: float, *, tol: float
) -> tuple[float, float] | None:
    """Return ``rng``, widened to include ``[lo, hi]`` when it overshoots by more than ``tol`` of the span."""
    if rng is None:
        return None
    a, b = rng
    span = b - a
    if lo < a - tol * span:
        a = lo - PAD * span
    if hi > b + tol * span:
        b = hi + PAD * span
    return (a, b)


def _finite_extent(*arrays: np.ndarray) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    return float(vals.min()), float(vals.max())


def _draw_curves(ax: Axes, series: list[_Series]) -> None:
    for s in series:
        key = s.name.lower()
        dashed = key == "threshold" or key.startswith("ci_")
        raw = key == "raw"
        ax.plot(
            s.x,
            s.y,
            label=s.name,
            color=_readable(s.color),
            linewidth=0.9 if raw else 1.6,
            alpha=0.5 if raw else 1.0,
            linestyle="--" if dashed else "-",
            marker=None if dashed or len(s.x) > 30 else "o",
            markersize=2.5,
        )
    ax.legend(fontsize=8, frameon=False)


def _draw_boxplot(ax: Axes, stats: list[dict[str, float]]) -> None:
    ax.bxp(
        stats,
        positions=[1, 2],
        widths=0.5,
        showmeans=True,
        meanprops={"marker": "x", "markeredgecolor": _figure_style().NEUTRAL, "markersize": 6},
    )
    ax.set_xticks([1, 2], ["PD Off", "PD On"])
    ax.legend(
        handles=[Line2D([0], [0], marker="x", color=_figure_style().NEUTRAL, linestyle="", label="mean")],
        fontsize=8,
        frameon=False,
    )


def _figure_style():
    """The panels' shared style module (transparent background, neutral grey chrome)."""
    if "figure_style" in sys.modules:
        return sys.modules["figure_style"]
    spec = importlib.util.spec_from_file_location("figure_style", FIGURE_STYLE)
    if spec is None or spec.loader is None:
        msg = f"cannot load figure style from {FIGURE_STYLE}"
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def draw_panel(key: str, panel: Panel, *, artifacts: Path, out_root: Path, version: int) -> Path | None:
    """Draw one panel and save it. Returns the output path, or ``None`` if an input is missing."""
    sources = [artifacts / ax.source for ax in panel.axes]
    missing = [p for p in sources if not p.is_file()]
    if missing:
        for p in missing:
            _warn(f"skip {key}: missing input {p}")
        return None

    style = _figure_style()
    style.apply()
    rows, cols = _GRID[len(panel.axes)]
    fig = Figure(figsize=_FIGSIZE[len(panel.axes)], dpi=DPI)
    grid = fig.subplots(rows, cols, squeeze=False).ravel()

    for ax, spec, src in zip(grid, panel.axes, sources, strict=True):
        if spec.boxplot:
            stats = _read_boxplot(src)
            lo = min(s["whislo"] for s in stats)
            hi = max(s["whishi"] for s in stats)
            _draw_boxplot(ax, stats)
            x_rng = None
        else:
            series = _read_curves(src, bin_near_duplicates=spec.bin_near_duplicates)
            if not series:
                _warn(f"skip {key}: no series in {src}")
                return None
            _draw_curves(ax, series)
            xs = np.concatenate([s.x for s in series])
            ys = np.concatenate([s.y for s in series])
            x_lo, x_hi = _finite_extent(xs)
            lo, hi = _finite_extent(ys)
            x_rng = _fit(_config_range(spec.config, "x"), x_lo, x_hi, tol=X_TOLERANCE)
        y_rng = _fit(_config_range(spec.config, "y") or spec.ylim, lo, hi, tol=0.0)
        if x_rng is not None:
            ax.set_xlim(*x_rng)
        if y_rng is not None:
            ax.set_ylim(*y_rng)
        if spec.xlabel:
            ax.set_xlabel(spec.xlabel)
        ax.set_ylabel(spec.ylabel)

    fig.suptitle(f"Digitized from Fig. {panel.fig} of {CITATIONS[panel.paper]}", fontsize=12)
    fig.text(0.5, 0.012, FOOTER, ha="center", va="bottom", fontsize=8.5, color=style.NEUTRAL)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))

    out = doc_paths.figure_images_dir(panel.paper, panel.fig, out_root) / f"digitized_v{version}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    style.finalize(fig)
    fig.savefig(out, dpi=DPI)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--panel",
        action="append",
        default=[],
        metavar="PAPER/PANEL",
        help="e.g. mehregan/4a or nguyen/6 (repeatable)",
    )
    parser.add_argument("--all", action="store_true", help="draw every known panel")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help=f"artifacts root holding <paper>/... digitization JSON (default {DEFAULT_ARTIFACTS})",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help=(
            "repo-shaped root; writes docs/controllers/<controller>/figures/images/<panel>/ "
            f"under it (default {DEFAULT_OUT_ROOT})"
        ),
    )
    parser.add_argument("--version", type=int, default=1, help="output suffix: digitized_v<N>.png (default 1)")
    args = parser.parse_args(argv)

    if args.all == bool(args.panel):
        parser.error("pass --all or one or more --panel PAPER/PANEL")
    if args.version < 1:
        parser.error("--version must be >= 1")
    keys = list(PANELS) if args.all else args.panel
    unknown = [k for k in keys if k not in PANELS]
    if unknown:
        parser.error(f"unknown panel(s) {unknown}; known: {sorted(PANELS)}")

    drawn: list[str] = []
    for key in keys:
        out = draw_panel(
            key,
            PANELS[key],
            artifacts=args.artifacts,
            out_root=args.out_root,
            version=args.version,
        )
        if out is not None:
            drawn.append(key)
            print(f"wrote {out}")

    skipped = [k for k in keys if k not in drawn]
    if skipped:
        print(f"skipped: {', '.join(skipped)}", file=sys.stderr)
    if not drawn:
        print("error: nothing drawn", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
