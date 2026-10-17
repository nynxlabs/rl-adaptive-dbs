"""Read the Nguyen Fig 3 boxplot (panel b) and mean lines (panel a) off the paper image.

Fig 3 is a scatter + boxplot, not a curve, so WebPlotDigitizer's curve tracing does not
apply. This reads the five box statistics per group (whisker caps, Q1, median, Q3) and
the two mean lines from pixel rows, with the y axis calibrated on the eight tick marks
(50–400). One pixel is ~1.4 units of α–β power.

    uv run python scripts/digitization/nguyen_fig3_boxplot.py

Writes ``artifacts/figures/papers/nguyen/paper_digitization/curves_fig3_boxplot.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PAPER_PNG = REPO / "figures" / "nguyen" / "images" / "3" / "paper.png"
OUT = REPO / "artifacts" / "figures" / "papers" / "nguyen" / "paper_digitization" / "curves_fig3_boxplot.json"

TICK_VALUES = (400, 350, 300, 250, 200, 150, 100, 50)  # top to bottom
BOX_AXIS_X = 401  # panel (b) left spine
SCATTER_X = (70, 340)  # panel (a) plot area columns for the mean lines
GROUP_COLUMNS = {"pd_off": (440, 545), "pd_on": (570, 680)}


def _calibration(dark: np.ndarray) -> tuple[float, float]:
    rows = [y for y in range(dark.shape[0]) if dark[y, BOX_AXIS_X - 4 : BOX_AXIS_X].sum() >= 3]
    if len(rows) != len(TICK_VALUES):
        msg = f"expected {len(TICK_VALUES)} tick rows, found {rows}"
        raise RuntimeError(msg)
    slope, intercept = np.polyfit(rows, TICK_VALUES, 1)
    return float(slope), float(intercept)


def _row_centres(mask: np.ndarray, min_px: int) -> list[float]:
    """Centres of runs of consecutive rows that have >= min_px marked pixels."""
    rows = [y for y in range(mask.shape[0]) if mask[y].sum() >= min_px]
    out: list[list[int]] = []
    for y in rows:
        if out and y - out[-1][-1] == 1:
            out[-1].append(y)
        else:
            out.append([y])
    return [float(np.mean(r)) for r in out]


def read_fig3(png: Path = PAPER_PNG) -> dict:
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    dark = im.sum(axis=2) < 500
    slope, intercept = _calibration(dark)

    def value(y: float) -> float:
        return round(slope * y + intercept, 1)

    groups: dict[str, dict[str, float]] = {}
    for name, (x0, x1) in GROUP_COLUMNS.items():
        seg = im[:, x0:x1]
        seg_dark = dark[:, x0:x1]
        # Box edges are ~22 px wide; caps ~11 px; frame lines span the whole window.
        width = seg_dark.sum(axis=1)
        frame = width > 0.8 * (x1 - x0)
        edges = _row_centres(seg_dark & (width[:, None] >= 18) & ~frame[:, None], 18)
        caps = _row_centres(seg_dark & (width[:, None] >= 9) & (width[:, None] < 18), 9)
        caps = [c for c in caps if c < 320]  # drop the x tick labels below the axis (frame at y≈323)
        orange = (seg[..., 0] > 180) & (seg[..., 2] < 150) & (seg[..., 0] - seg[..., 2] > 60)
        medians = _row_centres(orange, 5)
        if len(edges) != 2 or len(caps) != 2 or len(medians) != 1:
            msg = f"{name}: edges {edges}, caps {caps}, medians {medians}"
            raise RuntimeError(msg)
        groups[name] = {
            "whisker_high": value(min(caps)),
            "q3": value(min(edges)),
            "median": value(medians[0]),
            "q1": value(max(edges)),
            "whisker_low": value(max(caps)),
        }
    seg = im[:, SCATTER_X[0] : SCATTER_X[1]]
    red = (seg[..., 0] > 200) & (seg[..., 1] < 80) & (seg[..., 2] < 80)
    black = seg.sum(axis=2) < 120
    (red_y,) = _row_centres(red, 100)
    (black_y,) = _row_centres(black, 100)
    groups["pd_off"]["mean"] = value(red_y)
    groups["pd_on"]["mean"] = value(black_y)
    return {
        "figure": "nguyen_fig3",
        "method": "pil-pixel-read",
        "source": "figures/nguyen/images/3/paper.png",
        "units": "GPi alpha-beta oscillation power (paper y axis)",
        "y_per_pixel": round(abs(slope), 3),
        "note": (
            "Box statistics from panel (b) (caps, box edges, orange median); means from the "
            "red (PD Off) and black (PD On) lines of panel (a). Axis calibrated on 8 ticks."
        ),
        "groups": groups,
    }


def main() -> int:
    data = read_fig3()
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data["groups"], indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
