"""Automated paper-curve extraction via PIL color-mask tracing.

Reusable implementation of the approach the cursor agent used for the
ravivarapu fig 4a/4b curves.json (jul 2026). Works on clean paper PNGs:
white background, distinct saturated curve colors, small images.

Usage (after `cd` to repo root, with venv active):

    uv run python scripts/digitization/extract_pil.py \
        --png figures/mehregan/images/1b/paper.png \
        --config scripts/digitization/figs/mehregan_fig1b.json \
        --out artifacts/figures/papers/mehregan/1b/paper_digitization/curves_pil.json

The config JSON describes the axis box (pixels), tick calibration
(data ranges), and one color mask per series. See
`scripts/digitization/figs/mehregan_fig1b.json` for the shape.

Pitfall: OCR of tick labels is unreliable on small figures. Use known
tick values from the paper / visual inspection (see
`references/pil-color-mask-digitization.md`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from schema import series_stats


def load_mask(mask: dict) -> "callable":
    """Build a mask function from a config entry.

    Supports either explicit RGB threshold bounds or a target color
    with a tolerance.
    """
    if "r_min" in mask:
        return lambda rgb: (
            (rgb[..., 0] >= mask["r_min"]) & (rgb[..., 0] <= mask["r_max"])
            & (rgb[..., 1] >= mask["g_min"]) & (rgb[..., 1] <= mask["g_max"])
            & (rgb[..., 2] >= mask["b_min"]) & (rgb[..., 2] <= mask["b_max"])
        )
    color = np.asarray(mask["color"], dtype=int)
    tol = mask.get("tol", 60)
    return lambda rgb: np.all(np.abs(rgb.astype(int) - color) <= tol, axis=-1)


def digitize_series(
    rgb: np.ndarray,
    mask_fn,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_bins: int,
    legend_exclude_frac: float,
    legend_exclude_top_frac: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one series inside the axis box; returns (x_vals, y_vals)."""
    top, bottom, left, right = box
    h = bottom - top
    w = right - left
    if h <= 0 or w <= 0:
        raise ValueError(f"bad axis box: {box}")
    mask = mask_fn(rgb)
    mask_roi = np.zeros_like(mask)
    mask_roi[top : bottom + 1, left : right + 1] = mask[
        top : bottom + 1, left : right + 1
    ]
    # exclude legend region(s). two options:
    #  - top band: `legend_exclude_top_frac` of box height, full width
    #    (legends that run across the top of the plot)
    #  - corner: top-right `legend_exclude_frac` of width (default)
    top_band = int(h * legend_exclude_top_frac)
    if top_band > 0:
        mask_roi[top : top + top_band, left : right + 1] = False
    if legend_exclude_frac > 0:
        lx0 = left + int(w * (1 - legend_exclude_frac))
        ly1 = top + int(h * 0.28)
        mask_roi[top:ly1, lx0 : right + 1] = False

    xs: list[float] = []
    ys: list[float] = []
    for bi in range(n_bins):
        x0 = left + int(bi * w / n_bins)
        x1 = left + int((bi + 1) * w / n_bins)
        ys_pix = np.where(mask_roi[:, x0:x1].any(axis=1))[0]
        if len(ys_pix) == 0:
            continue
        y_pix = float(np.median(ys_pix))
        x_mid = 0.5 * (x0 + x1)
        x_val = x_range[0] + (x_mid - left) / w * (x_range[1] - x_range[0])
        y_val = y_range[1] - (y_pix - top) / h * (y_range[1] - y_range[0])
        xs.append(x_val)
        ys.append(y_val)
    return np.asarray(xs), np.asarray(ys)


def extract(config_path: Path, png_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    rgb = np.asarray(Image.open(png_path).convert("RGB"))

    panel = cfg["panel"]
    box = tuple(panel["box"])
    x_range = tuple(panel["x_range"])
    y_range = tuple(panel["y_range"])
    n_bins = panel.get("n_bins", 150)
    legend_exclude_frac = panel.get("legend_exclude_frac", 0.22)
    legend_exclude_top_frac = panel.get("legend_exclude_top_frac", 0.0)

    series = {}
    for name, mask_cfg in panel["series"].items():
        mask_fn = load_mask(mask_cfg)
        x, y = digitize_series(
            rgb, mask_fn, box, x_range, y_range, n_bins,
            legend_exclude_frac, legend_exclude_top_frac,
        )
        if len(x) < 4:
            print(f"  warn: '{name}' produced only {len(x)} points", file=sys.stderr)
            continue
        stats = series_stats(x, y)
        stats["xy"] = {"x": x.tolist(), "y": y.tolist()}
        series[name] = stats

    return {
        "figure": cfg.get("figure", png_path.stem),
        "method": "pil-color-mask",
        "source_png": str(png_path),
        "config": str(config_path),
        "box": list(box),
        "x_range": list(x_range),
        "y_range": list(y_range),
        "series": series,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--png", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = extract(args.config, args.png)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}: {len(result['series'])} series")


if __name__ == "__main__":
    main()
