"""Normalize a WebPlotDigitizer export into the digitization schema.

Accepts either:
- a WPD project JSON (series names, colors, calibration, pixel+data values), or
- a multi-series CSV export (header row of series names, then X,Y,X,Y...)

Output series keep ``n``, full ``xy`` traces, and ``color_rgba`` when the
source provides a color (JSON only). No convenience stats — gates should
slice ``xy`` by real x.

Usage:

    uv run python scripts/digitization/normalize_wpd.py \\
        --input artifacts/.../fig1b_refined.wpd.json \\
        --out artifacts/.../curves_wpd_refined.json \\
        --figure mehregan_fig1b
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from schema import series_record

DEFAULT_SERIES_MAP = {
    "PD no Treatment": "pd",
    "Healthy Control": "healthy",
    "Healthly Control": "healthy",
    "PD 130 Hz Treatment": "pd_130hz",
    "PD 130Hz Treatment": "pd_130hz",
}


def parse_wpd_json(path: Path) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Return ({name: (x, y)}, {name: color_rgba}).

    Uses the point ``value`` fields WPD writes at click/export time. Do **not**
    recompute from pixels + axes here: many refined projects have stale or
    non-XY calibration metadata, and remapping can shift traces a lot (see
    Mehregan 4b/5a). If Automeris canvas and exported values disagree, re-save
    the project in WPD so ``value`` matches the axes, then re-normalize.
    """
    payload = json.loads(path.read_text())
    out: dict[str, np.ndarray] = {}
    colors: dict[str, list[int]] = {}
    for ds in payload.get("datasetColl", []):
        name = ds.get("name", "series")
        pts = ds.get("data", [])
        arr = np.asarray([p["value"] for p in pts], dtype=float)
        if arr.size == 0:
            continue
        out[name] = arr.T
        col = ds.get("colorRGB")
        if isinstance(col, list) and len(col) >= 3:
            rgba = [int(col[0]), int(col[1]), int(col[2]), int(col[3]) if len(col) > 3 else 255]
            colors[name] = rgba
    return out, colors



def parse_wpd_csv(path: Path) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Read WPD multi-series CSV: return ({name: (x, y)}, {}) — CSV has no colors."""
    rows = list(csv.reader(open(path)))
    names = [rows[0][0], rows[0][2], rows[0][4]]
    data: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    for r in rows[2:]:
        for i, n in enumerate(names):
            x, y = r[i * 2], r[i * 2 + 1]
            if x and y:
                data[n].append((float(x), float(y)))
    return {n: np.asarray(v).T for n, v in data.items() if v}, {}


def apply_series_map(
    data: dict[str, np.ndarray],
    colors: dict[str, list[int]],
    series_map: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    out: dict[str, np.ndarray] = {}
    out_colors: dict[str, list[int]] = {}
    for name, xy in data.items():
        key = series_map.get(name, name)
        if key != name:
            print(f"  mapped '{name}' -> '{key}'", file=__import__("sys").stderr)
        out[key] = xy
        if name in colors:
            out_colors[key] = colors[name]
    return out, out_colors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", default="paper_figure")
    ap.add_argument("--series-map", default=None,
                    help="comma-separated 'src=dst' pairs; defaults cover mehregan fig 1b")
    args = ap.parse_args()

    series_map = dict(DEFAULT_SERIES_MAP)
    if args.series_map:
        series_map.update(
            dict(pair.split("=", 1) for pair in args.series_map.split(",") if "=" in pair)
        )

    if args.input.suffix.lower() == ".json":
        data, colors = parse_wpd_json(args.input)
        source = "wpd-project-json"
    else:
        data, colors = parse_wpd_csv(args.input)
        source = "wpd-csv"

    data, colors = apply_series_map(data, colors, series_map)

    series = {
        name: series_record(xy[0], xy[1], color_rgba=colors.get(name))
        for name, xy in data.items()
    }
    result = {
        "figure": args.figure,
        "method": source,
        "source": str(args.input),
        "series": series,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}: {len(result['series'])} series ({source})")


if __name__ == "__main__":
    main()
