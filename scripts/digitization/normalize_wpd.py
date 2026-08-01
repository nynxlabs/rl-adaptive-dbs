"""Normalize a WebPlotDigitizer export into the digitization schema.

Accepts either:
- a WPD project JSON (the .json exported from WebPlotDigitizer, which
  carries series names, calibration points, and pixel+data values), or
- a multi-series CSV export (header row of series names, then X,Y,X,Y...)

Both are converted to the shared `curves.json` schema (see
`schema.py` / `digitization-schema.md` reference).

Usage:

    uv run python scripts/digitization/normalize_wpd.py \
        --input artifacts/figures/papers/mehregan/1b/paper_digitization/mehregan_fig_1b.wpd.json \
        --out /tmp/curves_wpd.json

    uv run python scripts/digitization/normalize_wpd.py \
        --input artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_paper_digitized.csv \
        --series-map 'PD no Treatment=pd,Healthy Control=healthy,PD 130 Hz Treatment=pd_130hz' \
        --out /tmp/curves_wpd.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from schema import series_stats

DEFAULT_SERIES_MAP = {
    "PD no Treatment": "pd",
    "Healthy Control": "healthy",
    "PD 130 Hz Treatment": "pd_130hz",
}


def parse_wpd_json(path: Path) -> dict[str, np.ndarray]:
    """Read WPD project JSON: return {series_name: (x, y)}."""
    payload = json.loads(path.read_text())
    out: dict[str, np.ndarray] = {}
    for ds in payload.get("datasetColl", []):
        name = ds.get("name", "series")
        pts = ds.get("data", [])
        arr = np.asarray([p["value"] for p in pts], dtype=float)
        if arr.size == 0:
            continue
        out[name] = arr.T  # (2, n) -> (x, y)
    return out


def parse_wpd_csv(path: Path) -> dict[str, np.ndarray]:
    """Read WPD multi-series CSV: return {series_name: (x, y)}."""
    rows = list(csv.reader(open(path)))
    names = [rows[0][0], rows[0][2], rows[0][4]]
    data: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    for r in rows[2:]:
        for i, n in enumerate(names):
            x, y = r[i * 2], r[i * 2 + 1]
            if x and y:
                data[n].append((float(x), float(y)))
    return {n: np.asarray(v).T for n, v in data.items() if v}


def apply_series_map(data: dict[str, np.ndarray], series_map: dict[str, str]) -> dict[str, np.ndarray]:
    """Rename series to canonical keys; warn on unmapped series."""
    out: dict[str, np.ndarray] = {}
    for name, xy in data.items():
        key = series_map.get(name, name)
        if key != name:
            print(f"  mapped '{name}' -> '{key}'", file=__import__("sys").stderr)
        out[key] = xy
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figure", default="paper_figure")
    ap.add_argument("--series-map", default=None,
                    help="comma-separated 'src=dst' pairs; defaults apply for mehregan fig 1b")
    args = ap.parse_args()

    series_map = DEFAULT_SERIES_MAP
    if args.series_map:
        series_map = dict(
            pair.split("=", 1) for pair in args.series_map.split(",") if "=" in pair
        )

    if args.input.suffix.lower() == ".json":
        data = parse_wpd_json(args.input)
        source = "wpd-project-json"
    else:
        data = parse_wpd_csv(args.input)
        source = "wpd-csv"

    data = apply_series_map(data, series_map)

    result = {
        "figure": args.figure,
        "method": source,
        "source": str(args.input),
        "series": {name: series_stats(xy[0], xy[1]) for name, xy in data.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}: {len(result['series'])} series ({source})")


if __name__ == "__main__":
    main()
