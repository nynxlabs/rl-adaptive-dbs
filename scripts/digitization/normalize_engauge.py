"""Normalize an Engauge Digitizer CSV export into the digitization schema.

Supports the common Engauge export layouts (Settings → Export Setup):

1. **Shared-x multi-column** (default-ish):
       x,Curve1,Curve2
       0.1,1.2,3.4
   First column is x; later columns are y per named curve.

2. **Raw Xs and Ys, all curves on each line** (ccp / petrobras style):
       x,Curve1,<optional>
       0.1,1.2
       0.2,1.3
       x,Curve2,<optional>
       ...
   A header row whose first cell is ``x`` starts a new named curve;
   following rows are ``x,y`` pairs until the next header.

3. **Simple two-column** (single curve):
       x,y
       0.1,1.2
   Use ``--curve-name`` (default ``series``) for the schema key.

Also accepts a comma-separated ``--series-map`` to rename Engauge curve
labels to canonical keys (same convention as ``normalize_wpd.py``).

Usage:

    uv run python scripts/digitization/normalize_engauge.py \\
        --input artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_engauge.csv \\
        --out artifacts/figures/papers/mehregan/1b/paper_digitization/curves_engauge.json \\
        --figure mehregan_fig1b \\
        --series-map 'PD no Treatment=pd,Healthy Control=healthy,PD 130 Hz Treatment=pd_130hz'
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
    "Healthly Control": "healthy",  # common typo from morning WPD export
    "PD 130 Hz Treatment": "pd_130hz",
}


def _is_float(s: str) -> bool:
    try:
        float(s.replace(",", "."))
        return True
    except (TypeError, ValueError):
        return False


def _f(s: str) -> float:
    return float(s.replace(",", "."))


def parse_engauge_csv(path: Path, default_curve: str = "series") -> dict[str, np.ndarray]:
    """Parse Engauge CSV → {curve_name: (x, y)} arrays."""
    rows = [r for r in csv.reader(path.open()) if r and not (r[0].startswith("#"))]
    if not rows:
        raise ValueError(f"empty Engauge CSV: {path}")

    # Detect layout from the first non-empty row.
    header = rows[0]
    header_l = [c.strip() for c in header]

    if header_l[0].lower() == "x" and len(header_l) >= 2:
        named = [c for c in header_l[1:] if c and not _is_float(c)]
        # Shared-x multi-column: x,Curve1,Curve2,... (2+ named y columns)
        if len(named) >= 2:
            names = header_l[1:]
            data: dict[str, list[tuple[float, float]]] = {n: [] for n in names if n}
            for r in rows[1:]:
                if not r or not _is_float(r[0]):
                    continue
                x = _f(r[0])
                for i, n in enumerate(names):
                    if not n:
                        continue
                    if i + 1 < len(r) and r[i + 1] and _is_float(r[i + 1]):
                        data[n].append((x, _f(r[i + 1])))
            return {n: np.asarray(v).T for n, v in data.items() if v}
        # Raw Xs/Ys blocks: x,<one curve name>[,optional numeric]
        if len(named) == 1:
            return _parse_raw_blocks(rows)

    # Layout 3: simple two-column (possibly with x,y header)
    if len(header_l) >= 2:
        start = 1 if (not _is_float(header_l[0]) or header_l[0].lower() in {"x", "x"}) else 0
        if start == 0 and _is_float(header_l[0]) and _is_float(header_l[1]):
            start = 0
        elif not _is_float(header_l[0]):
            start = 1
        pts: list[tuple[float, float]] = []
        for r in rows[start:]:
            if len(r) >= 2 and _is_float(r[0]) and _is_float(r[1]):
                pts.append((_f(r[0]), _f(r[1])))
        if not pts:
            raise ValueError(f"could not parse Engauge CSV layout: {path}")
        return {default_curve: np.asarray(pts).T}

    raise ValueError(f"unrecognized Engauge CSV layout: {path}")


def _parse_raw_blocks(rows: list[list[str]]) -> dict[str, np.ndarray]:
    """Parse 'Raw Xs and Ys; all curves on each line' block format."""
    curves: dict[str, list[tuple[float, float]]] = {}
    current: str | None = None
    for r in rows:
        cells = [c.strip() for c in r]
        if not cells:
            continue
        if cells[0].lower() == "x" and len(cells) >= 2 and not _is_float(cells[1]):
            current = cells[1]
            curves.setdefault(current, [])
            continue
        if current is None:
            continue
        if len(cells) >= 2 and _is_float(cells[0]) and _is_float(cells[1]):
            curves[current].append((_f(cells[0]), _f(cells[1])))
    return {n: np.asarray(v).T for n, v in curves.items() if v}


def apply_series_map(data: dict[str, np.ndarray], series_map: dict[str, str]) -> dict[str, np.ndarray]:
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
    ap.add_argument("--curve-name", default="series",
                    help="name for simple 2-column exports (default: series)")
    ap.add_argument("--series-map", default=None,
                    help="comma-separated 'src=dst' pairs; defaults cover mehregan fig 1b")
    args = ap.parse_args()

    series_map = dict(DEFAULT_SERIES_MAP)
    if args.series_map:
        series_map.update(
            dict(pair.split("=", 1) for pair in args.series_map.split(",") if "=" in pair)
        )

    data = parse_engauge_csv(args.input, default_curve=args.curve_name)
    data = apply_series_map(data, series_map)

    series = {name: series_record(xy[0], xy[1]) for name, xy in data.items()}
    result = {
        "figure": args.figure,
        "method": "engauge-csv",
        "source": str(args.input),
        "series": series,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}: {len(result['series'])} series (engauge-csv)")


if __name__ == "__main__":
    main()
