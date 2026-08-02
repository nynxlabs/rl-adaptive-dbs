"""Compare two digitization-schema JSON files (PIL / WPD / Engauge).

Uses full ``xy`` traces only (no convenience stats). Reports n, RMSE,
Pearson r on the reference x-grid, optional beta-band means, and
Mehregan-1b-style ordering when ``pd`` / ``healthy`` / ``pd_130hz`` keys
are present.

Usage:

    uv run python scripts/digitization/compare_curves.py \\
        --ref artifacts/.../curves_wpd_refined.json \\
        --hyp artifacts/.../curves_pil.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _xy(series: dict) -> tuple[np.ndarray, np.ndarray] | None:
    xy = series.get("xy")
    if not xy or "x" not in xy or "y" not in xy:
        return None
    return np.asarray(xy["x"], float), np.asarray(xy["y"], float)


def _band_mean(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    m = (x >= lo) & (x <= hi)
    if m.sum() == 0:
        return float("nan")
    return float(y[m].mean())


def compare(ref: dict, hyp: dict, *, beta_lo: float = 13.0, beta_hi: float = 35.0) -> dict:
    out: dict = {"series": {}, "gate": {}}
    ref_s = ref.get("series", {})
    hyp_s = hyp.get("series", {})
    for k in sorted(set(ref_s) | set(hyp_s)):
        row: dict = {"in_ref": k in ref_s, "in_hyp": k in hyp_s}
        if k in ref_s:
            row["n_ref"] = ref_s[k].get("n")
            if "color_rgba" in ref_s[k]:
                row["color_ref"] = ref_s[k]["color_rgba"]
        if k in hyp_s:
            row["n_hyp"] = hyp_s[k].get("n")
            if "color_rgba" in hyp_s[k]:
                row["color_hyp"] = hyp_s[k]["color_rgba"]
        if k in ref_s and k in hyp_s:
            rxy, hxy = _xy(ref_s[k]), _xy(hyp_s[k])
            if rxy is not None and hxy is not None:
                rx, ry = rxy
                hx, hy = hxy
                if rx.size >= 4 and hx.size >= 4 and (rx.max() - rx.min()) > 0:
                    hyp_on_ref = np.interp(rx, hx, hy)
                    row["rmse"] = float(np.sqrt(np.mean((ry - hyp_on_ref) ** 2)))
                    row["pearson_r"] = float(np.corrcoef(ry, hyp_on_ref)[0, 1])
                    row["beta_mean"] = {
                        "ref": _band_mean(rx, ry, beta_lo, beta_hi),
                        "hyp": _band_mean(hx, hy, beta_lo, beta_hi),
                    }
        out["series"][k] = row

    def band(src: dict, key: str) -> float | None:
        s = src.get("series", {}).get(key)
        if not s:
            return None
        xy = _xy(s)
        if xy is None:
            return None
        return _band_mean(xy[0], xy[1], beta_lo, beta_hi)

    for label, src in (("ref", ref), ("hyp", hyp)):
        pd_v, healthy, hz = band(src, "pd"), band(src, "healthy"), band(src, "pd_130hz")
        if pd_v is not None and healthy is not None:
            out["gate"][f"{label}_pd_gt_healthy"] = pd_v > healthy
        if pd_v is not None and hz is not None:
            out["gate"][f"{label}_130_lt_pd"] = hz < pd_v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--hyp", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="optional machine-readable report path")
    args = ap.parse_args()

    ref = json.loads(args.ref.read_text())
    hyp = json.loads(args.hyp.read_text())
    report = compare(ref, hyp)

    print(f"=== compare: ref={args.ref.name} ({ref.get('method')})  "
          f"hyp={args.hyp.name} ({hyp.get('method')}) ===")
    for k, row in report["series"].items():
        print(f"\n[{k}] in_ref={row['in_ref']} in_hyp={row['in_hyp']}  "
              f"n_ref={row.get('n_ref')} n_hyp={row.get('n_hyp')}")
        if "rmse" in row:
            print(f"  rmse={row['rmse']:.4f}  pearson_r={row['pearson_r']:.4f}")
            bm = row["beta_mean"]
            print(f"  beta_mean  ref={bm['ref']:.4f}  hyp={bm['hyp']:.4f}")
    print("\n=== gates (beta-band means from xy) ===")
    for k, v in report["gate"].items():
        print(f"  {k}: {v}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
