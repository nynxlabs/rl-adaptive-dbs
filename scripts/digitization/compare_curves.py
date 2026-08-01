"""Compare two digitization-schema JSON files (PIL / WPD / Engauge).

Reports per-series n, early/late means, drop, min/max, and interpolated
RMSE + Pearson r on the reference x-grid. Also prints Mehregan-1b-style
ordering checks when the expected keys are present.

Usage:

    uv run python scripts/digitization/compare_curves.py \\
        --ref artifacts/.../curves_wpd.json \\
        --hyp artifacts/.../curves_pil.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _series_xy_from_stats(stats: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort: schema currently stores summary stats, not full traces.

    Prefer an optional ``xy`` payload ``{"x": [...], "y": [...]}`` when present;
    otherwise fall back to sparse ``at`` samples keyed by index (lossier).
    """
    if "xy" in stats and "x" in stats["xy"] and "y" in stats["xy"]:
        return np.asarray(stats["xy"]["x"], float), np.asarray(stats["xy"]["y"], float)
    at = stats.get("at")
    if not at:
        return None
    idxs = sorted(int(k) for k in at)
    # reconstruct a unitless index axis; enough for shape/order checks only
    x = np.asarray(idxs, float)
    y = np.asarray([at[str(i)] for i in idxs], float)
    return x, y


def _beta_mean_from_xy(x: np.ndarray, y: np.ndarray, lo: float = 13.0, hi: float = 35.0) -> float:
    m = (x >= lo) & (x <= hi)
    if m.sum() == 0:
        return float("nan")
    return float(y[m].mean())


def compare(ref: dict, hyp: dict) -> dict:
    out: dict = {"series": {}, "gate": {}}
    ref_s = ref.get("series", {})
    hyp_s = hyp.get("series", {})
    keys = sorted(set(ref_s) | set(hyp_s))
    for k in keys:
        row: dict = {"in_ref": k in ref_s, "in_hyp": k in hyp_s}
        if k in ref_s and k in hyp_s:
            rs, hs = ref_s[k], hyp_s[k]
            for field in (
                "n", "early_mean", "late_mean", "drop_early_to_late",
                "min", "max", "mean", "slope",
            ):
                if field in rs and field in hs:
                    row[field] = {
                        "ref": rs[field],
                        "hyp": hs[field],
                        "delta": hs[field] - rs[field],
                    }
            rxy = _series_xy_from_stats(rs)
            hxy = _series_xy_from_stats(hs)
            if rxy is not None and hxy is not None:
                rx, ry = rxy
                hx, hy = hxy
                # interpolate hyp onto ref x when both look like real x axes
                if rx.size >= 4 and hx.size >= 4 and (rx.max() - rx.min()) > 1:
                    grid = rx
                    hyp_on_ref = np.interp(grid, hx, hy)
                    rmse = float(np.sqrt(np.mean((ry - hyp_on_ref) ** 2)))
                    r = float(np.corrcoef(ry, hyp_on_ref)[0, 1])
                    row["rmse"] = rmse
                    row["pearson_r"] = r
                    row["beta_mean"] = {
                        "ref": _beta_mean_from_xy(rx, ry),
                        "hyp": _beta_mean_from_xy(hx, hy),
                    }
        out["series"][k] = row

    # Ordering gate when mehregan 1b keys present
    def late(d: dict, key: str) -> float | None:
        s = d.get("series", {}).get(key)
        if not s:
            return None
        return float(s.get("late_mean", s.get("mean", float("nan"))))

    for label, src in (("ref", ref), ("hyp", hyp)):
        pd_v = late(src, "pd")
        healthy = late(src, "healthy")
        hz = late(src, "pd_130hz")
        if pd_v is not None and healthy is not None:
            out["gate"][f"{label}_pd_gt_healthy"] = pd_v > healthy
        if pd_v is not None and hz is not None:
            out["gate"][f"{label}_130_lt_pd"] = hz < pd_v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True, help="reference curves.json (e.g. WPD)")
    ap.add_argument("--hyp", type=Path, required=True, help="hypothesis curves.json (e.g. PIL)")
    ap.add_argument("--json", type=Path, help="optional machine-readable report path")
    args = ap.parse_args()

    ref = json.loads(args.ref.read_text())
    hyp = json.loads(args.hyp.read_text())
    report = compare(ref, hyp)

    print(f"=== compare: ref={args.ref.name} ({ref.get('method')})  "
          f"hyp={args.hyp.name} ({hyp.get('method')}) ===")
    for k, row in report["series"].items():
        print(f"\n[{k}] in_ref={row['in_ref']} in_hyp={row['in_hyp']}")
        for field in ("n", "early_mean", "late_mean", "drop_early_to_late", "mean", "max"):
            if field in row:
                d = row[field]
                print(f"  {field:20s}  ref={d['ref']:10.4f}  hyp={d['hyp']:10.4f}  "
                      f"delta={d['delta']:+10.4f}")
        if "rmse" in row:
            print(f"  {'rmse':20s}  {row['rmse']:.4f}")
            print(f"  {'pearson_r':20s}  {row['pearson_r']:.4f}")
            bm = row["beta_mean"]
            print(f"  {'beta_mean':20s}  ref={bm['ref']:.4f}  hyp={bm['hyp']:.4f}")
    print("\n=== gates ===")
    for k, v in report["gate"].items():
        print(f"  {k}: {v}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
