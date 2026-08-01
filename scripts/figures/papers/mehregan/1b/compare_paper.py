#!/usr/bin/env python3
"""Compare WPD-digitized paper curves against our replication curves.

Mehregan fig 1b reference implementation. Digitized CSV comes from
WebPlotDigitizer (aligned XY axes, 0-50 Hz / raw PSD); replication
curves come from the panel's curves.json cache.

Outputs a per-condition comparison table: RMSE, Pearson r, beta-band
(13-35 Hz) means and ratios, and the ordering gate (PD > healthy,
130 Hz < PD on beta power).

Usage:
  uv run python scripts/figures/papers/mehregan/1b/compare_paper.py
  uv run python scripts/figures/papers/mehregan/1b/compare_paper.py \
      --csv artifacts/figures/papers/mehregan/1b/paper_digitization/fig1b_paper_digitized.csv \
      --curves artifacts/figures/papers/mehregan/1b/curves.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FIGURE_DIR = Path("artifacts/figures/papers/mehregan/1b")
DEFAULT_CSV = FIGURE_DIR / "paper_digitization" / "fig1b_paper_digitized.csv"
DEFAULT_CURVES = FIGURE_DIR / "curves.json"

BETA_LO, BETA_HI = 13.0, 35.0
WINDOW_LO, WINDOW_HI = 1.0, 50.0

# WPD series name -> our curves.json key
SERIES_MAP = {
    "PD no Treatment": "pd",
    "Healthy Control": "healthy",
    "PD 130 Hz Treatment": "pd_130hz",
}


def load_wpd(path: Path) -> dict[str, np.ndarray]:
    """Parse WPD multi-series CSV into {name: (x, y)}."""
    rows = list(csv.reader(open(path)))
    names = [rows[0][0], rows[0][2], rows[0][4]]
    data: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    for r in rows[2:]:
        for i, n in enumerate(names):
            x, y = r[i * 2], r[i * 2 + 1]
            if x and y:
                data[n].append((float(x), float(y)))
    return {n: np.asarray(v).T for n, v in data.items() if v}


def load_replication(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load curves.json into {key: {freqs_hz, psd}} restricted to 1-50 Hz."""
    payload = json.loads(path.read_text())
    out: dict[str, dict[str, np.ndarray]] = {}
    for c in payload["curves"]:
        f = np.asarray(c["freqs_hz"])
        psd = np.asarray(c["psd"])
        m = (f >= WINDOW_LO) & (f <= WINDOW_HI)
        out[c["key"]] = {"freqs_hz": f[m], "psd": psd[m]}
    return out


def interp_to_grid(rep_x: np.ndarray, rep_y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate replication curve onto the digitized x positions."""
    return np.interp(grid, rep_x, rep_y)


def compare_pair(wpd_x: np.ndarray, wpd_y: np.ndarray,
                 rep_x: np.ndarray, rep_y: np.ndarray) -> dict[str, float]:
    """RMSE, Pearson r, and peak metrics between one paper curve and ours."""
    grid = wpd_x
    ours = interp_to_grid(rep_x, rep_y, grid)
    rmse = float(np.sqrt(np.mean((wpd_y - ours) ** 2)))
    r = float(np.corrcoef(wpd_y, ours)[0, 1]) if len(wpd_y) > 2 else float("nan")
    return {"n": len(wpd_y), "rmse": rmse, "pearson_r": r}


def beta_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    m = (x >= BETA_LO) & (x <= BETA_HI)
    if m.sum() == 0:
        return {"beta_mean": float("nan"), "beta_max": float("nan")}
    return {"beta_mean": float(y[m].mean()), "beta_max": float(y[m].max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    ap.add_argument("--json", type=Path, help="write machine-readable report")
    args = ap.parse_args()

    wpd = load_wpd(args.csv)
    rep = load_replication(args.curves)

    print("=== fig 1b: paper (WPD) vs replication ===")
    print(f"{'condition':24s} {'n':>3s} {'rmse':>8s} {'pearson_r':>9s}  {'paper beta':>22s} {'our beta':>22s}")
    report = {"figure": "mehregan_fig1b", "conditions": {}}
    beta_paper: dict[str, float] = {}
    beta_ours: dict[str, float] = {}

    for wpd_name, our_key in SERIES_MAP.items():
        if wpd_name not in wpd or our_key not in rep:
            print(f"  (skip {wpd_name} / {our_key}: missing)")
            continue
        wx, wy = wpd[wpd_name]
        rk = rep[our_key]
        cmp = compare_pair(wx, wy, rk["freqs_hz"], rk["psd"])
        pb = beta_stats(wx, wy)
        ob = beta_stats(rk["freqs_hz"], rk["psd"])
        beta_paper[wpd_name] = pb["beta_mean"]
        beta_ours[our_key] = ob["beta_mean"]
        print(f"{wpd_name:24s} {cmp['n']:3d} {cmp['rmse']:8.2f} {cmp['pearson_r']:9.3f}  "
              f"{pb['beta_mean']:10.2f} (max {pb['beta_max']:.2f})  "
              f"{ob['beta_mean']:10.2f} (max {ob['beta_max']:.2f})")
        report["conditions"][wpd_name] = {**cmp, **pb, **ob}

    # ordering gate: PD > healthy on beta, 130 Hz < PD
    print("\n=== ordering gate (beta 13-35 Hz mean) ===")
    gate = {}
    if all(k in beta_paper for k in ("PD no Treatment", "Healthy Control")):
        gate["paper_pd_gt_healthy"] = bool(beta_paper["PD no Treatment"] > beta_paper["Healthy Control"])
        gate["ours_pd_gt_healthy"] = bool(beta_ours["pd"] > beta_ours["healthy"])
    if all(k in beta_paper for k in ("PD no Treatment", "PD 130 Hz Treatment")):
        gate["paper_130_lt_pd"] = bool(beta_paper["PD 130 Hz Treatment"] < beta_paper["PD no Treatment"])
        gate["ours_130_lt_pd"] = bool(beta_ours["pd_130hz"] < beta_ours["pd"])
    for k, v in gate.items():
        print(f"  {k}: {v}")
    report["gate"] = gate
    report["gate_pass"] = all(v for v in gate.values())

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nreport written: {args.json}")


if __name__ == "__main__":
    main()
