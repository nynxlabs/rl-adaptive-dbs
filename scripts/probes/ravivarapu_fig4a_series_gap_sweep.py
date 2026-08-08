#!/usr/bin/env python3
"""Counterfactual baseline-PSD window sweeps on cached Fig 4a series.

Loads ``artifacts/figures/papers/ravivarapu/4/series.json`` (or ``--series``)
and applies piecewise constant shifts to the baseline ``episode_psd`` trace.
Reports which deltas pass ``gates.shape_pass`` without a full retrain.

Example::

  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig4a_series_gap_sweep.py
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig4a_series_gap_sweep.py \\
    --midlate-delta -0.012 --late-delta 0.003
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_PLOT = ROOT / "scripts" / "figures" / "papers" / "ravivarapu" / "4a" / "plot.py"
_spec = importlib.util.spec_from_file_location("ravivarapu_4a_plot", _PLOT)
assert _spec and _spec.loader
_plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plot)

DEFAULT_SERIES = ROOT / "artifacts/figures/papers/ravivarapu/4/series.json"


def _load_series(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_copy(series: dict) -> np.ndarray:
    return np.asarray(series["variants"]["baseline"]["episode_psd"], dtype=float).copy()


def evaluate_series(baseline: np.ndarray, series: dict) -> dict:
    mod = {
        "seed": series.get("seed", 0),
        "smoke": False,
        "variants": {
            "baseline": {"episode_psd": baseline.tolist()},
            "paper": series["variants"]["paper"],
        },
    }
    return _plot.evaluate_gates(mod)


def _metrics(gates: dict) -> dict[str, float]:
    m = gates.get("paper_gate_metrics") or gates.get("gate_metrics") or {}
    return {
        "gap_midlate": float(m.get("gap_midlate", float("nan"))),
        "gap_late": float(m.get("gap_late_window", m.get("late_gap", float("nan")))),
        "pearson_baseline": float(m.get("pearson_baseline", float("nan"))),
    }


def _apply_windows(
    baseline: np.ndarray,
    *,
    midlate_lo: int,
    midlate_hi: int,
    midlate_delta: float,
    late_lo: int,
    late_hi: int,
    late_delta: float,
) -> np.ndarray:
    out = baseline.copy()
    if midlate_delta != 0.0:
        out[midlate_lo:midlate_hi] += midlate_delta
    if late_delta != 0.0:
        out[late_lo:late_hi] += late_delta
    return out


def _print_row(
    label: str,
    gates: dict,
    *,
    midlate_delta: float = 0.0,
    late_delta: float = 0.0,
) -> None:
    met = _metrics(gates)
    fails = [
        k
        for k, v in gates.items()
        if k.startswith("dig_") and isinstance(v, bool) and not v
    ]
    print(
        f"{label}: shape_pass={gates.get('shape_pass')} "
        f"midlate={met['gap_midlate']:.4f} late={met['gap_late']:.4f} "
        f"pearson={met['pearson_baseline']:.4f} "
        f"d_ml={midlate_delta:+.4f} d_l={late_delta:+.4f} "
        f"fails={fails or 'none'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--midlate-lo", type=int, default=80)
    parser.add_argument("--midlate-hi", type=int, default=120)
    parser.add_argument("--late-lo", type=int, default=120)
    parser.add_argument("--late-hi", type=int, default=150)
    parser.add_argument("--midlate-delta", type=float, default=None)
    parser.add_argument("--late-delta", type=float, default=None)
    parser.add_argument("--grid", action="store_true", help="Sweep 2D delta grid")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/probes/ravivarapu_fig4a_series_gap_sweep.json"),
    )
    args = parser.parse_args()

    series = _load_series(args.series)
    baseline = _baseline_copy(series)
    raw_gates = evaluate_series(baseline, series)
    print(f"series={args.series}", flush=True)
    _print_row("raw", raw_gates)

    if args.midlate_delta is not None or args.late_delta is not None:
        d_ml = 0.0 if args.midlate_delta is None else args.midlate_delta
        d_l = 0.0 if args.late_delta is None else args.late_delta
        mod = _apply_windows(
            baseline,
            midlate_lo=args.midlate_lo,
            midlate_hi=args.midlate_hi,
            midlate_delta=d_ml,
            late_lo=args.late_lo,
            late_hi=args.late_hi,
            late_delta=d_l,
        )
        _print_row("single", evaluate_series(mod, series), midlate_delta=d_ml, late_delta=d_l)
        return

    if not args.grid:
        parser.error("Specify --grid or at least one of --midlate-delta / --late-delta")

    passing: list[dict] = []
    for d_ml in np.arange(-0.018, 0.002, 0.001):
        for d_l in np.arange(0.0, 0.014, 0.001):
            mod = _apply_windows(
                baseline,
                midlate_lo=args.midlate_lo,
                midlate_hi=args.midlate_hi,
                midlate_delta=float(d_ml),
                late_lo=args.late_lo,
                late_hi=args.late_hi,
                late_delta=float(d_l),
            )
            gates = evaluate_series(mod, series)
            if gates.get("shape_pass"):
                met = _metrics(gates)
                passing.append(
                    {
                        "midlate_delta": float(d_ml),
                        "late_delta": float(d_l),
                        **met,
                    }
                )

    passing.sort(key=lambda r: (-r["pearson_baseline"], r["midlate_delta"]))
    print(f"shape_pass combos: {len(passing)}", flush=True)
    for row in passing[:12]:
        print(
            f"  d_ml={row['midlate_delta']:+.3f} d_l={row['late_delta']:+.3f} "
            f"pearson={row['pearson_baseline']:.4f} "
            f"midlate={row['gap_midlate']:.4f} late={row['gap_late']:.4f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "series": str(args.series),
        "raw": {**_metrics(raw_gates), "shape_pass": raw_gates.get("shape_pass")},
        "passing": passing,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
