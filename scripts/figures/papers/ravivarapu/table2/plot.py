#!/usr/bin/env python3
"""Ravivarapu Table II — average PSD with the plant reseeded every n steps.

Paper §V.B: "changing the environment seed every n ∈ {10, 20, 50, 75} step";
Table II reports average PSD and average reward for Baseline and SEA-DBS.
Eval-only on the Fig 4 checkpoints: one deterministic (argmax) rollout of 30
blocks of n steps per variant and interval at 50 Hz on the training plant
setup, the plant reset with the next seed at every block, so every n visits
plant seeds 0–29 (``evaluate_reseeded``;
convention in ``docs/controllers/sea_dbs/replication.md`` §14.13). Avg PSD is
the mean Eq. (5) observation × 1000 (paper units). Avg Reward is reported
both as the mean step reward and as Eq. (7) of Avg PSD (the paper's column
equals the latter in all eight cells); gates use PSD only.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/table2/plot.py --export-notes --update-report
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/table2/plot.py --plot-only
"""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("RL_DBS_MAX_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.config import INFERENCE_CARRIER_50HZ
from controllers.sea_dbs.eval import TABLE2_N_BLOCKS, TABLE2_RESEED_INTERVALS, table2_horizon

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_PARALLEL_SERIES = Path(__file__).resolve().parents[2] / "parallel_series.py"
_parallel_series = load_script_module("figure_parallel_series", _PARALLEL_SERIES)

_COMMON = Path(__file__).resolve().parents[1] / "common.py"
_common = load_script_module("ravivarapu_common", _COMMON)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import (  # noqa: E402
    TABLE2_PAPER_PSD,
    TABLE2_PAPER_REWARD_E2,
    merge_gate_report,
    ravivarapu_table2_gates,
)

CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/table2")
FIGURES_DIR = Path("figures/ravivarapu/images/table2")
OUT_STEM = "avg_psd"
VARIANTS = ("baseline", "paper")
PSD_UNITS = 1000.0


def _rows(cells: dict[str, dict[str, Any]]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for n in TABLE2_RESEED_INTERVALS:
        b = cells.get(f"baseline_{n}")
        s = cells.get(f"paper_{n}")
        if b is None or s is None:
            continue
        rows[n] = {
            "baseline_psd": PSD_UNITS * float(b["avg_p_beta_norm"]),
            "sea_psd": PSD_UNITS * float(s["avg_p_beta_norm"]),
            "baseline_reward_e2": 100.0 * float(b["reward_of_avg_psd"]),
            "sea_reward_e2": 100.0 * float(s["reward_of_avg_psd"]),
            "baseline_mean_step_reward_e2": 100.0 * float(b["avg_reward"]),
            "sea_mean_step_reward_e2": 100.0 * float(s["avg_reward"]),
            "baseline_stim_frac": float(b["stim_frac"]),
            "sea_stim_frac": float(s["stim_frac"]),
        }
    return rows


def _plot(rows: dict[int, dict[str, float]], png_path: Path, *, promote: bool) -> None:
    ns = [n for n in TABLE2_RESEED_INTERVALS if n in rows]
    xs = np.arange(len(ns), dtype=float)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs - w / 2, [rows[n]["baseline_psd"] for n in ns], w, label="Baseline (ours)", color="#1f77b4")
    ax.bar(xs + w / 2, [rows[n]["sea_psd"] for n in ns], w, label="SEA-DBS (ours)", color="#2ca02c")
    ax.plot(
        xs - w / 2, [TABLE2_PAPER_PSD[n][0] for n in ns], "k_", markersize=22, markeredgewidth=2.5,
        label="Baseline (paper)",
    )
    ax.plot(
        xs + w / 2, [TABLE2_PAPER_PSD[n][1] for n in ns], "_", color="0.45", markersize=22,
        markeredgewidth=2.5, label="SEA-DBS (paper)",
    )
    all_y = [rows[n][k] for n in ns for k in ("baseline_psd", "sea_psd")]
    all_y += [v for n in ns for v in TABLE2_PAPER_PSD[n]]
    lo, hi = min(all_y), max(all_y)
    pad = 0.15 * (hi - lo + 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(xs, [str(n) for n in ns])
    ax.set_xlabel("Environment seed changed every n steps")
    ax.set_ylabel("Avg PSD")
    ax.set_title("Table II — avg PSD (50 Hz, 300-step rollout)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(_common.linked_png(png_path, _figure_promote, promote=promote), dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="2 blocks, n = 10 and 20 only")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached series.json (no eval)")
    _resume_cli.add_export_notes_arg(parser)
    _resume_cli.add_update_report3_arg(parser)
    _parallel_series.add_parallel_series_argument(parser)
    _common.add_no_promote_arg(parser)
    args = parser.parse_args()
    if args.no_promote:
        args.export_notes = False
        args.update_report = False
    _resume_cli.configure_promote_publish(args, _figure_promote)
    series_path = CACHE_DIR / "series.json"
    n_blocks = 2 if args.smoke else TABLE2_N_BLOCKS
    intervals = (10, 20) if args.smoke else TABLE2_RESEED_INTERVALS
    if args.plot_only:
        if not series_path.is_file():
            raise SystemExit(f"missing series cache: {series_path}")
        payload = json.loads(series_path.read_text(encoding="utf-8"))
        cells = payload["cells"]
        n_blocks = int(payload.get("n_blocks", n_blocks))
    else:
        jobs = [
            _parallel_series.RavivarapuTable2Job(
                variant=variant,
                seed=args.seed,
                checkpoint=str(_common.fig4_checkpoint(variant, args.seed)),
                reseed_every=n,
                horizon=table2_horizon(n, n_blocks),
                carrier_hz=INFERENCE_CARRIER_50HZ,
            )
            for variant in VARIANTS
            for n in intervals
        ]
        results = _parallel_series.run_series_parallel(
            jobs, _parallel_series.ravivarapu_table2_worker, args.parallel_series
        )
        cells = {f"{r['variant']}_{r['reseed_every']}": r for r in results}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        series_path.write_text(
            json.dumps({"n_blocks": n_blocks, "seed": args.seed, "smoke": args.smoke, "cells": cells}, indent=2)
            + "\n"
        )

    rows = _rows(cells)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    _plot(rows, png_path, promote=not args.no_promote)
    if args.smoke:
        gates: dict[str, Any] = {"pass": False, "smoke_override": True}
    else:
        dig = ravivarapu_table2_gates(rows)
        gates = merge_gate_report(dig, {"n_intervals": len(rows)})
    table = {
        str(n): {
            **{k: round(v, 3) for k, v in row.items()},
            "paper_baseline_psd": TABLE2_PAPER_PSD[n][0],
            "paper_sea_psd": TABLE2_PAPER_PSD[n][1],
            "paper_baseline_reward_e2": TABLE2_PAPER_REWARD_E2[n][0],
            "paper_sea_reward_e2": TABLE2_PAPER_REWARD_E2[n][1],
        }
        for n, row in rows.items()
    }
    caption = (
        f"Table II avg PSD (×1000 normalized) with the plant reseeded every n steps; "
        f"plant seeds {args.seed}–{args.seed + n_blocks - 1} once per n ({n_blocks} blocks of n steps), "
        f"argmax rollout @ 50 Hz on the Fig 4 checkpoints; "
        f"pass={gates.get('pass')}; bars ours, dashes paper."
    )
    manifest = {
        "panel": "table2",
        "carrier_hz": INFERENCE_CARRIER_50HZ,
        "n_blocks": n_blocks,
        "intervals": list(intervals),
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "table": table,
        "caption": caption,
        "series_cache": series_path.as_posix(),
    }
    manifest = _panel.stamp_manifest(manifest)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_promote and not args.smoke and hasattr(_figure_promote, "promote_ravivarapu_table2"):
        _figure_promote.promote_ravivarapu_table2(manifest=manifest, png_path=png_path)
    print(json.dumps({k: manifest[k] for k in ("table", "gates", "png")}, indent=2))
    print(f"wrote {png_path}")
    return _panel.exit_code(manifest, smoke=bool(args.smoke))


if __name__ == "__main__":
    raise SystemExit(main())
