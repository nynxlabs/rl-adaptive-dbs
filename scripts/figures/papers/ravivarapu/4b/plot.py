#!/usr/bin/env python3
"""Ravivarapu Fig 4b — training reward vs episode (paired with Fig 4a).

Training and checkpoint resume live in Fig 4a ``plot.py`` (``--resume``, periodic
checkpoints). This panel replots cached 4a metrics only.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py --smoke
"""
from __future__ import annotations

from rl_adaptive_dbs import doc_paths
from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import (  # noqa: E402
    merge_gate_report,
    ravivarapu_fig4b_attach_tiered_pass,
    ravivarapu_fig4b_gates,
)

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import = load_script_module("figure_overlay_import", _OVERLAY_IMPORT)
_paper_overlay = _overlay_import.load_paper_overlay()

FIGURES_DIR = Path(doc_paths.figure_images_rel("ravivarapu", "4b"))
CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")
SHARED_SERIES = CACHE_DIR / "series.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest_4b.json"
OUT_STEM = "training_reward"
PLOT_4A = Path(__file__).resolve().parents[1] / "4a" / "plot.py"
DISPLAY_ROLL_WINDOW = 10
EPISODE_STEPS = 30

_COMMON = Path(__file__).resolve().parents[1] / "common.py"
_common = load_script_module("ravivarapu_common", _COMMON)


def _per_step_rewards(series: dict[str, Any], variant: str) -> np.ndarray:
    """Episode reward as the mean Eq. (7) reward per step (paper Fig 4b units).

    The paper's Fig 4b y-range (~-1.4 → 0) is one Eq. (7) step reward, not a
    30-step sum; series.json stores the sum.
    """
    entry = series["variants"][variant]
    steps = int((entry.get("config") or {}).get("max_episode_steps", EPISODE_STEPS))
    return np.asarray(entry["episode_rewards"], dtype=float) / float(max(1, steps))


def evaluate_gates(series: dict[str, Any]) -> dict[str, Any]:
    if series.get("smoke"):
        return {"pass": False, "shape_pass": False, "smoke_override": True}
    baseline = _per_step_rewards(series, "baseline")
    sea = _per_step_rewards(series, "paper")
    dig = ravivarapu_fig4b_gates(
        baseline,
        sea,
        n_expected=150,
        baseline_psd=series["variants"]["baseline"]["episode_psd"],
        sea_psd=series["variants"]["paper"]["episode_psd"],
    )
    merged = merge_gate_report(dig, {"n_episodes": min(len(baseline), len(sea))})
    return ravivarapu_fig4b_attach_tiered_pass(merged)


def plot_series(series: dict[str, Any], png_path: Path, *, promote: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    repl_ys: list[np.ndarray] = []
    roll_w = DISPLAY_ROLL_WINDOW
    for variant, label in (
        ("baseline", f"Baseline (DDPG, roll{roll_w})"),
        ("paper", f"SEA-DBS (roll{roll_w})"),
    ):
        raw = _per_step_rewards(series, variant)
        rewards = _common.rolling_mean(raw, roll_w)
        episodes = np.arange(1, len(rewards) + 1)
        (line,) = ax.plot(episodes, rewards, label=label, linewidth=1.6 if variant == "baseline" else 1.8)
        # Raw episodes (what the gates read) drawn faintly under the rolling mean.
        ax.plot(episodes, raw, linewidth=0.6, alpha=0.3, color=line.get_color())
        repl_ys.extend([rewards, raw])
    paper = _paper_overlay.overlay_ravivarapu_fig4b(ax, y_scale=1.0)
    paper_ys = [py for _px, py in paper.values()]
    all_y = np.concatenate(repl_ys + paper_ys)
    lo, hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = 0.06 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean step reward (Eq. 7)")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, loc="lower right", fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_common.linked_png(png_path, _figure_promote, promote=promote), dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--export-notes",
        action="store_true",
        help="After promote, copy replication PNGs into the notes folder (RL_DBS_NOTES_DIR / export.notes_dir)",
    )
    # Former name, kept so existing commands keep working.
    parser.add_argument("--push-kb", dest="export_notes", action="store_true", help=argparse.SUPPRESS)
    _common.add_no_promote_arg(parser)
    args = parser.parse_args()
    if args.no_promote:
        args.export_notes = False
    _figure_promote.set_export_notes_images(args.export_notes)

    t0 = time.time()
    if args.plot_only:
        if not SHARED_SERIES.is_file():
            raise SystemExit(f"missing paired cache from Fig 4a: {SHARED_SERIES}")
    elif not SHARED_SERIES.is_file():
        cmd = [
            sys.executable,
            "-m",
            "rl_adaptive_dbs.run",
            str(PLOT_4A),
        ]
        if args.smoke:
            cmd.append("--smoke")
        if args.episodes is not None:
            cmd.extend(["--episodes", str(args.episodes)])
        cmd.extend(["--seed", str(args.seed)])
        subprocess.run(cmd, check=True)
    # else: reuse existing Fig 4a series.json (paired lineage)

    if not SHARED_SERIES.is_file():
        raise SystemExit(f"missing paired cache from Fig 4a: {SHARED_SERIES}")
    series = json.loads(SHARED_SERIES.read_text(encoding="utf-8"))

    gates = evaluate_gates(series)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    plot_series(series, png_path, promote=not args.no_promote)

    manifest = {
        "panel": "4b",
        "seed": args.seed,
        "smoke": series.get("smoke", False),
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "elapsed_s": round(time.time() - t0, 1),
        "caption": (
            f"Training episode reward vs episode (seed {args.seed}); "
            "mean Eq. (7) reward per step; paired with Fig 4a cache; "
            "display roll10 (gates on raw)."
        ),
        "series_cache": SHARED_SERIES.as_posix(),
        "display_roll_window": DISPLAY_ROLL_WINDOW,
    }
    manifest = _panel.stamp_manifest(manifest)
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.no_promote and hasattr(_figure_promote, "promote_ravivarapu_4b"):
        _figure_promote.promote_ravivarapu_4b(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))

if __name__ == "__main__":
    raise SystemExit(main())
