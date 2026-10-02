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

FIGURES_DIR = Path("figures/ravivarapu/images/4b")
CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")
SHARED_SERIES = CACHE_DIR / "series.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest_4b.json"
OUT_STEM = "training_reward"
PLOT_4A = Path(__file__).resolve().parents[1] / "4a" / "plot.py"
DISPLAY_ROLL_WINDOW = 10


def _rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if window <= 1 or y.size == 0:
        return y
    out = np.empty_like(y)
    for i in range(y.size):
        lo = max(0, i - window + 1)
        out[i] = float(y[lo : i + 1].mean())
    return out


def _linked_png(path: Path) -> Path:
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    linked_dir = paper.resolve().parent
    linked_target = linked_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    if not linked_target.exists():
        linked_target.touch()
    path.symlink_to(linked_target)
    return path


def evaluate_gates(series: dict[str, Any]) -> dict[str, Any]:
    if series.get("smoke"):
        return {"pass": False, "shape_pass": False, "smoke_override": True}
    baseline = series["variants"]["baseline"]["episode_rewards"]
    sea = series["variants"]["paper"]["episode_rewards"]
    dig = ravivarapu_fig4b_gates(baseline, sea, n_expected=150)
    merged = merge_gate_report(dig, {"n_episodes": min(len(baseline), len(sea))})
    return ravivarapu_fig4b_attach_tiered_pass(merged)


def plot_series(series: dict[str, Any], png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    repl_ys: list[np.ndarray] = []
    roll_w = DISPLAY_ROLL_WINDOW
    for variant, label in (
        ("baseline", f"Baseline (DDPG, roll{roll_w})"),
        ("paper", f"SEA-DBS (roll{roll_w})"),
    ):
        raw_rewards = series["variants"][variant]["episode_rewards"]
        rewards = _rolling_mean(raw_rewards, roll_w)
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, label=label, linewidth=1.6 if variant == "baseline" else 1.8)
        repl_ys.append(rewards)
    early = float(np.mean(repl_ys[0][: max(1, len(repl_ys[0]) // 10)])) if repl_ys else -95.0
    paper = _paper_overlay.overlay_ravivarapu_fig4b(ax, replication_early_mean=early)
    paper_ys = [v[0] for v in paper.values()]
    all_y = np.concatenate(repl_ys + paper_ys)
    lo = float(np.nanmin(all_y))
    span = 20.0 - lo
    pad = 0.08 * (span + 1e-6)
    ax.set_ylim(lo - pad, 20.0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (Eq. 7)")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, loc="lower right", fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_linked_png(png_path), dpi=150)
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
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="After promote, refresh Report 3 gallery image links in the notes folder",
    )
    args = parser.parse_args()
    _figure_promote.set_export_notes_images(args.export_notes)
    _figure_promote.set_update_report3(args.update_report)

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
    plot_series(series, png_path)

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
            "paired with Fig 4a cache; display roll10 (gates on raw)."
        ),
        "series_cache": SHARED_SERIES.as_posix(),
        "display_roll_window": DISPLAY_ROLL_WINDOW,
    }
    manifest = _panel.stamp_manifest(manifest)
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if hasattr(_figure_promote, "promote_ravivarapu_4b"):
        _figure_promote.promote_ravivarapu_4b(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")

    return _panel.exit_code(manifest, smoke=bool(getattr(args, "smoke", False)))

if __name__ == "__main__":
    raise SystemExit(main())
