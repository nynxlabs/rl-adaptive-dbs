#!/usr/bin/env python3
"""Ravivarapu Fig 4b — training reward vs episode (paired with Fig 4a).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4b/plot.py --smoke
"""
from __future__ import annotations

import argparse
import importlib.util
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
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import merge_gate_report, ravivarapu_fig4b_gates  # noqa: E402

FIGURES_DIR = Path("figures/ravivarapu/images/4b")
CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")
SHARED_SERIES = CACHE_DIR / "series.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest_4b.json"
OUT_STEM = "training_reward"
PLOT_4A = Path(__file__).resolve().parents[1] / "4a" / "plot.py"


def _vault_backed_png(path: Path) -> Path:
    path = Path(path)
    paper = path.parent / "paper.png"
    if not paper.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    vault_dir = paper.resolve().parent
    vault_target = vault_dir / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        return path
    if not vault_target.exists():
        vault_target.touch()
    path.symlink_to(vault_target)
    return path


def evaluate_gates(series: dict[str, Any]) -> dict[str, Any]:
    if series.get("smoke"):
        return {"pass": True, "smoke_override": True}
    baseline = series["variants"]["baseline"]["episode_rewards"]
    sea = series["variants"]["paper"]["episode_rewards"]
    dig = ravivarapu_fig4b_gates(baseline, sea)
    return merge_gate_report(dig, {"n_episodes": min(len(baseline), len(sea))})


def plot_series(series: dict[str, Any], png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, label in (("baseline", "Baseline (DDPG)"), ("paper", "SEA-DBS")):
        rewards = series["variants"][variant]["episode_rewards"]
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, label=label, linewidth=1.5)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (Eq. 7)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_vault_backed_png(png_path), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

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
            "paired with Fig 4a cache."
        ),
        "series_cache": SHARED_SERIES.as_posix(),
    }
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if hasattr(_figure_promote, "promote_ravivarapu_4b"):
        _figure_promote.promote_ravivarapu_4b(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
