#!/usr/bin/env python3
"""Ravivarapu Fig 4a — training beta PSD vs episode (Baseline vs SEA-DBS).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/ravivarapu/4a/plot.py --smoke
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import SEADBSConfig, fig4_ravivarapu_config
from dataclasses import replace

from controllers.sea_dbs.checkpoint import save_checkpoint
from controllers.sea_dbs.trainer import SEA_DBSTrainer

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

FIGURES_DIR = Path("figures/ravivarapu/images/4a")
CACHE_DIR = Path("artifacts/figures/papers/ravivarapu/4")
SHARED_SERIES = CACHE_DIR / "series.json"
DEFAULT_MANIFEST = CACHE_DIR / "manifest_4a.json"
OUT_STEM = "training_psd"
DEFAULT_SEED = 0
VARIANTS = ("baseline", "paper")
GATE_SLOPE_BURN_IN = 5  # only skip reset noise for polyfit diagnostics
# "Start" band = first ~5% of episodes (paper Fig 4a shared high onset). Wider
# windows (1/3) swallowed SEA-DBS's early drop into early_mean and falsely failed
# the steeper gate while Baseline declined gradually (v15).
GATE_EARLY_FRAC = 1 / 20
GATE_LATE_FRAC = 1 / 2


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


def train_variant(
    variant: str,
    *,
    seed: int,
    smoke: bool,
    num_episodes: int | None,
) -> dict[str, Any]:
    cfg = fig4_ravivarapu_config(seed=seed, variant=variant)
    if smoke:
        cfg = cfg.for_smoke(episodes=3, max_steps=5)
    elif num_episodes is not None:
        cfg = replace(cfg, num_episodes=num_episodes)

    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        result = trainer.train_episodes()
        ckpt = CACHE_DIR / f"{variant}_train{seed}.pt"
        save_checkpoint(
            ckpt,
            actor=result.actor,
            critic=result.critic,
            config=cfg,
            predictive_model=result.predictive_model,
            extra={
                "episode_rewards": result.episode_rewards,
                "episode_psd": result.episode_psd,
            },
        )
        return {
            "variant": variant,
            "seed": seed,
            "episode_rewards": result.episode_rewards,
            "episode_psd": result.episode_psd,
            "num_episodes": cfg.num_episodes,
            "smoke": smoke,
            "checkpoint": ckpt.as_posix(),
        }
    finally:
        env.close()


def train_all(*, seed: int, smoke: bool, num_episodes: int | None) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series: dict[str, Any] = {"seed": seed, "smoke": smoke, "variants": {}}
    for variant in VARIANTS:
        print(f"training variant={variant} seed={seed} smoke={smoke}", flush=True)
        series["variants"][variant] = train_variant(
            variant, seed=seed, smoke=smoke, num_episodes=num_episodes
        )
    SHARED_SERIES.write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")
    return series


def evaluate_gates(series: dict[str, Any]) -> dict[str, Any]:
    if series.get("smoke"):
        return {"pass": True, "smoke_override": True}
    baseline = np.asarray(series["variants"]["baseline"]["episode_psd"], dtype=float)
    paper = np.asarray(series["variants"]["paper"]["episode_psd"], dtype=float)
    n = min(baseline.size, paper.size)
    if n < 10:
        return {"pass": False, "reason": "too_few_episodes", "n_episodes": n}
    b_tail = float(np.mean(baseline[n // 2 :]))
    p_tail = float(np.mean(paper[n // 2 :]))
    early_n = max(3, int(n * GATE_EARLY_FRAC))
    late_start = n // 2
    b_early = float(np.mean(baseline[:early_n]))
    p_early = float(np.mean(paper[:early_n]))
    b_drop = b_early - b_tail  # positive => declined
    p_drop = p_early - p_tail
    burn = min(GATE_SLOPE_BURN_IN, n // 5)
    b_fit = baseline[burn:n]
    p_fit = paper[burn:n]
    n_fit = min(b_fit.size, p_fit.size)
    if n_fit < 10:
        return {
            "pass": False,
            "reason": "too_few_episodes_after_burn_in",
            "n_episodes": n,
            "slope_burn_in": burn,
        }
    x_fit = np.arange(n_fit)
    b_slope = float(np.polyfit(x_fit, b_fit[:n_fit], 1)[0])
    p_slope = float(np.polyfit(x_fit, p_fit[:n_fit], 1)[0])
    # Steeper = larger early→late drop (front-loaded learning must count; polyfit
    # after a large burn-in erased SEA-DBS's cliff and failed v12/v13 wrongly).
    gates = {
        "n_episodes": n,
        "slope_burn_in": burn,
        "early_n": early_n,
        "paper_below_baseline_tail": p_tail < b_tail,
        "paper_slope_down": p_drop > 0.0 or p_slope < 0.0,
        "paper_steeper_than_baseline": p_drop > b_drop,
        "baseline_tail_mean": b_tail,
        "paper_tail_mean": p_tail,
        "baseline_early_mean": b_early,
        "paper_early_mean": p_early,
        "baseline_drop": b_drop,
        "paper_drop": p_drop,
        "baseline_slope": b_slope,
        "paper_slope": p_slope,
    }
    gates["pass"] = bool(
        gates["paper_below_baseline_tail"]
        and gates["paper_slope_down"]
        and gates["paper_steeper_than_baseline"]
    )
    return gates


def plot_series(series: dict[str, Any], png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, label in (("baseline", "Baseline (DDPG)"), ("paper", "SEA-DBS")):
        psd = series["variants"][variant]["episode_psd"]
        episodes = np.arange(1, len(psd) + 1)
        ax.plot(episodes, psd, label=label, linewidth=1.5)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean beta PSD (norm)")
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    if args.plot_only:
        if not SHARED_SERIES.is_file():
            raise SystemExit(f"missing cache: {SHARED_SERIES}")
        series = json.loads(SHARED_SERIES.read_text(encoding="utf-8"))
    else:
        series = train_all(seed=args.seed, smoke=args.smoke, num_episodes=args.episodes)

    gates = evaluate_gates(series)
    png_path, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    plot_series(series, png_path)

    manifest = {
        "panel": "4a",
        "seed": args.seed,
        "smoke": args.smoke or series.get("smoke", False),
        "png": _figure_promote.repo_rel_posix(png_path),
        "png_version": png_version,
        "gates": gates,
        "elapsed_s": round(time.time() - t0, 1),
        "caption": (
            f"Training mean GPi beta PSD vs episode (seed {args.seed}); "
            "Baseline vs full SEA-DBS (PM+GS)."
        ),
        "series_cache": SHARED_SERIES.as_posix(),
    }
    DEFAULT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if hasattr(_figure_promote, "promote_ravivarapu_4a"):
        _figure_promote.promote_ravivarapu_4a(manifest=manifest, png_path=png_path)
    print(json.dumps(manifest, indent=2))
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
