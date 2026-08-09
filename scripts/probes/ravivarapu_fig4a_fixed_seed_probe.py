#!/usr/bin/env python3
"""Compare Fig 4a early PSD: per-episode plant reseed vs fixed episode seed.

Runs Baseline + SEA-DBS with v87 ``fig4_ravivarapu_config`` under:
  - reseed (default trainer: ``seed + episode`` → new IC draw each episode)
  - fixed (``fixed_episode_seed=True`` → same plant seed every episode)

Writes JSON + overlay PNG under ``artifacts/probes/``. With ``--full`` (150
episodes), also evaluates manifest gates for the fixed-seed run.

Example::

  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig4a_fixed_seed_probe.py
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_fig4a_fixed_seed_probe.py --full
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.config import fig4_ravivarapu_config
from controllers.sea_dbs.trainer import SEA_DBSTrainer

_PLOT = ROOT / "scripts" / "figures" / "papers" / "ravivarapu" / "4a" / "plot.py"
_spec = importlib.util.spec_from_file_location("ravivarapu_4a_plot", _PLOT)
assert _spec and _spec.loader
_plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plot)

DIG_PATH = ROOT / "artifacts/figures/papers/ravivarapu/paper_digitization/curves_fig4a.json"
OUT_DIR = ROOT / "artifacts/probes"
DEFAULT_EPISODES = 25
VARIANTS = ("baseline", "paper")


def _train_variant(
    variant: str,
    *,
    seed: int,
    num_episodes: int,
    fixed_episode_seed: bool,
) -> list[float]:
    cfg = replace(
        fig4_ravivarapu_config(seed=seed, num_episodes=num_episodes, variant=variant),
        fixed_episode_seed=fixed_episode_seed,
        log_episodes=True,
    )
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        result = trainer.train_episodes()
        return list(result.episode_psd)
    finally:
        env.close()


def _train_all(
    *,
    seed: int,
    num_episodes: int,
    fixed_episode_seed: bool,
    label: str,
) -> dict[str, Any]:
    print(f"=== mode={label} fixed_episode_seed={fixed_episode_seed} episodes={num_episodes} ===", flush=True)
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        print(f"training variant={variant}", flush=True)
        psd = _train_variant(
            variant,
            seed=seed,
            num_episodes=num_episodes,
            fixed_episode_seed=fixed_episode_seed,
        )
        variants[variant] = {"variant": variant, "episode_psd": psd}
    return {"seed": seed, "fixed_episode_seed": fixed_episode_seed, "label": label, "variants": variants}


def _paper_curve(label: str, n: int) -> np.ndarray:
    dig = json.loads(DIG_PATH.read_text(encoding="utf-8"))
    xy = dig["series"][label]["xy"]
    xs = np.asarray(xy["x"], dtype=float)
    ys = np.asarray(xy["y"], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    eps = np.arange(1, n + 1, dtype=float)
    return np.interp(eps, xs, ys)


def _paper_sea_early(n: int = 15) -> np.ndarray:
    return _paper_curve("SEA-DBS", n)


def _plot_overlay(
    reseed: dict[str, Any],
    fixed: dict[str, Any],
    *,
    n_plot: int,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    paper_labels = {"baseline": "Baseline", "paper": "SEA-DBS"}
    for ax, variant, title in zip(
        axes,
        VARIANTS,
        ("Baseline (DDPG)", "SEA-DBS"),
        strict=True,
    ):
        paper_early = _paper_curve(paper_labels[variant], n_plot)
        n = min(
            len(reseed["variants"][variant]["episode_psd"]),
            len(fixed["variants"][variant]["episode_psd"]),
            len(paper_early),
        )
        eps = np.arange(1, n + 1)
        ax.plot(eps, reseed["variants"][variant]["episode_psd"][:n], label="reseed (seed+ep)", lw=1.5)
        ax.plot(eps, fixed["variants"][variant]["episode_psd"][:n], label="fixed seed", lw=1.5)
        ax.plot(eps, paper_early[:n], label="paper dig", lw=1.2, ls="--", color="k", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean beta PSD (norm)")
    fig.suptitle("Fig 4a early episodes: reseed vs fixed plant seed")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _early_stats(psd: list[float], n: int = 15) -> dict[str, float]:
    arr = np.asarray(psd[:n], dtype=float)
    return {
        "mean_ep1_n": float(np.mean(arr)),
        "min_ep1_n": float(np.min(arr)),
        "min_ep": int(np.argmin(arr[:n]) + 1),
        "ep2": float(arr[1]) if len(arr) > 1 else float("nan"),
        "std_ep1_n": float(np.std(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--full", action="store_true", help="150 episodes + gate eval on fixed-seed run")
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="Train only fixed_episode_seed=True (use with --full for gate check)",
    )
    args = parser.parse_args()

    num_episodes = 150 if args.full else args.episodes
    paper_early = _paper_sea_early(min(20, num_episodes))

    reseed: dict[str, Any] | None = None
    if not args.fixed_only:
        reseed = _train_all(
            seed=args.seed,
            num_episodes=num_episodes,
            fixed_episode_seed=False,
            label="reseed",
        )
    fixed = _train_all(
        seed=args.seed,
        num_episodes=num_episodes,
        fixed_episode_seed=True,
        label="fixed",
    )

    report: dict[str, Any] = {
        "seed": args.seed,
        "num_episodes": num_episodes,
        "paper_early_sea": paper_early.tolist(),
        "fixed": {
            "baseline": _early_stats(fixed["variants"]["baseline"]["episode_psd"]),
            "paper": _early_stats(fixed["variants"]["paper"]["episode_psd"]),
        },
        "fixed_ep1_20": {
            v: fixed["variants"][v]["episode_psd"][:20] for v in VARIANTS
        },
    }
    if reseed is not None:
        report["reseed"] = {
            "baseline": _early_stats(reseed["variants"]["baseline"]["episode_psd"]),
            "paper": _early_stats(reseed["variants"]["paper"]["episode_psd"]),
        }
        report["reseed_ep1_20"] = {
            v: reseed["variants"][v]["episode_psd"][:20] for v in VARIANTS
        }

    if args.full:
        series_fixed = {
            "seed": args.seed,
            "smoke": False,
            "fixed_episode_seed": True,
            "variants": fixed["variants"],
        }
        gates = _plot.evaluate_gates(series_fixed)
        report["fixed_gates"] = gates
        print(json.dumps({"fixed_gates_pass": gates.get("pass"), "fixed_shape_pass": gates.get("shape_pass")}, indent=2), flush=True)

    tag = "full150" if args.full else f"ep{num_episodes}"
    if args.fixed_only:
        tag = f"{tag}_fixed_only"
    out_json = OUT_DIR / f"ravivarapu_fig4a_fixed_seed_{tag}.json"
    out_png = OUT_DIR / f"ravivarapu_fig4a_fixed_seed_{tag}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if reseed is not None:
        _plot_overlay(reseed, fixed, n_plot=min(20, num_episodes), out_png=out_png)
    else:
        _plot_overlay(fixed, fixed, n_plot=min(20, num_episodes), out_png=out_png)

    print(json.dumps(report, indent=2))
    print(f"wrote {out_json}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
