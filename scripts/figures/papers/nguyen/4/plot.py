#!/usr/bin/env python3
"""Nguyen et al.  Figure 4 — training episode rewards and lengths.

Paper §IV / Fig. 4: **500** DSQN training episodes with init DBS **40 Hz / 0.3 ms /
300 nA/cm²**. Panel (a) episode return; (b) episode length (early max **25** steps,
shorter once α–β sub-threshold termination kicks in).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --smoke

Each run writes ``figures/nguyen/images/4/training_reward_length_vN.png`` (N
auto-increments), caches training series + checkpoint under
``artifacts/figures/papers/nguyen/4/``, and updates ``figures/nguyen/replications.md``.

Long run (~hours). Prefer tmux:

  tmux new-session -d -s fig2-4-train \\
    "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
      scripts/figures/papers/nguyen/4/plot.py >> logs/fig2-4-train.log 2>&1 < /dev/null"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from nguyen_gates import (  # noqa: E402
    attach_digitization,
    fig4_length_gates,
    fig4_reward_gates,
    fig4_timing_shape_gates,
)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

import matplotlib.pyplot as plt
import numpy as np

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer
from controllers.snn.config import SNNConfig, fig4_nguyen_config
from controllers.snn.networks import DSQN
from controllers.snn.trainer import (
    DSQNTrainer,
    TrainResult,
    load_checkpoint,
    resume_dsqn_trainer,
    save_checkpoint,
    train_result_from_payload,
    write_train_metrics,
)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_spec = importlib.util.spec_from_file_location("figure_resume_cli", _RESUME_CLI)
assert _resume_spec and _resume_spec.loader
_resume_cli = importlib.util.module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(_resume_cli)

_AXES = Path(__file__).resolve().parents[2] / "plot_axes.py"
_axes_spec = importlib.util.spec_from_file_location("figure_plot_axes", _AXES)
assert _axes_spec and _axes_spec.loader
_plot_axes = importlib.util.module_from_spec(_axes_spec)
_axes_spec.loader.exec_module(_plot_axes)
data_ylim = _plot_axes.data_ylim

_OVERLAY = Path(__file__).resolve().parents[2] / "paper_overlay.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_paper_overlay", _OVERLAY)
assert _overlay_spec and _overlay_spec.loader
_paper_overlay = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_paper_overlay)

FIGURES_DIR = Path("figures/nguyen/images/4")
CACHE_DIR = Path("artifacts/figures/papers/nguyen/4")
DEFAULT_SERIES = CACHE_DIR / "series.json"
DEFAULT_CHECKPOINT = CACHE_DIR / "checkpoint.pt"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "training_reward_length"

DEFAULT_SEED = 0
DEFAULT_EPISODES = 500
SMOOTH_WINDOW = 20
EARLY_END = 100
LATE_START = 150
# Paper Fig. 4 panel (a): rewards in millions, converging toward 0 from below.
PAPER_REWARD_EARLY_MAG = 5e4
PAPER_REWARD_LATE_FLOOR = -2e5
PAPER_LENGTH_LATE_MAX = 12.0

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#cccccc",
    "font.size": 10,
}

COLOR_REWARD_RAW = "#9ecae1"
COLOR_REWARD_SMOOTH = "#08519c"
COLOR_LENGTH_RAW = "#fcbba1"
COLOR_LENGTH_SMOOTH = "#a50f15"


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


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; edges use available samples."""
    if y.size == 0:
        return y.copy()
    window = max(1, int(window))
    if window == 1 or y.size == 1:
        return y.astype(float, copy=True)
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(y.astype(float), (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: y.size]


def train_series(
    *,
    seed: int,
    num_episodes: int,
    smoke: bool = False,
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
) -> tuple[dict[str, Any], SNNConfig, DSQNTrainer, TrainResult]:
    if smoke:
        cfg = SNNConfig(seed=seed).for_smoke(episodes=min(5, num_episodes), max_steps=8)
    else:
        cfg = fig4_nguyen_config(seed=seed, num_episodes=num_episodes)

    env = NguyenEnvAdapter(config=cfg)
    try:
        initial_result: TrainResult | None = None
        resume_start = 0
        if resume_path is not None:
            payload = load_checkpoint(resume_path, map_location=cfg.device)
            metrics_path = resume_path.with_suffix(".metrics.json")
            trainer, resume_start = resume_dsqn_trainer(
                payload,
                config=cfg,
                metrics_path=metrics_path,
                start_episode=start_episode,
            )
            initial_result = train_result_from_payload(payload, dsqn=trainer.dsqn, config=cfg)
        else:
            dsqn = DSQN(cfg)
            buffer = ReplayBuffer(cfg, seed=cfg.seed)
            trainer = DSQNTrainer(dsqn, buffer, cfg)

        result = trainer.train_episodes(
            env,
            start_episode=resume_start,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=checkpoint_interval,
            initial_result=initial_result,
        )
        payload: dict[str, Any] = {
            "seed": cfg.seed,
            "num_episodes": cfg.num_episodes,
            "max_episode_steps": cfg.max_episode_steps,
            "episode_rewards": result.episode_rewards,
            "episode_lengths": result.episode_lengths,
            "episode_spike_totals": result.episode_spike_totals,
            "episode_energies": result.episode_energies,
            "episode_alpha_beta_means": result.episode_alpha_beta_means,
            "episode_early_stops": result.episode_early_stops,
            "episode_amplitudes": result.episode_amplitudes,
            "episode_frequencies": result.episode_frequencies,
            "episode_pulse_widths": result.episode_pulse_widths,
            "update_count": result.update_count,
            "smoke": smoke,
            "config": {
                "epsilon_decay_steps": cfg.epsilon_decay_steps,
                "subthreshold_steps_required": cfg.subthreshold_steps_required,
                "alpha_beta_threshold": cfg.alpha_beta_threshold,
            },
        }
        return payload, cfg, trainer, result
    finally:
        env.close()


# late_reward_near_zero is diagnostic only (paper approaches ~0 from below;
# positive progress shaping makes the floor check a free pass).
REWARD_HEURISTIC_KEYS = (
    "reward_scale_paper",
    "late_reward_above_early",
)
REWARD_SHAPE_KEYS = (
    "reward_scale_paper",
    "late_reward_above_early",
    "reward_post100_plateau",
)
REWARD_SHAPE_PAPER_KEYS = (
    "paper_reward_improves_like_paper",
)
# Full ship = shape (+ dig direction). Mag/ratio dig gates are info-only.
REWARD_FULL_KEYS = REWARD_SHAPE_KEYS + REWARD_SHAPE_PAPER_KEYS
LENGTH_HEURISTIC_KEYS = (
    "length_decreases",
    "late_length_paper_band",
    "early_near_max_length",
)
LENGTH_SHAPE_KEYS = (
    "length_decreases",
    "late_length_paper_band",
    "early_near_max_length",
    "length_mid_glide_like_paper",
    "length_post100_plateau",
)
LENGTH_SHAPE_PAPER_KEYS = (
    "paper_length_decreases_like_paper",
)


def _group_pass(group: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return all(bool(group.get(key)) for key in keys)


def evaluate_gates(
    series: dict[str, Any],
    *,
    max_episode_steps: int,
) -> dict[str, Any]:
    rewards = np.asarray(series["episode_rewards"], dtype=float)
    lengths = np.asarray(series["episode_lengths"], dtype=float)
    n = int(rewards.size)
    if n < 10:
        reward = {
            "pass": False,
            "shape_pass": False,
            "reason": "too_few_episodes",
            "early_reward_mean": float(np.mean(rewards)) if n else float("nan"),
            "late_reward_mean": float(np.mean(rewards)) if n else float("nan"),
        }
        length = {
            "pass": False,
            "shape_pass": False,
            "reason": "too_few_episodes",
            "early_length_mean": float(np.mean(lengths)) if n else float("nan"),
            "late_length_mean": float(np.mean(lengths)) if n else float("nan"),
        }
        gates = {
            "pass": False,
            "shape_pass": False,
            "n_episodes": n,
            "reward": reward,
            "length": length,
        }
        if series.get("smoke"):
            gates["pass"] = True
            gates["shape_pass"] = True
            gates["smoke_override"] = True
            reward["pass"] = True
            reward["shape_pass"] = True
            length["pass"] = True
            length["shape_pass"] = True
        return gates

    early_end = min(EARLY_END, n // 2)
    late_start = min(LATE_START, max(early_end + 1, n - 50))

    early_rewards = rewards[:early_end]
    late_rewards = rewards[late_start:]
    early_lengths = lengths[: min(75, n)]
    late_lengths = lengths[late_start:]

    early_std = float(np.std(early_rewards))
    early_mean = float(np.mean(early_rewards))
    late_mean_reward = float(np.mean(late_rewards))
    first50_mean = float(np.mean(rewards[: min(50, n)]))
    early_length_mean = float(np.mean(early_lengths))
    late_length_mean = float(np.mean(late_lengths))

    reward_heur = {
        "reward_scale_paper": abs(first50_mean) >= PAPER_REWARD_EARLY_MAG,
        "late_reward_above_early": late_mean_reward > first50_mean,
        "late_reward_near_zero": late_mean_reward > PAPER_REWARD_LATE_FLOOR,
        "early_high_variance": early_std > 0.05 * max(abs(early_mean), 1.0),
        "early_reward_mean": early_mean,
        "late_reward_mean": late_mean_reward,
        "first50_reward_mean": first50_mean,
        "pass": False,
        "shape_pass": False,
    }

    length_heur = {
        "length_decreases": late_length_mean < early_length_mean - 1.0,
        "late_length_paper_band": late_length_mean <= PAPER_LENGTH_LATE_MAX,
        "early_near_max_length": float(np.median(early_lengths[: min(50, n)])) >= max_episode_steps - 2,
        "early_length_mean": early_length_mean,
        "late_length_mean": late_length_mean,
        "pass": False,
        "shape_pass": False,
    }

    timing = fig4_timing_shape_gates(lengths, rewards)
    for key, value in timing["length_gates"].items():
        length_heur[key] = bool(value)
    for key, value in timing["reward_gates"].items():
        reward_heur[key] = bool(value)
    length_heur["timing_metrics"] = timing.get("metrics", {})
    reward_heur["timing_metrics"] = timing.get("metrics", {})
    reward_heur["shape_pass"] = _group_pass(reward_heur, REWARD_SHAPE_KEYS)
    reward_heur["pass"] = _group_pass(reward_heur, REWARD_HEURISTIC_KEYS)
    length_heur["shape_pass"] = _group_pass(length_heur, LENGTH_SHAPE_KEYS)
    length_heur["pass"] = _group_pass(length_heur, LENGTH_HEURISTIC_KEYS)

    if series.get("smoke"):
        reward_heur["pass"] = True
        reward_heur["shape_pass"] = True
        length_heur["pass"] = True
        length_heur["shape_pass"] = True
        return {
            "pass": True,
            "shape_pass": True,
            "n_episodes": n,
            "reward": reward_heur,
            "length": length_heur,
            "smoke_override": True,
        }

    dig_reward = fig4_reward_gates(rewards)
    dig_length = fig4_length_gates(lengths, max_episode_steps=max_episode_steps)
    reward = attach_digitization(reward_heur, dig_reward)
    length = attach_digitization(length_heur, dig_length)
    reward["shape_pass"] = _group_pass(reward, REWARD_SHAPE_KEYS + REWARD_SHAPE_PAPER_KEYS)
    reward["pass"] = _group_pass(reward, REWARD_FULL_KEYS)
    length["shape_pass"] = _group_pass(length, LENGTH_SHAPE_KEYS + LENGTH_SHAPE_PAPER_KEYS)
    # length full pass keeps attach_digitization (heuristics + dig levels)

    return {
        "pass": bool(reward["pass"] and length["pass"]),
        "shape_pass": bool(reward["shape_pass"] and length["shape_pass"]),
        "n_episodes": n,
        "reward": reward,
        "length": length,
    }


def plot_series(series: dict[str, Any], out_path: Path, *, smooth_window: int) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    rewards = np.asarray(series["episode_rewards"], dtype=float)
    lengths = np.asarray(series["episode_lengths"], dtype=float)
    episodes = np.arange(rewards.size, dtype=float)

    reward_smooth = moving_average(rewards, smooth_window)
    length_smooth = moving_average(lengths, smooth_window)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True, constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(episodes, rewards, color=COLOR_REWARD_RAW, linewidth=0.8, alpha=0.85, label="Raw", zorder=3)
    ax0.plot(
        episodes,
        reward_smooth,
        color=COLOR_REWARD_SMOOTH,
        linewidth=2.0,
        label="Smoothed",
        zorder=4,
    )
    ax0.set_ylabel("Reward")
    ax0.set_title("Episode Rewards")
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(-6, 6))
    ax0.grid(True, linestyle="--", alpha=0.6)

    ax1 = axes[1]
    ax1.plot(episodes, lengths, color=COLOR_LENGTH_RAW, linewidth=0.8, alpha=0.85, label="Raw", zorder=3)
    ax1.plot(
        episodes,
        length_smooth,
        color=COLOR_LENGTH_SMOOTH,
        linewidth=2.0,
        label="Smoothed",
        zorder=4,
    )
    paper_y = _paper_overlay.overlay_nguyen_fig4(ax0, axes[1])
    ax0.set_ylim(
        *data_ylim(
            rewards,
            reward_smooth,
            paper_y["reward"][0],
            paper_y["reward"][1],
            extra_values=(0.0,),
        )
    )
    _paper_overlay.place_legend(ax0, fontsize=8, loc="lower right")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Length")
    ax1.set_title("Episode Lengths")
    ax1.set_ylim(
        *data_ylim(
            lengths,
            length_smooth,
            paper_y["length"][0],
            paper_y["length"][1],
            integer_snap=True,
        )
    )
    ax1.grid(True, linestyle="--", alpha=0.6)
    _paper_overlay.place_legend(ax1, fontsize=8, loc="upper right")

    last_ep = max(0, rewards.size - 1)
    for ax in axes:
        ax.set_xlim(0.0, float(last_ep))
        if last_ep >= 100:
            ax.set_xticks(np.arange(0, last_ep + 1, 100))
        else:
            ax.set_xticks(np.arange(0, last_ep + 1, max(1, last_ep // 5)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "n_episodes": int(rewards.size),
        "reward_min": float(rewards.min()) if rewards.size else float("nan"),
        "reward_max": float(rewards.max()) if rewards.size else float("nan"),
        "length_min": float(lengths.min()) if lengths.size else float("nan"),
        "length_max": float(lengths.max()) if lengths.size else float("nan"),
        "smooth_window": int(smooth_window),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=SMOOTH_WINDOW,
        help=f"moving-average window for smoothed curves (default {SMOOTH_WINDOW})",
    )
    parser.add_argument("--smoke", action="store_true", help="5-episode smoke train for CI/dev")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output PNG (default auto {FIGURES_DIR}/{OUT_STEM}_vN.png)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--no-update-docs", action="store_true")
    _resume_cli.add_training_resume_args(parser)
    args = parser.parse_args(argv)
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    t0 = time.perf_counter()
    if args.plot_only:
        if not args.series.is_file():
            print(f"missing series cache: {args.series}", file=sys.stderr)
            return 2
        series = json.loads(args.series.read_text(encoding="utf-8"))
        max_steps = int(series.get("max_episode_steps", 25))
    else:
        print(
            f"training DSQN seed={args.seed} episodes={args.episodes} smoke={args.smoke} "
            f"resume={args.resume}",
            flush=True,
        )
        series, cfg, trainer, train_result = train_series(
            seed=args.seed,
            num_episodes=args.episodes,
            smoke=args.smoke,
            resume_path=args.resume,
            start_episode=args.start_episode,
            checkpoint_path=args.checkpoint,
            checkpoint_interval=args.checkpoint_interval,
        )
        write_json(args.series, series)
        max_steps = cfg.max_episode_steps
        if not args.smoke:
            save_checkpoint(
                args.checkpoint,
                dsqn=trainer.dsqn,
                config=cfg,
                optimizer=trainer.optimizer,
                trainer=trainer,
                extra={
                    "completed_episodes": len(series["episode_rewards"]),
                    "episode_rewards": series["episode_rewards"],
                    "episode_lengths": series["episode_lengths"],
                    "episode_spike_totals": series.get("episode_spike_totals", []),
                    "episode_energies": series.get("episode_energies", []),
                    "update_count": series["update_count"],
                },
            )
            write_train_metrics(train_result, args.checkpoint.with_suffix(".metrics.json"))

    gates = evaluate_gates(series, max_episode_steps=max_steps)
    panel = plot_series(series, args.out, smooth_window=args.smooth_window)

    caption = (
        f"DSQN train {series['num_episodes']} ep, seed={series['seed']}; "
        f"late_reward={gates['reward']['late_reward_mean']:.0f}, "
        f"late_len={gates['length']['late_length_mean']:.1f}; "
        f"shape_pass={gates['shape_pass']} pass={gates['pass']} "
        f"(reward shape={gates['reward']['shape_pass']} full={gates['reward']['pass']}, "
        f"length shape={gates['length']['shape_pass']} full={gates['length']['pass']})"
    )
    manifest = {
        "panel": "2/4",
        "out": args.out.as_posix(),
        "series": args.series.as_posix(),
        "checkpoint": args.checkpoint.as_posix(),
        "gates": gates,
        "panel_stats": panel,
        "elapsed_s": time.perf_counter() - t0,
        "png_version": png_version,
        "caption": caption,
        "smoke": bool(series.get("smoke")),
    }
    write_json(args.manifest, manifest)

    if not args.no_update_docs:
        updated = _figure_promote.promote_nguyen_4(
            manifest=manifest,
            png_path=args.out,
        )
        print(f"updated comparison doc: {updated['doc']}", flush=True)

    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.out}")
    if png_version is not None:
        print(f"output PNG version={png_version}", flush=True)
    return 0 if gates["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
