#!/usr/bin/env python3
"""Nguyen et al.  Figure 4 — training episode rewards and lengths.

Paper §IV / Fig. 4: **500** DSQN training episodes in the paper. Gates
(``nguyen_gates.fig4_gates``) are scale-free and anchored to the digitized
curves: **shape_pass** = transition timing and ep 0–200 shape, **pass** (ship) =
all required gates through ep 500.
Default train horizon is **500** episodes (paper figure length).

Run:
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run scripts/figures/papers/nguyen/4/plot.py --smoke

Each run writes ``figures/nguyen/images/4/training_reward_length_vN.png`` (N
auto-increments), caches training series + checkpoint under
``artifacts/figures/papers/nguyen/4/``, and updates ``figures/nguyen/replications.md``.

First-100 trains are ~15–20 min. Prefer tmux:

  tmux new-session -d -s fig2-4-train \\
    "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
      scripts/figures/papers/nguyen/4/plot.py >> logs/fig2-4-train.log 2>&1 < /dev/null"
"""
from __future__ import annotations

from rl_adaptive_dbs.panel import load_script_module

from rl_adaptive_dbs import panel as _panel

import argparse
import json
import os
import sys
import tempfile
import time
import dataclasses
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from nguyen_gates import (  # noqa: E402
    FIG4_ABORT_EPISODE,
    FIG4_REQUIRED_KEYS,
    FIG4_SMOOTH,
    FIG4_TIER2_KEYS,
    centered_smooth,
    fig4_abort_check,
    fig4_gates,
)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_figure_promote = load_script_module("figure_promote", _PROMOTE)

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
    episode_extra,
    training_budget,
)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_cli = load_script_module("figure_resume_cli", _RESUME_CLI)

_AXES = Path(__file__).resolve().parents[2] / "plot_axes.py"
_plot_axes = load_script_module("figure_plot_axes", _AXES)
data_ylim = _plot_axes.data_ylim

_OVERLAY = Path(__file__).resolve().parents[2] / "paper_overlay.py"
_paper_overlay = load_script_module("figure_paper_overlay", _OVERLAY)

FIGURES_DIR = Path("figures/nguyen/images/4")
CACHE_DIR = Path("artifacts/figures/papers/nguyen/4")
DEFAULT_SERIES = CACHE_DIR / "series.json"
DEFAULT_CHECKPOINT = CACHE_DIR / "checkpoint.pt"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "training_reward_length"

DEFAULT_SEED = 0
DEFAULT_EPISODES = 500  # paper Fig 4 horizon; shape_pass uses first 100
SMOOTH_WINDOW = FIG4_SMOOTH  # centred moving average, same as the gates

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


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, shrinking at the edges (same smoother as the gates)."""
    return centered_smooth(np.asarray(y, dtype=float), window)


def parse_config_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``--set field=value`` into typed ``SNNConfig`` overrides."""
    fields = {f.name: f for f in dataclasses.fields(SNNConfig)}
    defaults = SNNConfig()
    out: dict[str, Any] = {}
    for item in items or []:
        name, sep, raw = item.partition("=")
        name = name.strip().replace("-", "_")
        if not sep or name not in fields:
            raise SystemExit(f"--set expects FIELD=VALUE with an SNNConfig field, got {item!r}")
        current = getattr(defaults, name)
        if isinstance(current, bool):
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise SystemExit(f"--set {name}: expected true/false, got {raw!r}")
            out[name] = raw.lower() in {"true", "1"}
        elif isinstance(current, int):
            out[name] = int(raw)
        elif isinstance(current, float) or current is None:
            out[name] = float(raw)
        else:
            out[name] = raw
    return out


def config_record(cfg: SNNConfig) -> dict[str, Any]:
    """JSON-safe dump of every scalar ``SNNConfig`` field, so a run can be reproduced."""
    return {
        f.name: getattr(cfg, f.name)
        for f in dataclasses.fields(cfg)
        if isinstance(getattr(cfg, f.name), (bool, int, float, str, type(None)))
    }


class EarlyAbort(Exception):
    """Raised from the checkpoint probe when the doomed-run check fails."""

    def __init__(self, completed: int, failed: list[str], probe: dict[str, Any] | None = None) -> None:
        super().__init__(f"doomed-run check failed at episode {completed}: {failed}")
        self.completed = completed
        self.failed = failed
        self.probe = probe


def abort_probe(completed: int, result: TrainResult) -> dict[str, Any]:
    """Doomed-run check on the partial series (``nguyen_gates.fig4_abort_check``).

    Decidable from episode 150: abort only when length or reward progress over
    ep 120–150 (toward the paper's late level, scale-free) is below 0.5. The
    full gate set is never used to abort.
    """
    probe = fig4_abort_check(result.episode_rewards[:completed], result.episode_lengths[:completed])
    probe["completed_episodes"] = completed
    return probe


def find_repeat_run(cfg_record: dict[str, Any], search_dir: Path) -> Path | None:
    """A manifest under ``search_dir`` from the same clean commit with the identical config."""
    git = _panel.git_state()
    if git["dirty"] or not git["commit"] or not search_dir.is_dir():
        return None
    for path in sorted(search_dir.rglob("*manifest*.json")):
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        prov_git = (prior.get("provenance") or {}).get("git") or {}
        if (
            prior.get("config") == cfg_record
            and prov_git.get("commit") == git["commit"]
            and not prov_git.get("dirty")
            and not prior.get("smoke")
        ):
            return path
    return None


def train_series(
    *,
    seed: int,
    num_episodes: int,
    smoke: bool = False,
    double_dqn: bool = False,
    replay_timeout_weight: float | None = None,
    replay_short_stop_max_steps: int | None = None,
    replay_short_stop_weight: float | None = None,
    learning_rate: float | None = None,
    epsilon_end: float | None = None,
    target_update_period: int | None = None,
    pulse_width_min: float | None = None,
    pulse_width_min_early: float | None = None,
    pulse_width_min_early_episodes: int | None = None,
    pulse_width_min_ramp_end_episode: int | None = None,
    pulse_width_sensitivity: float | None = None,
    pulse_width_sensitivity_early: float | None = None,
    pulse_width_sensitivity_early_episodes: int | None = None,
    frequency_sensitivity_early_episodes: int | None = None,
    subthreshold_steps_required: int | None = None,
    overrides: dict[str, Any] | None = None,
    on_checkpoint: Any = None,
    allow_repeat: bool = True,
    repeat_search_dir: Path | None = None,
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
) -> tuple[dict[str, Any], SNNConfig, DSQNTrainer, TrainResult]:
    if smoke:
        cfg = SNNConfig(seed=seed).for_smoke(episodes=min(5, num_episodes), max_steps=8)
    else:
        cfg = fig4_nguyen_config(seed=seed, num_episodes=num_episodes)
        if double_dqn:
            cfg = replace(cfg, double_dqn=True)
        if replay_timeout_weight is not None:
            cfg = replace(cfg, replay_timeout_weight=replay_timeout_weight)
        if replay_short_stop_max_steps is not None:
            cfg = replace(cfg, replay_short_stop_max_steps=replay_short_stop_max_steps)
        if replay_short_stop_weight is not None:
            cfg = replace(cfg, replay_short_stop_weight=replay_short_stop_weight)
        if learning_rate is not None:
            cfg = replace(cfg, learning_rate=learning_rate)
        if epsilon_end is not None:
            cfg = replace(cfg, epsilon_end=epsilon_end)
        if target_update_period is not None:
            cfg = replace(cfg, target_update_period=target_update_period)
        if pulse_width_min is not None:
            cfg = replace(cfg, pulse_width_min=pulse_width_min)
        if pulse_width_min_early is not None:
            cfg = replace(cfg, pulse_width_min_early=pulse_width_min_early)
        if pulse_width_min_early_episodes is not None:
            cfg = replace(cfg, pulse_width_min_early_episodes=pulse_width_min_early_episodes)
        if pulse_width_min_ramp_end_episode is not None:
            cfg = replace(cfg, pulse_width_min_ramp_end_episode=pulse_width_min_ramp_end_episode)
        if pulse_width_sensitivity is not None:
            cfg = replace(cfg, pulse_width_sensitivity=pulse_width_sensitivity)
        if pulse_width_sensitivity_early is not None:
            cfg = replace(cfg, pulse_width_sensitivity_early=pulse_width_sensitivity_early)
        if pulse_width_sensitivity_early_episodes is not None:
            cfg = replace(
                cfg,
                pulse_width_sensitivity_early_episodes=pulse_width_sensitivity_early_episodes,
            )
        if frequency_sensitivity_early_episodes is not None:
            cfg = replace(
                cfg,
                frequency_sensitivity_early_episodes=frequency_sensitivity_early_episodes,
            )
        if subthreshold_steps_required is not None:
            cfg = replace(cfg, subthreshold_steps_required=subthreshold_steps_required)
        if overrides:
            cfg = replace(cfg, **overrides)

    budget = training_budget(cfg)
    print(
        f"preflight: env steps {budget['env_steps_range']}, SGD updates {budget['sgd_updates_range']}, "
        f"target syncs {budget['target_syncs_range'] if budget['target_network'] else 'n/a (no target network)'} "
        f"(cadence {budget['replay_update_cadence']} x{budget['replay_update_steps']} steps, "
        f"target_update_period {budget['target_update_period']})",
        flush=True,
    )
    for warning in budget["warnings"]:
        print(f"preflight WARNING: {warning}", flush=True)
    if not smoke and not allow_repeat and resume_path is None and repeat_search_dir is not None:
        prior = find_repeat_run(config_record(cfg), repeat_search_dir)
        if prior is not None:
            print(
                f"identical config already ran at this commit ({prior}); "
                "change a knob or pass --allow-repeat",
                file=sys.stderr,
            )
            raise SystemExit(_panel.EXIT_MISSING_INPUT)

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
            on_checkpoint=on_checkpoint,
        )
        payload: dict[str, Any] = {
            "seed": cfg.seed,
            "num_episodes": cfg.num_episodes,
            "max_episode_steps": cfg.max_episode_steps,
            "episode_rewards": result.episode_rewards,
            "episode_lengths": result.episode_lengths,
            "episode_spikes_per_step": result.episode_spikes_per_step,
            "episode_energies": result.episode_energies,
            "episode_alpha_beta_means": result.episode_alpha_beta_means,
            "episode_alpha_beta_finals": result.episode_alpha_beta_finals,
            "episode_early_stops": result.episode_early_stops,
            "episode_amplitudes": result.episode_amplitudes,
            "episode_frequencies": result.episode_frequencies,
            "episode_pulse_widths": result.episode_pulse_widths,
            "update_count": result.update_count,
            "smoke": smoke,
            "config": config_record(cfg),
            "training_budget": budget,
            "target_syncs": (
                result.update_count // cfg.target_update_period if cfg.target_update_period > 0 else 0
            ),
        }
        return payload, cfg, trainer, result
    finally:
        env.close()


def evaluate_gates(
    series: dict[str, Any],
    *,
    max_episode_steps: int,
) -> dict[str, Any]:
    """Fig 4 gates for a saved or in-memory series (``nguyen_gates.fig4_gates``).

    ``fig4_gates`` raises ``Fig4GateKeyError`` if a required key is not computed,
    so a deleted gate can never silently read as a fail (or a pass).
    """
    rewards = np.asarray(series["episode_rewards"], dtype=float)
    lengths = np.asarray(series["episode_lengths"], dtype=float)
    gates = fig4_gates(rewards, lengths, max_episode_steps=max_episode_steps)
    for group in ("reward", "length"):
        missing = [k for k in (*FIG4_REQUIRED_KEYS[group], "shape_pass", "pass") if k not in gates[group]]
        missing += [k for k in FIG4_TIER2_KEYS[group] if k not in gates[group]["tier2"]]
        if missing:
            msg = f"Fig 4 {group} gate output missing keys: {missing}"
            raise KeyError(msg)
    gates["reward"]["late_reward_mean"] = gates["reward"]["metrics"]["mean_150_end"]
    gates["length"]["late_length_mean"] = gates["length"]["metrics"]["mean_150_end"]
    if series.get("smoke"):
        gates["pass"] = False
        gates["shape_pass"] = False
        gates["smoke_override"] = True
    return gates


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
    parser.add_argument(
        "--double-dqn",
        action="store_true",
        help="Double DQN bootstrap (v83 recipe + online argmax / target eval)",
    )
    parser.add_argument(
        "--timeout-replay-weight",
        type=float,
        default=None,
        metavar="W",
        help="TD loss weight for max-horizon timeout-episode transitions (v89: 0.25 with --double-dqn)",
    )
    parser.add_argument(
        "--short-stop-max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Mark early-stop episodes with length <= N for short-stop replay down-weight (v99: 6)",
    )
    parser.add_argument(
        "--short-stop-replay-weight",
        type=float,
        default=None,
        metavar="W",
        help="TD loss weight for short early-stop episodes (v99: 0.25 with --short-stop-max-steps 6)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        metavar="LR",
        help="Adam learning rate override (v90: 2e-4 with --double-dqn for smoother post-100 ptp)",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=None,
        metavar="EPS",
        help="Late ε floor override (v91: 0.04 with v90 recipe to trim late timeouts)",
    )
    parser.add_argument(
        "--target-update-period",
        type=int,
        default=None,
        metavar="N",
        help="Hard target-network copy period (v92: 225 on v90 recipe for late timeout trim)",
    )
    parser.add_argument(
        "--pulse-width-min",
        type=float,
        default=None,
        metavar="MS",
        help="Minimum pulse width (ms); v108: 0.8 toward paper Fig 6 ~1 ms for Fig 5 energy",
    )
    parser.add_argument(
        "--pulse-width-min-early",
        type=float,
        default=None,
        metavar="MS",
        help="Early episodes minimum pulse width (ms)",
    )
    parser.add_argument(
        "--pulse-width-min-early-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Episodes using pulse_width_min_early before exploit floor",
    )
    parser.add_argument(
        "--pulse-width-min-ramp-end-episode",
        type=int,
        default=None,
        metavar="N",
        help="Episode where pulse_width_min linear ramp reaches full pulse_width_min",
    )
    parser.add_argument(
        "--pulse-width-sensitivity",
        type=float,
        default=None,
        metavar="S",
        help="Pulse width delta per +1 action (ms); v108: 0.4 for mid-training pw ramp",
    )
    parser.add_argument(
        "--pulse-width-sensitivity-early",
        type=float,
        default=None,
        metavar="S",
        help="Early episodes pulse width delta per +1 action (ms)",
    )
    parser.add_argument(
        "--pulse-width-sensitivity-early-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Episodes using pulse_width_sensitivity_early before exploit ms/step",
    )
    parser.add_argument(
        "--frequency-sensitivity-early-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Episodes using frequency_sensitivity_early before exploit Hz/step (v108: 100)",
    )
    parser.add_argument(
        "--subthreshold-steps",
        type=int,
        default=None,
        metavar="TU",
        help="Consecutive sub-threshold steps for early stop (t_u); default from fig4_nguyen_config",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        metavar="FIELD=VALUE",
        help="Override any SNNConfig field (repeatable), e.g. --set amplitude_min=265",
    )
    parser.add_argument(
        "--abort-on-fail",
        action="store_true",
        help=(
            "Stop training (exit 3) at the first checkpoint from episode "
            f"{FIG4_ABORT_EPISODE} where length or reward progress over ep 120–150 is below 0.5 "
            "(doomed-run check only)"
        ),
    )
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Train even if an identical config already ran at this clean commit",
    )
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
    overrides = parse_config_overrides(args.config_overrides)
    if args.smoke:
        # Smoke is a plumbing check: never touch the panel's real series/checkpoint/manifest
        # (Fig 5–7 read them) or allocate a tracked PNG version.
        smoke_dir = Path(tempfile.mkdtemp(prefix="nguyen4-smoke-"))
        args.series = smoke_dir / "series.json"
        args.checkpoint = smoke_dir / "checkpoint.pt"
        args.manifest = smoke_dir / "manifest.json"
        args.out = smoke_dir / f"{OUT_STEM}_v1.png"
        args.no_update_docs = True
        args.export_notes = False
        args.update_report = False
        print(f"smoke outputs -> {smoke_dir}", flush=True)
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _linked_png(args.out)

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
            f"double_dqn={args.double_dqn} timeout_replay_weight={args.timeout_replay_weight} "
            f"short_stop_max_steps={args.short_stop_max_steps} "
            f"short_stop_replay_weight={args.short_stop_replay_weight} "
            f"learning_rate={args.learning_rate} epsilon_end={args.epsilon_end} "
            f"target_update_period={args.target_update_period} "
            f"pulse_width_min={args.pulse_width_min} "
            f"pulse_width_sensitivity={args.pulse_width_sensitivity} "
            f"freq_sens_early_ep={args.frequency_sensitivity_early_episodes} "
            f"subthreshold_steps={args.subthreshold_steps} overrides={overrides} resume={args.resume}",
            flush=True,
        )
        partial_path = args.manifest.with_name(args.manifest.stem + ".partial.json")

        def _on_checkpoint(completed: int, result: TrainResult) -> None:
            probe = abort_probe(completed, result)
            write_json(partial_path, _panel.stamp_manifest({"partial": probe}))
            if probe.get("decidable"):
                print(
                    f"checkpoint probe ep {completed}: abort={probe['abort']} "
                    f"len_progress={probe['length_progress_120_150']:.2f} "
                    f"reward_progress={probe['reward_progress_120_150']:.2f} "
                    f"failed={probe['failed']}",
                    flush=True,
                )
            if args.abort_on_fail and probe.get("decidable") and probe["abort"]:
                raise EarlyAbort(completed, probe["failed"], probe)

        try:
            series, cfg, trainer, train_result = train_series(
                seed=args.seed,
                num_episodes=args.episodes,
                smoke=args.smoke,
                double_dqn=args.double_dqn,
                replay_timeout_weight=args.timeout_replay_weight,
                replay_short_stop_max_steps=args.short_stop_max_steps,
                replay_short_stop_weight=args.short_stop_replay_weight,
                learning_rate=args.learning_rate,
                epsilon_end=args.epsilon_end,
                target_update_period=args.target_update_period,
                pulse_width_min=args.pulse_width_min,
                pulse_width_min_early=args.pulse_width_min_early,
                pulse_width_min_early_episodes=args.pulse_width_min_early_episodes,
                pulse_width_min_ramp_end_episode=args.pulse_width_min_ramp_end_episode,
                pulse_width_sensitivity=args.pulse_width_sensitivity,
                pulse_width_sensitivity_early=args.pulse_width_sensitivity_early,
                pulse_width_sensitivity_early_episodes=args.pulse_width_sensitivity_early_episodes,
                frequency_sensitivity_early_episodes=args.frequency_sensitivity_early_episodes,
                subthreshold_steps_required=args.subthreshold_steps,
                overrides=overrides,
                on_checkpoint=None if args.smoke else _on_checkpoint,
                allow_repeat=args.allow_repeat,
                repeat_search_dir=args.manifest.parent,
                resume_path=args.resume,
                start_episode=args.start_episode,
                checkpoint_path=args.checkpoint,
                checkpoint_interval=args.checkpoint_interval,
            )
        except EarlyAbort as abort:
            manifest = {
                "panel": "2/4",
                "aborted": True,
                "aborted_at_episode": abort.completed,
                "shape_failed": abort.failed,
                "abort_probe": abort.probe,
                "checkpoint": args.checkpoint.as_posix(),
                "gates": {"pass": False, "shape_pass": False},
                "smoke": False,
            }
            write_json(args.manifest, _panel.stamp_manifest(manifest))
            print(f"EARLY ABORT: {abort}; checkpoint kept at {args.checkpoint}", flush=True)
            return _panel.EXIT_ABORTED
        if series.get("target_syncs", 1) == 0 and cfg.target_update_period > 0 and not args.smoke:
            print(
                "WARNING: the target network never synced during this run "
                f"({series['update_count']} SGD updates < target_update_period)",
                flush=True,
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
                extra=episode_extra(
                    train_result,
                    completed_episodes=len(train_result.episode_rewards),
                    update_count=trainer.update_count,
                ),
            )
            write_train_metrics(train_result, args.checkpoint.with_suffix(".metrics.json"))

    gates = evaluate_gates(series, max_episode_steps=max_steps)
    panel = plot_series(series, args.out, smooth_window=args.smooth_window)

    caption = (
        f"DSQN train {series['num_episodes']} ep, seed={series['seed']}; "
        f"reward ep150+={gates['reward']['late_reward_mean']:.0f}, "
        f"len ep150+={gates['length']['late_length_mean']:.1f}; "
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
        "config": series.get("config"),
    }
    manifest = _panel.stamp_manifest(
        manifest,
        inputs={"series": args.series, "checkpoint": args.checkpoint},
    )
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
    return 0 if gates["pass"] or manifest["smoke"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
