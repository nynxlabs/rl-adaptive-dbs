#!/usr/bin/env python3
"""Mehregan et al.  Figure 4a — training GPi beta power vs step.

Paper §IV.A.1: DDPG on the **45 Hz** fixed-mean pattern alphabet, 10 episodes ×
30 steps (300 env steps). Y-axis **PSD(x10³)** = ``p_beta_norm`` =
raw $P_\\beta$ / 1000.

Defaults (see ``figures/mehregan/replications.md``):
  seed 0, state_length=1, fixed_mean_pattern, softmax + one_hot critic,
  init_bias=0.0 (phase-1 ep0; library fig4a profile is still 0.5),
  jitter_fraction=0.5 (library alphabet default 1/3), plant.dt_ms=0.02.

Run:
  uv run python scripts/figures/papers/mehregan/4a/plot.py
  uv run python scripts/figures/papers/mehregan/4a/plot.py --plot-only

Each run writes ``figures/mehregan/images/4a/training_beta_vN.png`` (N auto-increments),
``artifacts/figures/papers/mehregan/4a/checkpoint.pt`` (fp32 actor for Fig 5a/6a eval),
and updates the replication image link in ``figures/mehregan/replications.md``.

Long run (~30–60 min). Prefer tmux:

  tmux new-session -d -s fig4a-train \\
    "setsid nohup uv run python scripts/figures/papers/mehregan/4a/plot.py >> logs/fig4a-train.log 2>&1 < /dev/null"
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from controllers.ddpg.checkpoint import (
    _config_from_checkpoint_payload,
    infer_ddpg_start_episode,
    load_checkpoint,
    save_checkpoint,
    validate_resume_config,
)
from controllers.ddpg.config import DDPGConfig, fig4a_ddpg_config
from controllers.ddpg.quantization import unwrap_actor
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig4a_gates  # noqa: E402

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

_RESUME_CLI = Path(__file__).resolve().parents[2] / "resume_cli.py"
_resume_spec = importlib.util.spec_from_file_location("figure_resume_cli", _RESUME_CLI)
assert _resume_spec and _resume_spec.loader
_resume_cli = importlib.util.module_from_spec(_resume_spec)
_resume_spec.loader.exec_module(_resume_cli)

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_import_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_import_spec and _overlay_import_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_import_spec)
_overlay_import_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

FIGURES_DIR = Path("figures/mehregan/images/4a")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/4a")
DEFAULT_SERIES = CACHE_DIR / "series.json"
DEFAULT_CHECKPOINT = CACHE_DIR / "checkpoint.pt"
OUT_STEM = "training_beta"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"

PAPER_DT_MS = 0.02
MEAN_HZ = 45.0
STATE_LENGTH = 1
NUM_EPISODES = 10
STEPS_PER_EPISODE = 30
DEFAULT_SEED = 0
EARLY_END = 130
LATE_START = 150
WINDOW = 30
# Paper-silent Fig 4a knobs for the ep0-first revisit. Library alphabet jitter
# is 1/3; init_bias in fig4a_ddpg_config remains 0.5. Regular 45 Hz (pattern 0)
# is a *suppressor* on this plant (~0.29), so biasing toward it cannot raise ep0.
FIG4A_JITTER_FRACTION = 0.5
FIG4A_INIT_BIAS_SCALE = 0.0
# Sequential 2 s segments (Alg. 1). Disconnected cold-starts make repeated
# pattern 0 bit-identical (ruler-flat late traces). Fresh train — material.
FIG4A_PLANT_INTEGRATION = "continuous"

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
}


def _vault_backed_png(path: Path) -> Path:
    """Ensure ``path`` is a symlink into the vault figure dir when ``paper.png`` is.

    Fig assets under ``figures/<paper>/images/`` are usually vault symlinks. New
    versioned PNGs must follow the same pattern so the main checkout and docs see them.
    """
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
    # Create/overwrite vault file on savefig through this symlink.
    if not vault_target.exists():
        vault_target.touch()
    path.symlink_to(vault_target)
    return path


def _make_env(
    *,
    seed: int,
    state_length: int = STATE_LENGTH,
    jitter_fraction: float = FIG4A_JITTER_FRACTION,
    plant_integration_mode: str = FIG4A_PLANT_INTEGRATION,
) -> tuple[MehreganEnv, Any]:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=state_length,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        plant_integration_mode=plant_integration_mode,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
        jitter_fraction=jitter_fraction,
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)
    _ = seed  # seed applied per-episode in train loop
    return env, plant_cfg


def _train_trace(
    env: MehreganEnv,
    *,
    seed: int,
    num_episodes: int,
    exploration_mode: str = "softmax",
    init_bias_scale: float = FIG4A_INIT_BIAS_SCALE,
    temperature_start: float = 2.0,
    temperature_end: float = 1.0,
    logit_noise_std: float = 0.05,
    entropy_coeff: float = 0.08,
    critic_action_input: str = "one_hot",
    critic_warmup_steps: int = 15,
    actor_lr: float = 9.0e-4,
    action_persistence: int = 3,
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
    prior_beta_trace: list[float] | None = None,
    prior_actions: list[int] | None = None,
) -> tuple[list[float], list[int], list[float], dict[str, Any], DDPGTrainer, DDPGConfig]:
    """Train and return beta_norm per step, actions, episode rewards, and trainer."""
    config = fig4a_ddpg_config(
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=STEPS_PER_EPISODE,
        exploration_mode=exploration_mode,
        init_bias_scale=init_bias_scale,
        exploration_temperature_start=temperature_start,
        exploration_temperature_end=temperature_end,
        logit_noise_std=logit_noise_std,
        entropy_coeff=entropy_coeff,
        critic_action_input=critic_action_input,
        critic_warmup_steps=critic_warmup_steps,
        actor_lr=actor_lr,
    )
    trainer = DDPGTrainer(env, config)
    beta_trace: list[float] = list(prior_beta_trace or [])
    actions: list[int] = list(prior_actions or [])
    episode_rewards: list[float] = list(trainer.metrics.episode_rewards)
    resume_start = 0

    if resume_path is not None:
        payload = load_checkpoint(resume_path, device=config.device)
        saved_cfg = _config_from_checkpoint_payload(payload["ddpg_config"])
        metrics_path = resume_path.with_suffix(".metrics.json")
        resume_start = infer_ddpg_start_episode(
            payload,
            metrics_path=metrics_path,
            start_episode=start_episode,
        )
        validate_resume_config(saved_cfg, config, resume_start=resume_start)
        unwrap_actor(trainer.actor).load_state_dict(payload["actor_state_dict"])
        trainer.load_resume_state(payload)
        episode_rewards = list(trainer.metrics.episode_rewards)

    if resume_start == 0:
        env_step = trainer._random_warmup()
    else:
        env_step = trainer._env_step

    for episode in range(resume_start, num_episodes):
        state, info0 = env.reset(seed=seed + episode)
        trainer._update_obs_stats(state)
        episode_reward = float(info0.get("reward", 0.0))
        terminated = truncated = False
        action = None
        logits = None
        steps_remaining_on_action = 0

        while not (terminated or truncated):
            if steps_remaining_on_action <= 0 or action is None:
                action, logits = trainer._select_action(state, env_step=env_step)
                steps_remaining_on_action = action_persistence
            steps_remaining_on_action -= 1

            env_step += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            trainer._update_obs_stats(next_state)
            beta_trace.append(float(info["p_beta_norm"]))
            actions.append(int(action))
            episode_reward += float(reward)
            normalized_reward = trainer._normalize_reward(reward)
            dw = float(info.get("dw", 1.0 if truncated else 0.0))
            trainer.buffer.add(
                state=state,
                action=action,
                action_logits=logits,
                reward=normalized_reward,
                next_state=next_state,
                dw=dw,
            )
            state = next_state
            if len(trainer.buffer) >= config.min_buffer_size:
                for _ in range(config.update_frequency):
                    trainer._update_step()
        episode_rewards.append(episode_reward)
        trainer._env_step = env_step
        completed = episode + 1
        if checkpoint_path is not None and checkpoint_interval > 0:
            if completed % checkpoint_interval == 0 or completed == num_episodes:
                _save_training_checkpoint(
                    trainer=trainer,
                    config=config,
                    env=env,
                    path=checkpoint_path,
                    extra={
                        "completed_episodes": completed,
                        "figure": "mehregan_fig4a",
                        "seed": seed,
                        "episode_rewards": episode_rewards,
                    },
                )
        print(
            f"episode {episode + 1}/{num_episodes} "
            f"reward={episode_reward:.2f} steps={STEPS_PER_EPISODE}",
            flush=True,
        )

    counts = Counter(actions)
    dominant, dom_n = counts.most_common(1)[0] if counts else (-1, 0)
    meta = {
        "exploration_mode": config.exploration_mode,
        "init_bias_scale": config.init_bias_scale,
        "jitter_fraction": getattr(env.alphabet, "jitter_fraction", None),
        "temperature_start": config.exploration_temperature_start,
        "temperature_end": config.exploration_temperature_end,
        "logit_noise_std": config.logit_noise_std,
        "entropy_coeff": config.entropy_coeff,
        "critic_warmup_steps": config.critic_warmup_steps,
        "actor_lr": config.actor_lr,
        "critic_action_input": config.critic_action_input,
        "plant_integration_mode": getattr(
            env.config, "plant_integration_mode", "disconnected"
        ),
        "unique_actions": len(counts),
        "dominant_action": int(dominant),
        "dominant_fraction": dom_n / len(actions) if actions else 0.0,
        "action_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "episode_rewards": episode_rewards,
    }
    return beta_trace, actions, episode_rewards, meta, trainer, config


def _save_training_checkpoint(
    *,
    trainer: DDPGTrainer,
    config: DDPGConfig,
    env: MehreganEnv,
    path: Path,
    extra: dict[str, Any],
) -> Path:
    """Persist fp32 actor weights for Fig 5a/6a eval on the same trained model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path,
        actor=unwrap_actor(trainer.actor),
        config=config,
        state_length=int(env.config.state_length),
        n_actions=int(env.action_space.n),
        policy=trainer.actor,
        critic=trainer.critic,
        trainer=trainer,
        extra=extra,
    )
    return path


def _window_mean(trace: list[float], start: int, end: int) -> float:
    chunk = np.asarray(trace[start:end], dtype=float)
    if chunk.size == 0:
        return float("nan")
    return float(chunk.mean())


def _gate_summary(beta_trace: list[float]) -> dict[str, Any]:
    """Digitization-anchored gates (paper early/late x windows + drop ratio).

    Absolute early/late bands are intentionally dropped: seed changes level;
    paper is one realization. Require trend down and a drop that tracks the
    digitized paper drop / late-to-early ratio.
    """
    n = len(beta_trace)
    early = _window_mean(beta_trace, 0, min(EARLY_END, n))
    late = _window_mean(beta_trace, min(LATE_START, n), n)
    start_w = _window_mean(beta_trace, 0, min(WINDOW, n))
    end_w = _window_mean(beta_trace, max(0, n - WINDOW), n)
    mid = _window_mean(beta_trace, min(120, n), min(150, n))
    dig = fig4a_gates(beta_trace, n_expected=NUM_EPISODES * STEPS_PER_EPISODE)
    gates = dict(dig["gates"])
    return {
        "n_steps": n,
        "early_mean_0_130": early,
        "late_mean_150_end": late,
        "start_window_mean": start_w,
        "end_window_mean": end_w,
        "delta_end_minus_start": end_w - start_w,
        "mid_mean_120_150": mid,
        "trend_down": gates.get("overall_trend_down"),
        "paper_gate_metrics": dig["metrics"],
        "paper_ref": dig["paper_ref"],
        "gates": gates,
        "gates_pass": all(gates.values()),
    }


def _ylim_for_trace(y: np.ndarray) -> tuple[float, float, list[float]]:
    """Y limits that include every sample (paper panel uses ~0.3–0.6; we extend if needed)."""
    if y.size == 0 or not np.isfinite(y).any():
        return 0.3, 0.6, [0.3, 0.4, 0.5, 0.6]
    pad = 0.02
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    # Keep the paper's usual window when possible, but never clip the trace.
    lo = min(0.3, y_min - pad)
    hi = max(0.6, y_max + pad)
    lo = float(np.floor(lo * 20.0) / 20.0)  # 0.05 grid
    hi = float(np.ceil(hi * 20.0) / 20.0)
    if hi <= lo:
        hi = lo + 0.3
    ticks = [float(t) for t in np.arange(lo, hi + 1e-9, 0.1)]
    if not ticks or ticks[-1] < hi - 1e-9:
        ticks.append(hi)
    return lo, hi, ticks


def plot_fig4a(cache: dict[str, Any], *, out_path: Path) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    y = np.asarray(cache["beta_norm_trace"], dtype=float)
    x = np.arange(y.size)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(x, y, color="#1f6f6f", linewidth=1.0, label="Replication")
    paper_y = _paper_overlay.overlay_mehregan_fig4a(ax)
    paper_vals = np.concatenate([arr[0] for arr in paper_y.values() if arr[0].size]) if paper_y else np.array([])
    y_combined = np.concatenate([y, paper_vals]) if paper_vals.size else y
    ax.set_xlim(0, 300)
    ax.set_xticks([0, 60, 120, 180, 240, 300])
    y0, y1, yticks = _ylim_for_trace(y_combined)
    ax.set_ylim(y0, y1)
    ax.set_yticks(yticks)
    ax.set_xlabel("Steps")
    ax.set_ylabel(r"PSD($x10^3$)")
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    fig.tight_layout()
    _paper_overlay.place_legend(ax, loc="upper right", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return {
        "n_steps": int(y.size),
        "y_min": float(y.min()) if y.size else float("nan"),
        "y_max": float(y.max()) if y.size else float("nan"),
        "y_mean": float(y.mean()) if y.size else float("nan"),
        "ylim": [y0, y1],
    }


def _checklist_rows(gates: dict[str, Any], summary: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(check, paper, replication, match) rows for docs."""
    early = summary.get("early_mean_0_130")
    late = summary.get("late_mean_150_end")
    mid = summary.get("mid_mean_120_150")
    trend = summary.get("trend_down")

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    return [
        (
            "**Plot style**",
            "Single noisy line, 0–300 steps",
            f"Line trace, {summary.get('n_steps', '—')} steps",
            "✓" if gates.get("plot_style") else "✗",
        ),
        (
            "**Axes**",
            "Steps 0–300; y **PSD(x10³)** ~0.3–0.6",
            "Same labels; y auto-fits full trace (extends below 0.3 if needed)",
            "✓",
        ),
        (
            "**Episode 0**",
            "First 30 steps near digitized paper ~0.50 (untreated-like)",
            f"ep0 mean {_fmt(summary.get('paper_gate_metrics', {}).get('ep0_mean'))}; "
            f"paper {_fmt(summary.get('paper_gate_metrics', {}).get('paper_ep0'))}",
            "✓" if gates.get("ep0_near_paper") else "✗",
        ),
        (
            "**Drop vs paper**",
            "Digitized early→late drop (seed-robust)",
            f"early {_fmt(early)} → late {_fmt(late)}; "
            f"paper metrics={summary.get('paper_gate_metrics')}",
            "✓" if gates.get("drop_vs_paper") else "✗",
        ),
        (
            "**Drop timing / mid fade**",
            "Digitized mid fade ~120–150 (modest, not a cliff)",
            f"mid(120–150) mean {_fmt(mid)}; gate mid_fade_vs_paper",
            "✓" if gates.get("mid_fade_vs_paper") else "✗",
        ),
        (
            "**Late/early ratio**",
            "Near digitized paper late/early ratio",
            f"ours late/early; gate late_early_ratio_near_paper",
            "✓" if gates.get("late_early_ratio_near_paper") else "✗",
        ),
        (
            "**Overall trend**",
            "Mean beta **decreases** over training",
            f"end−start window Δ={_fmt(summary.get('delta_end_minus_start'))}",
            "✓" if trend else "✗",
        ),
    ]


def update_checklist_in_doc(rows: list[tuple[str, str, str, str]]) -> None:
    """Rewrite the Fig 4a checklist Match? column in replications/mehregan.md."""
    doc = _figure_promote.PAPER_1_DOC
    text = doc.read_text()
    start = text.find("## Fig 4a — training beta power vs step")
    if start < 0:
        return
    table_hdr = (
        "| Check | Paper | Replication | Match? |\n"
        "|-------|-------|-------------|--------|\n"
    )
    hdr_pos = text.find(table_hdr, start)
    if hdr_pos < 0:
        return
    body_start = hdr_pos + len(table_hdr)
    # Table ends at blank line before **Run:**
    run_pos = text.find("\n**Run:**", body_start)
    if run_pos < 0:
        return
    new_rows = "".join(f"| {a} | {b} | {c} | {d} |\n" for a, b, c, d in rows)
    doc.write_text(text[:body_start] + new_rows + text[run_pos:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--series", type=Path, default=DEFAULT_SERIES)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Save fp32 actor here after training (Fig 5a/6a eval source)",
    )
    parser.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Skip writing --checkpoint (training trace + PNG only)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: auto-increment "
            f"{FIGURES_DIR.as_posix()}/{OUT_STEM}_vN.png"
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Re-plot from cached series JSON (skip training)",
    )
    parser.add_argument(
        "--exploration",
        choices=("greedy", "softmax", "epsilon"),
        default="softmax",
        help="Training action selection (default: softmax Fig 4a profile)",
    )
    parser.add_argument(
        "--init-bias-scale",
        type=float,
        default=FIG4A_INIT_BIAS_SCALE,
        help=(
            "Actor head bias toward pattern 0 (Fig 4a phase-1 default 0.0; "
            "library fig4a profile is 0.5). Pattern 0 suppresses on this plant."
        ),
    )
    parser.add_argument(
        "--jitter-fraction",
        type=float,
        default=FIG4A_JITTER_FRACTION,
        help=(
            "Interior-onset jitter as a fraction of the regular ISI "
            f"(default {FIG4A_JITTER_FRACTION}; library alphabet is 1/3). "
            "Paper-silent; larger jitter weakens the irregular mix vs untreated."
        ),
    )
    parser.add_argument(
        "--temperature-start",
        type=float,
        default=2.0,
        help="Softmax temperature at step 0 (default: 2.0)",
    )
    parser.add_argument(
        "--temperature-end",
        type=float,
        default=1.0,
        help="Softmax temperature at final step (default: 1.0)",
    )
    parser.add_argument(
        "--logit-noise-std",
        type=float,
        default=0.05,
        help="Gaussian noise on actor logits during training (default: 0.05)",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.08,
        help="Policy entropy bonus (default: 0.08)",
    )
    parser.add_argument(
        "--critic-warmup-steps",
        type=int,
        default=15,
        help=(
            "Gradient steps that update the critic only (default: 15). "
            "Lower values let the actor start learning earlier for organic mid-fade."
        ),
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=9.0e-4,
        help="Adam learning rate for the actor (default: 9.0e-4).",
    )
    parser.add_argument(
        "--action-persistence",
        type=int,
        default=3,
        help="Number of consecutive 2s steps each selected action is held (Option C, default: 3).",
    )
    parser.add_argument(
        "--critic-action-input",
        choices=("one_hot", "logits"),
        default="one_hot",
        help="Critic action encoding (default one_hot — required for learning under exploration)",
    )
    parser.add_argument(
        "--plant-integration",
        choices=("disconnected", "continuous"),
        default=FIG4A_PLANT_INTEGRATION,
        help=(
            "disconnected = cold 2 s from episode ICs (bit-identical repeats); "
            "continuous = sequential stitched idbs (Alg. 1; default)"
        ),
    )
    parser.add_argument(
        "--state-length",
        type=int,
        default=STATE_LENGTH,
        help="Observation length L (1 = scalar P_beta per paper Fig 4a)",
    )
    parser.add_argument(
        "--no-update-docs",
        dest="update_docs",
        action="store_false",
        help="Skip figures/mehregan/replications.md caption + replication image link",
    )
    parser.add_argument(
        "--update-checklist",
        action="store_true",
        help="Rewrite the Fig 4a side-by-side checklist in replications/mehregan.md (off by default)",
    )
    parser.set_defaults(update_docs=True)
    _resume_cli.add_training_resume_args(parser)
    args = parser.parse_args()
    _resume_cli.configure_promote_publish(args, _figure_promote)

    if args.out is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        args.out, png_version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    else:
        png_version = _figure_promote.parse_png_version(args.out)
    args.out = _vault_backed_png(args.out)

    if args.plot_only:
        if not args.series.exists():
            print(f"missing series cache: {args.series}", file=sys.stderr)
            print("Run without --plot-only once to build the cache.", file=sys.stderr)
            return 2
        cache = json.loads(args.series.read_text())
        print(f"loaded traces from {args.series}", flush=True)
    else:
        expected = args.episodes * STEPS_PER_EPISODE
        print(
            f"Fig 4a train — {args.episodes} ep × {STEPS_PER_EPISODE} "
            f"({expected} steps), 45 Hz, L={args.state_length}, "
            f"exploration={args.exploration}, init_bias={args.init_bias_scale}, "
            f"jitter={args.jitter_fraction}, plant={args.plant_integration}, "
            f"critic={args.critic_action_input}",
            flush=True,
        )
        prior_beta: list[float] | None = None
        prior_actions: list[int] | None = None
        if args.resume is not None and args.series.exists():
            prior_cache = json.loads(args.series.read_text())
            prior_beta = list(prior_cache.get("beta_norm_trace", []))
            prior_actions = list(prior_cache.get("actions", []))

        env, plant_cfg = _make_env(
            seed=args.seed,
            state_length=args.state_length,
            jitter_fraction=args.jitter_fraction,
            plant_integration_mode=args.plant_integration,
        )
        t0 = time.time()
        trainer: DDPGTrainer | None = None
        train_config: DDPGConfig | None = None
        try:
            beta_trace, actions, episode_rewards, train_meta, trainer, train_config = (
                _train_trace(
                    env,
                    seed=args.seed,
                    num_episodes=args.episodes,
                    exploration_mode=args.exploration,
                    init_bias_scale=args.init_bias_scale,
                    temperature_start=args.temperature_start,
                    temperature_end=args.temperature_end,
                    logit_noise_std=args.logit_noise_std,
                    entropy_coeff=args.entropy_coeff,
                    critic_action_input=args.critic_action_input,
                    critic_warmup_steps=args.critic_warmup_steps,
                    actor_lr=args.actor_lr,
                    action_persistence=args.action_persistence,
                    resume_path=args.resume,
                    start_episode=args.start_episode,
                    checkpoint_path=args.checkpoint,
                    checkpoint_interval=args.checkpoint_interval,
                    prior_beta_trace=prior_beta,
                    prior_actions=prior_actions,
                )
            )
            if not args.no_save_checkpoint and trainer is not None and train_config is not None:
                ckpt_extra = {
                    "figure": "mehregan_fig4a",
                    "seed": args.seed,
                    "png_version": png_version,
                    "dominant_action": train_meta.get("dominant_action"),
                    "dominant_fraction": train_meta.get("dominant_fraction"),
                    "episode_rewards": episode_rewards,
                }
                _save_training_checkpoint(
                    trainer=trainer,
                    config=train_config,
                    env=env,
                    path=args.checkpoint,
                    extra=ckpt_extra,
                )
                versioned_ckpt = args.checkpoint.with_name(
                    f"checkpoint_v{png_version}.pt"
                )
                _save_training_checkpoint(
                    trainer=trainer,
                    config=train_config,
                    env=env,
                    path=versioned_ckpt,
                    extra=ckpt_extra,
                )
                print(f"wrote checkpoint {args.checkpoint}", flush=True)
                print(f"wrote {versioned_ckpt}", flush=True)
        finally:
            env.close()
        elapsed = time.time() - t0
        summary = _gate_summary(beta_trace)
        cache = {
            "figure": "mehregan_fig4a",
            "seed": args.seed,
            "num_episodes": args.episodes,
            "steps_per_episode": STEPS_PER_EPISODE,
            "mean_hz": MEAN_HZ,
            "state_length": args.state_length,
            "plant_dt_ms": plant_cfg.dt_ms,
            "exploration_mode": args.exploration,
            "init_bias_scale": args.init_bias_scale,
            "jitter_fraction": args.jitter_fraction,
            "temperature_start": args.temperature_start,
            "temperature_end": args.temperature_end,
            "logit_noise_std": args.logit_noise_std,
            "entropy_coeff": args.entropy_coeff,
            "critic_warmup_steps": args.critic_warmup_steps,
            "actor_lr": args.actor_lr,
            "critic_action_input": args.critic_action_input,
            "plant_integration_mode": args.plant_integration,
            "elapsed_s": elapsed,
            "beta_norm_trace": beta_trace,
            "actions": actions,
            "episode_rewards": episode_rewards,
            "training": train_meta,
            "summary": summary,
            "checkpoint": str(args.checkpoint) if args.checkpoint.exists() else None,
        }
        args.series.parent.mkdir(parents=True, exist_ok=True)
        args.series.write_text(json.dumps(cache, indent=2) + "\n")
        versioned_series = args.series.with_name(f"series_v{png_version}.json")
        versioned_series.write_text(json.dumps(cache, indent=2) + "\n")
        print(f"wrote series cache {args.series}", flush=True)
        print(f"wrote {versioned_series}", flush=True)
        print(
            f"gates: trend_down={summary['trend_down']} "
            f"ep0={summary['paper_gate_metrics'].get('ep0_mean', float('nan')):.3f} "
            f"early={summary['early_mean_0_130']:.3f} "
            f"late={summary['late_mean_150_end']:.3f} "
            f"unique_actions={train_meta['unique_actions']} "
            f"dominant={train_meta['dominant_action']} "
            f"gates_pass={summary['gates_pass']} "
            f"({elapsed:.0f}s)",
            flush=True,
        )

    print(f"output PNG: {args.out} (version={png_version})", flush=True)
    panel = plot_fig4a(cache, out_path=args.out)
    print(f"wrote {args.out}", flush=True)

    # Always recompute digitization gates from the trace (series cache may hold a stale summary).
    summary = _gate_summary(cache["beta_norm_trace"])
    gates = summary.get("gates", {})
    manifest = {
        "figure": "mehregan_fig4a",
        "seed": cache.get("seed", args.seed),
        "num_episodes": cache.get("num_episodes", args.episodes),
        "steps_per_episode": cache.get("steps_per_episode", STEPS_PER_EPISODE),
        "mean_hz": cache.get("mean_hz", MEAN_HZ),
        "state_length": cache.get("state_length", args.state_length),
        "plant_dt_ms": cache.get("plant_dt_ms", PAPER_DT_MS),
        "exploration_mode": cache.get("exploration_mode", "greedy"),
        "init_bias_scale": cache.get("init_bias_scale"),
        "jitter_fraction": cache.get("jitter_fraction"),
        "temperature_start": cache.get("temperature_start"),
        "temperature_end": cache.get("temperature_end"),
        "logit_noise_std": cache.get("logit_noise_std"),
        "entropy_coeff": cache.get("entropy_coeff", 0.01),
        "critic_warmup_steps": cache.get("critic_warmup_steps", 100),
        "actor_lr": cache.get("actor_lr", 5e-4),
        "critic_action_input": cache.get("critic_action_input", "logits"),
        "plant_integration_mode": cache.get(
            "plant_integration_mode", FIG4A_PLANT_INTEGRATION
        ),
        "elapsed_s": cache.get("elapsed_s"),
        "png_version": png_version,
        "summary": summary,
        "training": cache.get("training"),
        "panel": panel,
        "output_png": str(args.out),
        "series": str(args.series),
        "checkpoint": cache.get("checkpoint") or (
            str(args.checkpoint) if args.checkpoint.exists() else None
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {args.manifest}", flush=True)

    if args.update_docs:
        updated = _figure_promote.promote_4a(
            manifest=manifest,
            series_path=args.series,
            png_path=args.out,
            update_docs=True,
        )
        print(f"updated docs caption: {updated.get('caption')}", flush=True)
        print(f"updated docs image: {updated.get('png_repo_rel')}", flush=True)

    if args.update_checklist:
        update_checklist_in_doc(_checklist_rows(gates, summary))
        print("updated docs checklist", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
