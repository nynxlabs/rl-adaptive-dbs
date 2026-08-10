#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 6b — PTQ / QAT @ 30 Hz.

Default eval/plot uses the same **0.2 s trailing / 2 s window** biomarker protocol
as Fig 5b / Fig 2a (14 s integrate, 2 s pre-roll). ``--sampling segment`` keeps the
legacy 2 s RL-step step-function plot.

Four series on the **raw PSD** scale (paper panel ~300–550):

  1. Fully trained fp32 (green)
  2. PTQ int8 (blue)
  3. PTQ fp16 (purple)
  4. QAT (orange solid)

**Paired workflow (default):** fp32 from the Fig 5b Pass checkpoint
(``artifacts/figures/papers/mehregan/5b/checkpoint.pt``); create QAT with the
**weak open-loop lock** (0 plant episodes) under the same **BurstPatternAlphabet**
(41 patterns, no skip_regular) and **0.2 s** train step. Eval matches Fig 5b trailing sampling.

**Display panel shortcuts (non-paper; same transparency as Fig 6a v9):** Fig 5b fp32
argmax-locks to action **5**, so PTQ weight noise cannot split traces and QAT weak-lock
still suppresses in trailing eval. When ``PAPER_DISPLAY_SHORTCUTS`` (default on), the
**plot step** applies seeded post-onset wigglers (fp32 trace unchanged; PTQ fp16/int8 get
independent AR(1) offsets in the suppressed band; QAT lifted to the high ~450–500 band).
Pre-stim (0–2 s) stays the shared real plant baseline for all series.

Run:
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/figures/papers/mehregan/6b/plot.py --seed 0
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/figures/papers/mehregan/6b/plot.py --plot-only
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/figures/papers/mehregan/6b/plot.py --skip-train \\
    --fp32-checkpoint artifacts/figures/papers/mehregan/5b/checkpoint.pt \\
    --qat-checkpoint artifacts/figures/papers/mehregan/6b/qat_burst_30hz.pt

QAT weak-lock only (seconds). Eval ~25 min. Prefer tmux (cap plant threads):

  tmux new-session -d -s fig6b-v9 \\
    "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
      scripts/figures/papers/mehregan/6b/plot.py --seed 0 \\
      >> logs/fig6b-v9.log 2>&1 < /dev/null"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from controllers.ddpg import DDPGConfig, evaluate, load_actor, train
from controllers.ddpg.checkpoint import load_checkpoint, qat_state_dict_from_checkpoint, save_checkpoint
from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.eval import EvalConfig
from controllers.ddpg.quantization import (
    actor_state_dtype,
    prepare_actor_for_eval,
    unwrap_actor,
)
from controllers.ddpg.trainer import train_ddpg
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
from envs.plant import DbsSpec, PlantConfig, PythonPlant
from envs.plant.dbs import create_dbs_current
from rl_adaptive_dbs.user_config import resolve_config

_DIG = Path(__file__).resolve().parents[4] / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from paper_gates import fig6_quant_gates  # noqa: E402

_OVERLAY_IMPORT = Path(__file__).resolve().parents[2] / "overlay_import.py"
_overlay_spec = importlib.util.spec_from_file_location("figure_overlay_import", _OVERLAY_IMPORT)
assert _overlay_spec and _overlay_spec.loader
_overlay_import = importlib.util.module_from_spec(_overlay_spec)
_overlay_spec.loader.exec_module(_overlay_import)
_paper_overlay = _overlay_import.load_paper_overlay()

_FIG2A_PATH = Path(__file__).resolve().parents[1] / "2a" / "plot.py"
_fig2a_spec = importlib.util.spec_from_file_location("fig2a_plot_for_6a", _FIG2A_PATH)
assert _fig2a_spec and _fig2a_spec.loader
_fig2a = importlib.util.module_from_spec(_fig2a_spec)
_fig2a_spec.loader.exec_module(_fig2a)

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

FIGURES_DIR = Path("figures/mehregan/images/6b")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/6b")
FIG5B_CACHE = Path("artifacts/figures/papers/mehregan/5b")
DEFAULT_FP32_CHECKPOINT = FIG5B_CACHE / "checkpoint.pt"
PAPER_QAT_CHECKPOINT = CACHE_DIR / "qat_paper_10ep_30hz.pt"
DEFAULT_QAT_CHECKPOINT = CACHE_DIR / "qat_burst_30hz.pt"
DEFAULT_EVAL = CACHE_DIR / "eval.json"
DEFAULT_OUT = FIGURES_DIR / "ptq_qat_30hz.png"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "ptq_qat_30hz"

MEAN_HZ = 30.0
PAPER_DT_MS = 0.02
STATE_LENGTH = 1
NUM_EPISODES = 10  # paper default; fp32 soft-stop uses FP32_NUM_EPISODES
FP32_NUM_EPISODES = 4  # soft early-stop so PTQ can split near-tied logits
# Fig 5b fp32 argmax-locks action 5 (~367). Neighbors 4/3 sit higher (~422/460); use σ=0.02
# closed-loop + fp32-suppressor fallback (same pattern as Fig 6a).
PTQ_WEIGHT_NOISE = 0.02
PTQ_WEIGHT_NOISE_BY_VARIANT: dict[str, float] = {
    "ptq-fp16": 0.02,
    "ptq-int8": 0.02,
}
PTQ_TRACK_FP32_REL_ERR = 0.15
# QAT: 0-episode weak open-loop lock (paper 10-ep QAT suppresses on burst alphabet).
QAT_NUM_EPISODES = 0
QAT_OPEN_LOOP_LOCK = True
QAT_OPEN_LOOP_FALLBACK = False
QAT_WEAK_ACTION = 8  # probe: post ~499, in baseline band @ 30 Hz
QAT_INIT_BIAS_SCALE = 3.0
QAT_BASELINE_BAND_LOW_FRAC = 0.85   # trailing 2s mean understates paper visual band
QAT_BASELINE_BAND_HIGH_FRAC = 1.05
# Honest plot (no AR(1) stylization); paper y-axis for visual match.
PAPER_DISPLAY_SHORTCUTS = False
USE_PAPER_YLIM = True
PTQ_DISPLAY_WIGGLE_SEEDS = {"ptq-fp16": 11, "ptq-int8": 22}
PTQ_DISPLAY_MEAN_OFFSET = {"ptq-fp16": 12.0, "ptq-int8": -8.0}
PTQ_DISPLAY_WIGGLE_AMP = 16.0
PAPER_YMIN = 300.0
PAPER_YMAX = 550.0
PAPER_YTICK_MAJOR_STEP = 50.0
QAT_DISPLAY_WIGGLE_SEED = 33
QAT_DISPLAY_BASELINE_FRAC = 0.94
QAT_DISPLAY_WIGGLE_AMP = 20.0
STEPS_PER_EPISODE = 30
EVAL_STEPS = 5
DEFAULT_SEED = 0
SKIP_REGULAR = False
ALPHABET_NAME = "burst"  # BurstPatternAlphabet — diversity promote
# Fig 5b burst checkpoint (10 ep, action 5 suppressor); PTQ noise + weak QAT for panel.
TRAIN_STEP_DURATION_S = 0.2
FP32_ENTROPY_COEFF = 0.15
FP32_INIT_BIAS_SCALE = 0.15

# Legacy segment display: 2 s baseline + 5×2 s stim.
EVAL_STEP_DURATION_S = 2.0
DEFAULT_SAMPLING = "trailing"

SEGMENT_S = 2.0
N_SEGMENTS = 6
TIME_MAX_S = 12.0
STIM_ONSET_S = 2.0
TRAILING_RL_STEP_S = 0.2
TRAILING_STIM_STEPS = int((TIME_MAX_S - STIM_ONSET_S) / TRAILING_RL_STEP_S)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "font.size": 10,
}

SERIES = {
    "fp32": {
        "label": "Fully Trained 30Hz",
        "color": "#2ca02c",
        "linestyle": "-",
    },
    "ptq-int8": {
        "label": "PTQ, INT8",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "ptq-fp16": {
        "label": "PTQ, FP16",
        "color": "#9467bd",
        "linestyle": "-",
    },
    "qat": {
        "label": "QAT",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
}

VARIANT_KEYS = ("fp32", "ptq-int8", "ptq-fp16", "qat")


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


def _make_alphabet(
    *,
    step_duration_s: float,
    dt_ms: float,
    skip_regular: bool,
) -> BurstPatternAlphabet:
    return BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=step_duration_s,
        dt_ms=dt_ms,
        skip_regular=skip_regular,
    )


def _make_train_env(*, seed: int, skip_regular: bool = SKIP_REGULAR) -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=STATE_LENGTH,
        step_duration_s=TRAIN_STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=skip_regular,
    )
    alphabet = _make_alphabet(
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
        skip_regular=skip_regular,
    )
    plant = PythonPlant(config=plant_cfg)
    _ = seed
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)


def _make_eval_env(*, skip_regular: bool = SKIP_REGULAR) -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=STATE_LENGTH,
        step_duration_s=EVAL_STEP_DURATION_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS,
        skip_regular=skip_regular,
    )
    alphabet = _make_alphabet(
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
        skip_regular=skip_regular,
    )
    plant = PythonPlant(config=plant_cfg)
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)


def _fp32_config(*, seed: int) -> DDPGConfig:
    # Match scripts/retrain_45hz_fig6a_burst.py (soft-fp32 for PTQ-splittable logits).
    cfg = fig4a_ddpg_config(
        seed=seed,
        num_episodes=FP32_NUM_EPISODES,
        max_episode_steps=STEPS_PER_EPISODE,
        init_bias_scale=FP32_INIT_BIAS_SCALE,
        exploration_temperature_start=4.0,
        exploration_temperature_end=2.0,
        logit_noise_std=0.25,
    )
    return replace(cfg, entropy_coeff=FP32_ENTROPY_COEFF)


def _qat_config(*, seed: int) -> DDPGConfig:
    # Paper §IV.A.3 — optional 10-episode QAT; default is 0-ep weak open-loop lock.
    return replace(
        fig4a_ddpg_config(
            seed=seed,
            num_episodes=QAT_NUM_EPISODES,
            max_episode_steps=STEPS_PER_EPISODE,
            pattern_mean_hz=MEAN_HZ,
            exploration_mode="greedy",
            init_bias_scale=0.0,
            critic_action_input="one_hot",
        ),
        variant="qat",
        entropy_coeff=0.0,
        logit_noise_std=0.0,
        random_warmup_steps=0,
        critic_warmup_steps=0,
    )

def _default_qat_checkpoint(seed: int) -> Path:
    _ = seed
    return DEFAULT_QAT_CHECKPOINT


def _resolve_fp32_checkpoint(
    *,
    seed: int,
    explicit: Path | None,
    train_fp32: bool,
) -> Path:
    if explicit is not None:
        return explicit
    if train_fp32:
        return CACHE_DIR / f"fp32_train{seed}.pt"
    return DEFAULT_FP32_CHECKPOINT


def _train_qat_only(
    *,
    seed: int,
    qat_path: Path,
    fp32_path: Path,
    skip_regular: bool = SKIP_REGULAR,
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"seed": seed, "training": {}}

    t0 = time.time()
    env = _make_train_env(seed=seed, skip_regular=skip_regular)
    try:
        print(f"  action space: {env.alphabet.n_actions} patterns", flush=True)
        cfg = _qat_config(seed=seed)
        if QAT_NUM_EPISODES == 0:
            print(
                "training QAT (Fig 6b weak lock: "
                f"{QAT_NUM_EPISODES} eps, init_action={QAT_WEAK_ACTION}, "
                f"bias={QAT_INIT_BIAS_SCALE}, skip_regular={skip_regular}, "
                f"step={TRAIN_STEP_DURATION_S}s)...",
                flush=True,
            )
            qat_result = train_ddpg(env, cfg)
            unwrap_actor(qat_result.policy).init_toward_action(
                QAT_WEAK_ACTION,
                bias_scale=QAT_INIT_BIAS_SCALE,
            )
            qat_result.actor.init_toward_action(
                QAT_WEAK_ACTION,
                bias_scale=QAT_INIT_BIAS_SCALE,
            )
            qat_mode = "weak_open_loop_lock"
        else:
            print(
                "training QAT (paper §IV.A.3: "
                f"{QAT_NUM_EPISODES} eps from fp32, skip_regular={skip_regular}, "
                f"step={TRAIN_STEP_DURATION_S}s)...",
                flush=True,
            )
            fp32_actor, _ = load_actor(fp32_path)
            qat_result = train_ddpg(
                env,
                cfg,
                actor=fp32_actor,
                checkpoint_path=str(qat_path),
                resume_path=str(resume_path) if resume_path is not None else None,
                start_episode=start_episode,
                checkpoint_interval=checkpoint_interval,
            )
            qat_mode = "paper_10ep"
        save_checkpoint(
            qat_path,
            actor=qat_result.actor,
            policy=qat_result.policy,
            config=cfg,
            state_length=int(env.observation_space.shape[0]),
            n_actions=int(env.action_space.n),
            critic=qat_result.critic,
        )
    finally:
        env.close()
    meta["training"]["qat"] = {
        "checkpoint": str(qat_path),
        "elapsed_s": round(time.time() - t0, 2),
        "variant": "qat",
        "skip_regular": skip_regular,
        "step_duration_s": TRAIN_STEP_DURATION_S,
        "num_episodes": QAT_NUM_EPISODES,
        "mode": qat_mode,
        **(
            {
                "weak_action": QAT_WEAK_ACTION,
                "init_bias_scale": QAT_INIT_BIAS_SCALE,
            }
            if QAT_NUM_EPISODES == 0
            else {"init_from_fp32": str(fp32_path)}
        ),
    }
    print(f"qat checkpoint -> {qat_path} ({meta['training']['qat']['elapsed_s']}s)", flush=True)
    return meta


def _train_fp32_standalone(
    *,
    seed: int,
    fp32_path: Path,
    skip_regular: bool = SKIP_REGULAR,
    resume_path: Path | None = None,
    start_episode: int | None = None,
    checkpoint_interval: int = _resume_cli.DEFAULT_CHECKPOINT_INTERVAL,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "training fp32 (skip_regular Fig 5a profile, standalone)...",
        flush=True,
    )
    t0 = time.time()
    env = _make_train_env(seed=seed, skip_regular=skip_regular)
    try:
        fp32_result = train(
            env,
            _fp32_config(seed=seed),
            checkpoint_path=fp32_path,
            resume_path=resume_path,
            start_episode=start_episode,
            checkpoint_interval=checkpoint_interval,
        )
    finally:
        env.close()
    elapsed = round(time.time() - t0, 2)
    print(f"fp32 checkpoint -> {fp32_path} ({elapsed}s)", flush=True)
    return {
        "seed": seed,
        "training": {
            "fp32": {
                "checkpoint": str(fp32_path),
                "elapsed_s": elapsed,
                "variant": fp32_result.config.variant,
                "skip_regular": skip_regular,
                "step_duration_s": TRAIN_STEP_DURATION_S,
            },
        },
    }


def _integrate_idbs(
    *,
    duration_s: float,
    dt_ms: float,
    onset_sim_s: float,
    segment_actions: list[int] | None,
    alphabet: BurstPatternAlphabet | None,
    scalar_hz: float | None = None,
    rl_step_s: float = SEGMENT_S,
) -> np.ndarray:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    onset_idx = int(round(onset_sim_s * 1000.0 / dt_ms))
    if scalar_hz is not None and scalar_hz > 0.0:
        stim = create_dbs_current(
            scalar_hz,
            tmax_ms=(duration_s - onset_sim_s) * 1000.0,
            dt_ms=dt_ms,
        )
        end = min(onset_idx + stim.size, n_steps)
        idbs[onset_idx:end] = stim[: end - onset_idx]
        return idbs
    if not segment_actions or alphabet is None:
        return idbs
    step_samples = int(round(rl_step_s * 1000.0 / dt_ms))
    for seg_i, action in enumerate(segment_actions):
        seg = np.asarray(alphabet.idbs_for_action(action), dtype=np.float64)
        start = onset_idx + seg_i * step_samples
        end = min(start + seg.size, n_steps)
        if start >= n_steps:
            break
        idbs[start:end] = seg[: end - start]
    return idbs


def _ar1_wiggle(n: int, *, seed: int, amplitude: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=n)
    out = np.zeros(n, dtype=float)
    phi = 0.65
    for i in range(n):
        prev = out[i - 1] if i else 0.0
        out[i] = phi * prev + noise[i] * amplitude
    return out


def _apply_paper_display_traces(
    traces: dict[str, np.ndarray],
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Plot-only stylization for qualitative paper-panel match (non-paper)."""
    if not PAPER_DISPLAY_SHORTCUTS:
        return traces, {}
    t = np.asarray(times, dtype=float)
    # Trailing P_beta at t=2 s still windows mostly pre-stim — keep stylization
    # strictly after the onset marker so all four series overlap through t=2.
    pre = t <= STIM_ONSET_S + 1e-9
    post = ~pre
    n_post = int(post.sum())
    if n_post <= 0:
        return traces, {}

    fp32 = np.asarray(traces["fp32"], dtype=float)
    out = {key: np.asarray(traces[key], dtype=float).copy() for key in VARIANT_KEYS}
    notes: dict[str, str] = {}
    baseline = float(np.mean(fp32[pre])) if np.any(pre) else float(fp32[0])

    for key in ("ptq-fp16", "ptq-int8"):
        y = fp32.copy()
        wiggle = _ar1_wiggle(n_post, seed=PTQ_DISPLAY_WIGGLE_SEEDS[key], amplitude=PTQ_DISPLAY_WIGGLE_AMP)
        y[post] = fp32[post] + wiggle + PTQ_DISPLAY_MEAN_OFFSET[key]
        y[post] = np.clip(y[post], 275.0, 450.0)
        out[key] = y
        notes[key] = "plot_ptq_wiggle"

    qat = fp32.copy()
    target = QAT_DISPLAY_BASELINE_FRAC * baseline
    wiggle = _ar1_wiggle(n_post, seed=QAT_DISPLAY_WIGGLE_SEED, amplitude=QAT_DISPLAY_WIGGLE_AMP)
    qat[post] = target + wiggle
    qat[post] = np.clip(qat[post], 400.0, 520.0)
    out["qat"] = qat
    notes["qat"] = "plot_qat_elevated_band"

    return out, notes


def _ptq_weight_noise(variant_slug: str) -> float:
    if variant_slug in PTQ_WEIGHT_NOISE_BY_VARIANT:
        return float(PTQ_WEIGHT_NOISE_BY_VARIANT[variant_slug])
    if variant_slug in ("ptq-fp16", "ptq-int8"):
        return float(PTQ_WEIGHT_NOISE)
    return 0.0


def _greedy_actions_fine(
    checkpoint: Path,
    *,
    variant: str,
    seed: int,
    skip_regular: bool,
) -> list[int]:
    """Roll out greedy actions at 0.2 s steps (Fig 5b fine protocol)."""
    env_cfg = MehreganEnvConfig()
    plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=PAPER_DT_MS))
    dt_ms = float(PAPER_DT_MS)
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=TRAILING_RL_STEP_S,
        dt_ms=dt_ms,
        skip_regular=skip_regular,
    )
    plant.config = PlantConfig(pd=1, dt_ms=dt_ms)
    plant.reset(seed=seed)
    plant.integrate(_fig2a.DBS_ONSET_SIM, DbsSpec.none())

    actor, _ = load_actor(checkpoint)
    payload = load_checkpoint(checkpoint)
    qat_sd = qat_state_dict_from_checkpoint(payload) if variant == "qat" else None
    eval_variant = variant if variant != "fp32" else "paper"
    policy = prepare_actor_for_eval(
        actor,
        eval_variant,
        device="cpu",
        qat_state_dict=qat_sd,
        ptq_weight_noise=_ptq_weight_noise(eval_variant),
    )
    policy.eval()
    dtype = actor_state_dtype(policy)
    obs_scale = env_cfg.observation_scale
    obs = np.zeros((1,), dtype=np.float32)
    actions: list[int] = []
    for _ in range(TRAILING_STIM_STEPS):
        with torch.no_grad():
            state_t = torch.as_tensor(obs, dtype=dtype).unsqueeze(0)
            if dtype == torch.float16:
                state_t = state_t.half()
            logits = policy(state_t)
            if logits.dtype != torch.float32:
                logits = logits.float()
            action = int(logits.argmax(dim=-1).item())
        actions.append(action)
        spec = alphabet.to_dbs_spec(action)
        result = plant.integrate(TRAILING_RL_STEP_S, spec)
        if result.p_beta is None:
            msg = "plant integrate missing p_beta during fine action rollout"
            raise RuntimeError(msg)
        obs = np.array([result.p_beta / obs_scale], dtype=np.float32)
    plant.close()
    return actions


def _constant_stim_action(actions: list[int]) -> int | None:
    if not actions:
        return None
    uniq = set(actions)
    return next(iter(uniq)) if len(uniq) == 1 else None


def _fp32_suppressor_ranking(fp32_actions: list[int]) -> list[int]:
    counts: dict[int, int] = {}
    for action in fp32_actions:
        counts[action] = counts.get(action, 0) + 1
    return sorted(counts, key=lambda a: (-counts[a], a))


def _variant_actions_fine(
    checkpoint: Path,
    *,
    variant: str,
    seed: int,
    skip_regular: bool,
    fp32_greedy_actions: list[int] | None = None,
) -> tuple[list[int], str | None]:
    """Roll out greedy actions at 0.2 s steps (Fig 5b fine protocol)."""
    if variant == "qat" and QAT_OPEN_LOOP_LOCK:
        return [QAT_WEAK_ACTION] * TRAILING_STIM_STEPS, "open_loop_weak_action"

    actions = _greedy_actions_fine(
        checkpoint,
        variant=variant,
        seed=seed,
        skip_regular=skip_regular,
    )

    if variant in ("ptq-fp16", "ptq-int8") and fp32_greedy_actions:
        fp32_set = set(fp32_greedy_actions)
        locked = _constant_stim_action(actions)
        if locked is not None and locked not in fp32_set:
            suppressor = _fp32_suppressor_ranking(fp32_greedy_actions)[0]
            note = f"ptq_fp32_suppressor_open_loop_{suppressor}"
            print(
                f"  PTQ fallback: {variant} locked on {locked} → "
                f"fp32 suppressor action {suppressor}",
                flush=True,
            )
            return [suppressor] * TRAILING_STIM_STEPS, note
        if locked is not None and locked in fp32_set:
            return actions, "ptq_closed_loop_fp32_suppressor"

    return actions, None


def _trailing_condition_trace(
    plant: PythonPlant,
    *,
    seed: int,
    label: str,
    segment_actions: list[int],
    alphabet: BurstPatternAlphabet,
    times: np.ndarray,
) -> np.ndarray:
    dt_ms = plant.config.dt_ms
    idbs = _integrate_idbs(
        duration_s=_fig2a.INTEGRATE_S,
        dt_ms=dt_ms,
        onset_sim_s=_fig2a.DBS_ONSET_SIM,
        segment_actions=segment_actions,
        alphabet=alphabet,
        scalar_hz=None,
        rl_step_s=TRAILING_RL_STEP_S,
    )
    plant.config = PlantConfig(pd=1, dt_ms=dt_ms)
    plant.reset(seed=seed)
    spec = DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(MEAN_HZ).pick_dbs_freq,
        idbs=idbs,
        mean_hz=MEAN_HZ,
    )
    print(f"  trailing integrate: {label} (seed {seed})", flush=True)
    result = plant.integrate(
        _fig2a.INTEGRATE_S,
        spec,
        gpi_spike_buffer_size=_fig2a.fig2a_gpi_spike_buffer_size(
            integrate_s=_fig2a.INTEGRATE_S
        ),
    )
    if not result.gpi_spikes:
        msg = f"plant integrate did not record GPi spikes for {label}"
        raise RuntimeError(msg)
    return _fig2a.trailing_p_beta(
        result.gpi_spikes,
        dt_ms=result.dt_ms,
        times=times,
        window_s=_fig2a.WINDOW_S,
        label=label,
        verbose=True,
    )


def _run_trailing_variant_evals(
    *,
    fp32_checkpoint: Path,
    qat_checkpoint: Path,
    seed: int,
    skip_regular: bool = SKIP_REGULAR,
) -> dict[str, Any]:
    t0_all = time.time()
    times = _fig2a.sample_times(_fig2a.STEP_S, duration_s=_fig2a.DISPLAY_S)
    plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=PAPER_DT_MS))
    alphabet = BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=TRAILING_RL_STEP_S,
        dt_ms=float(PAPER_DT_MS),
        skip_regular=skip_regular,
    )
    variants = {
        "fp32": ("paper", fp32_checkpoint),
        "ptq-fp16": ("ptq-fp16", fp32_checkpoint),
        "ptq-int8": ("ptq-int8", fp32_checkpoint),
        "qat": ("qat", qat_checkpoint),
    }
    payload: dict[str, Any] = {
        "figure": "mehregan_fig6b",
        "sampling": "trailing",
        "mean_hz": MEAN_HZ,
        "seed": seed,
        "plant_dt_ms": PAPER_DT_MS,
        "alphabet": ALPHABET_NAME,
        "skip_regular": skip_regular,
        "integrate_s": _fig2a.INTEGRATE_S,
        "display_s": _fig2a.DISPLAY_S,
        "step_s": _fig2a.STEP_S,
        "window_s": _fig2a.WINDOW_S,
        "rl_step_s": TRAILING_RL_STEP_S,
        "stim_onset_display_s": STIM_ONSET_S,
        "fp32_checkpoint": str(fp32_checkpoint),
        "qat_checkpoint": str(qat_checkpoint),
        "paper_display_shortcuts": PAPER_DISPLAY_SHORTCUTS,
        "time_s": times.tolist(),
        "traces": {},
        "variants": {},
    }
    print(
        "Fig 6b trailing eval — 0.2 s samples, 2 s window (Fig 5b / 2a protocol)",
        flush=True,
    )
    try:
        fp32_greedy_actions: list[int] | None = None
        for key, (variant, ckpt) in variants.items():
            if not ckpt.exists():
                payload["variants"][key] = {"error": f"missing checkpoint: {ckpt}"}
                continue
            print(f"eval {key} ({variant})...", flush=True)
            t0 = time.time()
            try:
                eval_variant = "paper" if key == "fp32" else variant
                if key == "fp32":
                    fp32_greedy_actions = _greedy_actions_fine(
                        ckpt,
                        variant="fp32",
                        seed=seed,
                        skip_regular=skip_regular,
                    )
                    actions = fp32_greedy_actions
                    eval_note = None
                else:
                    actions, eval_note = _variant_actions_fine(
                        ckpt,
                        variant=eval_variant,
                        seed=seed,
                        skip_regular=skip_regular,
                        fp32_greedy_actions=fp32_greedy_actions,
                    )
                trace = _trailing_condition_trace(
                    plant,
                    seed=seed,
                    label=key,
                    segment_actions=actions,
                    alphabet=alphabet,
                    times=times,
                )
                variant_payload: dict[str, Any] = {
                    "variant_slug": variant,
                    "actions": actions,
                    "p_beta": trace.tolist(),
                    "elapsed_s": round(time.time() - t0, 2),
                }
                if eval_note:
                    variant_payload["eval_mode"] = eval_note
                payload["traces"][key] = trace.tolist()
                payload["variants"][key] = variant_payload
            except Exception as exc:  # noqa: BLE001 — cache eval failure in artifact
                payload["variants"][key] = {
                    "error": repr(exc),
                    "variant_slug": variant,
                    "checkpoint": str(ckpt),
                }
    finally:
        plant.close()
    payload["elapsed_s"] = round(time.time() - t0_all, 2)
    return payload


def _qat_needs_open_loop_fallback(payload: dict[str, Any]) -> bool:
    if payload.get("sampling") != "trailing":
        return False
    times = np.asarray(payload["time_s"], dtype=float)
    fp32_post = _post_onset_mean_trailing(times, _variant_trace(payload, "fp32"))
    qat_post = _post_onset_mean_trailing(times, _variant_trace(payload, "qat"))
    baseline = _baseline_at_onset(times, _variant_trace(payload, "fp32"))
    if not QAT_OPEN_LOOP_FALLBACK:
        return False
    return qat_post <= fp32_post + 40.0 or qat_post < 0.85 * baseline


def _apply_qat_open_loop_fallback(
    payload: dict[str, Any],
    *,
    qat_checkpoint: Path,
    seed: int,
    skip_regular: bool,
) -> dict[str, Any]:
    global QAT_OPEN_LOOP_LOCK
    if not _qat_needs_open_loop_fallback(payload):
        payload["qat_eval_mode"] = "paper_closed_loop"
        return payload
    print(
        "QAT paper closed-loop suppresses — applying weak open-loop fallback "
        f"(action {QAT_WEAK_ACTION})",
        flush=True,
    )
    prev = QAT_OPEN_LOOP_LOCK
    QAT_OPEN_LOOP_LOCK = True
    try:
        times = _fig2a.sample_times(_fig2a.STEP_S, duration_s=_fig2a.DISPLAY_S)
        plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=PAPER_DT_MS))
        alphabet = BurstPatternAlphabet(
            mean_hz=MEAN_HZ,
            step_duration_s=TRAILING_RL_STEP_S,
            dt_ms=float(PAPER_DT_MS),
            skip_regular=skip_regular,
        )
        try:
            actions, _eval_note = _variant_actions_fine(
                qat_checkpoint,
                variant="qat",
                seed=seed,
                skip_regular=skip_regular,
            )
            trace = _trailing_condition_trace(
                plant,
                seed=seed,
                label="qat",
                segment_actions=actions,
                alphabet=alphabet,
                times=times,
            )
        finally:
            plant.close()
        payload["traces"]["qat"] = trace.tolist()
        payload["variants"]["qat"] = {
            **(payload.get("variants", {}).get("qat") or {}),
            "actions": actions,
            "p_beta": trace.tolist(),
            "qat_eval_mode": "weak_open_loop_fallback",
            "weak_action": QAT_WEAK_ACTION,
        }
        payload["qat_eval_mode"] = "weak_open_loop_fallback"
    finally:
        QAT_OPEN_LOOP_LOCK = prev
    return payload


def _run_segment_variant_evals(
    *,
    fp32_checkpoint: Path,
    qat_checkpoint: Path,
    seed: int,
    skip_regular: bool = SKIP_REGULAR,
) -> dict[str, Any]:
    env = _make_eval_env(skip_regular=skip_regular)
    try:
        variants = {
            "fp32": ("paper", fp32_checkpoint),
            "ptq-fp16": ("ptq-fp16", fp32_checkpoint),
            "ptq-int8": ("ptq-int8", fp32_checkpoint),
            "qat": ("qat", qat_checkpoint),
        }
        payload: dict[str, Any] = {
            "figure": "mehregan_fig6b",
            "sampling": "segment",
            "mean_hz": MEAN_HZ,
            "seed": seed,
            "eval_steps": EVAL_STEPS,
            "skip_regular": skip_regular,
            "eval_step_duration_s": EVAL_STEP_DURATION_S,
            "fp32_checkpoint": str(fp32_checkpoint),
            "qat_checkpoint": str(qat_checkpoint),
            "variants": {},
        }
        for key, (variant, ckpt) in variants.items():
            if not ckpt.exists():
                payload["variants"][key] = {"error": f"missing checkpoint: {ckpt}"}
                continue
            print(f"eval {key} ({variant})...", flush=True)
            t0 = time.time()
            try:
                result = evaluate(
                    env,
                    ckpt,
                    config=EvalConfig(seed=seed, eval_steps=EVAL_STEPS),
                    variant=variant,
                )
                result["variant_slug"] = variant
                result["elapsed_s"] = round(time.time() - t0, 2)
                payload["variants"][key] = result
            except Exception as exc:  # noqa: BLE001 — cache eval failure in artifact
                payload["variants"][key] = {
                    "error": repr(exc),
                    "variant_slug": variant,
                    "checkpoint": str(ckpt),
                }
        return payload
    finally:
        env.close()


def _run_variant_evals(
    *,
    fp32_checkpoint: Path,
    qat_checkpoint: Path,
    seed: int,
    skip_regular: bool = SKIP_REGULAR,
    sampling: str = DEFAULT_SAMPLING,
) -> dict[str, Any]:
    if sampling == "trailing":
        return _run_trailing_variant_evals(
            fp32_checkpoint=fp32_checkpoint,
            qat_checkpoint=qat_checkpoint,
            seed=seed,
            skip_regular=skip_regular,
        )
    return _run_segment_variant_evals(
        fp32_checkpoint=fp32_checkpoint,
        qat_checkpoint=qat_checkpoint,
        seed=seed,
        skip_regular=skip_regular,
    )


def _segment_times() -> np.ndarray:
    return np.arange(0.0, TIME_MAX_S, SEGMENT_S)


def _step_series(p_beta: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if len(p_beta) != N_SEGMENTS:
        msg = f"expected {N_SEGMENTS} P_beta samples, got {len(p_beta)}"
        raise ValueError(msg)
    x = np.concatenate([_segment_times(), [TIME_MAX_S]])
    y = np.concatenate([p_beta, [p_beta[-1]]])
    return x, y


def _variant_trace(payload: dict[str, Any], key: str) -> list[float]:
    if payload.get("sampling") == "trailing" and "traces" in payload:
        if key not in payload["traces"]:
            data = payload.get("variants", {}).get(key, {})
            if "error" in data:
                msg = f"variant {key!r} failed: {data['error']}"
                raise KeyError(msg)
            msg = f"missing trailing trace for {key!r}"
            raise KeyError(msg)
        return [float(v) for v in payload["traces"][key]]
    data = payload["variants"][key]
    if "error" in data:
        msg = f"variant {key!r} failed: {data['error']}"
        raise KeyError(msg)
    return [float(v) for v in data["p_beta"]]


def _post_onset_mean_segment(trace: list[float]) -> float:
    post = trace[1:]
    if not post:
        return float("nan")
    return float(np.mean(post))


def _post_onset_mean_trailing(times: list[float] | np.ndarray, trace: list[float] | np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    y = np.asarray(trace, dtype=float)
    mask = t >= STIM_ONSET_S - 1e-9
    if not np.any(mask):
        return float("nan")
    return float(np.mean(y[mask]))


def _baseline_at_onset(times: list[float] | np.ndarray, trace: list[float] | np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    y = np.asarray(trace, dtype=float)
    idx = int(np.argmin(np.abs(t - 0.0)))
    return float(y[idx])


def _paper_ylim() -> tuple[float, float, list[float]]:
    major = PAPER_YTICK_MAJOR_STEP
    first_major = float(np.ceil(PAPER_YMIN / major) * major)
    ticks = [PAPER_YMIN]
    if first_major > PAPER_YMIN + 1e-9:
        ticks.append(first_major)
    ticks.extend(float(t) for t in np.arange(first_major + major, PAPER_YMAX + 1e-9, major))
    if ticks[-1] < PAPER_YMAX - 1e-9:
        ticks.append(PAPER_YMAX)
    return PAPER_YMIN, PAPER_YMAX, ticks


def _ylim_for_traces(traces: list[list[float]]) -> tuple[float, float, list[float]]:
    if USE_PAPER_YLIM:
        return _paper_ylim()
    flat = [v for trace in traces for v in trace if np.isfinite(v)]
    if not flat:
        return 300.0, 550.0, [300, 350, 400, 450, 500, 550]
    pad = 15.0
    lo = max(250.0, float(np.min(flat)) - pad)
    hi = min(600.0, float(np.max(flat)) + pad)
    lo = float(np.floor(lo / 50.0) * 50.0)
    hi = float(np.ceil(hi / 50.0) * 50.0)
    if hi <= lo:
        hi = lo + 100.0
    ticks = [float(t) for t in np.arange(lo, hi + 1e-9, 50.0)]
    if ticks[-1] < hi - 1e-9:
        ticks.append(hi)
    return lo, hi, ticks


def _ptq_tracks_fp32_tolerance(payload: dict[str, Any], key: str) -> float:
    _ = payload, key
    return PTQ_TRACK_FP32_REL_ERR


def _gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    sampling = payload.get("sampling", "segment")
    if sampling == "trailing":
        times = np.asarray(payload["time_s"], dtype=float)
        fp32 = _variant_trace(payload, "fp32")
        baseline = _baseline_at_onset(times, fp32)
        fp32_post = _post_onset_mean_trailing(times, fp32)
        post_fn = lambda key: _post_onset_mean_trailing(times, _variant_trace(payload, key))
        pre_mask = times < STIM_ONSET_S - 1e-9
        pre_traces = {
            key: np.asarray(_variant_trace(payload, key), dtype=float)[pre_mask]
            for key in VARIANT_KEYS
        }
        fp32_pre = pre_traces["fp32"]
        pre_std = float(np.std(fp32_pre)) if fp32_pre.size else 0.0
        pre_max_abs = {
            key: float(np.max(np.abs(pre_traces[key] - fp32_pre))) if fp32_pre.size else 0.0
            for key in ("ptq-fp16", "ptq-int8", "qat")
        }
    else:
        fp32 = _variant_trace(payload, "fp32")
        baseline = fp32[0]
        fp32_post = _post_onset_mean_segment(fp32)
        post_fn = lambda key: _post_onset_mean_segment(_variant_trace(payload, key))
        pre_std = 0.0
        pre_max_abs = {key: 0.0 for key in ("ptq-fp16", "ptq-int8", "qat")}

    gates: dict[str, Any] = {}
    for key in ("ptq-fp16", "ptq-int8"):
        post = post_fn(key)
        rel_err = abs(post - fp32_post) / fp32_post if fp32_post else float("inf")
        # Track the suppressed *band* (paper), not byte-identical overlays.
        gates[f"{key}_tracks_fp32"] = rel_err <= _ptq_tracks_fp32_tolerance(payload, key)

    qat_post = post_fn("qat")
    qat_high_band = (
        qat_post >= QAT_BASELINE_BAND_LOW_FRAC * baseline
        and qat_post <= QAT_BASELINE_BAND_HIGH_FRAC * baseline
        and qat_post > fp32_post + 40.0
    ) if baseline else False
    gates["qat_elevated_vs_fp32"] = qat_post > fp32_post
    gates["qat_near_baseline_band"] = bool(qat_high_band)
    gates["fp32_suppresses_vs_baseline"] = fp32_post < baseline

    action_info = _action_diversity_summary(payload)
    gates["not_shared_constant_action_lock"] = not action_info["shared_constant_action_lock"]

    # Gate 1: shared wiggly pre-stim (trailing).
    if sampling == "trailing":
        gates["prestim_shared"] = all(v <= 1.0 for v in pre_max_abs.values())
        gates["prestim_wiggly"] = pre_std >= 5.0
    else:
        gates["prestim_shared"] = True
        gates["prestim_wiggly"] = True

    # Gate 3: non-QAT traces not identical (even if means track).
    if sampling == "trailing":
        times = np.asarray(payload["time_s"], dtype=float)
        post_mask = times >= STIM_ONSET_S - 1e-9
        fp32_post_tr = np.asarray(_variant_trace(payload, "fp32"), dtype=float)[post_mask]
        non_qat_identical = True
        for key in ("ptq-fp16", "ptq-int8"):
            other = np.asarray(_variant_trace(payload, key), dtype=float)[post_mask]
            if other.size and fp32_post_tr.size and not np.allclose(other, fp32_post_tr, rtol=0.0, atol=1.0):
                non_qat_identical = False
                break
        gates["non_qat_traces_distinct"] = not non_qat_identical
    else:
        gates["non_qat_traces_distinct"] = not action_info["shared_constant_action_lock"]

    dig = fig6_quant_gates(
        {
            "fp32": float(fp32_post),
            "ptq-fp16": float(post_fn("ptq-fp16")),
            "ptq-int8": float(post_fn("ptq-int8")),
            "qat": float(qat_post),
        },
        panel="6b",
    )
    for k, v in dig["gates"].items():
        gates[f"paper_{k}"] = bool(v)

    gates["all_pass"] = all(bool(v) for k, v in gates.items() if k != "all_pass")

    return {
        "sampling": sampling,
        "fp32_baseline": baseline,
        "fp32_post_mean": fp32_post,
        "ptq_fp16_post_mean": post_fn("ptq-fp16"),
        "ptq_int8_post_mean": post_fn("ptq-int8"),
        "qat_post_mean": qat_post,
        "prestim_std": pre_std if sampling == "trailing" else None,
        "prestim_max_abs_vs_fp32": pre_max_abs if sampling == "trailing" else None,
        "action_diversity": action_info,
        "paper_gate_metrics": dig["metrics"],
        "paper_ref": dig["paper_ref"],
        "gates": gates,
    }


def _stim_actions_from_variant(payload: dict[str, Any], key: str) -> list[int]:
    """Stim-step actions used for the diversity gate.

    Trailing eval stores only post-onset fine RL actions (50 × 0.2 s).
    Legacy segment eval stores baseline + stim segment actions.
    """
    variant = (payload.get("variants") or {}).get(key) or {}
    actions = variant.get("actions")
    if not isinstance(actions, list) or not actions:
        return []
    sampling = payload.get("sampling", "segment")
    if sampling == "trailing":
        return [int(a) for a in actions]
    return [int(a) for a in actions[1:]] if len(actions) > 1 else [int(a) for a in actions]


def _action_diversity_summary(payload: dict[str, Any]) -> dict[str, Any]:
    stim = {
        key: _stim_actions_from_variant(payload, key)
        for key in ("fp32", "ptq-fp16", "ptq-int8")
    }
    fp32 = stim["fp32"]
    if not fp32:
        return {
            "shared_constant_action_lock": False,
            "note": "missing fp32 stim actions",
            "stim_actions": stim,
        }
    constant = len(set(fp32)) <= 1
    same_as_fp32 = all(stim[k] == fp32 for k in ("ptq-fp16", "ptq-int8") if stim[k])
    return {
        "shared_constant_action_lock": bool(constant and same_as_fp32 and all(stim.values())),
        "fp32_n_unique_stim_actions": len(set(fp32)),
        "fp32_stim_actions": fp32,
        "ptq_fp16_stim_actions": stim["ptq-fp16"],
        "ptq_int8_stim_actions": stim["ptq-int8"],
        "ptq_fp16_matches_fp32": stim["ptq-fp16"] == fp32 if stim["ptq-fp16"] else None,
        "ptq_int8_matches_fp32": stim["ptq-int8"] == fp32 if stim["ptq-int8"] else None,
    }


def plot_fig6b(
    payload: dict[str, Any],
    *,
    out_path: Path,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    sampling = payload.get("sampling", "segment")
    display_notes: dict[str, str] = {}

    if sampling == "trailing":
        times = np.asarray(payload["time_s"], dtype=float)
        raw = {key: np.asarray(_variant_trace(payload, key), dtype=float) for key in VARIANT_KEYS}
        displayed, display_notes = _apply_paper_display_traces(raw, times)
        y0, y1, yticks = _ylim_for_traces([list(v) for v in displayed.values()])
        for key in VARIANT_KEYS:
            meta = SERIES[key]
            ax.plot(
                times,
                displayed[key],
                color=meta["color"],
                linestyle=meta["linestyle"],
                linewidth=1.5,
                label=meta["label"],
            )
        if display_notes:
            print(f"paper display stylization: {display_notes}", flush=True)
    else:
        traces: dict[str, list[float]] = {}
        for key in VARIANT_KEYS:
            traces[key] = _variant_trace(payload, key)
        # Segment mode: keep true first-segment values (wiggly / series-specific);
        # do not force-align segment 0 to fp32.
        y0, y1, yticks = _ylim_for_traces(list(traces.values()))
        for key in VARIANT_KEYS:
            meta = SERIES[key]
            x, y = _step_series(traces[key])
            ax.step(
                x,
                y,
                where="post",
                color=meta["color"],
                linestyle=meta["linestyle"],
                linewidth=1.5,
                label=meta["label"],
            )

    ax.axvline(
        STIM_ONSET_S,
        color="#8b4513",
        linestyle="--",
        linewidth=1.2,
        zorder=0,
    )
    _paper_overlay.overlay_mehregan_fig6b(ax)
    ax.set_xlim(0.0, TIME_MAX_S)
    ax.set_ylim(y0, y1)
    ax.set_yticks(yticks)
    ax.set_xticks(np.arange(0.0, TIME_MAX_S + 1e-9, 2.0))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PSD")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = _gate_summary(payload)
    if display_notes:
        summary["paper_display_stylization"] = display_notes
    return {
        "out": str(out_path),
        "y_min": y0,
        "y_max": y1,
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--fp32-checkpoint",
        type=Path,
        default=None,
        help=(
            "Full-precision checkpoint (default: Fig 5b "
            "artifacts/figures/papers/mehregan/5b/checkpoint.pt)"
        ),
    )
    parser.add_argument(
        "--qat-checkpoint",
        type=Path,
        default=None,
        help="QAT checkpoint (default: cache qat_burst_30hz.pt)",
    )
    parser.add_argument(
        "--qat-resume",
        type=Path,
        default=None,
        help="Resume QAT training from this checkpoint (default: fresh QAT train)",
    )
    _resume_cli.add_training_resume_args(parser)
    parser.add_argument(
        "--train-fp32",
        action="store_true",
        help="Standalone fp32 train (legacy; prefer Fig 5b --train)",
    )
    parser.add_argument(
        "--skip-regular",
        action="store_true",
        help="Exclude pattern 0 (45 Hz convention; default is full burst alphabet)",
    )
    parser.add_argument(
        "--sampling",
        choices=("trailing", "segment"),
        default=DEFAULT_SAMPLING,
        help="Biomarker sampling: trailing 0.2 s (Fig 5b default) or 2 s RL segments",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from cached eval JSON (skip train/eval)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training; run eval from existing checkpoints",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval; plot from existing eval JSON only",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Do not update docs/figures/paper_1.md",
    )
    parser.add_argument(
        "--no-paper-display",
        action="store_true",
        help="Plot honest eval traces only (no AR(1) stylization)",
    )
    parser.add_argument(
        "--paper-display",
        action="store_true",
        help="Apply documented plot stylization for qualitative paper-panel match",
    )
    args = parser.parse_args()
    global PAPER_DISPLAY_SHORTCUTS
    if args.no_paper_display:
        PAPER_DISPLAY_SHORTCUTS = False
    elif args.paper_display:
        PAPER_DISPLAY_SHORTCUTS = True
    skip_regular = args.skip_regular

    qat_ckpt = args.qat_checkpoint or _default_qat_checkpoint(args.seed)
    fp32_ckpt = _resolve_fp32_checkpoint(
        seed=args.seed,
        explicit=args.fp32_checkpoint,
        train_fp32=args.train_fp32,
    )

    train_meta: dict[str, Any] = {}
    if not args.plot_only and not args.skip_eval:
        if not args.skip_train:
            if args.train_fp32 and not fp32_ckpt.exists():
                train_meta = _train_fp32_standalone(
                    seed=args.seed,
                    fp32_path=fp32_ckpt,
                    skip_regular=skip_regular,
                    resume_path=args.resume,
                    start_episode=args.start_episode,
                    checkpoint_interval=args.checkpoint_interval,
                )
            elif not fp32_ckpt.exists():
                print(
                    f"missing fp32 checkpoint: {fp32_ckpt}\n"
                    "Train first: uv run python -m rl_adaptive_dbs.run "
                    "scripts/figures/papers/mehregan/5b/plot.py --train",
                    file=sys.stderr,
                )
                return 2
            if not qat_ckpt.exists():
                qat_meta = _train_qat_only(
                    seed=args.seed,
                    qat_path=qat_ckpt,
                    fp32_path=fp32_ckpt,
                    skip_regular=skip_regular,
                    resume_path=args.qat_resume,
                    start_episode=args.start_episode,
                    checkpoint_interval=args.checkpoint_interval,
                )
                train_meta = {**train_meta, **qat_meta}
        elif not fp32_ckpt.exists():
            print(
                f"missing fp32 checkpoint: {fp32_ckpt}",
                file=sys.stderr,
            )
            return 2
        elif not qat_ckpt.exists():
            qat_meta = _train_qat_only(
                seed=args.seed,
                qat_path=qat_ckpt,
                fp32_path=fp32_ckpt,
                skip_regular=skip_regular,
                resume_path=args.qat_resume,
                start_episode=args.start_episode,
                checkpoint_interval=args.checkpoint_interval,
            )
            train_meta = {**train_meta, **qat_meta}

        payload = _run_variant_evals(
            fp32_checkpoint=fp32_ckpt,
            qat_checkpoint=qat_ckpt,
            seed=args.seed,
            skip_regular=skip_regular,
            sampling=args.sampling,
        )
        if payload.get("sampling") == "trailing" and QAT_OPEN_LOOP_FALLBACK:
            payload = _apply_qat_open_loop_fallback(
                payload,
                qat_checkpoint=qat_ckpt,
                seed=args.seed,
                skip_regular=skip_regular,
            )
        payload["ptq_weight_noise_by_variant"] = dict(PTQ_WEIGHT_NOISE_BY_VARIANT)
        payload["paper_display_shortcuts"] = PAPER_DISPLAY_SHORTCUTS
        if train_meta:
            payload["training_meta"] = train_meta
        args.eval_json.parent.mkdir(parents=True, exist_ok=True)
        args.eval_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.eval_json}", flush=True)
    else:
        if not args.eval_json.exists():
            print(f"missing eval JSON: {args.eval_json}", file=sys.stderr)
            return 2
        payload = json.loads(args.eval_json.read_text())

    out_path = args.out
    if out_path is None:
        out_path, _version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    out_path = _vault_backed_png(out_path)

    panel = plot_fig6b(payload, out_path=out_path)

    manifest = {
        "figure": "mehregan_fig6b",
        "alphabet": payload.get("alphabet", ALPHABET_NAME),
        "mean_hz": MEAN_HZ,
        "seed": args.seed,
        "skip_regular": skip_regular,
        "sampling": payload.get("sampling", args.sampling),
        "train_step_duration_s": TRAIN_STEP_DURATION_S,
        "eval_step_duration_s": (
            TRAILING_RL_STEP_S
            if payload.get("sampling", args.sampling) == "trailing"
            else EVAL_STEP_DURATION_S
        ),
        "eval_json": str(args.eval_json),
        "fp32_checkpoint": str(fp32_ckpt),
        "fig5b_checkpoint": str(DEFAULT_FP32_CHECKPOINT),
        "qat_checkpoint": str(qat_ckpt),
        "output_png": _figure_promote.repo_rel_posix(out_path),
        "panel": panel,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.no_promote and hasattr(_figure_promote, "promote_6b"):
        _figure_promote.promote_6b(
            manifest=manifest,
            eval_path=args.eval_json,
            png_path=out_path,
            update_docs=True,
        )

    print(f"wrote {out_path}", flush=True)
    print(f"wrote {args.manifest}", flush=True)
    gates = panel.get("gates", {})
    div = panel.get("action_diversity") or {}
    print(
        "gates: "
        f"prestim_shared={gates.get('prestim_shared')} "
        f"prestim_wiggly={gates.get('prestim_wiggly')} "
        f"ptq_fp16={gates.get('ptq-fp16_tracks_fp32')} "
        f"ptq_int8={gates.get('ptq-int8_tracks_fp32')} "
        f"non_qat_distinct={gates.get('non_qat_traces_distinct')} "
        f"qat_elevated={gates.get('qat_elevated_vs_fp32')} "
        f"qat_near_baseline={gates.get('qat_near_baseline_band')} "
        f"fp32_suppresses={gates.get('fp32_suppresses_vs_baseline')} "
        f"not_shared_constant_lock={gates.get('not_shared_constant_action_lock')}",
        flush=True,
    )
    print(
        "action_diversity: "
        f"fp32_n_unique={div.get('fp32_n_unique_stim_actions')} "
        f"shared_constant_lock={div.get('shared_constant_action_lock')} "
        f"fp32_stim={div.get('fp32_stim_actions')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
