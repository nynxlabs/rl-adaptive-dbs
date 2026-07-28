#!/usr/bin/env python3
"""Honest Fig 6a-style panel on burst n=256 (no forced non-QAT / QAT tricks).

Disables the Fig 6a promote knobs that exist only to make lines look different:
  - PTQ weight noise (σ=0.05) before quantize
  - weak QAT lock (0 plant eps + bias to open-loop action 17)

Uses the existing soft-fp32 n=256 checkpoint, trains **real** QAT (10 episodes),
then trailing-eval fp32 / PTQ fp16 / PTQ int8 / QAT with ``ptq_weight_noise=0``.

  tmux new-session -d -s fig6a-n256-honest \\
    \"setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
      scripts/figures/papers/mehregan/6a/plot_n256_6a_honest.py \\
      >> logs/fig6a-n256-honest.log 2>&1 < /dev/null\"
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

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    sys.path.insert(0, str(_REPO_ROOT))

from controllers.ddpg import load_actor
from controllers.ddpg.checkpoint import (
    load_checkpoint,
    qat_state_dict_from_checkpoint,
    save_checkpoint,
)
from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.quantization import (
    actor_state_dtype,
    prepare_actor_for_eval,
)
from controllers.ddpg.trainer import train_ddpg
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.pattern_alternatives import BurstPatternAlphabet
from envs.plant import DbsSpec, PlantConfig, PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

_FIG2A_PATH = Path(__file__).resolve().parents[1] / "2a" / "plot.py"
_fig2a_spec = importlib.util.spec_from_file_location("fig2a_plot", _FIG2A_PATH)
assert _fig2a_spec and _fig2a_spec.loader
_fig2a = importlib.util.module_from_spec(_fig2a_spec)
_fig2a_spec.loader.exec_module(_fig2a)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

MEAN_HZ = 45.0
N_PATTERNS = 256
PAPER_DT_MS = 0.02
SEED = 0
SKIP_REGULAR = True
TRAIN_STEP_S = 0.2
STEPS_PER_EPISODE = 30
QAT_NUM_EPISODES = 10  # paper §IV.A.3 — real QAT, not weak lock
PTQ_WEIGHT_NOISE = 0.0  # honest — no forced argmax flips
TRAILING_RL_STEP_S = 0.2
STIM_ONSET_S = 2.0
TIME_MAX_S = 12.0
TRAILING_STIM_STEPS = int((TIME_MAX_S - STIM_ONSET_S) / TRAILING_RL_STEP_S)

FIGURES_DIR = Path("figures/mehregan/images/6a")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/6a")
FP32_CHECKPOINT = CACHE_DIR / "checkpoint_burst_n256_skip_regular_02s.pt"
QAT_CHECKPOINT = CACHE_DIR / "qat_burst_n256_honest_02s.pt"
EVAL_JSON = CACHE_DIR / "eval_burst_n256_6a_honest.json"
OUT_STEM = "ptq_qat_45hz_n256_honest"

SERIES = {
    "fp32": {"label": "Fully Trained 45Hz", "color": "#2ca02c", "linestyle": "-"},
    "ptq-int8": {"label": "PTQ, INT8", "color": "#1f77b4", "linestyle": "-"},
    "ptq-fp16": {"label": "PTQ, FP16", "color": "#9467bd", "linestyle": "-"},
    "qat": {"label": "QAT", "color": "#ff7f0e", "linestyle": "--"},
}
VARIANT_KEYS = ("fp32", "ptq-int8", "ptq-fp16", "qat")


def _alphabet(*, step_s: float) -> BurstPatternAlphabet:
    return BurstPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=step_s,
        dt_ms=PAPER_DT_MS,
        n_patterns=N_PATTERNS,
        skip_regular=SKIP_REGULAR,
    )


def _make_train_env() -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=TRAIN_STEP_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=SKIP_REGULAR,
    )
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=_alphabet(step_s=TRAIN_STEP_S),
    )


def _integrate_idbs(
    *,
    duration_s: float,
    dt_ms: float,
    onset_sim_s: float,
    segment_actions: list[int],
    alphabet: BurstPatternAlphabet,
    rl_step_s: float,
) -> np.ndarray:
    n_steps = int(round(duration_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    onset_idx = int(round(onset_sim_s * 1000.0 / dt_ms))
    step_samples = int(round(rl_step_s * 1000.0 / dt_ms))
    for seg_i, action in enumerate(segment_actions):
        seg = np.asarray(alphabet.idbs_for_action(int(action)), dtype=np.float64)
        start = onset_idx + seg_i * step_samples
        end = min(start + seg.size, n_steps)
        if start >= n_steps:
            break
        idbs[start:end] = seg[: end - start]
    return idbs


def _variant_actions(
    checkpoint: Path,
    *,
    variant: str,
    seed: int,
) -> list[int]:
    env_cfg = MehreganEnvConfig()
    plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=PAPER_DT_MS))
    alphabet = _alphabet(step_s=TRAILING_RL_STEP_S)
    plant.reset(seed=seed)
    plant.integrate(_fig2a.DBS_ONSET_SIM, DbsSpec.none())

    actor, _ = load_actor(checkpoint)
    payload = load_checkpoint(checkpoint)
    qat_sd = qat_state_dict_from_checkpoint(payload) if variant == "qat" else None
    eval_variant = "paper" if variant == "fp32" else variant
    policy = prepare_actor_for_eval(
        actor,
        eval_variant,
        device="cpu",
        qat_state_dict=qat_sd,
    )
    policy.eval()
    dtype = actor_state_dtype(policy)
    obs_scale = env_cfg.observation_scale
    obs = np.zeros((1,), dtype=np.float32)
    actions: list[int] = []
    for _ in range(TRAILING_STIM_STEPS):
        with torch.no_grad():
            state_t = torch.as_tensor(obs, dtype=dtype).unsqueeze(0)
            logits = policy(state_t)
            if logits.dtype != torch.float32:
                logits = logits.float()
            action = int(logits.argmax(dim=-1).item())
        actions.append(action)
        result = plant.integrate(TRAILING_RL_STEP_S, alphabet.to_dbs_spec(action))
        if result.p_beta is None:
            raise RuntimeError("missing p_beta during fine action rollout")
        obs = np.array([result.p_beta / obs_scale], dtype=np.float32)
    plant.close()
    return actions


def _trailing_trace(
    plant: PythonPlant,
    *,
    seed: int,
    label: str,
    times: np.ndarray,
    actions: list[int],
    alphabet: BurstPatternAlphabet,
) -> np.ndarray:
    idbs = _integrate_idbs(
        duration_s=_fig2a.INTEGRATE_S,
        dt_ms=plant.config.dt_ms,
        onset_sim_s=_fig2a.DBS_ONSET_SIM,
        segment_actions=actions,
        alphabet=alphabet,
        rl_step_s=TRAILING_RL_STEP_S,
    )
    plant.config = PlantConfig(pd=1, dt_ms=plant.config.dt_ms)
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
        raise RuntimeError(f"no GPi spikes for {label}")
    return _fig2a.trailing_p_beta(
        result.gpi_spikes,
        dt_ms=result.dt_ms,
        times=times,
        window_s=_fig2a.WINDOW_S,
        label=label,
        verbose=True,
    )


def _train_honest_qat() -> Path:
    if QAT_CHECKPOINT.is_file():
        print(f"reusing QAT checkpoint {QAT_CHECKPOINT}", flush=True)
        return QAT_CHECKPOINT
    print(
        f"=== honest QAT train n={N_PATTERNS} skip_regular "
        f"eps={QAT_NUM_EPISODES} (NO weak lock) ===",
        flush=True,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env = _make_train_env()
    t0 = time.time()
    try:
        print(f"action space: {env.alphabet.n_actions} irregular", flush=True)
        cfg = replace(
            fig4a_ddpg_config(
                seed=SEED,
                num_episodes=QAT_NUM_EPISODES,
                max_episode_steps=STEPS_PER_EPISODE,
                pattern_mean_hz=MEAN_HZ,
                exploration_mode="epsilon",
                init_bias_scale=0.5,
                critic_action_input="one_hot",
            ),
            variant="qat",
        )
        result = train_ddpg(env, cfg)
        # Save as-trained — do NOT init_toward_action.
        save_checkpoint(
            QAT_CHECKPOINT,
            actor=result.actor,
            policy=result.policy,
            config=cfg,
            state_length=int(env.observation_space.shape[0]),
            n_actions=int(env.action_space.n),
            critic=result.critic,
        )
    finally:
        env.close()
    print(f"qat -> {QAT_CHECKPOINT} ({time.time() - t0:.0f}s)", flush=True)
    return QAT_CHECKPOINT


def _run_eval(*, fp32: Path, qat: Path) -> dict[str, Any]:
    times = _fig2a.sample_times(_fig2a.STEP_S, duration_s=_fig2a.DISPLAY_S)
    plant = PythonPlant(config=PlantConfig(pd=1, dt_ms=PAPER_DT_MS))
    alphabet = _alphabet(step_s=TRAILING_RL_STEP_S)
    variants = {
        "fp32": ("fp32", fp32),
        "ptq-fp16": ("ptq-fp16", fp32),
        "ptq-int8": ("ptq-int8", fp32),
        "qat": ("qat", qat),
    }
    payload: dict[str, Any] = {
        "figure": "mehregan_fig6a_n256_honest",
        "sampling": "trailing",
        "n_patterns": N_PATTERNS,
        "n_actions": alphabet.n_actions,
        "skip_regular": SKIP_REGULAR,
        "ptq_weight_noise": PTQ_WEIGHT_NOISE,
        "qat_weak_lock": False,
        "mean_hz": MEAN_HZ,
        "seed": SEED,
        "fp32_checkpoint": str(fp32),
        "qat_checkpoint": str(qat),
        "time_s": times.tolist(),
        "traces": {},
        "variants": {},
    }
    print(
        "Honest Fig 6a trailing eval — PTQ noise=0, no QAT weak lock, "
        f"n_actions={alphabet.n_actions}",
        flush=True,
    )
    try:
        for key, (variant, ckpt) in variants.items():
            print(f"eval {key} ({variant})...", flush=True)
            actions = _variant_actions(ckpt, variant=variant, seed=SEED)
            uniq = sorted(set(actions))
            print(f"  actions unique={uniq} (n_unique={len(uniq)})", flush=True)
            trace = _trailing_trace(
                plant,
                seed=SEED,
                label=key,
                times=times,
                actions=actions,
                alphabet=alphabet,
            )
            payload["traces"][key] = trace.tolist()
            payload["variants"][key] = {
                "variant_slug": variant,
                "actions": actions,
                "n_unique": len(uniq),
                "unique_actions": uniq,
            }
    finally:
        plant.close()

    # Diversity summary (honest)
    stim = {
        k: payload["variants"][k]["actions"]
        for k in VARIANT_KEYS
        if k in payload["variants"]
    }
    fp32_a = stim.get("fp32")
    shared_lock = bool(
        fp32_a
        and all(stim[k] == fp32_a for k in ("fp32", "ptq-fp16", "ptq-int8") if k in stim)
    )
    payload["action_diversity"] = {
        "shared_constant_non_qat_lock": shared_lock,
        "fp32_unique": payload["variants"].get("fp32", {}).get("unique_actions"),
        "ptq_fp16_unique": payload["variants"].get("ptq-fp16", {}).get("unique_actions"),
        "ptq_int8_unique": payload["variants"].get("ptq-int8", {}).get("unique_actions"),
        "qat_unique": payload["variants"].get("qat", {}).get("unique_actions"),
    }
    print("action_diversity:", json.dumps(payload["action_diversity"]), flush=True)
    EVAL_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {EVAL_JSON}", flush=True)
    return payload


def _plot(payload: dict[str, Any]) -> Path:
    times = np.asarray(payload["time_s"], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for key in VARIANT_KEYS:
        meta = SERIES[key]
        ax.plot(
            times,
            payload["traces"][key],
            color=meta["color"],
            linestyle=meta["linestyle"],
            linewidth=1.5,
            label=meta["label"],
        )
    ax.axvline(STIM_ONSET_S, color="#8b4513", linestyle="--", linewidth=1.2, zorder=0)
    ax.set_xlim(0.0, TIME_MAX_S)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("PSD")
    div = payload.get("action_diversity", {})
    ax.set_title(
        f"Fig 6a-style honest n={N_PATTERNS} (PTQ noise=0, no QAT lock)\n"
        f"non-QAT shared_lock={div.get('shared_constant_non_qat_lock')}  "
        f"fp32={div.get('fp32_unique')} fp16={div.get('ptq_fp16_unique')} "
        f"int8={div.get('ptq_int8_unique')} qat={div.get('qat_unique')}"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.6, alpha=0.9)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path, _version = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    vault = Path(
        "/home/nynxbox/Insync/knowledge-base/neuroengineering/"
        "brain-stimulation-engineering/effort/figures/mehregan/images/6a"
    )
    if vault.is_dir():
        dest = vault / Path(out_path).name
        dest.write_bytes(Path(out_path).read_bytes())
        main = Path("figures/mehregan/images/6a") / Path(out_path).name
        if Path("figures/mehregan/images/6a").is_dir() and not main.exists():
            try:
                main.symlink_to(dest)
            except OSError:
                pass
        print(f"vault copy: {dest}", flush=True)
    print(f"wrote {out_path}", flush=True)
    return Path(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--retrain-qat", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    if not FP32_CHECKPOINT.is_file():
        print(
            f"missing fp32 checkpoint {FP32_CHECKPOINT}\n"
            "Run scripts/figures/papers/mehregan/6a/plot_large_alphabet.py first.",
            file=sys.stderr,
        )
        return 2

    if args.plot_only and EVAL_JSON.is_file():
        payload = json.loads(EVAL_JSON.read_text())
    else:
        if args.retrain_qat and QAT_CHECKPOINT.is_file():
            QAT_CHECKPOINT.unlink()
        qat = (
            QAT_CHECKPOINT
            if args.skip_train and QAT_CHECKPOINT.is_file()
            else _train_honest_qat()
        )
        payload = _run_eval(fp32=FP32_CHECKPOINT, qat=qat)

    out = _plot(payload)
    print(f"done in {time.time() - t0:.0f}s → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
