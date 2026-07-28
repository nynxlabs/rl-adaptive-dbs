#!/usr/bin/env python3
"""Mehregan Fig 6a — fully honest PTQ / QAT @ 45 Hz.

No PTQ weight noise, no weak-QAT lock, real 10-episode QAT. Trailing 0.2 s /
2 s window eval (Fig 2a protocol). fp32 from continuous skip_regular retrain.

Run:
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/retrain_45hz_fig6a_honest.py
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/figures/papers/mehregan/6a/plot_honest.py
  uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
    scripts/figures/papers/mehregan/6a/plot_honest.py --plot-only

tmux:
  tmux new-session -d -s fig6a-honest \\
    "setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
      scripts/figures/papers/mehregan/6a/plot_honest.py \\
      >> logs/fig6a-honest.log 2>&1 < /dev/null"
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

from controllers.ddpg import load_actor, save_checkpoint, train
from controllers.ddpg.checkpoint import load_checkpoint, qat_state_dict_from_checkpoint
from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.quantization import actor_state_dtype, prepare_actor_for_eval
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant import DbsSpec, PlantConfig, PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

_FIG2A_PATH = Path(__file__).resolve().parents[1] / "2a" / "plot.py"
_fig2a_spec = importlib.util.spec_from_file_location("fig2a_plot", _FIG2A_PATH)
assert _fig2a_spec and _fig2a_spec.loader
_fig2a = importlib.util.module_from_spec(_fig2a_spec)
_fig2a_spec.loader.exec_module(_fig2a)

_PLOT6A_PATH = Path(__file__).resolve().parent / "plot.py"
_plot6a_spec = importlib.util.spec_from_file_location("fig6a_plot", _PLOT6A_PATH)
assert _plot6a_spec and _plot6a_spec.loader
_plot6a = importlib.util.module_from_spec(_plot6a_spec)
_plot6a_spec.loader.exec_module(_plot6a)

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

MEAN_HZ = 45.0
PAPER_DT_MS = 0.02
SEED = 0
SKIP_REGULAR = True
TRAIN_STEP_S = 0.2
STEPS_PER_EPISODE = 30
QAT_NUM_EPISODES = 10
STATE_LENGTH = 15
TRAILING_RL_STEP_S = 0.2
STIM_ONSET_S = 2.0
TIME_MAX_S = 12.0
TRAILING_STIM_STEPS = int((TIME_MAX_S - STIM_ONSET_S) / TRAILING_RL_STEP_S)

FIGURES_DIR = Path("figures/mehregan/images/6a")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/6a")
FP32_CHECKPOINT = CACHE_DIR / "checkpoint_honest_continuous_02s.pt"
QAT_CHECKPOINT = CACHE_DIR / "qat_honest_continuous_02s.pt"
EVAL_JSON = CACHE_DIR / "eval_honest.json"
MANIFEST_JSON = CACHE_DIR / "manifest_honest.json"
OUT_STEM = "ptq_qat_45hz_honest"

VARIANT_KEYS = ("fp32", "ptq-int8", "ptq-fp16", "qat")


def _alphabet(*, step_s: float) -> FixedMeanPatternAlphabet:
    return FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=step_s,
        dt_ms=PAPER_DT_MS,
        skip_regular=SKIP_REGULAR,
    )


def _make_train_env() -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=STATE_LENGTH,
        step_duration_s=TRAIN_STEP_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=SKIP_REGULAR,
        plant_integration_mode="continuous",
        pre_stim_duration_s=2.0,
    )
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=_alphabet(step_s=TRAIN_STEP_S),
    )


def _fp32_config() -> Any:
    return replace(
        fig4a_ddpg_config(
            seed=SEED,
            num_episodes=10,
            max_episode_steps=STEPS_PER_EPISODE,
            pattern_mean_hz=MEAN_HZ,
            exploration_mode="softmax",
            init_bias_scale=0.5,
            exploration_temperature_start=3.0,
            exploration_temperature_end=1.0,
            logit_noise_std=0.15,
        ),
        entropy_coeff=0.05,
        obs_normalize=True,
        random_warmup_steps=150,
    )


def _qat_config() -> Any:
    return replace(_fp32_config(), variant="qat")


def _integrate_idbs(
    *,
    duration_s: float,
    dt_ms: float,
    onset_sim_s: float,
    segment_actions: list[int],
    alphabet: FixedMeanPatternAlphabet,
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


def _variant_actions(checkpoint: Path, *, variant: str, seed: int) -> list[int]:
    env_cfg = MehreganEnvConfig(state_length=STATE_LENGTH)
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
    obs_scale = env_cfg.observation_scale
    obs = np.zeros((STATE_LENGTH,), dtype=np.float32)
    actions: list[int] = []
    for _ in range(TRAILING_STIM_STEPS):
        with torch.no_grad():
            state_t = torch.as_tensor(obs, dtype=actor_state_dtype(policy)).unsqueeze(0)
            logits = policy(state_t).float()
            actions.append(int(logits.argmax(dim=-1).item()))
        result = plant.integrate(TRAILING_RL_STEP_S, alphabet.to_dbs_spec(actions[-1]))
        if result.p_beta is None:
            raise RuntimeError("missing p_beta during rollout")
        norm = float(result.p_beta / obs_scale)
        obs = np.roll(obs, -1)
        obs[-1] = norm
    plant.close()
    return actions


def _trailing_trace(
    plant: PythonPlant,
    *,
    seed: int,
    label: str,
    times: np.ndarray,
    actions: list[int],
    alphabet: FixedMeanPatternAlphabet,
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


def _train_fp32() -> Path:
    if FP32_CHECKPOINT.is_file():
        print(f"reusing fp32 {FP32_CHECKPOINT}", flush=True)
        return FP32_CHECKPOINT
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=== honest fp32 train (continuous plant, skip_regular) ===", flush=True)
    env = _make_train_env()
    t0 = time.time()
    try:
        train(env, _fp32_config(), checkpoint_path=FP32_CHECKPOINT)
    finally:
        env.close()
    print(f"fp32 -> {FP32_CHECKPOINT} ({time.time() - t0:.0f}s)", flush=True)
    return FP32_CHECKPOINT


def _train_qat() -> Path:
    if QAT_CHECKPOINT.is_file():
        print(f"reusing QAT {QAT_CHECKPOINT}", flush=True)
        return QAT_CHECKPOINT
    if not FP32_CHECKPOINT.is_file():
        _train_fp32()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== honest QAT train ({QAT_NUM_EPISODES} ep, no weak lock) ===", flush=True)
    env = _make_train_env()
    t0 = time.time()
    try:
        train(env, _qat_config(), checkpoint_path=QAT_CHECKPOINT)
    finally:
        env.close()
    print(f"qat -> {QAT_CHECKPOINT} ({time.time() - t0:.0f}s)", flush=True)
    return QAT_CHECKPOINT


def _post_onset_mean(times: list[float], trace: list[float]) -> float:
    t = np.asarray(times, dtype=float)
    y = np.asarray(trace, dtype=float)
    mask = t >= STIM_ONSET_S - 1e-9
    return float(np.mean(y[mask])) if np.any(mask) else float("nan")


def _honest_gates(payload: dict[str, Any]) -> dict[str, Any]:
    traces = payload["traces"]
    times = payload["time_s"]
    variants = payload["variants"]
    fp32_post = _post_onset_mean(times, traces["fp32"])
    qat_post = _post_onset_mean(times, traces["qat"])
    fp32_actions = variants["fp32"]["actions"]
    shared_lock = all(
        variants[k]["actions"] == fp32_actions for k in ("fp32", "ptq-fp16", "ptq-int8")
    )
    trace_equal = (
        traces["fp32"] == traces["ptq-fp16"] or traces["fp32"] == traces["ptq-int8"]
    )
    gates = {
        "fp32_suppresses_vs_baseline": fp32_post < traces["fp32"][0],
        "ptq_fp16_tracks_fp32": abs(_post_onset_mean(times, traces["ptq-fp16"]) - fp32_post)
        / fp32_post
        <= 0.15,
        "ptq_int8_tracks_fp32": abs(_post_onset_mean(times, traces["ptq-int8"]) - fp32_post)
        / fp32_post
        <= 0.15,
        "not_shared_constant_action_lock": not shared_lock,
        "non_qat_traces_distinct": traces["fp32"] != traces["ptq-fp16"]
        or traces["fp32"] != traces["ptq-int8"],
        "qat_elevated_vs_fp32": qat_post > fp32_post,
        "qat_near_baseline_band": qat_post >= 420.0,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "fp32_baseline": traces["fp32"][0],
        "fp32_post_mean": fp32_post,
        "qat_post_mean": qat_post,
        "shared_constant_non_qat_lock": shared_lock,
        "non_qat_identical_traces": trace_equal,
        "gates": gates,
    }


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
        "figure": "mehregan_fig6a_honest",
        "sampling": "trailing",
        "honest": True,
        "ptq_weight_noise": 0.0,
        "qat_weak_lock": False,
        "plant_integration_train": "continuous",
        "state_length": STATE_LENGTH,
        "skip_regular": SKIP_REGULAR,
        "mean_hz": MEAN_HZ,
        "seed": SEED,
        "fp32_checkpoint": str(fp32),
        "qat_checkpoint": str(qat),
        "time_s": times.tolist(),
        "traces": {},
        "variants": {},
    }
    print("Honest Fig 6a trailing eval (PTQ noise=0)", flush=True)
    try:
        for key, (variant, ckpt) in variants.items():
            print(f"eval {key}...", flush=True)
            actions = _variant_actions(ckpt, variant=variant, seed=SEED)
            uniq = sorted(set(actions))
            print(f"  unique actions={uniq}", flush=True)
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
                "unique_actions": uniq,
                "n_unique": len(uniq),
            }
    finally:
        plant.close()
    payload["honest_gates"] = _honest_gates(payload)
    EVAL_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {EVAL_JSON}", flush=True)
    print("gates:", json.dumps(payload["honest_gates"]["gates"]), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--retrain-qat", action="store_true")
    parser.add_argument("--fp32-checkpoint", type=Path, default=FP32_CHECKPOINT)
    parser.add_argument("--qat-checkpoint", type=Path, default=QAT_CHECKPOINT)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--train-fp32-only", action="store_true")
    parser.add_argument("--train-qat-only", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    if args.train_fp32_only:
        _train_fp32()
        print(f"done in {time.time() - t0:.0f}s", flush=True)
        return 0
    if args.train_qat_only:
        _train_qat()
        print(f"done in {time.time() - t0:.0f}s", flush=True)
        return 0

    if args.plot_only and EVAL_JSON.is_file():
        payload = json.loads(EVAL_JSON.read_text())
    else:
        if not args.skip_train:
            _train_fp32()
        elif not args.fp32_checkpoint.is_file():
            print(f"missing fp32 checkpoint {args.fp32_checkpoint}", file=sys.stderr)
            return 2
        if args.retrain_qat and args.qat_checkpoint.is_file():
            args.qat_checkpoint.unlink()
        if not args.skip_train or not args.qat_checkpoint.is_file():
            _train_qat()
        payload = _run_eval(fp32=args.fp32_checkpoint, qat=args.qat_checkpoint)

    out_path, _ = _figure_promote.next_versioned_png(FIGURES_DIR, OUT_STEM)
    panel = _plot6a.plot_fig6a(payload, out_path=out_path)
    panel["honest_gates"] = payload.get("honest_gates", _honest_gates(payload))

    manifest = {
        "figure": "mehregan_fig6a_honest",
        "eval_json": str(EVAL_JSON),
        "fp32_checkpoint": str(args.fp32_checkpoint),
        "qat_checkpoint": str(args.qat_checkpoint),
        "output_png": _figure_promote.repo_rel_posix(out_path),
        "panel": panel,
        "honest": True,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.no_promote and hasattr(_figure_promote, "promote_6a"):
        _figure_promote.promote_6a(
            manifest=manifest,
            eval_path=EVAL_JSON,
            png_path=out_path,
            update_docs=True,
        )
    print(f"wrote {out_path}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0 if panel.get("honest_gates", {}).get("gates", {}).get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
