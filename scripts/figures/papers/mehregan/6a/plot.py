#!/usr/bin/env python3
"""Mehregan et al. (paper 1) Figure 6a — PTQ / QAT @ 45 Hz.

Paper-protocol eval (2 s baseline + 5×2 s steps): step-function GPi P_beta on the
**raw PSD scale** (y ~300–550 in the paper panel). Four series:

  1. Fully trained fp32 (green)
  2. PTQ int8 (blue)
  3. PTQ fp16 (purple)
  4. QAT (orange dashed)

**Paired workflow (default):** fp32 from Fig 4a checkpoint
(``artifacts/figures/papers/mehregan/4a/checkpoint.pt``); train QAT only (10 episodes).
Train once with ``scripts/figures/papers/mehregan/4a/plot.py``, then run this script.

Run:
  uv run python scripts/figures/papers/mehregan/4a/plot.py --seed 1
  uv run python scripts/figures/papers/mehregan/6a/plot.py --seed 1
  uv run python scripts/figures/papers/mehregan/6a/plot.py --plot-only
  uv run python scripts/figures/papers/mehregan/6a/plot.py --skip-train \\
    --fp32-checkpoint artifacts/figures/papers/mehregan/4a/checkpoint.pt \\
    --qat-checkpoint artifacts/figures/papers/mehregan/6a/qat_train1.pt

QAT train only (~30–60 min). Prefer tmux:

  tmux new-session -d -s fig6a-train \\
    "setsid nohup uv run python scripts/figures/papers/mehregan/6a/plot.py >> logs/fig6a-train.log 2>&1 < /dev/null"
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

from controllers.ddpg import DDPGConfig, evaluate, save_checkpoint, train
from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.eval import EvalConfig
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

_PROMOTE = Path(__file__).resolve().parents[2] / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_figure_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_figure_promote)

FIGURES_DIR = Path("figures/mehregan/images/6a")
CACHE_DIR = Path("artifacts/figures/papers/mehregan/6a")
FIG4A_CACHE = Path("artifacts/figures/papers/mehregan/4a")
DEFAULT_FP32_CHECKPOINT = FIG4A_CACHE / "checkpoint.pt"
DEFAULT_EVAL = CACHE_DIR / "eval.json"
DEFAULT_OUT = FIGURES_DIR / "ptq_qat_45hz.png"
DEFAULT_MANIFEST = CACHE_DIR / "manifest.json"
OUT_STEM = "ptq_qat_45hz"

MEAN_HZ = 45.0
PAPER_DT_MS = 0.02
STATE_LENGTH = 1
NUM_EPISODES = 10
STEPS_PER_EPISODE = 30
EVAL_STEPS = 5
DEFAULT_SEED = 0

SEGMENT_S = 2.0
N_SEGMENTS = 6
TIME_MAX_S = 12.0
STIM_ONSET_S = 2.0

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
        "label": "Fully Trained 45Hz",
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
        "linestyle": "--",
    },
}


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


def _make_train_env(*, seed: int) -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=STATE_LENGTH,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
    )
    plant = PythonPlant(config=plant_cfg)
    _ = seed
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)


def _make_eval_env() -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=STATE_LENGTH,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS,
    )
    alphabet = FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=env_cfg.step_duration_s,
        dt_ms=plant_cfg.dt_ms,
    )
    plant = PythonPlant(config=plant_cfg)
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)


def _fp32_config(*, seed: int) -> DDPGConfig:
    return fig4a_ddpg_config(
        seed=seed,
        num_episodes=NUM_EPISODES,
        max_episode_steps=STEPS_PER_EPISODE,
    )


def _qat_config(*, seed: int) -> DDPGConfig:
    return replace(_fp32_config(seed=seed), variant="qat")


def _default_qat_checkpoint(seed: int) -> Path:
    return CACHE_DIR / f"qat_train{seed}.pt"


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
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"seed": seed, "training": {}}

    print("training QAT (10 episodes)...", flush=True)
    t0 = time.time()
    env = _make_train_env(seed=seed)
    try:
        qat_result = train(
            env,
            _qat_config(seed=seed),
            checkpoint_path=qat_path,
        )
    finally:
        env.close()
    meta["training"]["qat"] = {
        "checkpoint": str(qat_path),
        "elapsed_s": round(time.time() - t0, 2),
        "variant": qat_result.config.variant,
    }
    print(f"qat checkpoint -> {qat_path} ({meta['training']['qat']['elapsed_s']}s)", flush=True)
    return meta


def _train_fp32_standalone(
    *,
    seed: int,
    fp32_path: Path,
) -> dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("training fp32 (Fig 4a profile, standalone)...", flush=True)
    t0 = time.time()
    env = _make_train_env(seed=seed)
    try:
        fp32_result = train(
            env,
            _fp32_config(seed=seed),
            checkpoint_path=fp32_path,
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
            },
        },
    }


def _run_variant_evals(
    *,
    fp32_checkpoint: Path,
    qat_checkpoint: Path,
    seed: int,
) -> dict[str, Any]:
    env = _make_eval_env()
    try:
        variants = {
            "fp32": ("paper", fp32_checkpoint),
            "ptq-fp16": ("ptq-fp16", fp32_checkpoint),
            "ptq-int8": ("ptq-int8", fp32_checkpoint),
            "qat": ("qat", qat_checkpoint),
        }
        payload: dict[str, Any] = {
            "mean_hz": MEAN_HZ,
            "seed": seed,
            "eval_steps": EVAL_STEPS,
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
    data = payload["variants"][key]
    if "error" in data:
        msg = f"variant {key!r} failed: {data['error']}"
        raise KeyError(msg)
    return [float(v) for v in data["p_beta"]]


def _post_onset_mean(trace: list[float]) -> float:
    post = trace[1:]
    if not post:
        return float("nan")
    return float(np.mean(post))


def _ylim_for_traces(traces: list[list[float]]) -> tuple[float, float, list[float]]:
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


def _gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    fp32 = _variant_trace(payload, "fp32")
    fp32_post = _post_onset_mean(fp32)
    gates: dict[str, Any] = {}

    for key in ("ptq-fp16", "ptq-int8"):
        trace = _variant_trace(payload, key)
        post = _post_onset_mean(trace)
        rel_err = abs(post - fp32_post) / fp32_post if fp32_post else float("inf")
        gates[f"{key}_tracks_fp32"] = rel_err <= 0.15

    qat_post = _post_onset_mean(_variant_trace(payload, "qat"))
    gates["qat_elevated_vs_fp32"] = qat_post > fp32_post
    gates["fp32_suppresses_vs_baseline"] = fp32_post < fp32[0]

    return {
        "fp32_baseline": fp32[0],
        "fp32_post_mean": fp32_post,
        "ptq_fp16_post_mean": _post_onset_mean(_variant_trace(payload, "ptq-fp16")),
        "ptq_int8_post_mean": _post_onset_mean(_variant_trace(payload, "ptq-int8")),
        "qat_post_mean": qat_post,
        "gates": gates,
    }


def plot_fig6a(
    payload: dict[str, Any],
    *,
    out_path: Path,
) -> dict[str, Any]:
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)

    traces: dict[str, list[float]] = {}
    for key in ("fp32", "ptq-int8", "ptq-fp16", "qat"):
        traces[key] = _variant_trace(payload, key)

    # Shared pre-stim baseline: align all series to fp32 segment 0 (paper overlap 0–2 s).
    baseline = traces["fp32"][0]
    aligned = {
        key: [baseline] + trace[1:] for key, trace in traces.items()
    }

    y0, y1, yticks = _ylim_for_traces(list(aligned.values()))

    for key in ("fp32", "ptq-int8", "ptq-fp16", "qat"):
        meta = SERIES[key]
        x, y = _step_series(aligned[key])
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
        help="Full-precision checkpoint (default: Fig 4a artifacts/figures/papers/mehregan/4a/checkpoint.pt)",
    )
    parser.add_argument(
        "--qat-checkpoint",
        type=Path,
        default=None,
        help="QAT checkpoint (default: cache qat_train{seed}.pt)",
    )
    parser.add_argument(
        "--train-fp32",
        action="store_true",
        help="Standalone fp32 train (legacy; prefer Fig 4a paired workflow)",
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
        help="Do not update figures/mehregan/replications.md",
    )
    args = parser.parse_args()

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
                train_meta = _train_fp32_standalone(seed=args.seed, fp32_path=fp32_ckpt)
            elif not fp32_ckpt.exists():
                print(
                    f"missing fp32 checkpoint: {fp32_ckpt}\n"
                    "Run Fig 4a first: uv run python scripts/figures/papers/mehregan/4a/plot.py "
                    f"--seed {args.seed}",
                    file=sys.stderr,
                )
                return 2
            if not qat_ckpt.exists():
                qat_meta = _train_qat_only(seed=args.seed, qat_path=qat_ckpt)
                train_meta = {**train_meta, **qat_meta}
        elif not fp32_ckpt.exists() or not qat_ckpt.exists():
            print(
                f"missing checkpoints: fp32={fp32_ckpt.exists()} qat={qat_ckpt.exists()}",
                file=sys.stderr,
            )
            return 2

        payload = _run_variant_evals(
            fp32_checkpoint=fp32_ckpt,
            qat_checkpoint=qat_ckpt,
            seed=args.seed,
        )
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

    panel = plot_fig6a(payload, out_path=out_path)

    manifest = {
        "figure": "mehregan_fig6a",
        "mean_hz": MEAN_HZ,
        "seed": args.seed,
        "eval_json": str(args.eval_json),
        "fp32_checkpoint": str(fp32_ckpt),
        "fig4a_checkpoint": str(DEFAULT_FP32_CHECKPOINT),
        "qat_checkpoint": str(qat_ckpt),
        "output_png": _figure_promote.repo_rel_posix(out_path),
        "panel": panel,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    if not args.no_promote and hasattr(_figure_promote, "promote_6a"):
        _figure_promote.promote_6a(
            manifest=manifest,
            eval_path=args.eval_json,
            png_path=out_path,
            update_docs=True,
        )

    print(f"wrote {out_path}", flush=True)
    print(f"wrote {args.manifest}", flush=True)
    gates = panel.get("gates", {})
    print(
        "gates: "
        f"ptq_fp16={gates.get('ptq-fp16_tracks_fp32')} "
        f"ptq_int8={gates.get('ptq-int8_tracks_fp32')} "
        f"qat_elevated={gates.get('qat_elevated_vs_fp32')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
