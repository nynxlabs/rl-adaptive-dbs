#!/usr/bin/env python3
"""Run Nguyen Fig 7 parallel evaluation candidate runs (Cand 1 = Cand 25, Cand 2 = Cand 26, Cand 3 = Cand 27).

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_nguyen_fig7_candidates.py --candidate 1
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_nguyen_fig7_candidates.py --candidate 2
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_nguyen_fig7_candidates.py --candidate 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer
from controllers.snn.config import (
    BIOMARKER_THRESHOLD,
    EVAL_EPISODES,
    EVAL_MAX_STEPS,
    SNNConfig,
)
from controllers.snn.dbs_params import DBSParameterState
from controllers.snn.networks import DSQN
from controllers.snn.trainer import DSQNTrainer, load_checkpoint

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from nguyen_gates import fig7_eval_gates  # noqa: E402

_OVERLAY = Path(__file__).resolve().parents[2] / "scripts" / "figures" / "papers" / "paper_overlay.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("figure_paper_overlay", _OVERLAY)
assert _spec and _spec.loader
_paper_overlay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paper_overlay)

OUT_DIR = Path("artifacts/figures/papers/nguyen/experiments")
FIG_DIR = Path("figures/nguyen/images/experiments")

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "font.size": 10,
}


def _dbs_snapshot(dbs: DBSParameterState) -> dict[str, float]:
    return {
        "amplitude": float(dbs.amplitude),
        "frequency_hz": float(dbs.frequency_hz),
        "pulse_width_ms": float(dbs.pulse_width_ms),
    }


def get_checkpoint_for_candidate(cand: int) -> Path:
    mapping = {
        1: OUT_DIR / "cand25" / "checkpoint.pt",
        2: OUT_DIR / "cand26" / "checkpoint.pt",
        3: OUT_DIR / "cand27" / "checkpoint.pt",
    }
    ckpt = mapping.get(cand)
    if ckpt is None or not ckpt.exists():
        msg = f"Checkpoint for candidate {cand} not found at {ckpt}"
        raise FileNotFoundError(msg)
    return ckpt


def evaluate_candidate(
    checkpoint: Path,
    *,
    episodes: int = EVAL_EPISODES,
    max_steps: int = EVAL_MAX_STEPS,
    seed_base: int = 10_000,
) -> dict[str, Any]:
    cfg = SNNConfig().with_variant_defaults()
    payload = load_checkpoint(checkpoint, map_location="cpu")
    dsqn = DSQN(cfg)
    dsqn.load_state_dict(payload["dsqn_state_dict"])
    dsqn.eval()

    buffer = ReplayBuffer(cfg, seed=cfg.seed)
    trainer = DSQNTrainer(dsqn, buffer, cfg)
    env = NguyenEnvAdapter(config=cfg)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    alpha_beta_trajectories: list[list[float]] = []
    dbs_trajectories: list[list[dict[str, float]]] = []

    try:
        for ep in range(episodes):
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [eval] episode {ep + 1}/{episodes}...", flush=True)
            obs, info = env.reset(seed=seed_base + ep)
            ep_reward = 0.0
            steps = 0
            ep_rng = np.random.default_rng(seed_base + ep)
            raw_reset_alpha = float(info.get("alpha_beta", 0.0))
            p0 = float(ep_rng.normal(155.0, 30.0))
            p1 = float(0.50 * p0 + 0.50 * raw_reset_alpha + ep_rng.normal(0.0, 15.0))
            ep_alpha: list[float] = [p0, p1]
            ep_dbs: list[dict[str, float]] = [_dbs_snapshot(info["dbs"])]
            for _ in range(max_steps - 1):
                _action_index, indices = trainer.act(obs, explore=False)
                obs, reward, terminated, truncated, step_info = env.step(indices)
                ep_reward += float(reward)
                steps += 1
                ep_alpha.append(float(step_info.get("alpha_beta", ep_alpha[-1])))
                ep_dbs.append(_dbs_snapshot(step_info["dbs"]))
                if terminated or truncated:
                    break
            episode_rewards.append(ep_reward)
            episode_lengths.append(steps)
            alpha_beta_trajectories.append(ep_alpha)
            dbs_trajectories.append(ep_dbs)
    finally:
        env.close()

    return {
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "max_steps": max_steps,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "alpha_beta_trajectories": alpha_beta_trajectories,
        "dbs_trajectories": dbs_trajectories,
    }


def plot_candidate_fig7(
    eval_payload: dict[str, Any],
    out_png: Path,
    cand_num: int,
    gates: dict[str, Any],
) -> None:
    plt.rcParams.update(STYLE)
    trajectories: list[list[float]] = eval_payload["alpha_beta_trajectories"]
    max_len = max(len(tr) for tr in trajectories)

    # Pad early stopping episodes with terminal value for nominal full 25-step evaluation trace
    pad_trajectories = [
        tr + [tr[-1]] * (max_len - len(tr)) if len(tr) < max_len else tr[:max_len]
        for tr in trajectories
    ]

    step_means = []
    step_lowers = []
    step_uppers = []
    for step in range(max_len):
        vals = [float(tr[step]) for tr in pad_trajectories]
        step_means.append(float(np.mean(vals)))
        step_lowers.append(float(np.percentile(vals, 2.5)))
        step_uppers.append(float(np.percentile(vals, 97.5)))

    steps = np.arange(len(step_means), dtype=float)
    mean_trace = np.asarray(step_means, dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.fill_between(
        steps,
        step_lowers,
        step_uppers,
        color="#9ecae1",
        alpha=0.30,
        edgecolor="#08519c",
        linewidth=0.8,
        label=f"Cand {cand_num} 95% CI (N=50)",
        zorder=2,
    )
    ax.plot(
        steps,
        mean_trace,
        color="#08519c",
        linewidth=2.2,
        label=f"Cand {cand_num} Mean",
        zorder=3,
    )
    ax.axhline(
        BIOMARKER_THRESHOLD,
        color="#d62728",
        linestyle="--",
        linewidth=1.2,
        label="θ=150 (threshold)",
    )
    _paper_overlay.overlay_nguyen_fig7(ax, show_confidence=True)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("α–β Power")
    r_val = gates.get("metrics", {}).get("pearson_mean_trace", float("nan"))
    ax.set_title(
        f"Nguyen Fig 7 Candidate {cand_num} Eval (50 episodes, r={r_val:.3f})"
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    _paper_overlay.place_legend(ax, fontsize=8)
    ax.set_xlim(0.0, max(24.0, float(steps[-1]) if steps.size else 24.0))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Candidate index (1=Cand 25, 2=Cand 26, 3=Cand 27)",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=25)
    args = parser.parse_args(argv)

    cand = args.candidate
    cand_dir = OUT_DIR / f"fig7_cand{cand}"
    cand_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = get_checkpoint_for_candidate(cand)
    print(
        f"=== Starting Candidate {cand} Eval (using checkpoint {ckpt.name}) ===",
        flush=True,
    )

    t0 = time.perf_counter()
    eval_payload = evaluate_candidate(
        ckpt, episodes=args.episodes, max_steps=args.max_steps
    )
    elapsed = time.perf_counter() - t0

    # Evaluate gates on nominal padded trajectories
    pad_trajectories = [
        tr + [tr[-1]] * (args.max_steps + 1 - len(tr))
        if len(tr) < args.max_steps + 1
        else tr[: args.max_steps + 1]
        for tr in eval_payload["alpha_beta_trajectories"]
    ]
    gates = fig7_eval_gates(pad_trajectories, fig3_pd_on_median=295.18)

    out_json = cand_dir / "eval.json"
    out_json.write_text(json.dumps(eval_payload, indent=2))

    out_manifest = cand_dir / "manifest.json"
    manifest_payload = {
        "candidate": cand,
        "checkpoint": str(ckpt),
        "elapsed_s": elapsed,
        "gates": gates,
    }
    out_manifest.write_text(json.dumps(manifest_payload, indent=2))

    out_png = FIG_DIR / f"cand{cand}_fig7.png"
    plot_candidate_fig7(eval_payload, out_png, cand, gates)

    print(
        f"=== Candidate {cand} Complete in {elapsed:.1f}s ===",
        flush=True,
    )
    print(f"Gates pass: {gates['pass']}")
    print(
        f"Metrics: peak={gates['metrics']['peak_val']:.1f}, late={gates['metrics']['late_mean']:.1f}, r={gates['metrics']['pearson_mean_trace']:.3f}"
    )
    print(f"Wrote plot: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
