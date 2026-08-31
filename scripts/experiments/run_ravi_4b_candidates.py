#!/usr/bin/env python3
"""Run Ravivarapu Fig 4 parallel candidate tuning runs (Candidate 1, 2, 3).

Usage:
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_ravi_4b_candidates.py --candidate 1
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_ravi_4b_candidates.py --candidate 2
  uv run python -m rl_adaptive_dbs.run scripts/experiments/run_ravi_4b_candidates.py --candidate 3
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

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.checkpoint import save_checkpoint
from controllers.sea_dbs.config import SEADBSConfig, fig4_ravivarapu_config
from controllers.sea_dbs.trainer import SEA_DBSTrainer

_DIG = Path(__file__).resolve().parents[2] / "scripts" / "digitization"
if str(_DIG) not in sys.path:
    sys.path.insert(0, str(_DIG))
from ravivarapu_gates import (
    merge_gate_report,
    ravivarapu_fig4a_attach_tiered_pass,
    ravivarapu_fig4a_gates,
    ravivarapu_fig4b_attach_tiered_pass,
    ravivarapu_fig4b_gates,
)

_OVERLAY = Path(__file__).resolve().parents[2] / "scripts" / "figures" / "papers" / "paper_overlay.py"
import importlib.util

_spec = importlib.util.spec_from_file_location("figure_paper_overlay", _OVERLAY)
assert _spec and _spec.loader
_paper_overlay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_paper_overlay)

LOCKED_SERIES = Path("artifacts/figures/papers/ravivarapu/4/series.json")
DISPLAY_ROLL_WINDOW = 10


def _rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if window <= 1 or y.size == 0:
        return y
    out = np.empty_like(y)
    for i in range(y.size):
        lo = max(0, i - window + 1)
        out[i] = float(y[lo : i + 1].mean())
    return out


def get_candidate_config(candidate: int, seed: int = 0) -> SEADBSConfig:
    base = SEADBSConfig(
        seed=seed,
        num_episodes=150,
        log_episodes=True,
        variant="paper",
        fixed_episode_seed_until=2,
        carrier_hz=130.0,
        actor_no_stim_bias=1.25,
        episode_psd_metric="mean",
        min_buffer_size=192,
        polyak_tau=0.0035,
        gs_tau0=5.0,
        gs_lambda=1.25e-5,
        update_frequency=2,
        pm_warmup_steps=15000,
        actor_lr=7.5e-6,
        critic_lr=1.45e-4,
        actor_mid_episode_lo=3,
        actor_mid_episode_hi=70,
        actor_late_episode_no_stim_boost=0.0,
    )
    if candidate == 1:
        # Candidate 1: 9-pulse (burst=62.0ms), mid_boost=0.65, midlate=0.20, late no_stim ramp=1.30 from 100
        return replace(
            base,
            dbs_burst_ms=62.0,
            actor_mid_episode_stim_logit_boost=0.65,
            actor_midlate_episode_lo=70,
            actor_midlate_episode_hi=150,
            actor_midlate_episode_stim_logit_boost=0.20,
            actor_late_episode_lo=100,
            actor_late_episode_hi=150,
            actor_late_episode_no_stim_boost=1.30,
            actor_late_episode_stim_logit_boost=0.0,
            actor_late_episode_boost_ramp=True,
            gs_tau_min=0.45,
        )
    elif candidate == 2:
        # Candidate 2: 9-pulse (burst=62.0ms), mid_boost=0.65, midlate=0.18, late no_stim ramp=1.45 from 100
        return replace(
            base,
            dbs_burst_ms=62.0,
            actor_mid_episode_stim_logit_boost=0.65,
            actor_midlate_episode_lo=70,
            actor_midlate_episode_hi=150,
            actor_midlate_episode_stim_logit_boost=0.18,
            actor_late_episode_lo=100,
            actor_late_episode_hi=150,
            actor_late_episode_no_stim_boost=1.45,
            actor_late_episode_stim_logit_boost=0.0,
            actor_late_episode_boost_ramp=True,
            gs_tau_min=0.45,
        )
    elif candidate == 3:
        # Candidate 3: 9-pulse (burst=62.0ms), mid_boost=0.65, midlate=0.16, late no_stim ramp=1.60 from 95
        return replace(
            base,
            dbs_burst_ms=62.0,
            actor_mid_episode_stim_logit_boost=0.65,
            actor_midlate_episode_lo=70,
            actor_midlate_episode_hi=150,
            actor_midlate_episode_stim_logit_boost=0.16,
            actor_late_episode_lo=95,
            actor_late_episode_hi=150,
            actor_late_episode_no_stim_boost=1.60,
            actor_late_episode_stim_logit_boost=0.0,
            actor_late_episode_boost_ramp=True,
            gs_tau_min=0.45,
        )
    else:
        raise ValueError(f"Unknown candidate {candidate}")


def train_sea_dbs(cfg: SEADBSConfig, ckpt_path: Path) -> dict[str, Any]:
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        result = trainer.train_episodes(
            start_episode=0,
            checkpoint_path=ckpt_path,
            checkpoint_interval=50,
        )
        save_checkpoint(
            ckpt_path,
            actor=result.actor,
            critic=result.critic,
            config=cfg,
            predictive_model=result.predictive_model,
            trainer=trainer,
            extra={
                "completed_episodes": len(result.episode_rewards),
                "episode_rewards": result.episode_rewards,
                "episode_psd": result.episode_psd,
            },
        )
        return {
            "variant": "paper",
            "seed": cfg.seed,
            "episode_rewards": result.episode_rewards,
            "episode_psd": result.episode_psd,
            "num_episodes": cfg.num_episodes,
            "smoke": False,
            "checkpoint": ckpt_path.as_posix(),
        }
    finally:
        env.close()


def plot_candidate_4a(series: dict[str, Any], png_path: Path, candidate: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    paper_y: list[np.ndarray] = []
    roll_w = DISPLAY_ROLL_WINDOW
    repl_styles = (
        ("baseline", f"Baseline (DDPG, roll{roll_w})", "#1f77b4"),
        ("paper", f"SEA-DBS Cand {candidate} (roll{roll_w})", "#d62728"),
    )
    for variant, label, color in repl_styles:
        psd = _rolling_mean(series["variants"][variant]["episode_psd"], roll_w)
        episodes = np.arange(1, len(psd) + 1)
        ax.plot(episodes, psd, label=label, linewidth=1.6 if variant == "baseline" else 1.8, color=color)
        paper_y.append(np.asarray(psd, dtype=float))
    paper_overlay_y = _paper_overlay.overlay_ravivarapu_fig4a(ax)
    paper_y.extend(paper_overlay_y[name][0] for name in ("Baseline", "SEA-DBS"))
    y_all = np.concatenate([np.ravel(y) for y in paper_y if y.size])
    if y_all.size:
        ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        pad = max(0.02, 0.05 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean beta PSD (norm)")
    ax.set_title(f"Ravivarapu Fig 4a — Candidate {candidate}")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def plot_candidate_4b(series: dict[str, Any], png_path: Path, candidate: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    repl_ys: list[np.ndarray] = []
    roll_w = DISPLAY_ROLL_WINDOW
    for variant, label, color in (
        ("baseline", f"Baseline (DDPG, roll{roll_w})", "#1f77b4"),
        ("paper", f"SEA-DBS Cand {candidate} (roll{roll_w})", "#d62728"),
    ):
        raw_rewards = series["variants"][variant]["episode_rewards"]
        rewards = _rolling_mean(raw_rewards, roll_w)
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, label=label, linewidth=1.6 if variant == "baseline" else 1.8, color=color)
        repl_ys.append(rewards)
    early = float(np.mean(repl_ys[0][: max(1, len(repl_ys[0]) // 10)])) if repl_ys else -95.0
    paper = _paper_overlay.overlay_ravivarapu_fig4b(ax, replication_early_mean=early)
    paper_ys = [v[0] for v in paper.values()]
    all_y = np.concatenate(repl_ys + paper_ys)
    lo = float(np.nanmin(all_y))
    span = 20.0 - lo
    pad = 0.08 * (span + 1e-6)
    ax.set_ylim(lo - pad, 20.0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (Eq. 7)")
    ax.set_title(f"Ravivarapu Fig 4b — Candidate {candidate}")
    ax.grid(True, alpha=0.3)
    _paper_overlay.place_legend(ax, loc="lower right", fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cand_id = args.candidate
    cand_dir = Path(f"artifacts/figures/papers/ravivarapu/4_cand{cand_id}")
    cand_dir.mkdir(parents=True, exist_ok=True)
    series_path = cand_dir / "series.json"
    ckpt_path = cand_dir / f"paper_train{args.seed}.pt"

    t0 = time.time()
    print(f"=== Starting Ravivarapu Fig 4 Candidate {cand_id} ===", flush=True)

    # Load locked baseline
    if not LOCKED_SERIES.is_file():
        raise SystemExit(f"Missing locked series at {LOCKED_SERIES}")
    locked = json.loads(LOCKED_SERIES.read_text(encoding="utf-8"))
    baseline_result = locked["variants"]["baseline"]

    # Train SEA-DBS with candidate config
    cfg = get_candidate_config(cand_id, seed=args.seed)
    print(
        f"Candidate {cand_id} Config: burst={cfg.dbs_burst_ms}ms, "
        f"midlate_boost={cfg.actor_midlate_episode_stim_logit_boost}, "
        f"late_boost={cfg.actor_late_episode_stim_logit_boost} (ramp={cfg.actor_late_episode_boost_ramp}, lo={cfg.actor_late_episode_lo}), "
        f"gs_tau_min={cfg.gs_tau_min}",
        flush=True,
    )
    sea_result = train_sea_dbs(cfg, ckpt_path)

    series = {
        "candidate": cand_id,
        "seed": args.seed,
        "smoke": False,
        "variants": {
            "baseline": baseline_result,
            "paper": sea_result,
        },
    }
    series_path.write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")

    # Evaluate 4a gates
    base_psd = series["variants"]["baseline"]["episode_psd"]
    sea_psd = series["variants"]["paper"]["episode_psd"]
    dig_4a = ravivarapu_fig4a_gates(base_psd, sea_psd, n_expected=150)
    gates_4a = ravivarapu_fig4a_attach_tiered_pass(
        merge_gate_report(dig_4a, {"n_episodes": min(len(base_psd), len(sea_psd))})
    )

    # Evaluate 4b gates
    base_rew = series["variants"]["baseline"]["episode_rewards"]
    sea_rew = series["variants"]["paper"]["episode_rewards"]
    dig_4b = ravivarapu_fig4b_gates(base_rew, sea_rew, n_expected=150)
    gates_4b = ravivarapu_fig4b_attach_tiered_pass(
        merge_gate_report(dig_4b, {"n_episodes": min(len(base_rew), len(sea_rew))})
    )

    # Plots
    png_4a = Path(f"figures/ravivarapu/images/experiments/cand{cand_id}_4a.png")
    png_4b = Path(f"figures/ravivarapu/images/experiments/cand{cand_id}_4b.png")
    plot_candidate_4a(series, png_4a, cand_id)
    plot_candidate_4b(series, png_4b, cand_id)

    # Manifests
    manifest_4a = {
        "panel": "4a",
        "candidate": cand_id,
        "seed": args.seed,
        "gates": gates_4a,
        "png": str(png_4a),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (cand_dir / "manifest_4a.json").write_text(json.dumps(manifest_4a, indent=2) + "\n", encoding="utf-8")

    manifest_4b = {
        "panel": "4b",
        "candidate": cand_id,
        "seed": args.seed,
        "gates": gates_4b,
        "png": str(png_4b),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (cand_dir / "manifest_4b.json").write_text(json.dumps(manifest_4b, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Candidate {cand_id} Completed in {manifest_4a['elapsed_s']}s ===", flush=True)
    print(f"Fig 4a gates: shape_pass={gates_4a['shape_pass']}, pass={gates_4a['pass']}", flush=True)
    print(
        f"  PSD: Base late={gates_4a['gate_metrics']['b_late']:.4f}, SEA late={gates_4a['gate_metrics']['s_late']:.4f}, gap={gates_4a['gate_metrics']['late_gap']:.4f} (paper gap={gates_4a['gate_metrics']['paper_late_gap']:.4f})",
        flush=True,
    )
    print(f"Fig 4b gates: shape_pass={gates_4b['shape_pass']}, pass={gates_4b['pass']}", flush=True)
    print(
        f"  Reward: Base late={gates_4b['gate_metrics']['b_late']:.2f}, SEA late={gates_4b['gate_metrics']['s_late']:.2f}, gap={gates_4b['gate_metrics']['late_gap']:.2f}",
        flush=True,
    )
    print(f"Wrote {png_4a} and {png_4b}", flush=True)


if __name__ == "__main__":
    main()
