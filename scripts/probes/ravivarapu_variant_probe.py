#!/usr/bin/env python3
"""Train one SEA-DBS variant with config overrides, then roll it out at 50 / 30 Hz.

Diagnostic probe for paper-silent conventions (docs/controllers/sea-dbs/figures/*.md).
Trains on ``fig4_ravivarapu_config`` plus ``--set FIELD=VALUE`` overrides, saves
the checkpoint, and records the per-episode PSD / stim duty plus the deterministic
(argmax) 10-step inference traces the Figs 5–7 panels use. Panel results still
come from the panel scripts; this only screens candidates.

  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_variant_probe.py \\
      --variant baseline --set critic_action_input=logits --out /tmp/probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.checkpoint import save_checkpoint
from controllers.sea_dbs.config import fig4_ravivarapu_config
from controllers.sea_dbs.eval import evaluate
from controllers.sea_dbs.trainer import SEA_DBSTrainer

_FIG4A = Path(__file__).resolve().parents[1] / "figures" / "papers" / "ravivarapu" / "4a" / "plot.py"


def _parse_overrides(items: list[str]) -> dict:
    from rl_adaptive_dbs.panel import load_script_module

    mod = load_script_module("ravi_probe_fig4a", _FIG4A)
    return dict(mod.parse_overrides(items))


def _windows(y: list[float]) -> dict[str, float]:
    a = np.asarray(y, dtype=float)
    return {
        "early_0_15": float(a[:15].mean()),
        "mid_40_80": float(a[40:80].mean()) if a.size >= 80 else float("nan"),
        "midlate_70_110": float(a[70:110].mean()) if a.size >= 110 else float("nan"),
        "late_120_150": float(a[120:150].mean()) if a.size >= 150 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["baseline", "baseline-pm", "baseline-gs", "paper"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    overrides = _parse_overrides(args.set)
    cfg = replace(
        fig4_ravivarapu_config(seed=args.seed, variant=args.variant, num_episodes=args.episodes),
        **overrides,
    )
    t0 = time.time()
    env = SEA_DBSEnvAdapter(config=cfg)
    try:
        trainer = SEA_DBSTrainer(env, cfg)
        result = trainer.train_episodes()
    finally:
        env.close()
    ckpt = args.out.with_suffix(".pt")
    save_checkpoint(
        ckpt,
        actor=result.actor,
        critic=result.critic,
        config=cfg,
        predictive_model=result.predictive_model,
        trainer=trainer,
        extra={"episode_psd": result.episode_psd, "episode_rewards": result.episode_rewards},
    )
    out: dict = {
        "variant": args.variant,
        "seed": args.seed,
        "overrides": overrides,
        "config": asdict(cfg),
        "episode_psd": result.episode_psd,
        "episode_rewards": [r / cfg.max_episode_steps for r in result.episode_rewards],
        "episode_stim_frac": result.episode_stim_frac,
        "psd_windows": _windows(result.episode_psd),
        "stim_windows": _windows(result.episode_stim_frac),
        "train_s": round(time.time() - t0, 1),
        "checkpoint": ckpt.as_posix(),
        "eval": {},
    }
    for hz in (50.0, 30.0):
        ev = evaluate(ckpt, episodes=1, max_steps=10, carrier_hz=hz, action_mode="argmax")
        logits = np.asarray(ev["logit_trajectories"][0])
        out["eval"][f"{int(hz)}hz"] = {
            "p_beta": ev["p_beta_trajectories"][0],
            "actions": ev["action_trajectories"][0],
            "stim_margin": (logits[:, 1] - logits[:, 0]).tolist(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    e50 = out["eval"]["50hz"]
    print(json.dumps({k: out[k] for k in ("psd_windows", "stim_windows", "train_s")}, indent=1))
    print("50hz", [round(v, 3) for v in e50["p_beta"]], e50["actions"])
    print("margin", [round(v, 2) for v in e50["stim_margin"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
