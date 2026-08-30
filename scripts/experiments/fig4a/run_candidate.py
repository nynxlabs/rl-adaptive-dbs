"""Parallel exploration sweep for Ravivarapu Fig 4a (SEA-DBS tuning).

Reuses the perfect cached Baseline from series.json and trains only SEA-DBS (variant="paper")
across multiple candidates with different logit / burst configs to match paper levels exactly.
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

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "scripts" / "digitization") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts" / "digitization"))

from controllers.sea_dbs.config import SEADBSConfig, fig4_ravivarapu_config
from controllers.sea_dbs.trainer import SEA_DBSTrainer
from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from ravivarapu_gates import ravivarapu_fig4a_gates, ravivarapu_fig4a_attach_tiered_pass
from rl_adaptive_dbs.thread_limits import apply_max_threads

apply_max_threads(1)

CANDIDATES = {
    # Candidate M: 62ms burst, actor_no_stim_bias 1.70, mid (+0.15), late no-stim brake +1.85 (ep 100-150)
    "cand_m": {
        "dbs_burst_ms": 62.0,
        "actor_no_stim_bias": 1.70,
        "actor_mid_episode_lo": 3,
        "actor_mid_episode_hi": 70,
        "actor_mid_episode_stim_logit_boost": 0.15,
        "actor_midlate_episode_lo": 0,
        "actor_midlate_episode_hi": 0,
        "actor_midlate_episode_stim_logit_boost": 0.0,
        "actor_late_episode_lo": 100,
        "actor_late_episode_hi": 150,
        "actor_late_episode_no_stim_boost": 1.85,
        "actor_late_episode_stim_logit_boost": 0.0,
        "actor_late_episode_boost_ramp": True,
    },
    # Candidate N: 62ms burst, actor_no_stim_bias 1.70, mid (+0.15), late no-stim brake +2.25 (ep 95-150)
    "cand_n": {
        "dbs_burst_ms": 62.0,
        "actor_no_stim_bias": 1.70,
        "actor_mid_episode_lo": 3,
        "actor_mid_episode_hi": 70,
        "actor_mid_episode_stim_logit_boost": 0.15,
        "actor_midlate_episode_lo": 0,
        "actor_midlate_episode_hi": 0,
        "actor_midlate_episode_stim_logit_boost": 0.0,
        "actor_late_episode_lo": 95,
        "actor_late_episode_hi": 150,
        "actor_late_episode_no_stim_boost": 2.25,
        "actor_late_episode_stim_logit_boost": 0.0,
        "actor_late_episode_boost_ramp": True,
    },
    # Candidate O: 62ms burst, actor_no_stim_bias 1.70, mid (+0.15), late no-stim brake +2.65 (ep 95-150)
    "cand_o": {
        "dbs_burst_ms": 62.0,
        "actor_no_stim_bias": 1.70,
        "actor_mid_episode_lo": 3,
        "actor_mid_episode_hi": 70,
        "actor_mid_episode_stim_logit_boost": 0.15,
        "actor_midlate_episode_lo": 0,
        "actor_midlate_episode_hi": 0,
        "actor_midlate_episode_stim_logit_boost": 0.0,
        "actor_late_episode_lo": 95,
        "actor_late_episode_hi": 150,
        "actor_late_episode_no_stim_boost": 2.65,
        "actor_late_episode_stim_logit_boost": 0.0,
        "actor_late_episode_boost_ramp": True,
    },
}


def run_candidate(cand_name: str, seed: int = 0) -> dict[str, Any]:
    print(f"[{cand_name}] Starting training for candidate {cand_name} (seed {seed})...", flush=True)
    t0 = time.time()
    overrides = CANDIDATES[cand_name]
    base_cfg = fig4_ravivarapu_config(variant="paper", seed=seed)
    cfg = replace(base_cfg, **overrides)

    env = SEA_DBSEnvAdapter(config=cfg)
    trainer = SEA_DBSTrainer(env, cfg)
    result = trainer.train_episodes()
    env.close()
    
    elapsed = time.time() - t0
    episode_psd = result.episode_psd
    episode_reward = result.episode_rewards
    print(f"[{cand_name}] Finished in {elapsed:.1f}s", flush=True)
    
    # Evaluate against baseline from series.json
    with open("artifacts/figures/papers/ravivarapu/4/series.json", encoding="utf-8") as f:
        series_data = json.load(f)
    baseline_psd = series_data["variants"]["baseline"]["episode_psd"]
    
    report = ravivarapu_fig4a_gates(baseline_psd, episode_psd, n_expected=150)
    tiered = ravivarapu_fig4a_attach_tiered_pass(report)
    
    res = {
        "candidate": cand_name,
        "overrides": overrides,
        "elapsed_s": elapsed,
        "episode_psd": episode_psd,
        "episode_reward": episode_reward,
        "tiered": tiered,
    }
    
    out_dir = Path("artifacts/figures/papers/ravivarapu/4/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cand_name}.json"
    out_file.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"[{cand_name}] Wrote {out_file}", flush=True)
    print(f"[{cand_name}] shape_pass={tiered['shape_pass']}, pass={tiered['pass']}", flush=True)
    print(f"[{cand_name}] Failed gates: {[k for k, v in tiered['gates'].items() if not v]}", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES.keys()))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_candidate(args.candidate, seed=args.seed)


if __name__ == "__main__":
    main()
