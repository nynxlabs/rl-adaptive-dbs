#!/usr/bin/env python3
"""Re-roll spike/energy series from a Fig 4 checkpoint with current logging rules.

Updates ``episode_spike_totals`` and ``episode_energies`` in the shared
``artifacts/figures/papers/nguyen/4/series.json`` without retraining weights
(``learn=False`` rollout). Use after changing spike accounting or observation
spike sums for Fig 5.

Example:
  uv run python -m rl_adaptive_dbs.run scripts/probes/nguyen_fig5_relog_metrics.py \\
    --checkpoint artifacts/figures/papers/nguyen/4/checkpoint.pt \\
    --series artifacts/figures/papers/nguyen/4/series.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.config import SNNConfig
from controllers.snn.trainer import (
    load_checkpoint,
    resume_dsqn_trainer,
    train_result_from_payload,
    write_train_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relog Fig 5 spike/energy metrics from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/figures/papers/nguyen/4/checkpoint.pt"),
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=Path("artifacts/figures/papers/nguyen/4/series.json"),
    )
    parser.add_argument("--smoke", action="store_true", help="Roll 5 episodes only (does not write series)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write spike/energy even when rollout length != series episode count",
    )
    args = parser.parse_args()

    series = json.loads(args.series.read_text(encoding="utf-8"))
    num_episodes = int(series.get("num_episodes", 500))
    if args.smoke:
        num_episodes = min(5, num_episodes)

    payload = load_checkpoint(args.checkpoint)
    saved_cfg = payload["config"]
    if not isinstance(saved_cfg, SNNConfig):
        saved_cfg = SNNConfig(**saved_cfg)
    cfg = replace(saved_cfg, num_episodes=num_episodes)

    trainer, _ = resume_dsqn_trainer(payload, config=cfg, start_episode=0)
    env = NguyenEnvAdapter(config=cfg)

    rollout = trainer.train_episodes(env, start_episode=0, learn=False)

    n_series = len(series.get("episode_rewards", []))
    n_roll = len(rollout.episode_spike_totals)
    if args.smoke:
        print(
            f"smoke: {n_roll} episodes; spike mean={sum(rollout.episode_spike_totals)/n_roll:.1f} "
            f"energy mean={sum(rollout.episode_energies)/n_roll:.1f}"
        )
        return

    if n_roll != n_series and not args.force:
        raise SystemExit(
            f"rollout episodes ({n_roll}) != series rewards ({n_series}); "
            "use --force after a full 500-ep rollout"
        )

    series["episode_spike_totals"] = rollout.episode_spike_totals
    series["episode_energies"] = rollout.episode_energies
    args.series.write_text(json.dumps(series, indent=2) + "\n", encoding="utf-8")
    write_train_metrics(rollout, args.checkpoint.with_suffix(".metrics.json"))

    spikes = rollout.episode_spike_totals
    energies = rollout.episode_energies
    print(
        f"relogged {len(spikes)} episodes; spike mean={sum(spikes)/len(spikes):.1f} "
        f"energy mean={sum(energies)/len(energies):.1f}"
    )


if __name__ == "__main__":
    main()
