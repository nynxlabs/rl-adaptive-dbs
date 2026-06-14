#!/usr/bin/env python3
"""Run Mehregan DDPG paper replication on the Kumaravelu MATLAB plant.

Example::

    source scripts/matlab/env.sh
    uv run python scripts/replicate_mehregan_ddpg.py --variant paper --train-seed 0

Full paper defaults: 10 training episodes × 30 steps (~hours on MATLAB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.replication import ReplicationConfig, run_replication, write_replication_summary
from envs.mehregan import MehreganEnv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mehregan DDPG replication (train + eval + baselines)")
    parser.add_argument("--variant", default="paper", choices=("paper", "init-30hz"))
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override training episodes (default: 10 from paper §IV.A.1)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/ddpg"),
        help="Directory for actor checkpoint (default: artifacts/ddpg/)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="JSON summary path (default: <checkpoint-dir>/<variant>_summary.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ddpg_kwargs: dict[str, object] = {"variant": args.variant, "seed": args.train_seed}
    if args.episodes is not None:
        ddpg_kwargs["num_episodes"] = args.episodes
    ddpg = DDPGConfig(**ddpg_kwargs)

    repl = ReplicationConfig(
        variant=args.variant,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        ddpg=ddpg,
        checkpoint_dir=args.checkpoint_dir,
    )

    env = MehreganEnv()
    try:
        result = run_replication(env, repl)
    finally:
        env.close()

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.checkpoint_dir / f"{args.variant}_train{args.train_seed}_summary.json"
    write_replication_summary(result, summary_path)

    ddpg_eval = result.eval_metrics
    print(f"variant={result.variant} train_seed={result.train_seed} eval_seed={result.eval_seed}")
    print(f"checkpoint={result.checkpoint_path}")
    print(f"summary={summary_path}")
    print(
        f"ddpg  p_beta_mean={ddpg_eval['p_beta_mean']:.1f} "
        f"p_beta_final={ddpg_eval['p_beta_final']:.1f} reward_sum={ddpg_eval['reward_sum']:.2f}"
    )
    for name, metrics in result.baseline_metrics.items():
        print(
            f"{name:14} p_beta_mean={metrics['p_beta_mean']:.1f} "
            f"p_beta_final={metrics['p_beta_final']:.1f} reward_sum={metrics['reward_sum']:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
