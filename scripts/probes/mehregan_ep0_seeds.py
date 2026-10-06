"""Episode-0 training $P_\\beta$ spread across seeds (Mehregan Fig 4a/4b ep0 gate).

Episode 0 of the paper recipe is dominated by exploration over near-zero
initial logits, so its mean $P_\\beta$ is a property of the seed + alphabet, not
of learning. Diagnostic only.

  uv run python -m rl_adaptive_dbs.run scripts/probes/mehregan_ep0_seeds.py --seeds 0-9 \
      --out artifacts/probes/mehregan_ep0_seeds.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_adaptive_dbs.panel import load_script_module

_PP = load_script_module(
    "mehregan_paper_protocol",
    Path(__file__).resolve().parents[1] / "figures/papers/mehregan/paper_protocol.py",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mean-hz", type=float, default=45.0)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lo, _, hi = args.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    rows = {}
    for seed in seeds:
        rec = _PP.train_paper(mean_hz=args.mean_hz, seed=seed, knobs=_PP.PaperKnobs(), num_episodes=1)
        rows[str(seed)] = {
            "ep0_mean_beta": float(np.mean(rec["beta_trace"])),
            "ep0_reward": rec["episode_rewards"][0],
        }
        print(f"seed {seed}: {rows[str(seed)]}", flush=True)
    vals = np.array([r["ep0_mean_beta"] for r in rows.values()])
    out = {"mean_hz": args.mean_hz, "rows": rows, "mean": float(vals.mean()), "std": float(vals.std())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"ep0 mean {vals.mean():.3f} ± {vals.std():.3f}; wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
