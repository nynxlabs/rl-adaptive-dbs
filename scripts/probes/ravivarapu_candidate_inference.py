#!/usr/bin/env python3
"""Score a Ravivarapu Fig 4 candidate's checkpoints on the Fig 5a inference rollout.

Rolls ``baseline_train<seed>.pt`` and ``paper_train<seed>.pt`` from a candidate
cache (argmax actions, 50 Hz, each checkpoint's own plant setup — exactly the Fig
5a protocol) and prints the traces, the actions, and the Fig 5a gate count, so a
training reading can be checked for a Baseline–SEA split before promoting it.

Run:
  uv run python -m rl_adaptive_dbs.run scripts/probes/ravivarapu_candidate_inference.py \
      artifacts/figures/papers/ravivarapu/candidates/<name>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "digitization"))

from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, INFERENCE_CARRIER_50HZ  # noqa: E402
from controllers.sea_dbs.eval import evaluate_ablation_steps  # noqa: E402
from ravivarapu_gates import ravivarapu_inference_gates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    traces: dict[str, list[float]] = {}
    actions: dict[str, list[int]] = {}
    for variant in ("baseline", "paper"):
        out = evaluate_ablation_steps(
            args.cache_dir / f"{variant}_train{args.seed}.pt",
            n_steps=ABLATION_EVAL_STEPS,
            carrier_hz=INFERENCE_CARRIER_50HZ,
        )
        traces[variant] = out["p_beta_trajectories"][0]
        actions[variant] = out["action_trajectories"][0]
        print(variant, [round(v, 3) for v in traces[variant]], "actions", actions[variant], flush=True)
    rep = ravivarapu_inference_gates(
        traces["baseline"], traces["paper"], carrier_hz=INFERENCE_CARRIER_50HZ, n_expected=ABLATION_EVAL_STEPS + 1
    )
    gates = {k: bool(v) for k, v in rep["gates"].items()}
    fails = [k for k, v in gates.items() if not v]
    print(f"Fig 5a gates {len(gates) - len(fails)}/{len(gates)}; fails {fails}")
    (args.cache_dir / "inference_5a.json").write_text(
        json.dumps({"traces": traces, "actions": actions, "gates": gates}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
