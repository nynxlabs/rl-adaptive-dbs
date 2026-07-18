#!/usr/bin/env python3
"""TASK-158: Q6 sanity under within-step temporal state (no training)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.lib.pattern_reward_landscape import describe_pattern, run_landscape

PAPER_DT_MS = 0.02
ARTIFACTS = Path("artifacts/ddpg")
PROBE_SEED = 0
STATE_LENGTH = 16


def main() -> int:
    results: dict[str, object] = {
        "state_mode": "within_step",
        "state_length": STATE_LENGTH,
        "plant_dt_ms": PAPER_DT_MS,
        "seed": PROBE_SEED,
    }
    for mean_hz in (30.0, 45.0):
        landscape = run_landscape(
            seed=PROBE_SEED,
            mean_hz=mean_hz,
            state_length=STATE_LENGTH,
            plant_dt_ms=PAPER_DT_MS,
        )
        ranked = sorted(
            landscape["patterns"],
            key=lambda row: float(row["reward"]),
            reverse=True,
        )
        pattern0_rank = next(
            idx + 1
            for idx, row in enumerate(ranked)
            if int(row["action"]) == 0
        )
        best = ranked[0]
        results[f"{int(mean_hz)}hz"] = {
            "pattern0_rank": pattern0_rank,
            "pattern0_reward": next(
                float(row["reward"]) for row in ranked if int(row["action"]) == 0
            ),
            "best_action": int(best["action"]),
            "best_reward": float(best["reward"]),
            "best_semantics": describe_pattern(int(best["action"]), mean_hz=mean_hz),
            "n_patterns": len(ranked),
        }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"q6_within_step_L{STATE_LENGTH}_30hz.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {out}", file=sys.stderr)

    at_30 = results["30hz"]
    assert isinstance(at_30, dict)
    if int(at_30["pattern0_rank"]) == int(at_30["n_patterns"]):
        print("Q6 direction OK @ 30 Hz: pattern 0 worst", file=sys.stderr)
        return 0
    print(
        f"WARN: pattern 0 rank {at_30['pattern0_rank']} / {at_30['n_patterns']} (expected worst)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
