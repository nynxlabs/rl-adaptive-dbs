#!/usr/bin/env python3
"""Fig 5b alphabet redesign — 1-step open-loop oracle across constructions.

Motivation (TASK-176/177 + switching oracle):
  - Current ±1/3 ISI 41-pattern alphabet: 0/41 beat no-stim at dt=0.02
  - Switching among its top-8 also fails
  - Continuous periodic Hz: 14–34 Hz all lose; ≥35 Hz mostly wins
  - Paper Fig 5b: irregular 30 Hz-mean works because *instantaneous* rate
    leaves the beta band

This probe keeps mean rate = 30 Hz (fixed pulse count) and compares
constructions. 1-step only (~10 s/pattern); escalate Fig 5b / switching only
for families with ≥1 beater.

Families:
  baseline          FixedMeanPatternAlphabet ±1/3 ISI
  jitter_full       ±1 ISI jitter
  burst             TASK-177 BurstPatternAlphabet (60–120 Hz clusters)
  alternating       short/long ISI
  random            full-range interior jitter
  therapeutic_burst NEW: 90–160 Hz packets + silence (paper-aligned)

Run:
  tmux new-session -d -s fig5b-alphabet-oracle \\
    "setsid nohup uv run python scripts/probes/run_fig5b_alphabet_redesign_oracle.py \\
      >> logs/fig5b-alphabet-oracle.log 2>&1 < /dev/null"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.pattern_alternatives import (
    AlternatingPatternAlphabet,
    BurstPatternAlphabet,
    PatternAlphabetLike,
    RandomPulseAlphabet,
    TherapeuticBurstAlphabet,
)
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

PAPER_DT_MS = 0.02
MEAN_HZ = 30.0
SEED = 0
ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "fig5b_alphabet_redesign_oracle_30hz.json"
MAIN_OUT = Path("/home/nynxbox/neuroengineering/rl-adaptive-dbs") / OUT_JSON


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    factory: Callable[[], PatternAlphabetLike]


def _families() -> list[Family]:
    dt = PAPER_DT_MS
    hz = MEAN_HZ
    return [
        Family(
            "baseline",
            "±1/3 ISI jitter (current default)",
            lambda: FixedMeanPatternAlphabet(mean_hz=hz, dt_ms=dt, jitter_fraction=1.0 / 3.0),
        ),
        Family(
            "jitter_full",
            "±1 ISI jitter",
            lambda: FixedMeanPatternAlphabet(mean_hz=hz, dt_ms=dt, jitter_fraction=1.0),
        ),
        Family(
            "burst",
            "burst 2–6 @ 60–120 Hz (TASK-177)",
            lambda: BurstPatternAlphabet(mean_hz=hz, dt_ms=dt),
        ),
        Family(
            "alternating",
            "alternating short/long ISI",
            lambda: AlternatingPatternAlphabet(mean_hz=hz, dt_ms=dt),
        ),
        Family(
            "random",
            "random interior ISI (full range)",
            lambda: RandomPulseAlphabet(mean_hz=hz, dt_ms=dt),
        ),
        Family(
            "therapeutic_burst",
            "therapeutic packets 90–160 Hz + silence (NEW)",
            lambda: TherapeuticBurstAlphabet(mean_hz=hz, dt_ms=dt),
        ),
    ]


def _pulse_count(alphabet: PatternAlphabetLike, index: int) -> int:
    trace = np.asarray(alphabet.idbs_for_pattern(index))
    active = trace > 0.0
    rising = np.concatenate(([active[0]], active[1:] & ~active[:-1]))
    return int(np.count_nonzero(rising))


def _no_stim_p_beta(plant_cfg) -> float:
    env = MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=MehreganEnvConfig(
            state_length=1,
            action_space_mode="scalar_frequency",
        ),
    )
    try:
        env.reset(seed=SEED)
        _obs, _r, _t, _tr, info = env.step(0)
        return float(info["p_beta_raw"])
    finally:
        env.close()


def _one_step_landscape(
    *,
    plant_cfg,
    alphabet: PatternAlphabetLike,
    no_stim: float,
) -> dict[str, Any]:
    env = MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=MehreganEnvConfig(
            state_length=1,
            step_duration_s=2.0,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=MEAN_HZ,
            max_episode_steps=1,
        ),
        alphabet=alphabet,
    )
    rows: list[dict[str, Any]] = []
    try:
        for action in range(alphabet.n_actions):
            t0 = time.time()
            env.reset(seed=SEED)
            _obs, reward, _t, _tr, info = env.step(action)
            p_beta = float(info["p_beta_raw"])
            rows.append(
                {
                    "action": action,
                    "reward": float(reward),
                    "p_beta_raw": p_beta,
                    "beat_no_stim": p_beta < no_stim,
                    "pulse_count": _pulse_count(alphabet, action),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            flag = "BEAT" if rows[-1]["beat_no_stim"] else "lose"
            print(
                f"    a={action:3d} Pβ={p_beta:7.1f} pulses={rows[-1]['pulse_count']} {flag}",
                flush=True,
            )
    finally:
        env.close()

    beaters = [r for r in rows if r["beat_no_stim"]]
    best = min(rows, key=lambda r: r["p_beta_raw"])
    pulse0 = rows[0]["pulse_count"]
    pulse_ok = all(r["pulse_count"] == pulse0 for r in rows)
    return {
        "n_patterns": len(rows),
        "n_beat_no_stim": len(beaters),
        "best_action": int(best["action"]),
        "best_p_beta_raw": float(best["p_beta_raw"]),
        "pulse_count_pattern0": pulse0,
        "pulse_count_preserved": pulse_ok,
        "beaters": [
            {"action": r["action"], "p_beta_raw": r["p_beta_raw"]}
            for r in sorted(beaters, key=lambda x: x["p_beta_raw"])
        ],
        "patterns": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Family keys to run (default: all). Example: --only therapeutic_burst burst",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the known-fail ±1/3 ISI baseline family.",
    )
    args = parser.parse_args()

    t0 = time.time()
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    selected = _families()
    if args.skip_baseline:
        selected = [f for f in selected if f.key != "baseline"]
    if args.only:
        by_key = {f.key: f for f in selected}
        missing = [k for k in args.only if k not in by_key]
        if missing:
            raise SystemExit(f"unknown --only keys: {missing}")
        selected = [by_key[k] for k in args.only]

    print("=== Fig 5b alphabet redesign oracle @ 30 Hz (1-step) ===", flush=True)
    print(f"families: {[f.key for f in selected]}", flush=True)
    no_stim = _no_stim_p_beta(plant_cfg)
    print(f"no-stim Pβ={no_stim:.2f}", flush=True)

    families_out: dict[str, Any] = {}
    comparison: list[dict[str, Any]] = []

    for fam in selected:
        print(f"\n=== {fam.key}: {fam.label} ===", flush=True)
        alphabet = fam.factory()
        t1 = time.time()
        landscape = _one_step_landscape(
            plant_cfg=plant_cfg, alphabet=alphabet, no_stim=no_stim
        )
        landscape["elapsed_s"] = round(time.time() - t1, 2)
        families_out[fam.key] = {
            "label": fam.label,
            "landscape": landscape,
        }
        row = {
            "key": fam.key,
            "label": fam.label,
            "n_beat_no_stim": landscape["n_beat_no_stim"],
            "best_action": landscape["best_action"],
            "best_p_beta_raw": landscape["best_p_beta_raw"],
            "pulse_count_preserved": landscape["pulse_count_preserved"],
            "delta_vs_nostim": landscape["best_p_beta_raw"] - no_stim,
        }
        comparison.append(row)
        print(
            f"  → beat {landscape['n_beat_no_stim']}/{landscape['n_patterns']}; "
            f"best a={landscape['best_action']} Pβ={landscape['best_p_beta_raw']:.1f} "
            f"(Δ={row['delta_vs_nostim']:+.1f})",
            flush=True,
        )

        # Checkpoint after each family so a kill still leaves evidence.
        payload = {
            "probe": "fig5b_alphabet_redesign_oracle",
            "mean_hz": MEAN_HZ,
            "plant_dt_ms": PAPER_DT_MS,
            "seed": SEED,
            "protocol": "single 2s open-loop step per pattern",
            "no_stim_p_beta": no_stim,
            "families": families_out,
            "comparison_table": comparison,
            "in_progress": True,
            "elapsed_s": round(time.time() - t0, 2),
        }
        text = json.dumps(payload, indent=2) + "\n"
        OUT_JSON.write_text(text, encoding="utf-8")
        MAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
        MAIN_OUT.write_text(text, encoding="utf-8")

    winners = [r for r in comparison if r["n_beat_no_stim"] > 0]
    result = {
        "probe": "fig5b_alphabet_redesign_oracle",
        "mean_hz": MEAN_HZ,
        "plant_dt_ms": PAPER_DT_MS,
        "seed": SEED,
        "protocol": "single 2s open-loop step per pattern",
        "no_stim_p_beta": no_stim,
        "families": families_out,
        "comparison_table": comparison,
        "summary": {
            "n_families": len(comparison),
            "n_families_with_beater": len(winners),
            "winners": winners,
            "best_family": min(comparison, key=lambda r: r["best_p_beta_raw"]),
            "structural_blocker_all_families": len(winners) == 0,
        },
        "in_progress": False,
        "elapsed_s": round(time.time() - t0, 2),
    }
    text = json.dumps(result, indent=2) + "\n"
    OUT_JSON.write_text(text, encoding="utf-8")
    MAIN_OUT.write_text(text, encoding="utf-8")

    print(f"\nWrote {OUT_JSON}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"total elapsed={result['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
