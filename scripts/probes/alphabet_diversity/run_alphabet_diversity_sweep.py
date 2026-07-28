#!/usr/bin/env python3
"""Open-loop alphabet diversity sweep at 30 Hz and 45 Hz (1-step landscapes).

Goal (Fig 6a / quantization): avoid a single dominant action so fp32 and PTQ
do not all collapse to the exact same constant action. Traces may still overlap
visually; action sequences should not be identical locks.

For each construction × mean_hz, report:
  - best / 2nd / 3rd P_beta (and margins)
  - how many irregular patterns sit within ``near_best_frac`` of the best
  - skip_regular view (exclude pattern 0) — matches Fig 5a/6a training space
  - whether pattern 0 is best (45 Hz) or worst-ish (30 Hz paper claim)

Constructions reuse TASK-177 families in ``pattern_alternatives`` plus jitter widths.

Run (tmux, 2 threads):
  tmux new-session -d -s alphabet-diversity \\
    \"setsid nohup uv run python -m rl_adaptive_dbs.run --max-threads 2 \\
      scripts/probes/alphabet_diversity/run_alphabet_diversity_sweep.py \\
      >> logs/alphabet-diversity.log 2>&1 < /dev/null\"
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

from envs.mehregan.extensions.alphabet_diversity.config import WithinStepEnvConfig
from envs.mehregan.extensions.alphabet_diversity.env import WithinStepMehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.extensions.alphabet_diversity.near_hub import NearHubBurstAlphabet
from envs.mehregan.pattern_alternatives import (
    AlternatingPatternAlphabet,
    BurstPatternAlphabet,
    PatternAlphabetLike,
    RandomPulseAlphabet,
)
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "alphabet_diversity_sweep.json"
NEAR_HUB_OUT_JSON = ARTIFACTS / "alphabet_diversity_near_hub.json"
PAPER_DT_MS = 0.02
PROBE_SEED = 0
NEAR_BEST_FRAC = 0.02  # within 2% of best P_beta counts as "near-best"


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    construction: str
    factory: Callable[[float], PatternAlphabetLike]


def _specs() -> list[Spec]:
    def jitter(jf: float, n: int = 41) -> Callable[[float], PatternAlphabetLike]:
        return lambda hz: FixedMeanPatternAlphabet(
            mean_hz=hz,
            dt_ms=PAPER_DT_MS,
            n_patterns=n,
            jitter_fraction=jf,
        )

    def burst(n: int) -> Callable[[float], PatternAlphabetLike]:
        return lambda hz: BurstPatternAlphabet(
            mean_hz=hz, dt_ms=PAPER_DT_MS, n_patterns=n
        )

    def near_hub(n_per_hub: int) -> Callable[[float], PatternAlphabetLike]:
        return lambda hz: NearHubBurstAlphabet(
            mean_hz=hz, dt_ms=PAPER_DT_MS, n_per_hub=n_per_hub
        )

    return [
        Spec("jitter_third", "±1/3 ISI (current default)", "jitter", jitter(1.0 / 3.0)),
        Spec("jitter_half", "±1/2 ISI", "jitter", jitter(0.5)),
        Spec("jitter_two_thirds", "±2/3 ISI", "jitter", jitter(2.0 / 3.0)),
        Spec("jitter_full", "±1 ISI (full interior)", "jitter", jitter(1.0)),
        Spec(
            "burst",
            "burst clusters 2–6 @ 60–120 Hz (n=41)",
            "burst",
            burst(41),
        ),
        Spec(
            "burst_n128",
            "burst clusters + seeded phase/gap (n=128)",
            "burst",
            burst(128),
        ),
        Spec(
            "burst_n256",
            "burst clusters + seeded phase/gap (n=256)",
            "burst",
            burst(256),
        ),
        Spec(
            "jitter_third_n128",
            "±1/3 ISI (n=128)",
            "jitter",
            jitter(1.0 / 3.0, 128),
        ),
        Spec(
            "jitter_third_n256",
            "±1/3 ISI (n=256)",
            "jitter",
            jitter(1.0 / 3.0, 256),
        ),
        Spec(
            "near_hub_n257",
            "phase/gap near-copies of top burst hubs (8×32+reg)",
            "near_hub",
            near_hub(32),
        ),
        Spec(
            "near_hub_n513",
            "phase/gap near-copies of top burst hubs (8×64+reg)",
            "near_hub",
            near_hub(64),
        ),
        Spec(
            "random",
            "random ISI (full range)",
            "random",
            lambda hz: RandomPulseAlphabet(mean_hz=hz, dt_ms=PAPER_DT_MS),
        ),
        Spec(
            "alternating",
            "alternating short/long ISI",
            "alternating",
            lambda hz: AlternatingPatternAlphabet(mean_hz=hz, dt_ms=PAPER_DT_MS),
        ),
    ]


def _no_stim_p_beta(*, plant_cfg, mean_hz: float) -> float:
    _ = mean_hz
    env_cfg = WithinStepEnvConfig(
        state_length=1,
        action_space_mode="scalar_frequency",
        max_episode_steps=1,
    )
    env = WithinStepMehreganEnv(plant=PythonPlant(config=plant_cfg), config=env_cfg)
    try:
        env.reset(seed=PROBE_SEED)
        _obs, _r, _t, _tr, info = env.step(0)
        return float(info["p_beta_raw"])
    finally:
        env.close()


def _one_step_rows(
    *,
    plant_cfg,
    alphabet: PatternAlphabetLike,
    mean_hz: float,
) -> list[dict[str, Any]]:
    env_cfg = WithinStepEnvConfig(
        state_length=1,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
        max_episode_steps=1,
    )
    env = WithinStepMehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=alphabet,  # type: ignore[arg-type]
    )
    try:
        rows: list[dict[str, Any]] = []
        for action in range(alphabet.n_actions):
            env.reset(seed=PROBE_SEED)
            try:
                _obs, reward, _t, _tr, info = env.step(action)
                rows.append(
                    {
                        "action": int(action),
                        "p_beta_raw": float(info["p_beta_raw"]),
                        "reward": float(reward),
                    }
                )
            except Exception as exc:  # noqa: BLE001 — probe must finish the grid
                print(
                    f"  WARN action={action} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                rows.append(
                    {
                        "action": int(action),
                        "p_beta_raw": float("nan"),
                        "reward": float("nan"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return rows
    finally:
        env.close()


def _diversity_metrics(
    rows: list[dict[str, Any]],
    *,
    no_stim: float,
    skip_regular: bool,
    n_unique_traces: int | None = None,
) -> dict[str, Any]:
    if skip_regular:
        # pattern 0 = regular; agent space is actions 1..n-1 remapped, but here
        # rows use internal pattern indices 0..n-1 — drop index 0.
        use = [r for r in rows if r["action"] != 0]
    else:
        use = list(rows)
    # Drop failed plant steps (NaN) so one bad action cannot poison ranking.
    use = [
        r
        for r in use
        if r.get("p_beta_raw") is not None and np.isfinite(r["p_beta_raw"])
    ]
    # Larger alphabets: want more near-ties (≥5 when scoring ≥100 irregulars).
    min_near = 5 if len(use) >= 100 else 3
    if not use:
        return {
            "n_patterns_scored": 0,
            "best_action": None,
            "best_p_beta": None,
            "second_action": None,
            "second_p_beta": None,
            "third_action": None,
            "third_p_beta": None,
            "margin_best_second": None,
            "margin_best_third": None,
            "n_near_best": 0,
            "near_best_actions": [],
            "near_best_frac": NEAR_BEST_FRAC,
            "min_near_for_ok": min_near,
            "n_beat_no_stim": 0,
            "regular_p_beta": None,
            "regular_is_best": None,
            "best_irregular_p_beta": None,
            "spread_irr_max_minus_min": None,
            "n_unique_traces": n_unique_traces,
            "diversity_ok": False,
            "n_errors": sum(1 for r in rows if r.get("error")),
        }

    ranked = sorted(use, key=lambda r: r["p_beta_raw"])
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else best
    third = ranked[2] if len(ranked) > 2 else second
    best_pb = float(best["p_beta_raw"])
    near = [
        r
        for r in ranked
        if r["p_beta_raw"] <= best_pb * (1.0 + NEAR_BEST_FRAC)
    ]
    irr = [r for r in rows if r["action"] != 0]
    reg = next((r for r in rows if r["action"] == 0), None)
    return {
        "n_patterns_scored": len(use),
        "best_action": int(best["action"]),
        "best_p_beta": best_pb,
        "second_action": int(second["action"]),
        "second_p_beta": float(second["p_beta_raw"]),
        "third_action": int(third["action"]),
        "third_p_beta": float(third["p_beta_raw"]),
        "margin_best_second": float(second["p_beta_raw"] - best_pb),
        "margin_best_third": float(third["p_beta_raw"] - best_pb),
        "n_near_best": len(near),
        "near_best_actions": [int(r["action"]) for r in near[:12]],
        "near_best_frac": NEAR_BEST_FRAC,
        "min_near_for_ok": min_near,
        "n_beat_no_stim": sum(1 for r in use if r["p_beta_raw"] < no_stim),
        "regular_p_beta": float(reg["p_beta_raw"]) if reg else None,
        "regular_is_best": bool(reg and reg["action"] == best["action"])
        if not skip_regular
        else None,
        "best_irregular_p_beta": float(min(irr, key=lambda r: r["p_beta_raw"])["p_beta_raw"])
        if irr
        else None,
        "spread_irr_max_minus_min": (
            float(
                max(r["p_beta_raw"] for r in irr) - min(r["p_beta_raw"] for r in irr)
            )
            if irr
            else None
        ),
        "n_unique_traces": n_unique_traces,
        # Soft score: several near-best + small margin (not a single spike winner).
        "diversity_ok": (
            len(near) >= min_near and float(second["p_beta_raw"] - best_pb) <= 15.0
        ),
    }


def _count_unique_traces(alphabet: PatternAlphabetLike) -> int:
    """Count distinct STN drive waveforms (catches modulo-duplicate alphabets)."""
    seen: set[int] = set()
    n = int(getattr(alphabet, "n_patterns", alphabet.n_actions))
    for j in range(n):
        seen.add(hash(alphabet.idbs_for_pattern(j).tobytes()))
    return len(seen)


def run_one(
    spec: Spec,
    *,
    mean_hz: float,
    plant_cfg,
    no_stim: float,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n=== {mean_hz:g} Hz | {spec.key}: {spec.label} ===", flush=True)
    alphabet = spec.factory(mean_hz)
    n_unique = _count_unique_traces(alphabet)
    print(
        f"  n_actions={alphabet.n_actions} n_unique_traces={n_unique}",
        flush=True,
    )
    rows = _one_step_rows(plant_cfg=plant_cfg, alphabet=alphabet, mean_hz=mean_hz)
    full = _diversity_metrics(
        rows, no_stim=no_stim, skip_regular=False, n_unique_traces=n_unique
    )
    skip = _diversity_metrics(
        rows, no_stim=no_stim, skip_regular=True, n_unique_traces=n_unique
    )
    elapsed = round(time.time() - t0, 2)
    print(
        f"  full: best={full['best_action']} Pβ={full['best_p_beta']:.1f} "
        f"margin2={full['margin_best_second']:.1f} near={full['n_near_best']} "
        f"diversity_ok={full['diversity_ok']}",
        flush=True,
    )
    print(
        f"  skip_regular: best={skip['best_action']} Pβ={skip['best_p_beta']:.1f} "
        f"margin2={skip['margin_best_second']:.1f} near={skip['n_near_best']} "
        f"diversity_ok={skip['diversity_ok']}",
        flush=True,
    )
    print(f"  elapsed={elapsed}s", flush=True)
    return {
        "key": spec.key,
        "label": spec.label,
        "construction": spec.construction,
        "mean_hz": mean_hz,
        "n_actions": alphabet.n_actions,
        "n_unique_traces": n_unique,
        "no_stim_p_beta": no_stim,
        "elapsed_s": elapsed,
        "full_alphabet": full,
        "skip_regular": skip,
        "one_step": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hz",
        type=float,
        action="append",
        default=None,
        help="Mean Hz to sweep (repeatable). Default: 30 and 45.",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="KEY",
        help="Run only these construction keys (repeatable).",
    )
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()
    hz_list = args.hz or [30.0, 45.0]
    specs = _specs()
    if args.only:
        want = set(args.only)
        specs = [s for s in specs if s.key in want]
        missing = want - {s.key for s in specs}
        if missing:
            print(f"unknown keys: {sorted(missing)}", file=sys.stderr)
            return 2

    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "figure_goal": "action_diversity_for_fig6a",
        "plant_dt_ms": PAPER_DT_MS,
        "seed": PROBE_SEED,
        "near_best_frac": NEAR_BEST_FRAC,
        "exit_criteria": {
            "skip_regular_diversity_ok": (
                ">=3 near-best (or >=5 when scoring >=100 irregulars) within 2% of "
                "best AND best–2nd margin <= 15 raw P_beta"
            ),
            "note": (
                "PTQ may still track fp32; success is avoiding a single locked action "
                "for fp32/PTQ after retrain on the chosen alphabet. Large-n burst "
                "indices ≥41 use seeded phase/gap (legacy 1–40 unchanged)."
            ),
        },
        "runs": [],
    }

    t_all = time.time()
    for hz in hz_list:
        no_stim = _no_stim_p_beta(plant_cfg=plant_cfg, mean_hz=hz)
        print(f"\n##### mean_hz={hz:g}  no_stim Pβ={no_stim:.1f} #####", flush=True)
        for spec in specs:
            payload["runs"].append(
                run_one(spec, mean_hz=hz, plant_cfg=plant_cfg, no_stim=no_stim)
            )
            args.out.write_text(json.dumps(payload, indent=2) + "\n")

    # Ranking: prefer skip_regular diversity_ok at 45 Hz, then smaller margin, more near-best.
    ranked = sorted(
        [r for r in payload["runs"] if r["mean_hz"] == 45.0],
        key=lambda r: (
            not r["skip_regular"]["diversity_ok"],
            r["skip_regular"]["margin_best_second"],
            -r["skip_regular"]["n_near_best"],
            r["skip_regular"]["best_p_beta"],
        ),
    )
    payload["ranking_45hz_skip_regular"] = [
        {
            "key": r["key"],
            "n_actions": r["n_actions"],
            "n_unique_traces": r.get("n_unique_traces"),
            "diversity_ok": r["skip_regular"]["diversity_ok"],
            "margin_best_second": r["skip_regular"]["margin_best_second"],
            "n_near_best": r["skip_regular"]["n_near_best"],
            "best_action": r["skip_regular"]["best_action"],
            "best_p_beta": r["skip_regular"]["best_p_beta"],
        }
        for r in ranked
    ]
    payload["elapsed_s"] = round(time.time() - t_all, 2)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    print("45 Hz skip_regular ranking:", flush=True)
    for row in payload["ranking_45hz_skip_regular"]:
        print(
            f"  {row['key']:22s} n={row['n_actions']:<4} "
            f"uniq={row.get('n_unique_traces')} ok={row['diversity_ok']} "
            f"margin2={row['margin_best_second']:.1f} near={row['n_near_best']} "
            f"best={row['best_action']} Pβ={row['best_p_beta']:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
