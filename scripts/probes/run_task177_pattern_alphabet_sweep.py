#!/usr/bin/env python3
"""TASK-177: Sweep alternative pattern alphabet constructions at 30 Hz.

Compares jitter widths, alphabet sizes, burst/random/alternating constructions.
For each: 1-step Pβ landscape + Fig 5b constant-policy oracle vs no-stim.

Artifacts:
  artifacts/ddpg/task177_pattern_alphabet_sweep.json
  artifacts/ddpg/task177_pattern_alphabet_sweep.log

Run:
  uv run python scripts/probes/run_task177_pattern_alphabet_sweep.py
  uv run python scripts/probes/run_task177_pattern_alphabet_sweep.py --only jitter_half
  tmux new-session -d -s task177-sweep \\
    'uv run python scripts/probes/run_task177_pattern_alphabet_sweep.py 2>&1 | tee artifacts/ddpg/task177_pattern_alphabet_sweep.log'
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
)
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "task177_pattern_alphabet_sweep.json"

PAPER_DT_MS = 0.02
MEAN_HZ = 30.0
STATE_LENGTH = 16
PROBE_SEED = 0
EVAL_STEPS = 5
NO_STIM_REF = 503.4  # from TASK-176


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    construction: str
    factory: Callable[[], PatternAlphabetLike]


def _experiment_specs() -> list[ExperimentSpec]:
    dt = PAPER_DT_MS
    hz = MEAN_HZ

    def jitter(jf: float, n: int = 41) -> Callable[[], PatternAlphabetLike]:
        return lambda: FixedMeanPatternAlphabet(
            mean_hz=hz, dt_ms=dt, n_patterns=n, jitter_fraction=jf
        )

    specs = [
        ExperimentSpec("jitter_third", "±1/3 ISI jitter (baseline)", "jitter", jitter(1.0 / 3.0)),
        ExperimentSpec("jitter_half", "±1/2 ISI jitter", "jitter", jitter(0.5)),
        ExperimentSpec("jitter_two_thirds", "±2/3 ISI jitter", "jitter", jitter(2.0 / 3.0)),
        ExperimentSpec("jitter_full", "±1/1 ISI jitter (full random interior)", "jitter", jitter(1.0)),
        ExperimentSpec("n100", "100 patterns, ±1/3 jitter", "size", jitter(1.0 / 3.0, 100)),
        ExperimentSpec("n200", "200 patterns, ±1/3 jitter", "size", jitter(1.0 / 3.0, 200)),
        ExperimentSpec("n500", "500 patterns, ±1/3 jitter", "size", jitter(1.0 / 3.0, 500)),
        ExperimentSpec(
            "burst",
            "burst clusters (2–6 pulses @ 60–120 Hz)",
            "burst",
            lambda: BurstPatternAlphabet(mean_hz=hz, dt_ms=dt),
        ),
        ExperimentSpec(
            "random",
            "random ISI trains (full range)",
            "random",
            lambda: RandomPulseAlphabet(mean_hz=hz, dt_ms=dt),
        ),
        ExperimentSpec(
            "alternating",
            "alternating short/long ISI",
            "alternating",
            lambda: AlternatingPatternAlphabet(mean_hz=hz, dt_ms=dt),
        ),
    ]
    return specs


def _no_stim_p_beta(*, plant_cfg) -> float:
    env_cfg = MehreganEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        action_space_mode="scalar_frequency",
    )
    plant = PythonPlant(config=plant_cfg)
    env = MehreganEnv(plant=plant, config=env_cfg)
    try:
        env.reset(seed=PROBE_SEED)
        _obs, _reward, _term, _trunc, info = env.step(0)
        return float(info["p_beta_raw"])
    finally:
        env.close()


def _make_env(*, plant_cfg, alphabet: PatternAlphabetLike) -> MehreganEnv:
    env_cfg = MehreganEnvConfig(
        state_mode="within_step",
        state_length=STATE_LENGTH,
        reward_state_mode="full_segment",
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS,
    )
    plant = PythonPlant(config=plant_cfg)
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=alphabet)


def _one_step_landscape(
    env: MehreganEnv,
    alphabet: PatternAlphabetLike,
    *,
    no_stim: float,
) -> list[dict[str, Any]]:
    for i in range(alphabet.n_actions):
        alphabet.idbs_for_pattern(i)

    rows: list[dict[str, Any]] = []
    for action in range(alphabet.n_actions):
        env.reset(seed=PROBE_SEED)
        _obs, reward, _term, _trunc, info = env.step(action)
        p_beta = float(info["p_beta_raw"])
        rows.append(
            {
                "action": action,
                "reward": float(reward),
                "p_beta_raw": p_beta,
                "beat_no_stim": p_beta < no_stim,
            }
        )
    return rows


def _fig5b_eval(env: MehreganEnv, *, action: int) -> dict[str, Any]:
    _, info = env.reset(seed=PROBE_SEED)
    p_beta = [float(info.get("p_beta_raw", np.nan))]
    for _ in range(EVAL_STEPS):
        _, _reward, terminated, truncated, info = env.step(action)
        p_beta.append(float(info.get("p_beta_raw", np.nan)))
        if terminated or truncated:
            break
    return {
        "action": action,
        "p_beta": p_beta,
        "p_beta_mean": float(np.mean(p_beta)),
    }


def _summarize(
    *,
    spec: ExperimentSpec,
    one_step: list[dict[str, Any]],
    fig5b: list[dict[str, Any]],
    no_stim: float,
    elapsed_s: float,
) -> dict[str, Any]:
    beat_1 = [r for r in one_step if r["beat_no_stim"]]
    beat_f = [r for r in fig5b if r["p_beta_mean"] < no_stim]
    best_1 = min(one_step, key=lambda r: r["p_beta_raw"])
    best_f = min(fig5b, key=lambda r: r["p_beta_mean"])
    return {
        "key": spec.key,
        "label": spec.label,
        "construction": spec.construction,
        "alphabet_size": len(one_step),
        "no_stim_p_beta": no_stim,
        "one_step": {
            "n_beat_no_stim": len(beat_1),
            "best_action": int(best_1["action"]),
            "best_p_beta": float(best_1["p_beta_raw"]),
            "beaters": [
                {"action": int(r["action"]), "p_beta_raw": float(r["p_beta_raw"])}
                for r in sorted(beat_1, key=lambda x: x["p_beta_raw"])
            ],
        },
        "fig5b": {
            "n_beat_no_stim": len(beat_f),
            "best_action": int(best_f["action"]),
            "best_p_beta_mean": float(best_f["p_beta_mean"]),
            "best_p_beta_1step": float(
                next(r["p_beta_raw"] for r in one_step if r["action"] == best_f["action"])
            ),
            "beaters": [
                {
                    "action": int(r["action"]),
                    "p_beta_mean": float(r["p_beta_mean"]),
                }
                for r in sorted(beat_f, key=lambda x: x["p_beta_mean"])
            ],
        },
        "elapsed_s": round(elapsed_s, 2),
    }


def _comparison_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "construction": r["label"],
            "alphabet_size": r["alphabet_size"],
            "beat_no_stim_1step": r["one_step"]["n_beat_no_stim"],
            "beat_no_stim_fig5b": r["fig5b"]["n_beat_no_stim"],
            "best_p_beta_1step": r["one_step"]["best_p_beta"],
            "best_p_beta_fig5b": r["fig5b"]["best_p_beta_mean"],
            "best_action_1step": r["one_step"]["best_action"],
            "best_action_fig5b": r["fig5b"]["best_action"],
        }
        for r in results
    ]


def run_experiment(
    spec: ExperimentSpec,
    *,
    plant_cfg,
    no_stim: float,
    skip_fig5b: bool = False,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"\n=== {spec.key}: {spec.label} ===", flush=True)
    alphabet = spec.factory()
    env = _make_env(plant_cfg=plant_cfg, alphabet=alphabet)
    try:
        print(f"  1-step landscape ({alphabet.n_actions} patterns)...", flush=True)
        one_step = _one_step_landscape(env, alphabet, no_stim=no_stim)
        beat_1 = sum(1 for r in one_step if r["beat_no_stim"])
        best_1 = min(one_step, key=lambda r: r["p_beta_raw"])
        print(
            f"  1-step: {beat_1}/{alphabet.n_actions} beat no-stim; "
            f"best action {best_1['action']} Pβ={best_1['p_beta_raw']:.1f}",
            flush=True,
        )

        fig5b_rows: list[dict[str, Any]] = []
        if not skip_fig5b:
            print(f"  Fig5b oracle ({alphabet.n_actions} patterns)...", flush=True)
            for action in range(alphabet.n_actions):
                row = _fig5b_eval(env, action=action)
                row["p_beta_raw_1step"] = one_step[action]["p_beta_raw"]
                row["beat_no_stim_fig5b"] = row["p_beta_mean"] < no_stim
                fig5b_rows.append(row)
                if action % 10 == 0 or action == alphabet.n_actions - 1:
                    print(
                        f"    action {action:3d}/{alphabet.n_actions - 1}: "
                        f"fig5b_mean={row['p_beta_mean']:.1f}",
                        flush=True,
                    )
        else:
            for action, r in enumerate(one_step):
                fig5b_rows.append(
                    {
                        "action": action,
                        "p_beta_mean": float("nan"),
                        "p_beta_raw_1step": r["p_beta_raw"],
                        "beat_no_stim_fig5b": False,
                    }
                )
    finally:
        env.close()

    summary = _summarize(
        spec=spec,
        one_step=one_step,
        fig5b=fig5b_rows,
        no_stim=no_stim,
        elapsed_s=time.time() - t0,
    )
    if not skip_fig5b:
        print(
            f"  Fig5b: {summary['fig5b']['n_beat_no_stim']}/{alphabet.n_actions} beat; "
            f"best action {summary['fig5b']['best_action']} "
            f"Pβ={summary['fig5b']['best_p_beta_mean']:.1f}",
            flush=True,
        )
    print(f"  elapsed={summary['elapsed_s']}s", flush=True)
    return {
        "summary": summary,
        "one_step": one_step,
        "fig5b": fig5b_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        metavar="KEY",
        help="Run only these experiment keys (repeatable). Default: all.",
    )
    parser.add_argument(
        "--skip-fig5b",
        action="store_true",
        help="1-step only (faster smoke / large alphabets).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments already present in output JSON.",
    )
    args = parser.parse_args()

    t0 = time.time()
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)

    specs = _experiment_specs()
    if args.only:
        keys = set(args.only)
        specs = [s for s in specs if s.key in keys]
        if not specs:
            print(f"No experiments match --only {args.only}", file=sys.stderr)
            return 1

    existing: dict[str, Any] = {}
    if args.resume and OUT_JSON.is_file():
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        done_keys = {r["summary"]["key"] for r in existing.get("experiments", [])}
        specs = [s for s in specs if s.key not in done_keys]
        print(f"Resume: skipping {len(done_keys)} completed experiments", flush=True)

    print("=== TASK-177: pattern alphabet alternative sweep @ 30 Hz ===", flush=True)
    no_stim = _no_stim_p_beta(plant_cfg=plant_cfg)
    print(f"no-stim Pβ (measured): {no_stim:.2f} (ref {NO_STIM_REF})", flush=True)

    experiments: list[dict[str, Any]] = list(existing.get("experiments", []))
    for spec in specs:
        payload = run_experiment(
            spec,
            plant_cfg=plant_cfg,
            no_stim=no_stim,
            skip_fig5b=args.skip_fig5b,
        )
        experiments.append(payload)
        # Incremental save after each experiment
        partial = {
            "task": "TASK-177",
            "mean_hz": MEAN_HZ,
            "plant_dt_ms": PAPER_DT_MS,
            "state_length": STATE_LENGTH,
            "seed": PROBE_SEED,
            "no_stim_p_beta": no_stim,
            "experiments": experiments,
            "comparison_table": _comparison_table([e["summary"] for e in experiments]),
            "elapsed_s": round(time.time() - t0, 2),
            "in_progress": True,
        }
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    summaries = [e["summary"] for e in experiments]
    any_beat_1 = any(s["one_step"]["n_beat_no_stim"] > 0 for s in summaries)
    any_beat_f = any(s["fig5b"]["n_beat_no_stim"] > 0 for s in summaries)

    result = {
        "task": "TASK-177",
        "mean_hz": MEAN_HZ,
        "plant_dt_ms": PAPER_DT_MS,
        "state_length": STATE_LENGTH,
        "reward_state_mode": "full_segment",
        "seed": PROBE_SEED,
        "no_stim_p_beta": no_stim,
        "experiments": experiments,
        "comparison_table": _comparison_table(summaries),
        "conclusions": {
            "any_construction_beats_no_stim_1step": any_beat_1,
            "any_construction_beats_no_stim_fig5b": any_beat_f,
            "escalate_plant_parity_if_none": not any_beat_1 and not any_beat_f,
        },
        "elapsed_s": round(time.time() - t0, 2),
        "in_progress": False,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\n=== Comparison table ===", flush=True)
    for row in result["comparison_table"]:
        print(
            f"  {row['construction'][:40]:40s} "
            f"n={row['alphabet_size']:3d} "
            f"1-step beat={row['beat_no_stim_1step']:3d} "
            f"fig5b beat={row['beat_no_stim_fig5b']:3d} "
            f"best1={row['best_p_beta_1step']:.1f} "
            f"bestF={row['best_p_beta_fig5b']:.1f}",
            flush=True,
        )
    print(json.dumps(result["conclusions"], indent=2), flush=True)
    print(f"\nWrote {OUT_JSON} (total {result['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
