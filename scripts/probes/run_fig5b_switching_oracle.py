#!/usr/bin/env python3
"""Fig 5b multi-step switching oracle on the current 41-pattern alphabet.

Question: if we hand-pick a *sequence* of patterns (not just one constant
pattern), can post-onset mean Pβ beat no-stim at 30 Hz / plant.dt_ms=0.02?

Constant-policy oracle (TASK-176) already failed: 0/41 beat no-stim. This probe
adds open-loop *switching* search — still no DDPG.

Protocols (paper mehregan_eval style: 2 s reset + 5×2 s steps):
  1. no-stim baseline
  2. best constant from TASK-176 (action 19)
  3. greedy among TOP_K 1-step-best patterns (replay prefix each candidate)
  4. random sequences among TOP_K (N_RANDOM trials)

Run:
  tmux new-session -d -s fig5b-switch-oracle \\
    "setsid nohup uv run python scripts/probes/run_fig5b_switching_oracle.py \\
      >> logs/fig5b-switch-oracle.log 2>&1 < /dev/null"
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.plant.dbs import DbsSpec
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

PAPER_DT_MS = 0.02
MEAN_HZ = 30.0
SEED = 0
EVAL_STEPS = 5
# Top irregulars from task176 1-step / fig5b constant ranking at dt=0.02.
TOP_K_ACTIONS = (19, 12, 9, 37, 10, 4, 15, 11)
N_RANDOM = 16
RANDOM_SEED = 42

ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "fig5b_switching_oracle_30hz.json"


def _make_alphabet() -> FixedMeanPatternAlphabet:
    return FixedMeanPatternAlphabet(
        mean_hz=MEAN_HZ,
        step_duration_s=2.0,
        dt_ms=PAPER_DT_MS,
        skip_regular=False,
    )


def _make_env(alphabet: FixedMeanPatternAlphabet) -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = MehreganEnvConfig(
        state_length=1,
        step_duration_s=2.0,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=EVAL_STEPS,
        skip_regular=False,
    )
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=alphabet,
    )


def _summarize(p_beta: list[float]) -> dict[str, float]:
    arr = np.asarray(p_beta, dtype=np.float64)
    post = arr[1:]  # exclude reset segment
    return {
        "p_beta_mean_incl_reset": float(np.mean(arr)),
        "p_beta_mean_post_onset": float(np.mean(post)) if post.size else float("nan"),
        "p_beta_final": float(arr[-1]),
    }


def _rollout_actions(
    env: MehreganEnv,
    *,
    actions: list[int] | None,
    no_stim: bool = False,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """Reset + ``n_steps`` stim segments (default EVAL_STEPS)."""
    steps = EVAL_STEPS if n_steps is None else int(n_steps)
    _, info = env.reset(seed=SEED)
    p_beta = [float(info["p_beta_raw"])]
    taken: list[int | None] = []
    plant = env.unwrapped._plant  # noqa: SLF001 — probe-only no-stim path
    for i in range(steps):
        if no_stim:
            result = plant.integrate(
                env.config.step_duration_s,
                DbsSpec.none(),
                record_spikes=True,
            )
            if result.p_beta is None:
                raise RuntimeError("no-stim integrate missing p_beta")
            p_beta.append(float(result.p_beta))
            taken.append(None)
        else:
            assert actions is not None
            _obs, _r, term, trunc, info = env.step(int(actions[i]))
            p_beta.append(float(info["p_beta_raw"]))
            taken.append(int(actions[i]))
            if term or trunc:
                break
    return {
        "actions": taken,
        "p_beta": p_beta,
        **_summarize(p_beta),
    }


def _greedy_topk(env: MehreganEnv, *, candidates: tuple[int, ...]) -> dict[str, Any]:
    """At each step, try every candidate with the best prefix so far; pick lowest post Pβ."""
    t0 = time.time()
    prefix: list[int] = []
    step_choices: list[dict[str, Any]] = []

    for step_i in range(EVAL_STEPS):
        rows: list[dict[str, Any]] = []
        for action in candidates:
            trial = prefix + [action]
            roll = _rollout_actions(env, actions=trial, n_steps=len(trial))
            score = float(np.mean(roll["p_beta"][1:]))
            rows.append(
                {
                    "action": action,
                    "score_post_mean_so_far": score,
                    "p_beta_at_step": float(roll["p_beta"][-1]),
                }
            )
        best = min(rows, key=lambda r: r["score_post_mean_so_far"])
        prefix.append(int(best["action"]))
        step_choices.append(
            {
                "step": step_i,
                "chosen": int(best["action"]),
                "candidates": rows,
            }
        )
        print(
            f"  greedy step {step_i}: chose {best['action']} "
            f"(post_mean_so_far={best['score_post_mean_so_far']:.2f})",
            flush=True,
        )

    final = _rollout_actions(env, actions=prefix)
    return {
        "method": "greedy_topk",
        "candidates": list(candidates),
        "actions": prefix,
        "step_choices": step_choices,
        "elapsed_s": round(time.time() - t0, 2),
        **{
            k: final[k]
            for k in (
                "p_beta",
                "p_beta_mean_incl_reset",
                "p_beta_mean_post_onset",
                "p_beta_final",
            )
        },
    }


def _random_sequences(
    env: MehreganEnv,
    *,
    candidates: tuple[int, ...],
    n_trials: int,
) -> dict[str, Any]:
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    trials: list[dict[str, Any]] = []
    for i in range(n_trials):
        actions = [int(rng.choice(candidates)) for _ in range(EVAL_STEPS)]
        roll = _rollout_actions(env, actions=actions)
        trials.append({"trial": i, "actions": actions, **_summarize(roll["p_beta"]), "p_beta": roll["p_beta"]})
        print(
            f"  random {i}: actions={actions} post={trials[-1]['p_beta_mean_post_onset']:.2f}",
            flush=True,
        )
    best = min(trials, key=lambda r: r["p_beta_mean_post_onset"])
    return {
        "method": "random_topk",
        "candidates": list(candidates),
        "n_trials": n_trials,
        "best": best,
        "trials": trials,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main() -> int:
    t0 = time.time()
    alphabet = _make_alphabet()
    env = _make_env(alphabet)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=== Fig 5b switching oracle @ 30 Hz (current alphabet) ===", flush=True)
    print(f"plant.dt_ms={PAPER_DT_MS}, candidates={TOP_K_ACTIONS}", flush=True)

    try:
        print("\n--- no-stim baseline ---", flush=True)
        no_stim = _rollout_actions(env, actions=None, no_stim=True)
        print(
            f"  post_onset={no_stim['p_beta_mean_post_onset']:.2f} "
            f"incl_reset={no_stim['p_beta_mean_incl_reset']:.2f}",
            flush=True,
        )

        print("\n--- best constant (action 19) ---", flush=True)
        constant = _rollout_actions(env, actions=[19] * EVAL_STEPS)
        print(
            f"  post_onset={constant['p_beta_mean_post_onset']:.2f} "
            f"incl_reset={constant['p_beta_mean_incl_reset']:.2f}",
            flush=True,
        )

        print("\n--- greedy among TOP_K ---", flush=True)
        greedy = _greedy_topk(env, candidates=TOP_K_ACTIONS)
        print(
            f"  actions={greedy['actions']} "
            f"post_onset={greedy['p_beta_mean_post_onset']:.2f}",
            flush=True,
        )

        print(f"\n--- {N_RANDOM} random TOP_K sequences ---", flush=True)
        random = _random_sequences(env, candidates=TOP_K_ACTIONS, n_trials=N_RANDOM)
        print(
            f"  best actions={random['best']['actions']} "
            f"post_onset={random['best']['p_beta_mean_post_onset']:.2f}",
            flush=True,
        )
    finally:
        env.close()

    no_stim_post = float(no_stim["p_beta_mean_post_onset"])
    methods = {
        "constant_19": float(constant["p_beta_mean_post_onset"]),
        "greedy_topk": float(greedy["p_beta_mean_post_onset"]),
        "random_topk_best": float(random["best"]["p_beta_mean_post_onset"]),
    }
    beaters = {k: v for k, v in methods.items() if v < no_stim_post}

    result: dict[str, Any] = {
        "probe": "fig5b_switching_oracle",
        "mean_hz": MEAN_HZ,
        "plant_dt_ms": PAPER_DT_MS,
        "seed": SEED,
        "eval_steps": EVAL_STEPS,
        "protocol": "2s reset + 5×2s steps; score = mean Pβ on stim steps only",
        "alphabet": "FixedMeanPatternAlphabet (±1/3 ISI, 41 patterns)",
        "candidates": list(TOP_K_ACTIONS),
        "no_stim": no_stim,
        "constant_19": constant,
        "greedy": greedy,
        "random": random,
        "summary": {
            "no_stim_post_onset": no_stim_post,
            "methods_post_onset": methods,
            "n_methods_beat_no_stim": len(beaters),
            "beaters": beaters,
            "switching_beats_constant": methods["greedy_topk"] < methods["constant_19"]
            or methods["random_topk_best"] < methods["constant_19"],
            "structural_blocker_switching": len(beaters) == 0,
        },
        "elapsed_s": round(time.time() - t0, 2),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Also copy into main checkout artifacts for shared evidence.
    main_out = Path("/home/nynxbox/neuroengineering/rl-adaptive-dbs") / OUT_JSON
    main_out.parent.mkdir(parents=True, exist_ok=True)
    main_out.write_text(OUT_JSON.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\nWrote {OUT_JSON}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"total elapsed={result['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
