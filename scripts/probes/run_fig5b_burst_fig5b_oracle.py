#!/usr/bin/env python3
"""Fig 5b protocol constant-policy oracle on winning burst alphabets.

After 1-step redesign oracle: burst + therapeutic_burst each have 32/41
patterns with Pβ < no-stim. This probe checks the paper eval protocol
(2 s reset + 5×2 s steps) for:
  - no-stim
  - pattern 0 (regular 30 Hz)
  - top-K 1-step beaters per family
  - full 41-pattern screen for the best family (burst)

Run:
  tmux new-session -d -s fig5b-burst-fig5b \\
    "setsid nohup uv run python scripts/probes/run_fig5b_burst_fig5b_oracle.py \\
      >> logs/fig5b-burst-fig5b-oracle.log 2>&1 < /dev/null"
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.pattern_alternatives import (
    BurstPatternAlphabet,
    TherapeuticBurstAlphabet,
)
from envs.plant.dbs import DbsSpec
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

PAPER_DT_MS = 0.02
MEAN_HZ = 30.0
SEED = 0
EVAL_STEPS = 5
ARTIFACTS = Path("artifacts/ddpg")
OUT_JSON = ARTIFACTS / "fig5b_burst_fig5b_oracle_30hz.json"
MAIN_OUT = Path("/home/nynxbox/neuroengineering/rl-adaptive-dbs") / OUT_JSON

# Top beaters from fig5b_alphabet_redesign_oracle_30hz.json
TOP = {
    "burst": [5, 25, 2, 22, 3, 23, 4, 24],
    "therapeutic_burst": [12, 36, 5, 18, 11, 35, 4, 28],
}


def _factory(key: str):
    if key == "burst":
        return BurstPatternAlphabet(mean_hz=MEAN_HZ, dt_ms=PAPER_DT_MS)
    if key == "therapeutic_burst":
        return TherapeuticBurstAlphabet(mean_hz=MEAN_HZ, dt_ms=PAPER_DT_MS)
    raise ValueError(key)


def _make_env(alphabet) -> MehreganEnv:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    return MehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=MehreganEnvConfig(
            state_length=1,
            step_duration_s=2.0,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=MEAN_HZ,
            max_episode_steps=EVAL_STEPS,
        ),
        alphabet=alphabet,
    )


def _summarize(p_beta: list[float]) -> dict[str, float]:
    arr = np.asarray(p_beta, dtype=np.float64)
    return {
        "p_beta_mean_incl_reset": float(np.mean(arr)),
        "p_beta_mean_post_onset": float(np.mean(arr[1:])),
        "p_beta_final": float(arr[-1]),
    }


def _rollout(env: MehreganEnv, *, action: int | None) -> dict[str, Any]:
    _, info = env.reset(seed=SEED)
    p_beta = [float(info["p_beta_raw"])]
    plant = env.unwrapped._plant  # noqa: SLF001
    for _ in range(EVAL_STEPS):
        if action is None:
            result = plant.integrate(
                env.config.step_duration_s, DbsSpec.none(), record_spikes=True
            )
            if result.p_beta is None:
                raise RuntimeError("missing p_beta")
            p_beta.append(float(result.p_beta))
        else:
            _o, _r, term, trunc, info = env.step(int(action))
            p_beta.append(float(info["p_beta_raw"]))
            if term or trunc:
                break
    return {"action": action, "p_beta": p_beta, **_summarize(p_beta)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-family",
        default="burst",
        choices=["burst", "therapeutic_burst", "none"],
        help="Also screen all 41 patterns for this family (default: burst).",
    )
    args = parser.parse_args()

    t0 = time.time()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print("=== Fig 5b protocol oracle on burst alphabets ===", flush=True)

    # Shared no-stim via burst env plant
    env0 = _make_env(_factory("burst"))
    try:
        no_stim = _rollout(env0, action=None)
    finally:
        env0.close()
    no_stim_post = float(no_stim["p_beta_mean_post_onset"])
    print(f"no-stim post={no_stim_post:.2f}", flush=True)

    families: dict[str, Any] = {}
    for key, actions in TOP.items():
        print(f"\n=== {key}: top-{len(actions)} + pattern 0 ===", flush=True)
        alphabet = _factory(key)
        env = _make_env(alphabet)
        rows: list[dict[str, Any]] = []
        try:
            for action in [0, *actions]:
                t1 = time.time()
                row = _rollout(env, action=action)
                row["elapsed_s"] = round(time.time() - t1, 2)
                row["beat_no_stim"] = row["p_beta_mean_post_onset"] < no_stim_post
                rows.append(row)
                flag = "BEAT" if row["beat_no_stim"] else "lose"
                print(
                    f"  a={action:3d} post={row['p_beta_mean_post_onset']:.1f} {flag}",
                    flush=True,
                )
        finally:
            env.close()
        beaters = [r for r in rows if r["beat_no_stim"] and r["action"] != 0]
        best = min((r for r in rows if r["action"] != 0), key=lambda r: r["p_beta_mean_post_onset"])
        families[key] = {
            "top_actions": actions,
            "n_top_beat_no_stim": len(beaters),
            "best_action": best["action"],
            "best_p_beta_mean_post_onset": best["p_beta_mean_post_onset"],
            "pattern0_post_onset": next(r for r in rows if r["action"] == 0)[
                "p_beta_mean_post_onset"
            ],
            "rows": rows,
        }

    full_key = args.full_family
    full_out: dict[str, Any] | None = None
    if full_key != "none":
        print(f"\n=== full 41-pattern Fig5b screen: {full_key} ===", flush=True)
        alphabet = _factory(full_key)
        env = _make_env(alphabet)
        rows = []
        try:
            for action in range(alphabet.n_actions):
                t1 = time.time()
                row = _rollout(env, action=action)
                row["elapsed_s"] = round(time.time() - t1, 2)
                row["beat_no_stim"] = row["p_beta_mean_post_onset"] < no_stim_post
                rows.append(row)
                flag = "BEAT" if row["beat_no_stim"] else "lose"
                print(
                    f"  a={action:3d} post={row['p_beta_mean_post_onset']:.1f} {flag}",
                    flush=True,
                )
        finally:
            env.close()
        beaters = [r for r in rows if r["beat_no_stim"]]
        best = min(rows, key=lambda r: r["p_beta_mean_post_onset"])
        full_out = {
            "family": full_key,
            "n_beat_no_stim": len(beaters),
            "n_patterns": len(rows),
            "best_action": best["action"],
            "best_p_beta_mean_post_onset": best["p_beta_mean_post_onset"],
            "beaters": [
                {"action": r["action"], "p_beta_mean_post_onset": r["p_beta_mean_post_onset"]}
                for r in sorted(beaters, key=lambda x: x["p_beta_mean_post_onset"])
            ],
            "rows": rows,
        }

    result = {
        "probe": "fig5b_burst_fig5b_oracle",
        "mean_hz": MEAN_HZ,
        "plant_dt_ms": PAPER_DT_MS,
        "seed": SEED,
        "protocol": "2s reset + 5×2s constant action; score=post-onset mean Pβ",
        "no_stim": no_stim,
        "families_top": families,
        "full_screen": full_out,
        "summary": {
            "no_stim_post_onset": no_stim_post,
            "top_screen": {
                k: {
                    "n_top_beat": v["n_top_beat_no_stim"],
                    "best_action": v["best_action"],
                    "best_post": v["best_p_beta_mean_post_onset"],
                    "pattern0_post": v["pattern0_post_onset"],
                }
                for k, v in families.items()
            },
            "full_screen_beat": None
            if full_out is None
            else {
                "family": full_out["family"],
                "n_beat": full_out["n_beat_no_stim"],
                "best_action": full_out["best_action"],
                "best_post": full_out["best_p_beta_mean_post_onset"],
            },
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    text = json.dumps(result, indent=2) + "\n"
    OUT_JSON.write_text(text, encoding="utf-8")
    MAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    MAIN_OUT.write_text(text, encoding="utf-8")
    print(f"\nWrote {OUT_JSON}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
