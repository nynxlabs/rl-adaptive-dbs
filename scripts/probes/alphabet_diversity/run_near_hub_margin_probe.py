#!/usr/bin/env python3
"""Soft-fp32 train on NearHubBurstAlphabet + post-train logit-margin check.

Run only after open-loop ``diversity_ok`` at 45 Hz skip_regular for near_hub_*.

Gates (Fig 6a candidacy — do not promote on fail):
  - open-loop: already passed in alphabet_diversity_near_hub.json
  - post-train: mean top-1 vs top-2 logit margin ≲ 1.0 on probe states
  - optional: plain PTQ (noise=0) can flip argmax on ≥1 probe state

  tmux new-session -d -s near-hub-margin \\
    \"setsid nohup uv run python -m rl_adaptive_dbs.run \\
      scripts/probes/alphabet_diversity/run_near_hub_margin_probe.py \\
      >> logs/near-hub-margin.log 2>&1 < /dev/null\"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers.ddpg import load_actor, train
from controllers.ddpg.config import fig4a_ddpg_config
from controllers.ddpg.quantization import prepare_actor_for_eval
from envs.mehregan.extensions.alphabet_diversity.config import WithinStepEnvConfig
from envs.mehregan.extensions.alphabet_diversity.env import WithinStepMehreganEnv
from envs.mehregan.extensions.alphabet_diversity.near_hub import NearHubBurstAlphabet
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config

ARTIFACTS = Path("artifacts/ddpg")
DIVERSITY_JSON = ARTIFACTS / "alphabet_diversity_near_hub.json"
CACHE = Path("artifacts/figures/papers/1/6a")
OUT_JSON = ARTIFACTS / "near_hub_margin_probe.json"
MEAN_HZ = 45.0
PAPER_DT_MS = 0.02
SEED = 0
NUM_EPISODES = 4
STEPS_PER_EPISODE = 30
TRAIN_STEP_S = 0.2
ENTROPY_COEFF = 0.15
INIT_BIAS_SCALE = 0.15
MARGIN_TARGET = 1.0
N_PROBE_STATES = 24


def _alphabet(*, n_per_hub: int, step_duration_s: float) -> NearHubBurstAlphabet:
    return NearHubBurstAlphabet(
        mean_hz=MEAN_HZ,
        n_per_hub=n_per_hub,
        step_duration_s=step_duration_s,
        dt_ms=PAPER_DT_MS,
        skip_regular=True,
    )


def _pick_diversity_key(path: Path, *, prefer: str | None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run run_alphabet_diversity_sweep.py "
            "--hz 45 --only near_hub_n257 --only near_hub_n513 first"
        )
    payload = json.loads(path.read_text())
    cands = [
        r
        for r in payload["runs"]
        if r["mean_hz"] == MEAN_HZ and r["construction"] == "near_hub"
    ]
    if prefer:
        cands = [r for r in cands if r["key"] == prefer] or cands
    ok = [r for r in cands if r["skip_regular"]["diversity_ok"]]
    if not ok:
        raise RuntimeError(
            "no near_hub run with skip_regular diversity_ok at 45 Hz; "
            "do not soft-train — see " + str(path)
        )
    # Prefer more near-best, then smaller margin, then smaller n.
    ok.sort(
        key=lambda r: (
            -r["skip_regular"]["n_near_best"],
            r["skip_regular"]["margin_best_second"],
            r["n_actions"],
        )
    )
    return ok[0]


def _n_per_hub_from_key(key: str) -> int:
    if key.endswith("n513"):
        return 64
    if key.endswith("n257"):
        return 32
    raise ValueError(f"cannot infer n_per_hub from key={key}")


def _soft_train(*, n_per_hub: int, checkpoint: Path) -> Path:
    if checkpoint.is_file():
        print(f"reusing checkpoint {checkpoint}", flush=True)
        return checkpoint
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = WithinStepEnvConfig(
        state_length=1,
        step_duration_s=TRAIN_STEP_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=True,
    )
    alphabet = _alphabet(n_per_hub=n_per_hub, step_duration_s=TRAIN_STEP_S)
    env = WithinStepMehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=alphabet,
    )
    print(
        f"=== soft-fp32 near_hub n_per_hub={n_per_hub} "
        f"n_actions={alphabet.n_actions} eps={NUM_EPISODES} ===",
        flush=True,
    )
    ddpg_cfg = fig4a_ddpg_config(
        seed=SEED,
        num_episodes=NUM_EPISODES,
        max_episode_steps=STEPS_PER_EPISODE,
        pattern_mean_hz=MEAN_HZ,
        init_bias_scale=INIT_BIAS_SCALE,
        exploration_temperature_start=4.0,
        exploration_temperature_end=2.0,
        logit_noise_std=0.25,
    )
    ddpg_cfg = replace(ddpg_cfg, entropy_coeff=ENTROPY_COEFF)
    CACHE.mkdir(parents=True, exist_ok=True)
    train(config=ddpg_cfg, env=env, checkpoint_path=checkpoint)
    return checkpoint


@torch.no_grad()
def _logit_margins(
    *,
    checkpoint: Path,
    n_per_hub: int,
    n_states: int,
) -> dict[str, Any]:
    resolved = resolve_config()
    plant_cfg = replace(resolved.plant, dt_ms=PAPER_DT_MS)
    env_cfg = WithinStepEnvConfig(
        state_length=1,
        step_duration_s=TRAIN_STEP_S,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=MEAN_HZ,
        max_episode_steps=STEPS_PER_EPISODE,
        skip_regular=True,
    )
    alphabet = _alphabet(n_per_hub=n_per_hub, step_duration_s=TRAIN_STEP_S)
    env = WithinStepMehreganEnv(
        plant=PythonPlant(config=plant_cfg),
        config=env_cfg,
        alphabet=alphabet,
    )
    actor, _cfg = load_actor(checkpoint)
    actor.eval()
    margins: list[float] = []
    top1_actions: list[int] = []
    states: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=SEED)
        for i in range(n_states):
            # Alternate greedy and exploratory steps to visit a few plant states.
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = actor(x).squeeze(0).detach().cpu().numpy()
            order = np.argsort(-logits)
            m = float(logits[order[0]] - logits[order[1]])
            margins.append(m)
            top1_actions.append(int(order[0]))
            states.append(np.asarray(obs, dtype=np.float32).copy())
            action = int(order[0]) if i % 3 else int(order[min(3, len(order) - 1)])
            obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=SEED + i + 1)
    finally:
        env.close()

    # Plain PTQ (noise=0): does argmax flip on any probe state?
    flips = 0
    fp_actor, _ = load_actor(checkpoint)
    ptq_actor = prepare_actor_for_eval(
        fp_actor,
        "ptq-int8",
        device="cpu",
        ptq_weight_noise=0.0,
    )
    ptq_actor.eval()
    for st, a0 in zip(states, top1_actions, strict=True):
        x = torch.as_tensor(st, dtype=torch.float32).unsqueeze(0)
        a_ptq = int(torch.argmax(ptq_actor(x).squeeze(0)).item())
        if a_ptq != a0:
            flips += 1

    mean_m = float(np.mean(margins))
    return {
        "n_states": n_states,
        "mean_top1_top2_margin": mean_m,
        "median_top1_top2_margin": float(np.median(margins)),
        "max_top1_top2_margin": float(np.max(margins)),
        "min_top1_top2_margin": float(np.min(margins)),
        "unique_greedy_actions": sorted(set(top1_actions)),
        "n_unique_greedy": len(set(top1_actions)),
        "margin_target": MARGIN_TARGET,
        "margin_ok": mean_m <= MARGIN_TARGET,
        "ptq_noise0_argmax_flips": flips,
        "ptq_noise0_can_flip": flips > 0,
        "margins": margins,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diversity-json", type=Path, default=DIVERSITY_JSON)
    parser.add_argument("--prefer-key", default=None)
    parser.add_argument("--plot-only", action="store_true", help="reuse checkpoint")
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    args = parser.parse_args()

    t0 = time.time()
    chosen = _pick_diversity_key(args.diversity_json, prefer=args.prefer_key)
    n_per_hub = _n_per_hub_from_key(chosen["key"])
    ckpt = CACHE / f"checkpoint_near_hub_n{1 + 8 * n_per_hub}_skip_regular_02s.pt"
    print(
        f"chosen {chosen['key']}: near={chosen['skip_regular']['n_near_best']} "
        f"margin2={chosen['skip_regular']['margin_best_second']:.3f} "
        f"best={chosen['skip_regular']['best_action']} "
        f"Pβ={chosen['skip_regular']['best_p_beta']:.1f}",
        flush=True,
    )
    if not args.plot_only:
        _soft_train(n_per_hub=n_per_hub, checkpoint=ckpt)
    elif not ckpt.is_file():
        print(f"missing checkpoint {ckpt}", file=sys.stderr)
        return 2

    margins = _logit_margins(
        checkpoint=ckpt, n_per_hub=n_per_hub, n_states=N_PROBE_STATES
    )
    payload = {
        "diversity_key": chosen["key"],
        "diversity_skip_regular": chosen["skip_regular"],
        "n_per_hub": n_per_hub,
        "checkpoint": str(ckpt),
        "logit_margins": {
            k: v for k, v in margins.items() if k != "margins"
        },
        "margins_raw": margins["margins"],
        "fig6a_candidate": bool(
            chosen["skip_regular"]["diversity_ok"]
            and margins["margin_ok"]
            and margins["ptq_noise0_can_flip"]
        ),
        "elapsed_s": round(time.time() - t0, 2),
        "note": (
            "fig6a_candidate requires diversity_ok + mean logit margin <= 1 "
            "+ plain PTQ (noise=0) argmax flip on ≥1 probe state. "
            "Do not run full Fig 6a until fig6a_candidate is true."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["logit_margins"], indent=2), flush=True)
    print(
        f"fig6a_candidate={payload['fig6a_candidate']} wrote {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
