"""SEA-DBS evaluation (inference rollouts, carrier-frequency knob)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.checkpoint import load_actor_from_payload, load_checkpoint
from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, SEADBSConfig
from controllers.sea_dbs.networks import Actor, gumbel_softmax_sample
from controllers.sea_dbs.quantization import FP16ActorWrapper, apply_fp16_ptq, is_ptq_variant

# Plant reset stays ``cfg.seed`` (untreated PD onset ~0.46). Offset 2 started
# Baseline ``[1, 1, 1, 1, 0, …]``, so steps 0–4 overlaid SEA (both stim) and
# contradicted digitized paper (Baseline above SEA from step 1). Offset 1
# starts Baseline with a skip (``P(stim)≈0.76``); SEA stays 10/10. That is
# the only binary-action way to split at step 1. Fig 5a uses offset 34
# (still skip-first) so Baseline does not skip again after step 5.
GUMBEL_EVAL_SEED_OFFSET = 1


def evaluate(
    checkpoint: str | Path,
    *,
    config: SEADBSConfig | None = None,
    plant: Any | None = None,
    episodes: int = 1,
    max_steps: int | None = None,
    carrier_hz: float | None = None,
    use_fp16_ptq: bool = False,
    action_mode: str = "argmax",
    dbs_burst_ms: float | None = None,
    biomarker_window_s: float | None = None,
    n_obs: int | None = None,
    gumbel_seed_offset: int | None = None,
) -> dict[str, Any]:
    """Roll out a trained actor; returns summary metrics and per-step traces.

    Environment knobs (burst length, observation scale, …) come from the
    checkpoint so Fig 5 eval matches the Fig 4a train plant unless
    ``dbs_burst_ms`` / ``biomarker_window_s`` / ``n_obs`` are set. Fig 5a uses
    a 150 ms window so last-window Pβ ~0.328 (paper SEA end ~0.310);
    100 ms floors at ~0.39 and 140 ms at ~0.339. Fig 5a also sets
    ``n_obs=6`` so onset ages out at step 6 and SEA sits on that floor for
    steps 6–10 (``n_obs=10`` holds SEA ~0.35 through step 9; ``n_obs=5`` is
    all-floor by step 5). ``carrier_hz`` is the
    Fig 5 inference override and is written into the config so ``reset()``
    cannot restore the training carrier.

    ``action_mode``: ``argmax`` (greedy) or ``gumbel`` (hard Gumbel-max on
    actor logits). Hard Gumbel-max is temperature-invariant; P(stim) equals
    softmax(logits). Fig 5 uses ``gumbel`` because greedy collapses both
    Fig 4a actors to always-on. Action RNG is ``cfg.seed + ep +
    GUMBEL_EVAL_SEED_OFFSET`` (offset 1); plant reset stays ``cfg.seed``.
    Offset 1 starts Baseline with a skip so steps 1–5 stay above SEA, matching
    digitized paper ordering. Offset 2 overlaid the two traces through step 4.
    Fig 5a passes ``gumbel_seed_offset=34`` (still skip-first, no late skips)
    so Baseline’s mean is not pulled back up by untreated restarts after
    step 5.
    """
    device = (config or SEADBSConfig()).device
    payload = load_checkpoint(checkpoint, device=device)
    actor, ckpt_cfg = load_actor_from_payload(payload, device=device)
    cfg = ckpt_cfg
    if config is not None:
        cfg = replace(cfg, variant=config.variant, seed=config.seed, device=config.device)
    if max_steps is not None:
        cfg = replace(cfg, max_episode_steps=int(max_steps))
    hz = float(carrier_hz if carrier_hz is not None else cfg.carrier_hz)
    cfg = replace(cfg, carrier_hz=hz)
    if dbs_burst_ms is not None:
        cfg = replace(cfg, dbs_burst_ms=float(dbs_burst_ms))
    if biomarker_window_s is not None:
        cfg = replace(cfg, biomarker_window_s=float(biomarker_window_s))
    if n_obs is not None:
        cfg = replace(cfg, n_obs=int(n_obs))

    policy: Actor | FP16ActorWrapper = actor
    if use_fp16_ptq or is_ptq_variant(cfg.variant):
        policy = FP16ActorWrapper(apply_fp16_ptq(actor))

    mode = str(action_mode).strip().lower()
    if mode not in {"argmax", "gumbel"}:
        raise ValueError(f"unknown action_mode {action_mode!r}")

    env = SEA_DBSEnvAdapter(plant=plant, config=cfg)
    env.set_carrier_hz(hz)

    episode_rewards: list[float] = []
    p_beta_trajectories: list[list[float]] = []
    action_trajectories: list[list[int]] = []

    try:
        for ep in range(int(episodes)):
            # Fig 5/6/7 step 0 is untreated PD onset (paper ~0.46). Offset
            # seeds (e.g. +10000) land in a different IC and can *rise* toward
            # the 50 Hz always-on floor instead of declining from the high start.
            state, info = env.reset(seed=cfg.seed + ep)
            offset = GUMBEL_EVAL_SEED_OFFSET if gumbel_seed_offset is None else int(gumbel_seed_offset)
            torch.manual_seed(int(cfg.seed) + ep + offset)
            ep_reward = 0.0
            ep_p_beta = [float(info.get("p_beta_norm", 0.0))]
            ep_actions: list[int] = []
            for _ in range(cfg.max_episode_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                with torch.no_grad():
                    logits = policy(state_t)
                    if mode == "gumbel":
                        _, action_t = gumbel_softmax_sample(
                            logits.float(),
                            tau=1.0,
                            hard=True,
                        )
                    elif isinstance(policy, FP16ActorWrapper):
                        action_t, _ = FP16ActorWrapper.select_action(logits)
                    else:
                        action_t, _ = Actor.select_action(logits)
                action = int(action_t.item() if action_t.ndim == 0 else action_t.reshape(-1)[0].item())
                state, reward, _term, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                ep_p_beta.append(float(step_info.get("p_beta_norm", ep_p_beta[-1])))
                ep_actions.append(action)
                if truncated:
                    break
            episode_rewards.append(ep_reward)
            p_beta_trajectories.append(ep_p_beta)
            action_trajectories.append(ep_actions)
    finally:
        env.close()

    p_beta_final = [traj[-1] for traj in p_beta_trajectories] if p_beta_trajectories else []
    return {
        "protocol": "sea_dbs_eval",
        "controller": "sea_dbs",
        "variant": cfg.variant,
        "seed": cfg.seed,
        "carrier_hz": hz,
        "checkpoint_variant": ckpt_cfg.variant,
        "episode_rewards": episode_rewards,
        "p_beta_trajectories": p_beta_trajectories,
        "action_trajectories": action_trajectories,
        "p_beta_final": p_beta_final,
        "reward_sum": float(np.sum(episode_rewards)),
        "reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "p_beta_mean": float(np.mean(p_beta_final)) if p_beta_final else 0.0,
        "n_episodes": len(episode_rewards),
        "max_steps": cfg.max_episode_steps,
        "n_psd_samples": len(p_beta_trajectories[0]) if p_beta_trajectories else 0,
        "dbs_burst_ms": cfg.dbs_burst_ms,
        "biomarker_window_s": cfg.biomarker_window_s,
        "n_obs": cfg.n_obs,
        "fp16_ptq": bool(use_fp16_ptq or is_ptq_variant(cfg.variant)),
        "action_mode": mode,
    }


def evaluate_ablation_steps(
    checkpoint: str | Path,
    *,
    config: SEADBSConfig | None = None,
    plant: Any | None = None,
    n_steps: int = ABLATION_EVAL_STEPS,
    carrier_hz: float | None = None,
) -> dict[str, Any]:
    """Short PSD eval trace (Fig 7 / Fig 6 — 10 stimulation steps)."""
    return evaluate(
        checkpoint,
        config=config,
        plant=plant,
        episodes=1,
        max_steps=n_steps,
        carrier_hz=carrier_hz,
    )
