"""SEA-DBS evaluation (inference rollouts, carrier-frequency knob).

Figs 5–7 roll out a trained actor on the **same** plant / observation setup it
was trained on (window, burst, ``n_obs``, normalization, reset — all read from
the checkpoint). The only eval override is the paper-stated stimulation
frequency (50 Hz for Figs 5a/6/7, 30 Hz for Fig 5b). Actions are deterministic
(argmax on the actor logits, spec §14.2) unless ``action_mode="gumbel"`` is asked
for explicitly; FP16 PTQ is a plain ``.half()`` cast.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from controllers.sea_dbs.adapter import SEA_DBSEnvAdapter
from controllers.sea_dbs.checkpoint import load_actor_from_payload, load_checkpoint
from controllers.sea_dbs.config import ABLATION_EVAL_STEPS, INFERENCE_CARRIER_50HZ, SEADBSConfig
from controllers.sea_dbs.networks import Actor, gumbel_softmax_sample
from controllers.sea_dbs.quantization import FP16ActorWrapper, apply_fp16_ptq, is_ptq_variant


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
) -> dict[str, Any]:
    """Roll out a trained actor; returns summary metrics and per-step traces.

    Plant reset uses ``cfg.seed + ep`` (episode 0 = the panel seed). With
    ``action_mode="gumbel"`` the hard Gumbel-max sample (``P(stim)`` =
    softmax(logits)) uses torch seed ``cfg.seed + ep`` — fixed, never tuned.
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

    policy: Actor | FP16ActorWrapper = actor
    fp16 = bool(use_fp16_ptq or is_ptq_variant(cfg.variant))
    if fp16:
        policy = FP16ActorWrapper(apply_fp16_ptq(actor))

    mode = str(action_mode).strip().lower()
    if mode not in {"argmax", "gumbel"}:
        raise ValueError(f"unknown action_mode {action_mode!r}")

    env = SEA_DBSEnvAdapter(plant=plant, config=cfg)
    env.set_carrier_hz(hz)

    episode_rewards: list[float] = []
    p_beta_trajectories: list[list[float]] = []
    action_trajectories: list[list[int]] = []
    logit_trajectories: list[list[list[float]]] = []

    try:
        for ep in range(int(episodes)):
            state, info = env.reset(seed=cfg.seed + ep)
            torch.manual_seed(int(cfg.seed) + ep)
            ep_reward = 0.0
            ep_p_beta = [float(info.get("p_beta_norm", 0.0))]
            ep_actions: list[int] = []
            ep_logits: list[list[float]] = []
            for _step in range(cfg.max_episode_steps):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                with torch.no_grad():
                    logits = policy(state_t).float()
                    if mode == "gumbel":
                        _, action_t = gumbel_softmax_sample(logits, tau=1.0, hard=True)
                    else:
                        action_t, _ = Actor.select_action(logits)
                action = int(action_t.reshape(-1)[0].item())
                ep_logits.append([float(v) for v in logits.reshape(-1).tolist()])
                state, reward, _term, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                ep_p_beta.append(float(step_info.get("p_beta_norm", ep_p_beta[-1])))
                ep_actions.append(action)
                if truncated:
                    break
            episode_rewards.append(ep_reward)
            p_beta_trajectories.append(ep_p_beta)
            action_trajectories.append(ep_actions)
            logit_trajectories.append(ep_logits)
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
        "logit_trajectories": logit_trajectories,
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
        "prefill_obs_window": cfg.prefill_obs_window,
        "plant_integration_mode": cfg.plant_integration_mode,
        "fp16_ptq": fp16,
        "action_mode": mode,
    }


def evaluate_ablation_steps(
    checkpoint: str | Path,
    *,
    config: SEADBSConfig | None = None,
    plant: Any | None = None,
    n_steps: int = ABLATION_EVAL_STEPS,
    carrier_hz: float | None = None,
    use_fp16_ptq: bool = False,
) -> dict[str, Any]:
    """Short PSD eval trace (Figs 5–7 — untreated t=0 + 10 stimulation steps)."""
    return evaluate(
        checkpoint,
        config=config,
        plant=plant,
        episodes=1,
        max_steps=n_steps,
        carrier_hz=carrier_hz or INFERENCE_CARRIER_50HZ,
        use_fp16_ptq=use_fp16_ptq,
        action_mode="argmax",
    )


# Table II (paper §V.B): "changing the environment seed every n ∈ {10, 20, 50,
# 75} step". Convention (paper silent on the rest — replication.md §14.13):
# one deterministic rollout of TABLE2_N_BLOCKS blocks of n steps; at the start of
# each block the plant is reset with the next seed (cfg.seed, cfg.seed + 1, …),
# which also re-prefills the Eq. (5) window with that seed's untreated level.
# Every n visits the same TABLE2_N_BLOCKS plant seeds: the full-stimulation
# floor varies 221–479 across seeds 0–29, so a fixed horizon (300 steps until
# oct 17 2026: seeds 0–3 for n = 75, 0–29 for n = 10) made the n trend mostly a
# seed-pool effect. Avg PSD is the mean Eq. (5) observation over all steps.
TABLE2_RESEED_INTERVALS = (10, 20, 50, 75)
TABLE2_N_BLOCKS = 30


def table2_horizon(reseed_every: int, n_blocks: int = TABLE2_N_BLOCKS) -> int:
    """Steps in the Table II rollout for one interval (``n_blocks`` seeds of ``n`` steps)."""
    return int(n_blocks) * int(reseed_every)


def evaluate_reseeded(
    checkpoint: str | Path,
    *,
    reseed_every: int,
    config: SEADBSConfig | None = None,
    plant: Any | None = None,
    horizon: int | None = None,
    carrier_hz: float | None = None,
) -> dict[str, Any]:
    """Argmax rollout with the plant reseeded every ``reseed_every`` steps (Table II).

    ``horizon`` defaults to ``table2_horizon(reseed_every)``.
    """
    from controllers.sea_dbs.reward import sea_dbs_reward

    n = int(reseed_every)
    if horizon is None:
        horizon = table2_horizon(n) if n > 0 else 0
    if n <= 0 or int(horizon) <= 0 or int(horizon) % n:
        raise ValueError(f"horizon {horizon} must be a positive multiple of reseed_every {n}")
    device = (config or SEADBSConfig()).device
    payload = load_checkpoint(checkpoint, device=device)
    actor, ckpt_cfg = load_actor_from_payload(payload, device=device)
    cfg = ckpt_cfg
    if config is not None:
        cfg = replace(cfg, variant=config.variant, seed=config.seed, device=config.device)
    hz = float(carrier_hz if carrier_hz is not None else INFERENCE_CARRIER_50HZ)
    # The adapter truncates at max_episode_steps; each block is n steps.
    cfg = replace(cfg, carrier_hz=hz, max_episode_steps=n)

    env = SEA_DBSEnvAdapter(plant=plant, config=cfg)
    env.set_carrier_hz(hz)
    p_beta: list[float] = []
    rewards: list[float] = []
    actions: list[int] = []
    block_seeds: list[int] = []
    try:
        for block in range(int(horizon) // n):
            seed = int(cfg.seed) + block
            block_seeds.append(seed)
            state, _info = env.reset(seed=seed)
            for _step in range(n):
                state_t = torch.as_tensor(state, dtype=torch.float32, device=cfg.device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _ = Actor.select_action(actor(state_t).float())
                action = int(action_t.reshape(-1)[0].item())
                state, reward, _term, _trunc, step_info = env.step(action)
                p_beta.append(float(step_info["p_beta_norm"]))
                rewards.append(float(reward))
                actions.append(action)
    finally:
        env.close()

    avg_p = float(np.mean(p_beta))
    return {
        "protocol": "sea_dbs_table2_reseed",
        "variant": cfg.variant,
        "seed": cfg.seed,
        "carrier_hz": hz,
        "reseed_every": n,
        "horizon": int(horizon),
        "block_seeds": block_seeds,
        "p_beta_norm": p_beta,
        "rewards": rewards,
        "actions": actions,
        "avg_p_beta_norm": avg_p,
        "avg_reward": float(np.mean(rewards)),
        "reward_of_avg_psd": float(
            sea_dbs_reward(avg_p, beta_threshold=cfg.beta_threshold, reward_scale=cfg.reward_scale)
        ),
        "stim_frac": float(np.mean(actions)),
        "biomarker_window_s": cfg.biomarker_window_s,
        "n_obs": cfg.n_obs,
        "prefill_obs_window": cfg.prefill_obs_window,
        "plant_integration_mode": cfg.plant_integration_mode,
    }
