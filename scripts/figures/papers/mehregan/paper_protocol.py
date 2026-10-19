"""Shared paper protocol for Mehregan et al. Figs 4–6 (one model per mean rate).

Mehregan et al. train **one** 45 Hz DDPG model (Fig 4a/4b training curves,
Fig 5a efficacy, Fig 6a fp32 + PTQ) and **one** 30 Hz model with "all other
parameters fixed" (Fig 5b, Fig 6b fp32 + PTQ). QAT is a separate 10-episode run
"with the same settings" (Fig 6a/6b). This module holds that single recipe so
every panel trains and evaluates the same way.

Paper-stated (§III.B, §IV.A.1, Alg. 1) — implemented as written:

- 2 s environment step, plant step 0.02 ms, 30 steps per episode, 10 episodes;
  Alg. 1's "update frequency" read as one update every 2nd step.
- Replay buffer 8192, batch 32, actor lr 5e-4, critic lr 1e-3, Adam.
- Critic scores ``Q(s, a_logit)`` (Eq. 4); critic target uses the target actor's
  logits (Eq. 3); actor maximizes ``Q(s, μ(s))`` (Eq. 5); soft target updates
  (Eqs. 6–7); MSE critic loss.
- Actor = CNN → logits → softmax/argmax (Fig 3a); action space initialized to
  the regular train at the mean rate.
- Reward Eq. (8), β_t = 0.35 on $P_\\beta / 1000$; below β_t the Fig. 3c curve
  $+((β_t - s)·10)^2$ (the printed linear branch has the wrong sign).
- Eval: fixed seed, 2 s reset, then five 2 s steps of the trained policy; fixed
  periodic baselines on the same seed.
- PTQ: ``torch.ao.quantization.quantize_dynamic`` (int8) and fp16 casting of the
  fp32 actor; QAT: fake-quant stub on the actor input, dequant stub on logits.

Paper-silent conventions (labelled; tune these, not the stated mechanism):

- Exploration: Gaussian noise on the actor logits (the DDPG continuous action);
  the executed pattern is ``argmax`` of the noisy logits and the stored
  ``a_logit`` is that noisy vector, so the critic always scores what was played.
- γ, τ, replay start size, init bias strength, pattern alphabet, seeds.
- Plant state carries across the 2 s steps of an episode (Alg. 1 sequential
  steps); each episode resets to new initial conditions (seed + episode).
- Eval display: 2 s no-stim pre-roll so the trailing 2 s $P_\\beta$ window is full
  at display t=0, then the 2 s reset segment, then five policy steps; the trace
  is sampled every 0.2 s (Fig 2a protocol).
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from controllers.ddpg.checkpoint import save_checkpoint
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.networks import Actor
from controllers.ddpg.quantization import QATActor, unwrap_actor
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet
from envs.mehregan.pattern_alternatives import BurstPatternAlphabet, CuratedPatternAlphabet
from envs.plant.config import PlantConfig
from envs.plant.dbs import DbsSpec
from envs.plant import PythonPlant
from rl_adaptive_dbs.panel import load_script_module
from rl_adaptive_dbs.user_config import resolve_config

_FIG2A = load_script_module(
    "mehregan_fig2a_for_protocol",
    Path(__file__).resolve().parent / "2a" / "plot.py",
)

# --- Paper-stated -----------------------------------------------------------
PAPER_DT_MS = 0.02
STEP_S = 2.0
STEPS_PER_EPISODE = 30
NUM_EPISODES = 10
ACTOR_LR = 5e-4
CRITIC_LR = 1e-3
BUFFER_CAPACITY = 8192
BATCH_SIZE = 32
EVAL_STEPS = 5
STATE_LENGTH = 1

# --- Paper-silent defaults (conventions) ------------------------------------
DEFAULT_ALPHABET = "burst"
DEFAULT_LOGIT_NOISE = 1.0
# §I: "the action space is initialized using regular pulses" — head bias toward
# pattern 0 (strength paper-silent; 2.0 plays the regular train from episode 0).
DEFAULT_INIT_BIAS = 1.0
DEFAULT_GAMMA = 0.99
DEFAULT_TAU = 0.005

# Eval display (Fig 2a protocol).
PREROLL_S = 2.0
DISPLAY_S = 12.0
ONSET_DISPLAY_S = 2.0
SAMPLE_S = 0.2
WINDOW_S = 2.0

ALPHABETS = ("burst", "burst_legacy", "burst_nobeta", "burst_nobeta_rand20", "burst_rand20", "jitter")


def _log(msg: str) -> None:
    print(msg, flush=True)


@dataclass(frozen=True)
class PaperKnobs:
    """Paper-silent knobs. Everything the paper states is fixed above."""

    alphabet: str = DEFAULT_ALPHABET
    jitter_fraction: float = 1.0 / 3.0
    logit_noise_std: float = DEFAULT_LOGIT_NOISE
    init_bias_scale: float = DEFAULT_INIT_BIAS
    gamma: float = DEFAULT_GAMMA
    tau: float = DEFAULT_TAU
    # Not paper-silent: reading of Eq. (8) below beta_t. "square" = Fig. 3c's
    # plotted curve (master-approved Oct 17 2026); "linear" = printed branch,
    # sign flipped (kept to separate the reward effect in candidate runs).
    reward_below: str = "square"
    # Alg. 1 line 15 "for update frequency do": our reading (agreed with the
    # maintainer, Oct 17 2026) is one update on every 2nd environment step; the
    # literal loop (1 update every step) is update_every=1.
    update_every: int = 2
    # Candidate readings (off by default):
    # carry_plant = continue the plant across episode resets instead of new ICs;
    # logit_noise_end = linear per-episode anneal of the logit noise to this σ.
    carry_plant: bool = False
    logit_noise_end: float | None = None
    # Action head (candidate; the shipped panels use the menu). "perpulse" =
    # Figure 3a's n_pulses x l logits head with the Fig 3b critic on the raw
    # logits (controllers.ddpg.action_designs.PerPulseLogits); the knobs below
    # only apply to it.
    action_head: str = "menu"
    # Network input length: Figure 3a draws 1 x l (l = 2000, fc1 fan-in 32,000);
    # 1 = the shipped menu's length-1 state (fc1 fan-in 64). The state holds the
    # current step's P_beta/1000, broadcast to the length.
    state_len: int = 2000
    # Network-input normalization (paper-silent): net input = (s - offset) * scale.
    state_offset: float = 0.0
    state_scale: float = 1.0
    # Exploration noise family on the logits: "gauss" (as the menu) or "gumbel".
    pp_noise: str = "gauss"
    # Deviation (candidate): Adam lr x min(1, fanin_ref / fan_in) per layer of the
    # networks in fanin_scope ("both" / "actor" / "critic"); 0 = the stated lrs.
    fanin_ref: int = 0
    fanin_scope: str = "both"
    # Adam epsilon (paper-silent; PyTorch default 1e-8). A large epsilon makes the
    # step scale with the gradient instead of its sign.
    actor_adam_eps: float = 1e-8
    critic_adam_eps: float = 1e-8
    # Deviation (candidate): critic scores logits minus the regular-pulse init spike.
    critic_center: bool = False
    # Deviation candidates (oct 19 2026; controllers.ddpg.perpulse_agents). Any
    # pp_design other than "faithful" deviates from Figure 3a / Eqs. 4-5:
    # design: "faithful" (above) | "logits" | "residual" | "offset" | "mixture";
    # algo: "ddpg" | "td3" | "reinforce"; view: critic input ("raw", "delta",
    # "regprob", "offhist"). actor_lr overrides the stated 5e-4 (0 = stated).
    pp_design: str = "faithful"
    pp_algo: str = "ddpg"
    pp_view: str = "raw"
    delta_max: float = 3.0
    mix_init: float = -3.0
    actor_lr: float = 0.0
    twin_critic: bool = False
    target_noise: float = 0.0
    policy_delay: int = 1
    grad_clip: float = 0.0
    pull_l2: float = 0.0
    bc_steps: int = 0
    updates_per_step: int = 1
    pg_batch: int = 8
    # Onset candidates (oct 19 2026): init bias spread over +-init_width ms;
    # per-step exploration temperature sigma * exp(U(-sigma_jitter, sigma_jitter));
    # critic-only updates before the first actor update.
    init_width: int = 0
    sigma_jitter: float = 0.0
    critic_warmup: int = 0
    # QAT activation observer averaging constant for per-pulse QAT (0 = PyTorch default 0.01).
    qat_avg_const: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        head = {}
        if self.action_head != "menu":
            head = {
                "action_head": self.action_head,
                "state_len": self.state_len,
                "state_offset": self.state_offset,
                "state_scale": self.state_scale,
                "pp_noise": self.pp_noise,
                "fanin_ref": self.fanin_ref,
                "fanin_scope": self.fanin_scope,
                "actor_adam_eps": self.actor_adam_eps,
                "critic_adam_eps": self.critic_adam_eps,
                "critic_center": self.critic_center,
            }
            if self.pp_design != "faithful":
                head.update(
                    {
                        "pp_design": self.pp_design,
                        "pp_algo": self.pp_algo,
                        "pp_view": self.pp_view,
                        "delta_max": self.delta_max,
                        "mix_init": self.mix_init,
                        "actor_lr": self.actor_lr,
                        "twin_critic": self.twin_critic,
                        "target_noise": self.target_noise,
                        "policy_delay": self.policy_delay,
                        "grad_clip": self.grad_clip,
                        "pull_l2": self.pull_l2,
                        "bc_steps": self.bc_steps,
                        "updates_per_step": self.updates_per_step,
                        "pg_batch": self.pg_batch,
                        "init_width": self.init_width,
                        "sigma_jitter": self.sigma_jitter,
                        "critic_warmup": self.critic_warmup,
                        "qat_avg_const": self.qat_avg_const,
                        "deviates_from_paper": "our best guess of what the authors may have done",
                    }
                )
        return {**head, 
            "alphabet": self.alphabet,
            "jitter_fraction": self.jitter_fraction,
            "logit_noise_std": self.logit_noise_std,
            "init_bias_scale": self.init_bias_scale,
            "gamma": self.gamma,
            "tau": self.tau,
            "reward_below": self.reward_below,
            "carry_plant": self.carry_plant,
            "update_every": self.update_every,
            "logit_noise_end": self.logit_noise_end,
        }


def apply_overrides(knobs: PaperKnobs, pairs: list[str] | None) -> PaperKnobs:
    """Apply ``FIELD=VALUE`` overrides (``--set``) to paper-silent knobs only."""
    if not pairs:
        return knobs
    out = knobs
    types = {
        "alphabet": str,
        "jitter_fraction": float,
        "logit_noise_std": float,
        "init_bias_scale": float,
        "gamma": float,
        "tau": float,
        "reward_below": str,
        "carry_plant": lambda v: str(v).lower() in ("1", "true", "yes"),
        "update_every": int,
        "logit_noise_end": float,
        "action_head": str,
        "state_len": int,
        "state_offset": float,
        "state_scale": float,
        "pp_noise": str,
        "fanin_ref": int,
        "fanin_scope": str,
        "actor_adam_eps": float,
        "critic_adam_eps": float,
        "critic_center": lambda v: str(v).lower() in ("1", "true", "yes"),
        "pp_design": str,
        "pp_algo": str,
        "pp_view": str,
        "delta_max": float,
        "mix_init": float,
        "actor_lr": float,
        "twin_critic": lambda v: str(v).lower() in ("1", "true", "yes"),
        "target_noise": float,
        "policy_delay": int,
        "grad_clip": float,
        "pull_l2": float,
        "bc_steps": int,
        "updates_per_step": int,
        "pg_batch": int,
        "init_width": int,
        "sigma_jitter": float,
        "critic_warmup": int,
        "qat_avg_const": float,
    }
    for raw in pairs:
        key, sep, value = raw.partition("=")
        key = key.strip()
        if not sep or key not in types:
            msg = f"--set expects one of {sorted(types)} as FIELD=VALUE, got {raw!r}"
            raise ValueError(msg)
        out = replace(out, **{key: types[key](value)})
    if out.reward_below not in ("linear", "square"):
        msg = f"reward_below must be 'linear' or 'square', got {out.reward_below!r}"
        raise ValueError(msg)
    if out.action_head not in ("menu", "perpulse"):
        msg = f"action_head must be 'menu' or 'perpulse', got {out.action_head!r}"
        raise ValueError(msg)
    if out.fanin_scope not in ("both", "actor", "critic"):
        msg = f"fanin_scope must be 'both', 'actor' or 'critic', got {out.fanin_scope!r}"
        raise ValueError(msg)
    if out.alphabet not in ALPHABETS:
        msg = f"alphabet must be one of {ALPHABETS}, got {out.alphabet!r}"
        raise ValueError(msg)
    return out


def make_alphabet(mean_hz: float, knobs: PaperKnobs) -> Any:
    if knobs.alphabet == "burst":
        return BurstPatternAlphabet(mean_hz=mean_hz, step_duration_s=STEP_S, dt_ms=PAPER_DT_MS)
    if knobs.alphabet in ("burst_nobeta", "burst_nobeta_rand20", "burst_rand20"):
        # Candidate menus (paper-silent construction; see author_questions Mehregan).
        return CuratedPatternAlphabet(
            mean_hz=mean_hz,
            step_duration_s=STEP_S,
            dt_ms=PAPER_DT_MS,
            exclude_beta_bursts="nobeta" in knobs.alphabet,
            n_random=20 if "rand20" in knobs.alphabet else 0,
        )
    if knobs.alphabet == "burst_legacy":
        return BurstPatternAlphabet(
            mean_hz=mean_hz,
            step_duration_s=STEP_S,
            dt_ms=PAPER_DT_MS,
            n_patterns=41,
            layout="legacy",
        )
    return FixedMeanPatternAlphabet(
        mean_hz=mean_hz,
        step_duration_s=STEP_S,
        dt_ms=PAPER_DT_MS,
        jitter_fraction=knobs.jitter_fraction,
    )


def plant_config() -> PlantConfig:
    return replace(resolve_config().plant, dt_ms=PAPER_DT_MS)


def make_env(mean_hz: float, knobs: PaperKnobs) -> MehreganEnv:
    env_cfg = MehreganEnvConfig(
        step_duration_s=STEP_S,
        state_length=STATE_LENGTH,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
        max_episode_steps=STEPS_PER_EPISODE,
        plant_integration_mode="continuous",
        reward_below_threshold=knobs.reward_below,
    )
    plant = PythonPlant(config=plant_config())
    return MehreganEnv(plant=plant, config=env_cfg, alphabet=make_alphabet(mean_hz, knobs))


def paper_ddpg_config(
    *,
    mean_hz: float,
    seed: int,
    knobs: PaperKnobs,
    variant: str = "paper",
    num_episodes: int = NUM_EPISODES,
) -> DDPGConfig:
    """Alg. 1 as written; only paper-silent fields come from ``knobs``."""
    return DDPGConfig(
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        num_episodes=num_episodes,
        max_episode_steps=STEPS_PER_EPISODE,
        gamma=knobs.gamma,
        tau=knobs.tau,
        update_frequency=1,
        variant=variant,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=mean_hz,
        min_buffer_size=BATCH_SIZE,
        seed=seed,
        critic_action_input="logits",
        exploration_mode="greedy",
        logit_noise_std=knobs.logit_noise_std,
        init_bias_scale=knobs.init_bias_scale,
        critic_warmup_steps=0,
        reward_normalize=False,
        critic_loss_fn="mse",
        entropy_coeff=0.0,
        obs_normalize=False,
        random_warmup_steps=0,
        log_episodes=True,
    )


def train_paper(
    *,
    mean_hz: float,
    seed: int,
    knobs: PaperKnobs,
    variant: str = "paper",
    checkpoint_path: Path | None = None,
    num_episodes: int = NUM_EPISODES,
    log: Callable[[str], None] = _log,
    extra: dict[str, Any] | None = None,
    qat_freeze_observer_after: int | None = None,
    episode_hook: Callable[[DDPGTrainer, int], None] | None = None,
    trainer_hook: Callable[[DDPGTrainer], None] | None = None,
) -> dict[str, Any]:
    """Run Alg. 1 and return the per-step training record.

    ``episode_hook(trainer, episode)`` runs after each episode and
    ``trainer_hook(trainer)`` once before episode 0 (probe diagnostics).
    Returns a JSON-able dict with ``beta_trace`` (``P_beta/1000`` per step),
    ``actions``, ``episode_rewards``, ``step_rewards``, and run metadata.
    """
    if knobs.action_head == "perpulse":
        qat_ok = variant == "qat" and knobs.pp_design != "faithful"
        if (variant != "paper" and not qat_ok) or qat_freeze_observer_after is not None or episode_hook or trainer_hook:
            msg = "action_head=perpulse supports the fp32 train, and QAT for the deviation designs"
            raise NotImplementedError(msg)
        return _train_perpulse(
            mean_hz=mean_hz,
            seed=seed,
            qat=variant == "qat",
            knobs=knobs,
            checkpoint_path=checkpoint_path,
            num_episodes=num_episodes,
            log=log,
            extra=extra,
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = make_env(mean_hz, knobs)
    config = paper_ddpg_config(
        mean_hz=mean_hz, seed=seed, knobs=knobs, variant=variant, num_episodes=num_episodes
    )
    trainer = DDPGTrainer(env, config)
    if trainer_hook is not None:
        trainer_hook(trainer)
    beta_trace: list[float] = []
    actions: list[int] = []
    step_rewards: list[float] = []
    episode_rewards: list[float] = []
    reset_beta: list[float] = []
    env_step = 0
    try:
        for episode in range(num_episodes):
            if qat_freeze_observer_after is not None and episode == qat_freeze_observer_after:
                # PyTorch QAT recipe: stop updating quantization ranges partway
                # through training (paper-silent; probe knob, off by default).
                for module in (trainer.actor, trainer.actor_target):
                    module.apply(torch.ao.quantization.disable_observer)
                log(f"froze QAT observers at episode {episode}")
            if knobs.logit_noise_end is not None and num_episodes > 1:
                frac = episode / (num_episodes - 1)
                sigma = knobs.logit_noise_std + frac * (knobs.logit_noise_end - knobs.logit_noise_std)
                trainer.config = replace(trainer.config, logit_noise_std=sigma)
            if knobs.carry_plant and episode > 0:
                state, info0 = env.reset(seed=seed + episode, options={"keep_plant": True})
            else:
                state, info0 = env.reset(seed=seed + episode)
            reset_beta.append(float(info0["p_beta_norm"]))
            # Alg. 1 line 7: the reset segment's reward is collected too.
            ep_reward = float(info0.get("reward", 0.0))
            done = False
            while not done:
                action, logits = trainer._select_action(state, env_step=env_step)
                env_step += 1
                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                beta_trace.append(float(info["p_beta_norm"]))
                actions.append(int(action))
                step_rewards.append(float(reward))
                ep_reward += float(reward)
                trainer.buffer.add(
                    state=state,
                    action=action,
                    action_logits=logits,
                    reward=float(reward),
                    next_state=next_state,
                    dw=float(info.get("dw", 0.0)),
                )
                state = next_state
                if len(trainer.buffer) >= config.min_buffer_size and env_step % knobs.update_every == 0:
                    for _ in range(config.update_frequency):
                        c_loss, a_loss = trainer._update_step()
                        trainer.metrics.critic_losses.append(c_loss)
                        trainer.metrics.actor_losses.append(a_loss)
            episode_rewards.append(ep_reward)
            trainer.metrics.episode_rewards.append(ep_reward)
            ep_actions = actions[-STEPS_PER_EPISODE:]
            top, top_n = Counter(ep_actions).most_common(1)[0]
            log(
                f"episode {episode + 1}/{num_episodes} reward={ep_reward:.2f} "
                f"mean_beta={np.mean(beta_trace[-STEPS_PER_EPISODE:]):.3f} "
                f"unique={len(set(ep_actions))} top={top}x{top_n}"
            )
            if episode_hook is not None:
                episode_hook(trainer, episode)
        trainer._env_step = env_step
        if checkpoint_path is not None:
            save_checkpoint(
                checkpoint_path,
                actor=unwrap_actor(trainer.actor),
                policy=trainer.actor,
                config=config,
                state_length=STATE_LENGTH,
                n_actions=int(env.action_space.n),
                critic=trainer.critic,
                trainer=trainer,
                extra={
                    "completed_episodes": num_episodes,
                    "mean_hz": mean_hz,
                    "knobs": knobs.as_dict(),
                    **(extra or {}),
                },
            )
            log(f"wrote checkpoint {checkpoint_path}")
    finally:
        env.close()

    counts = Counter(actions)
    return {
        "mean_hz": mean_hz,
        "seed": seed,
        "variant": variant,
        "num_episodes": num_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "knobs": knobs.as_dict(),
        "paper_stated": {
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "buffer_capacity": BUFFER_CAPACITY,
            "batch_size": BATCH_SIZE,
            "step_s": STEP_S,
            "dt_ms": PAPER_DT_MS,
            "critic_action_input": "logits",
        },
        "beta_trace": beta_trace,
        "actions": actions,
        "step_rewards": step_rewards,
        "episode_rewards": episode_rewards,
        "reset_beta": reset_beta,
        "unique_actions": len(counts),
        "action_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


def _train_perpulse(
    *,
    mean_hz: float,
    seed: int,
    qat: bool = False,
    knobs: PaperKnobs,
    checkpoint_path: Path | None,
    num_episodes: int,
    log: Callable[[str], None],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Alg. 1 on Figure 3a's per-pulse head (candidate; see ``PaperKnobs.action_head``).

    Same protocol as the menu: continuous plant within an episode, new initial
    conditions per episode, reset reward collected, one update every
    ``update_every`` steps once the buffer holds a batch. Each pulse row of the
    noisy logits plays its argmax slot; the critic scores the stored noisy logits.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_pulses = int(round(mean_hz * STEP_S))
    if knobs.pp_design != "faithful":
        agent = _deviation_agent(knobs, n_pulses=n_pulses, seed=seed, qat=qat)
    else:
        agent = _FaithfulPulseAgent(knobs, n_pulses=n_pulses, seed=seed)
    env = make_env(mean_hz, knobs)
    return _perpulse_loop(
        agent=agent, env=env, knobs=knobs, mean_hz=mean_hz, seed=seed, n_pulses=n_pulses,
        checkpoint_path=checkpoint_path, num_episodes=num_episodes, log=log,
        extra={**(extra or {}), "qat": qat},
    )


class _FaithfulPulseAgent:
    """Figure 3a head + Alg. 1 (``AlgOneAgent``) behind the per-pulse agent interface."""

    on_policy = False

    def __init__(self, knobs: PaperKnobs, *, n_pulses: int, seed: int) -> None:
        from controllers.ddpg.action_designs import AlgOneAgent, PerPulseLogits

        self.design = PerPulseLogits(
            n_pulses=n_pulses, init_bias=knobs.init_bias_scale, noise=knobs.pp_noise, center=knobs.critic_center
        )
        self.inner = AlgOneAgent(
            self.design,
            state_len=knobs.state_len,
            actor_lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            gamma=knobs.gamma,
            tau=knobs.tau,
            seed=seed,
            fanin_ref=knobs.fanin_ref,
            fanin_scope=knobs.fanin_scope,
            actor_eps=knobs.actor_adam_eps,
            critic_eps=knobs.critic_adam_eps,
        )
        self.actor = self.inner.actor

    def act(self, s: torch.Tensor, sigma: float) -> tuple[torch.Tensor, np.ndarray]:
        with torch.no_grad():
            stored = self.design.explore(self.actor(s), sigma, self.inner.gen)[0]
        return stored, self.design.decode_stored(stored)

    def update(self, s, a, r, s2, d) -> dict:
        return self.inner.update(s, a, r, s2, d)

    def noise_free(self, s: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.design.decode_raw(self.actor(s))

    def diag(self, s: torch.Tensor) -> dict:
        with torch.no_grad():
            raw = self.actor(s)
        return {"margin": self.design.regular_margin(raw), "logit_std": float(raw.std())}


def _deviation_agent(knobs: PaperKnobs, *, n_pulses: int, seed: int, qat: bool = False):
    """Candidate agents that deviate from the paper (controllers.ddpg.perpulse_agents)."""
    from controllers.ddpg.perpulse_agents import AgentKnobs, make_agent

    k = AgentKnobs(
        state_len=knobs.state_len,
        actor_lr=knobs.actor_lr or ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=knobs.gamma,
        tau=knobs.tau,
        seed=seed,
        sigma=knobs.logit_noise_std,
        twin=knobs.twin_critic,
        target_noise=knobs.target_noise,
        policy_delay=knobs.policy_delay,
        grad_clip=knobs.grad_clip,
        pull_l2=knobs.pull_l2,
        bc_steps=knobs.bc_steps,
        critic_warmup=knobs.critic_warmup,
        qat=qat,
        qat_avg_const=knobs.qat_avg_const,
    )
    agent, _ = make_agent(
        design=knobs.pp_design,
        algo=knobs.pp_algo,
        n_pulses=n_pulses,
        k=k,
        init_bias=knobs.init_bias_scale,
        noise=knobs.pp_noise,
        view=knobs.pp_view,
        delta_max=knobs.delta_max,
        mix_init=knobs.mix_init,
        init_width=knobs.init_width,
    )
    return agent


def _perpulse_loop(
    *,
    agent: Any,
    env: MehreganEnv,
    knobs: PaperKnobs,
    mean_hz: float,
    seed: int,
    n_pulses: int,
    checkpoint_path: Path | None,
    num_episodes: int,
    log: Callable[[str], None],
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    from controllers.ddpg.action_designs import SlotAlphabet, regular_onsets

    slots_alpha = SlotAlphabet(mean_hz, step_s=STEP_S, dt_ms=PAPER_DT_MS)
    env.alphabet = slots_alpha  # env.step(0) plays slots_alpha.pending
    reg = regular_onsets(n_pulses)
    rng = np.random.default_rng(seed)
    rng_sigma = np.random.default_rng(seed + 31)

    def net_state(obs: np.ndarray) -> np.ndarray:
        v = (float(np.asarray(obs).reshape(-1)[-1]) - knobs.state_offset) * knobs.state_scale
        return np.full(knobs.state_len, v, dtype=np.float32)

    def on_regular(slots: np.ndarray) -> float:
        return float(np.isin(reg, slots).mean())

    def isi_cv(slots: np.ndarray) -> float:
        d = np.diff(np.unique(slots))
        return float(d.std() / d.mean()) if d.size > 1 else float("nan")

    buf: list[tuple[np.ndarray, torch.Tensor, float, np.ndarray, float]] = []
    beta_trace: list[float] = []
    actions: list[list[int]] = []
    step_rewards: list[float] = []
    episode_rewards: list[float] = []
    reset_beta: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    losses: list[dict[str, float]] = []
    env_step = 0
    try:
        for episode in range(num_episodes):
            sigma = knobs.logit_noise_std
            if knobs.logit_noise_end is not None and num_episodes > 1:
                frac = episode / (num_episodes - 1)
                sigma = knobs.logit_noise_std + frac * (knobs.logit_noise_end - knobs.logit_noise_std)
            if knobs.carry_plant and episode > 0:
                obs, info0 = env.reset(seed=seed + episode, options={"keep_plant": True})
            else:
                obs, info0 = env.reset(seed=seed + episode)
            if hasattr(agent, "k"):
                agent.k.sigma = sigma
            reset_beta.append(float(info0["p_beta_norm"]))
            ep_reward = float(info0.get("reward", 0.0))
            s = net_state(obs)
            ep_losses: list[dict[str, float]] = []
            done = False
            while not done:
                sig_t = sigma * float(np.exp(rng_sigma.uniform(-knobs.sigma_jitter, knobs.sigma_jitter))) if knobs.sigma_jitter else sigma
                stored, slots = agent.act(torch.as_tensor(s).unsqueeze(0), sig_t)
                slots_alpha.pending = slots
                obs2, reward, terminated, truncated, info = env.step(0)
                done = bool(terminated or truncated)
                env_step += 1
                s2 = net_state(obs2)
                buf.append((s, stored, float(reward), s2, float(info.get("dw", 0.0))))
                if len(buf) > BUFFER_CAPACITY:
                    buf.pop(0)
                beta_trace.append(float(info["p_beta_norm"]))
                actions.append([int(v) for v in slots])
                step_rewards.append(float(reward))
                ep_reward += float(reward)
                s = s2
                if agent.on_policy:
                    ready = len(buf) >= knobs.pg_batch and env_step % knobs.update_every == 0
                else:
                    ready = len(buf) >= BATCH_SIZE and env_step % knobs.update_every == 0
                for _ in range(knobs.updates_per_step if ready else 0):
                    if agent.on_policy:  # the most recent transitions
                        idx = np.arange(len(buf) - knobs.pg_batch, len(buf))
                    else:
                        idx = rng.choice(len(buf), size=BATCH_SIZE, replace=False)
                    ep_losses.append(
                        agent.update(
                            torch.as_tensor(np.stack([buf[i][0] for i in idx])),
                            torch.stack([buf[i][1] for i in idx]),
                            torch.as_tensor([buf[i][2] for i in idx], dtype=torch.float32),
                            torch.as_tensor(np.stack([buf[i][3] for i in idx])),
                            torch.as_tensor([buf[i][4] for i in idx], dtype=torch.float32),
                        )
                    )
            episode_rewards.append(ep_reward)
            losses.extend(ep_losses)
            ep_slots = [np.asarray(a) for a in actions[-STEPS_PER_EPISODE:]]
            s_mean = torch.as_tensor(net_state(np.array([np.mean(beta_trace[-STEPS_PER_EPISODE:])]))).unsqueeze(0)
            nf = agent.noise_free(s_mean)
            extra_diag = agent.diag(s_mean)
            diag = {
                "episode": episode,
                "sigma": sigma,
                "mean_beta": float(np.mean(beta_trace[-STEPS_PER_EPISODE:])),
                "played_on_regular": float(np.mean([on_regular(a) for a in ep_slots])),
                "played_isi_cv": float(np.nanmean([isi_cv(a) for a in ep_slots])),
                "nf_on_regular": on_regular(nf),
                "nf_isi_cv": isi_cv(nf),
                "nf_distinct_slots": int(np.unique(nf).size),
                **{f"nf_{k}": v for k, v in extra_diag.items()},
                "critic_loss": float(np.mean([x["loss_c"] for x in ep_losses])) if ep_losses else float("nan"),
                "q": float(np.nanmean([x["q"] for x in ep_losses])) if ep_losses and any(np.isfinite(x["q"]) for x in ep_losses) else float("nan"),
                "updates": len(losses),
            }
            diagnostics.append(diag)
            log(
                f"episode {episode + 1}/{num_episodes} reward={ep_reward:.2f} mean_beta={diag['mean_beta']:.3f} "
                f"played_on_reg={diag['played_on_regular']:.2f} nf_on_reg={diag['nf_on_regular']:.2f} "
                f"nf_isi_cv={diag['nf_isi_cv']:.2f} nf_slots={diag['nf_distinct_slots']} "
                + " ".join(f"{k}={v:.3g}" for k, v in extra_diag.items())
                + f" critic_loss={diag['critic_loss']:.3g} q={diag['q']:.3g} updates={diag['updates']}"
            )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "action_head": "perpulse",
                    "actor_state_dict": agent.actor.state_dict(),
                    "state_len": knobs.state_len,
                    "n_pulses": n_pulses,
                    "mean_hz": mean_hz,
                    "knobs": knobs.as_dict(),
                    "completed_episodes": num_episodes,
                    **(extra or {}),
                },
                checkpoint_path,
            )
            log(f"wrote checkpoint {checkpoint_path}")
    finally:
        env.close()

    return {
        "mean_hz": mean_hz,
        "seed": seed,
        "variant": "paper",
        "num_episodes": num_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "knobs": knobs.as_dict(),
        "paper_stated": {
            "actor_lr": ACTOR_LR,
            "critic_lr": CRITIC_LR,
            "buffer_capacity": BUFFER_CAPACITY,
            "batch_size": BATCH_SIZE,
            "step_s": STEP_S,
            "dt_ms": PAPER_DT_MS,
            "critic_action_input": "logits",
            "action_head": "Figure 3a per-pulse logits",
        },
        "beta_trace": beta_trace,
        "actions": actions,
        "step_rewards": step_rewards,
        "episode_rewards": episode_rewards,
        "reset_beta": reset_beta,
        "unique_actions": len({tuple(a) for a in actions}),
        "action_counts": {},
        "perpulse_diagnostics": diagnostics,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }


# --- Policies for eval ------------------------------------------------------

Policy = Callable[[np.ndarray], tuple[int, np.ndarray]]


def _module_dtype(module: nn.Module) -> torch.dtype:
    for p in module.parameters():
        if torch.is_floating_point(p):
            return p.dtype
    return torch.float32


def module_policy(module: nn.Module) -> Policy:
    """Greedy ``argmax`` over the module's logits (deployed policy)."""
    module.eval()
    dtype = _module_dtype(module)

    def act(state: np.ndarray) -> tuple[int, np.ndarray]:
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(dtype)
            logits = module(x).float().squeeze(0)
        return int(torch.argmax(logits).item()), logits.numpy().copy()

    return act


def load_fp32_actor(path: Path) -> Actor:
    from controllers.ddpg import load_actor

    actor, _ = load_actor(path)
    actor.eval()
    return actor


def ptq_actor(actor: Actor, kind: str) -> nn.Module:
    """§III.D PTQ: fp16 cast, or PyTorch ``quantize_dynamic`` int8."""
    src = copy.deepcopy(actor).cpu().eval()
    if kind == "fp16":
        return src.half()
    if kind == "int8":
        return torch.ao.quantization.quantize_dynamic(src, {nn.Linear}, dtype=torch.qint8)
    msg = f"unknown PTQ kind {kind!r}"
    raise ValueError(msg)


def load_qat_actor(path: Path) -> nn.Module:
    """QAT actor with its trained fake-quant parameters, observers frozen."""
    from controllers.ddpg.checkpoint import load_checkpoint, qat_state_dict_from_checkpoint

    payload = load_checkpoint(path)
    base = load_fp32_actor(path)
    qat = QATActor(base)
    state = qat_state_dict_from_checkpoint(payload)
    if state is None:
        msg = f"{path} has no qat_state_dict — not a QAT checkpoint"
        raise ValueError(msg)
    qat.load_state_dict(state)
    qat.apply(torch.ao.quantization.disable_observer)
    qat.eval()
    return qat


# --- Per-pulse checkpoints (candidate; see PaperKnobs.action_head) -----------


def is_perpulse_checkpoint(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return isinstance(payload, dict) and payload.get("action_head") == "perpulse"


def load_perpulse(path: Path) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Rebuild a per-pulse actor from its checkpoint: ``(actor, design, payload)``.

    QAT checkpoints come back as the fake-quant ``QATActor`` with observers off.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    knobs = apply_overrides(PaperKnobs(), [f"{k}={v}" for k, v in payload["knobs"].items()
                                            if k in PaperKnobs.__dataclass_fields__ and v is not None])
    n_pulses = int(payload["n_pulses"])
    seed = 0
    if knobs.pp_design == "faithful":
        agent = _FaithfulPulseAgent(knobs, n_pulses=n_pulses, seed=seed)
        design = agent.design
    else:
        agent = _deviation_agent(knobs, n_pulses=n_pulses, seed=seed, qat=bool(payload.get("qat")))
        design = agent.d
    agent.actor.load_state_dict(payload["actor_state_dict"])
    if payload.get("qat"):
        agent.actor.apply(torch.ao.quantization.disable_observer)
    agent.actor.eval()
    return agent.actor, design, payload


def perpulse_policy(module: nn.Module, design: Any, *, state_len: int, knobs: dict[str, Any]) -> Policy:
    """Deployed per-pulse policy: argmax slot of each pulse row, no exploration."""
    module.eval()
    dtype = _module_dtype(module)
    offset, scale = float(knobs.get("state_offset", 0.0)), float(knobs.get("state_scale", 1.0))

    def act(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = (float(np.asarray(state).reshape(-1)[-1]) - offset) * scale
        with torch.no_grad():
            raw = module(torch.full((1, state_len), v, dtype=dtype)).float()
        return design.decode_raw(raw), np.zeros(0)

    return act


def perpulse_core(module: nn.Module) -> nn.Module:
    return module.actor if isinstance(module, QATActor) else module


def perpulse_ptq(module: nn.Module, kind: str, *, state_len: int, calib: np.ndarray | None = None) -> nn.Module:
    """PTQ of a per-pulse actor: ``fp16``, ``int8`` (dynamic) or ``int8_static``.

    ``int8_static`` (calibrated PTQ, qnnpack): quant/dequant stubs around the
    actor, per-tensor int8 weights and activations, calibrated on the network
    inputs ``calib`` ($P_\\beta/1000$ values; default 0.25-0.60).
    """
    src = copy.deepcopy(perpulse_core(module)).cpu().eval()
    if kind == "fp16":
        return src.half()
    if kind == "int8":
        return torch.ao.quantization.quantize_dynamic(src, {nn.Linear}, dtype=torch.qint8)
    if kind == "int8_static":
        torch.backends.quantized.engine = "qnnpack"
        wrapped = torch.ao.quantization.QuantWrapper(src)
        wrapped.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        torch.ao.quantization.prepare(wrapped, inplace=True)
        values = np.linspace(0.25, 0.60, 36) if calib is None else np.asarray(calib, dtype=float)
        with torch.no_grad():
            for v in values:
                wrapped(torch.full((1, state_len), float(v)))
        return torch.ao.quantization.convert(wrapped, inplace=False)
    msg = f"unknown PTQ kind {kind!r}"
    raise ValueError(msg)


# --- Closed-loop eval -------------------------------------------------------


def sample_times() -> np.ndarray:
    return _FIG2A.sample_times(SAMPLE_S, duration_s=DISPLAY_S)


def run_condition(
    *,
    seed: int,
    alphabet: Any,
    policy: Policy | None = None,
    fixed_spec: DbsSpec | None = None,
    obs_scale: float = 1000.0,
) -> dict[str, Any]:
    """One eval condition on a fixed seed with plant state carried across steps.

    ``policy`` → closed loop: each 2 s step plays ``argmax`` of the actor on the
    previous step's $P_\\beta$. ``fixed_spec`` → the same DBS every step.
    Neither → no stimulation.
    """
    plant = PythonPlant(config=plant_config())
    plant.reset(seed=seed)
    spikes: list[list[np.ndarray]] = []
    seg_beta: list[float] = []
    actions: list[int] = []
    logits_rows: list[list[float]] = []

    def _seg(spec: DbsSpec) -> float:
        res = plant.integrate(STEP_S, spec, carry=True, fixed_network=True)
        if res.p_beta is None:
            raise RuntimeError("plant integrate returned no p_beta")
        offset = len(spikes) * STEP_S
        spikes.append([np.asarray(s, dtype=float).reshape(-1) + offset for s in res.gpi_spikes])
        seg_beta.append(float(res.p_beta))
        return float(res.p_beta)

    _seg(DbsSpec.none())  # pre-roll (display −2..0)
    beta = _seg(DbsSpec.none())  # reset segment → s0 (display 0..2)
    for _ in range(EVAL_STEPS):
        if policy is not None:
            state = np.array([beta / obs_scale], dtype=np.float32)
            action, logits = policy(state)
            actions.append(action if np.ndim(action) == 0 else [int(v) for v in action])
            if len(logits):
                logits_rows.append([float(v) for v in logits])
            spec = alphabet.to_dbs_spec(action)
        elif fixed_spec is not None:
            spec = fixed_spec
        else:
            spec = DbsSpec.none()
        beta = _seg(spec)
    plant.close()

    n_neurons = len(spikes[0])
    merged = [np.concatenate([seg[i] for seg in spikes]) for i in range(n_neurons)]
    times = sample_times()
    trace = _FIG2A.trailing_p_beta(
        merged,
        dt_ms=PAPER_DT_MS,
        times=times,
        window_s=WINDOW_S,
        verbose=False,
    )
    return {
        "time_s": times.tolist(),
        "trace": [float(v) for v in trace],
        "segment_p_beta": seg_beta,
        "actions": actions,
        "logits": logits_rows,
        "closed_loop": policy is not None,
    }


def regular_spec(alphabet: Any) -> DbsSpec:
    """Periodic train at the alphabet's mean rate (pattern 0 / regular grid)."""
    idbs = np.asarray(alphabet.idbs_for_pattern(0), dtype=np.float64)
    return DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(alphabet.mean_hz).pick_dbs_freq,
        idbs=idbs,
        mean_hz=alphabet.mean_hz,
    )


def post_mean(cond: dict[str, Any], *, lo: float = 4.0, hi: float | None = None) -> float:
    t = np.asarray(cond["time_s"], dtype=float)
    y = np.asarray(cond["trace"], dtype=float)
    m = t >= lo
    if hi is not None:
        m &= t <= hi
    return float(np.mean(y[m]))


def pre_mean(cond: dict[str, Any], *, onset: float = ONSET_DISPLAY_S) -> float:
    t = np.asarray(cond["time_s"], dtype=float)
    y = np.asarray(cond["trace"], dtype=float)
    return float(np.mean(y[t <= onset]))
