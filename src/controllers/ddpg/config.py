"""DDPG hyperparameters and variant slugs (Mehregan et al. §IV.A.1)."""

from __future__ import annotations

from dataclasses import dataclass, replace


def init_baseline_for_variant(
    variant: str,
    *,
    action_space_mode: str = "scalar_frequency",
    pattern_mean_hz: float = 45.0,
) -> str:
    """Map benchmark variant slug to periodic init frequency.

    In ``fixed_mean_pattern`` mode the init baseline is always the regular
    train at ``pattern_mean_hz`` (pattern 0), regardless of variant.
    """
    if action_space_mode == "fixed_mean_pattern":
        return f"periodic-{int(pattern_mean_hz)}hz"
    if variant == "init-30hz":
        return "periodic-30hz"
    if variant in ("ptq-fp16", "ptq-int8", "qat", "paper"):
        return "periodic-45hz"
    return "periodic-45hz"


@dataclass(frozen=True)
class DDPGConfig:
    """Training defaults aligned with [replication.md](../../docs/controllers/ddpg/replication.md)."""

    # Paper §IV.A.1 (fixed)
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    buffer_capacity: int = 8192
    batch_size: int = 32
    num_episodes: int = 10
    max_episode_steps: int = 30

    # Not in §IV.A.1 — standard DDPG defaults (documented in replication.md §4.1, §9.3)
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = 1

    # CNN topology (§III.B, Figure 3a/3b — TASK-146)
    conv1_out: int = 32
    conv2_out: int = 64
    fc_hidden: int = 256
    pool_kernel: int = 2

    # Benchmark variant (`paper`, `init-30hz`, …)
    variant: str = "paper"

    # Option C (TASK-83/84): ``fixed_mean_pattern`` trains on irregular pulse patterns at a
    # fixed mean rate; ``scalar_frequency`` keeps the Kumaravelu 0:5:200 Hz grid (default).
    action_space_mode: str = "scalar_frequency"  # scalar_frequency | fixed_mean_pattern
    pattern_mean_hz: float = 45.0

    # Training control
    min_buffer_size: int = 32
    seed: int = 0
    device: str = "cpu"

    # Critic action encoding: ``one_hot`` uses the executed discrete action index (required
    # for correct Q targets under exploration); ``logits`` uses stored actor logits (paper
    # tuple layout; breaks when executed action != argmax(logits)).
    critic_action_input: str = "one_hot"  # one_hot | logits

    # Exploration during training (paper uses greedy argmax at deploy; not specified for
    # online interaction — see replication.md §4.2). Linear decay over total env steps.
    exploration_mode: str = "epsilon"  # epsilon | softmax | greedy
    exploration_epsilon_start: float = 0.5
    exploration_epsilon_end: float = 0.1
    exploration_temperature_start: float = 2.0
    exploration_temperature_end: float = 0.5

    # Periodic init bias on actor head (§IV.A.1). Default 2.0 matches benchmark checkpoints;
    # lower values (e.g. 0.5) for learning_v1 experiments — see phase4-results.md §10.4.
    init_bias_scale: float = 2.0

    # Critic warmup: train critic for N gradient steps before actor starts updating.
    # Standard DDPG practice — lets Q-values stabilize before the actor exploits them.
    critic_warmup_steps: int = 0  # 0 = disabled; 100 is a good starting point

    # Reward normalization: running z-score on rewards so Q-values learn at the
    # right scale (fixes q_pred_std << reward_std mismatch observed in TASK-72/74).
    reward_normalize: bool = False

    # Loss function for critic: "mse" (original) or "huber" (smooth L1, more robust
    # to reward outliers and can help Q-learning converge faster).
    critic_loss_fn: str = "mse"  # mse | huber

    # Gaussian noise std added to actor logits during training action selection.
    # Prevents logit margin collapse by maintaining diversity in the actor's output.
    # 0 = disabled. Try 0.1–0.3.
    logit_noise_std: float = 0.0

    # Entropy regularization coefficient for the actor loss.
    # Prevents logit collapse by penalizing low-entropy action distributions.
    # 0 = disabled. Try 0.01–0.1.
    entropy_coeff: float = 0.0

    # Observation normalization: running per-element z-score so the CNN sees
    # amplified differences even when the raw P_beta window changes slowly.
    # Fixes constant-policy collapse when state_length > 1 (TASK-67).
    obs_normalize: bool = False

    # Random warmup: run N random-action steps to fill replay buffer before
    # any policy training. Prevents mode collapse by ensuring the critic sees
    # diverse state-action-reward transitions before the actor starts exploiting.
    # 0 = disabled. 200–500 recommended for state_length > 1.
    random_warmup_steps: int = 0

    # Print episode reward after each episode (long PythonPlant runs).
    log_episodes: bool = False

    @property
    def effective_pattern_mean_hz(self) -> float:
        """Mean Hz implied by variant + pattern mode (``init-30hz`` → 30 Hz)."""
        if self.variant == "init-30hz":
            return 30.0
        return self.pattern_mean_hz

    def with_variant_defaults(self) -> DDPGConfig:
        """Return a copy with variant-implied fields applied (e.g. ``init-30hz`` → 30 Hz)."""
        effective = self.effective_pattern_mean_hz
        if effective != self.pattern_mean_hz:
            return replace(self, pattern_mean_hz=effective)
        return self

    @property
    def init_baseline(self) -> str:
        return init_baseline_for_variant(
            self.variant,
            action_space_mode=self.action_space_mode,
            pattern_mean_hz=self.effective_pattern_mean_hz,
        )


def fig4a_ddpg_config(
    *,
    seed: int = 0,
    num_episodes: int = 10,
    max_episode_steps: int = 30,
    pattern_mean_hz: float = 45.0,
    exploration_mode: str = "softmax",
    init_bias_scale: float = 0.5,
    exploration_temperature_start: float = 3.0,
    exploration_temperature_end: float = 1.4,
    logit_noise_std: float = 0.1,
    entropy_coeff: float = 0.01,
    critic_action_input: str = "one_hot",
    critic_warmup_steps: int = 100,
    actor_lr: float = 5e-4,
) -> DDPGConfig:
    """Mehregan Fig 4a — 45 Hz pattern DDPG.

    Default profile: softmax + one_hot critic, τ 3→1.4 (softer late mix than
    the locked v18 τ→1.0 collapse). For paper-faithful
    Alg. 1 interaction use ``exploration_mode="greedy"`` + ``critic_action_input="logits"``
    (no online exploration noise, no replay warmup extensions).
    """
    if exploration_mode == "greedy":
        return DDPGConfig(
            variant="paper",
            seed=seed,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            action_space_mode="fixed_mean_pattern",
            pattern_mean_hz=pattern_mean_hz,
            exploration_mode="greedy",
            init_bias_scale=init_bias_scale,
            critic_action_input=critic_action_input,
            logit_noise_std=0.0,
            entropy_coeff=0.0,
            random_warmup_steps=0,
            critic_warmup_steps=0,
            log_episodes=True,
        )
    return DDPGConfig(
        variant="paper",
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        action_space_mode="fixed_mean_pattern",
        pattern_mean_hz=pattern_mean_hz,
        exploration_mode=exploration_mode,
        exploration_temperature_start=exploration_temperature_start,
        exploration_temperature_end=exploration_temperature_end,
        init_bias_scale=init_bias_scale,
        critic_action_input=critic_action_input,
        critic_warmup_steps=critic_warmup_steps,
        actor_lr=actor_lr,
        logit_noise_std=logit_noise_std,
        entropy_coeff=entropy_coeff,
        random_warmup_steps=100,
        log_episodes=True,
    )
