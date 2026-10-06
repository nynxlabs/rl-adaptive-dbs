"""SNN / DSQN hyperparameters (Nguyen et al., docs/controllers/snn/replication.md)."""

from __future__ import annotations

from dataclasses import dataclass, replace

# Paper-fixed defaults (replication.md §4–§6).
STEP_DURATION_MS: float = 100.0
BIOMARKER_THRESHOLD: float = 150.0
LIF_LEAK: float = 0.95
HIDDEN_SIZE: int = 128
N_ACTION_OUTPUTS: int = 9
REPLAY_UPDATE_CADENCE: int = 128
TRAIN_EPISODES: int = 500
EVAL_EPISODES: int = 50
EVAL_MAX_STEPS: int = 25

# Initial DBS triple (§IV).
INIT_FREQUENCY_HZ: float = 40.0
INIT_PULSE_WIDTH_MS: float = 0.3
INIT_AMPLITUDE_NA_PER_CM2: float = 300.0


@dataclass(frozen=True)
class SNNConfig:
    """Nguyen DSQN defaults; open hyperparameters are config fields (replication.md §10)."""

    # RL timing
    step_duration_ms: float = STEP_DURATION_MS
    # One continuous plant simulation per episode (state + pulse phase carried across
    # 100 ms steps) vs a fresh integrate from initial conditions every step. The
    # follow-up paper (arXiv 2606.28600) randomizes initial conditions per episode.
    plant_carry: bool = False
    # With plant_carry: keep one network (wiring, conductances) for the whole episode
    # instead of redrawing it every 100 ms segment. Off by default: the redrawing plant
    # reproduces Figs 4-6 and the fixed network does not (docs/figures/nguyen/4.md).
    plant_fixed_network: bool = False
    max_episode_steps: int = EVAL_MAX_STEPS
    num_episodes: int = TRAIN_EPISODES

    # Observation layout (intentionally open — fixed across train/eval once chosen)
    sequence_steps: int = 10
    neurons_per_region: int = 10
    n_regions: int = 1  # scaffold: GPi only; expand when encoder covers full CBGT

    # Biomarker / termination
    alpha_beta_threshold: float = BIOMARKER_THRESHOLD
    subthreshold_steps_required: int = 3  # t_u — open in paper

    # DSQN topology
    hidden_size: int = HIDDEN_SIZE
    n_action_outputs: int = N_ACTION_OUTPUTS
    lif_leak: float = LIF_LEAK
    lif_threshold: float = 1.0  # θ_th — open in paper
    internal_unroll_steps: int = 5
    # Spike nonlinearity gradient. "none": hard threshold with no gradient, so only the
    # output layer's weights learn. "atan": arctan surrogate (snnTorch default, alpha 2),
    # the surrogate-gradient training the authors' follow-up (arXiv 2606.28600) cites.
    surrogate_gradient: str = "none"
    # "static": the flattened n×N spike matrix drives every layer for internal_unroll_steps,
    # and each layer passes spike counts on. "sequence": row t of the matrix drives SNN step t
    # (n steps) and spikes propagate through all layers every step.
    snn_input_mode: str = "static"

    # DQN / replay (open stabilizers documented beside code)
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_capacity: int = 10_000
    replay_update_cadence: int = REPLAY_UPDATE_CADENCE
    # Gradient minibatches per replay flush (paper silent). 4 × batch 32 = one pass's worth of
    # samples for the 128 transitions collected since the last flush.
    replay_update_steps: int = 4
    # No gradient updates until the buffer holds this many transitions (DQN "learning
    # starts"); 0 = start as soon as a batch fits. Paper-silent.
    replay_warmup_transitions: int = 0
    batch_size: int = 32
    # Bellman TD loss: ``mse`` or ``huber`` (smooth L1 — dampens timeout Q spikes).
    q_loss_fn: str = "mse"
    # Double DQN: online net picks argmax a' at s'; target net evaluates Q(s', a').
    double_dqn: bool = False
    # Per-transition loss weight for max-horizon timeout episodes (1.0 = uniform sampling).
    replay_timeout_weight: float = 1.0
    # Down-weight early-stop episodes with length <= replay_short_stop_max_steps (0 = off).
    replay_short_stop_max_steps: int = 0
    replay_short_stop_weight: float = 1.0
    # Hard-copy target network every N gradient updates. 0 = no separate target network:
    # bootstrap from the online weights θ, as in the paper's Q-target (Nguyen et al. §II).
    target_update_period: int = 0
    # Polyak-average the target network after every gradient update:
    # θ' ← τ θ + (1 − τ) θ'. 0 = off. The group's published DSQN implementation uses 0.005
    # (Nguyen et al. 2026, arXiv 2606.28600, Algorithm S1 / Table S3). Overrides
    # ``target_update_period`` when > 0.
    target_soft_update_tau: float = 0.0
    # ``adam`` or ``adamw`` (the group's published implementation uses AdamW, torch defaults).
    optimizer: str = "adam"
    # Gradient clipping: ``norm`` (clip_grad_norm_) or ``value`` (clip_grad_value_, per element;
    # the group's published implementation uses value 100). ``grad_clip <= 0`` disables it.
    grad_clip_mode: str = "norm"
    grad_clip: float = 10.0

    # Exploration (ε-greedy on spike-count argmax)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2_500
    # "linear": ramp start→end over epsilon_decay_steps. "exp": PyTorch DQN-tutorial
    # end + (start − end)·exp(−steps / epsilon_decay_steps) — the reading of the
    # follow-up's "ε decay steps 2,000" (arXiv 2606.28600 Table S3 uses that stack).
    epsilon_schedule: str = "linear"
    # Hold ε at start for this many env steps, then linear decay (0 = no delay).
    epsilon_decay_delay_steps: int = 0
    # After this many env steps, dump remaining ε to epsilon_end faster (0 = off).
    epsilon_accelerate_after_steps: int = 0
    epsilon_accelerate_decay_steps: int = 0

    # Logging
    log_episodes: bool = False

    # Action selection: ``factored`` (3× argmax over ternary groups) or ``joint`` (9-way)
    action_scheme: str = "factored"

    # Per-parameter ternary delta sensitivities (open — keep params in plausible ranges)
    amplitude_sensitivity: float = 10.0  # nA/cm² per +1
    frequency_sensitivity: float = 5.0  # Hz per +1
    # Episodes [0, N) use early Hz/step; episode N+ uses frequency_sensitivity.
    frequency_sensitivity_early: float = 0.0
    frequency_sensitivity_early_episodes: int = 0
    # When > 0, linearly schedule frequency_sensitivity from this value at
    # epsilon_start down to frequency_sensitivity at epsilon_end (Fig 4 v66).
    frequency_sensitivity_explore: float = 0.0
    # When explore scheduling is on, hold exploit Hz/step while ε is above this
    # (protects ep1–~30 from lucky 80 Hz random walks; v66 FAIL at ε≈1).
    frequency_sensitivity_explore_epsilon_max: float = 0.7
    pulse_width_sensitivity: float = 0.05  # ms per +1
    # Episodes [0, N) use early ms/step; episode N+ uses pulse_width_sensitivity.
    pulse_width_sensitivity_early: float = 0.0
    pulse_width_sensitivity_early_episodes: int = 0

    # Reward Eq. (7) coefficients (open)
    energy_penalty: float = 0.01  # δ
    threshold_reward: float = 1.0  # τ
    # Dense shaping when α–β drops but remains above θ (paper-silent; Fig 4 only).
    alpha_beta_progress_coef: float = 0.0
    # Cap per-step progress bonus (0 = no cap); blocks α–β wiggle farming on timeouts.
    alpha_beta_progress_cap_per_step: float = 0.0
    # Bonus when α–β is above θ but within warm_zone_upper (approaching suppression).
    warm_zone_upper: float = 0.0
    warm_zone_bonus_coef: float = 0.0
    # Penalty on max-length timeout without early stop (Fig 4 learnability).
    truncation_penalty: float = 0.0
    # Bellman reward scale for DQN updates only (episode logs stay raw).
    reward_learning_scale: float = 1.0

    # DBS parameter bounds (adapter clamping)
    amplitude_min: float = 0.0
    amplitude_max: float = 500.0
    frequency_min: float = 0.0
    frequency_max: float = 200.0
    pulse_width_min: float = 0.05
    pulse_width_min_early: float = 0.05
    pulse_width_min_early_episodes: int = 0
    pulse_width_min_ramp_end_episode: int = 0
    pulse_width_max: float = 2.0

    # Stimulated neuron count N in Eq. (6) — single STN contact (paper Fig. 5 scale).
    stimulated_neurons: int = 1

    variant: str = "paper"
    seed: int = 0
    device: str = "cpu"

    @property
    def step_duration_s(self) -> float:
        return self.step_duration_ms / 1000.0

    @property
    def observation_shape(self) -> tuple[int, int]:
        n_neurons = self.neurons_per_region * self.n_regions
        return (self.sequence_steps, n_neurons)

    @property
    def flat_observation_dim(self) -> int:
        rows, cols = self.observation_shape
        return rows * cols

    def with_variant_defaults(self) -> SNNConfig:
        """Return config unchanged for ``paper``; hook for future benchmark variants."""
        if self.variant == "paper":
            return self
        return replace(self)

    def frequency_sensitivity_at_epsilon(
        self,
        epsilon: float,
        *,
        episode: int | None = None,
    ) -> float:
        """Effective Hz/step for ternary +freq (episode curriculum, then ε schedule)."""
        early_eps = int(self.frequency_sensitivity_early_episodes)
        early_hz = float(self.frequency_sensitivity_early)
        if early_eps > 0 and early_hz > 0.0 and episode is not None and episode < early_eps:
            return early_hz

        explore = float(self.frequency_sensitivity_explore)
        exploit = float(self.frequency_sensitivity)
        if explore <= 0.0 or abs(explore - exploit) < 1e-9:
            return exploit
        eps = float(epsilon)
        end = float(self.epsilon_end)
        if eps <= end:
            return exploit
        eps_hi = float(self.frequency_sensitivity_explore_epsilon_max)
        if eps_hi > end and eps > eps_hi:
            return exploit
        start = float(self.epsilon_start)
        if start <= end:
            return exploit
        # Linear ramp within [epsilon_end, epsilon_hi] (mid-anneal band only).
        hi = min(start, eps_hi) if eps_hi > end else start
        span = hi - end
        if span <= 0.0:
            return exploit
        t = (eps - end) / span
        t = max(0.0, min(1.0, t))
        return exploit + t * (explore - exploit)

    def pulse_width_sensitivity_at_epsilon(
        self,
        epsilon: float,
        *,
        episode: int | None = None,
    ) -> float:
        """Effective ms/step for ternary +pulse_width (early episode curriculum)."""
        del epsilon
        early_eps = int(self.pulse_width_sensitivity_early_episodes)
        early_pw = float(self.pulse_width_sensitivity_early)
        if early_eps > 0 and early_pw > 0.0 and episode is not None and episode < early_eps:
            return early_pw
        return float(self.pulse_width_sensitivity)

    def pulse_width_min_at_episode(
        self,
        episode: int | None = None,
    ) -> float:
        """Effective minimum pulse width (early episode curriculum + linear ramp)."""
        if episode is None:
            return float(self.pulse_width_min)
        early_eps = int(self.pulse_width_min_early_episodes)
        ramp_end = int(self.pulse_width_min_ramp_end_episode)
        early_min = float(self.pulse_width_min_early)
        late_min = float(self.pulse_width_min)
        if early_eps <= 0 and ramp_end <= 0:
            return late_min
        if episode < early_eps:
            return early_min
        if ramp_end > early_eps:
            if episode >= ramp_end:
                return late_min
            t = (episode - early_eps) / float(ramp_end - early_eps)
            return early_min + t * (late_min - early_min)
        return late_min

    def for_smoke(
        self,
        *,
        episodes: int = 2,
        max_steps: int = 10,
    ) -> SNNConfig:
        """Tiny DSQN + short rollouts for CLI/pytest smoke (Python plant)."""
        return replace(
            self,
            sequence_steps=4,
            neurons_per_region=4,
            n_regions=1,
            hidden_size=16,
            internal_unroll_steps=2,
            num_episodes=int(episodes),
            max_episode_steps=int(max_steps),
            batch_size=8,
            replay_update_cadence=8,
            replay_update_steps=1,
            replay_capacity=128,
            target_update_period=2,
            epsilon_decay_steps=max(1, int(max_steps) * 10),
            log_episodes=True,
            frequency_min=10.0,
            amplitude_min=50.0,
        )


def fig4_nguyen_config(
    seed: int = 0,
    *,
    num_episodes: int = TRAIN_EPISODES,
) -> SNNConfig:
    """Nguyen Fig. 4–6 training config (docs/figures/nguyen/4.md, 6.md).

    Paper-stated: Eq. (7) reward with unnormalized d, θ = 150, init 300 nA/cm² / 40 Hz /
    0.3 ms, 25-step episodes, replay update every 128 transitions, LIF DSQN, binary spikes.
    From the authors' follow-up (arXiv 2606.28600): soft target τ 0.005, AdamW, SmoothL1,
    value clipping 100, γ 0.99, lr 1e-3, buffer 100k, batch 128, ε 0.9 → 0.05 over 2,000
    steps, surrogate-gradient training, per-episode continuous simulation, all eight CBGT
    populations observed. Step sizes 5 Hz / 0.1 ms read off Fig 6 (40 + 8 × 5 = 80 Hz,
    0.3 + 8 × 0.1 = 1.1 ms in ~8-step episodes). Paper-silent conventions: t_u = 3,
    amplitude step 10 nA/cm², δ = 1, τ = 330, 32 minibatches per update, 1,500-transition
    replay warm-up, rewards scaled by 1e-4 for learning only. 13 seeds: 6 pass every
    Fig 4 gate and the 13-seed mean passes every gate; the shipped figure is seed 1.
    """
    return SNNConfig(
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=EVAL_MAX_STEPS,
        alpha_beta_threshold=BIOMARKER_THRESHOLD,
        subthreshold_steps_required=3,
        energy_penalty=1.0,
        threshold_reward=330.0,
        reward_learning_scale=1e-4,
        amplitude_sensitivity=10.0,
        frequency_sensitivity=5.0,
        pulse_width_sensitivity=0.1,
        plant_carry=True,
        n_regions=8,
        neurons_per_region=10,
        sequence_steps=10,
        surrogate_gradient="atan",
        snn_input_mode="static",
        replay_update_cadence=128,
        replay_update_steps=32,
        replay_warmup_transitions=1_500,
        replay_capacity=100_000,
        batch_size=128,
        q_loss_fn="huber",
        optimizer="adamw",
        grad_clip_mode="value",
        grad_clip=100.0,
        learning_rate=1e-3,
        gamma=0.99,
        target_update_period=0,
        target_soft_update_tau=0.005,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay_steps=2_000,
        epsilon_schedule="linear",
        stimulated_neurons=1,
        log_episodes=True,
    )
