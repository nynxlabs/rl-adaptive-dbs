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

    # DQN / replay (open stabilizers documented beside code)
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_capacity: int = 10_000
    replay_update_cadence: int = REPLAY_UPDATE_CADENCE
    batch_size: int = 32
    # Hard-copy target network every N gradient updates (paper silent — convention).
    target_update_period: int = 100

    # Exploration (ε-greedy on spike-count argmax)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2_500

    # Logging
    log_episodes: bool = False

    # Action selection: ``factored`` (3× argmax over ternary groups) or ``joint`` (9-way)
    action_scheme: str = "factored"

    # Per-parameter ternary delta sensitivities (open — keep params in plausible ranges)
    amplitude_sensitivity: float = 10.0  # nA/cm² per +1
    frequency_sensitivity: float = 5.0  # Hz per +1
    pulse_width_sensitivity: float = 0.05  # ms per +1

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
            replay_capacity=128,
            target_update_period=2,
            epsilon_decay_steps=max(1, int(episodes) * int(max_steps)),
            log_episodes=True,
            frequency_min=10.0,
            amplitude_min=50.0,
        )


def fig4_nguyen_config(
    seed: int = 0,
    *,
    num_episodes: int = TRAIN_EPISODES,
) -> SNNConfig:
    """Nguyen Fig. 4 train defaults (figures/nguyen/replications.md § Fig 4).

    Paper Eq. (7) + probe-driven shaping (v9): progress/warm-zone bonuses,
    truncation penalty for 25-step timeouts, faster freq ramp. v11 removed
    shaping for negative-million digitization band only — regressed to energy
    collapse (v13); keep v9 shaping for learnable early-stop.

    v10c: subthreshold_steps_required=2 for easier early-stop.
    v43 FAIL: unscaled Bellman (`reward_learning_scale=1.0`) blew up Q
    (loss ~10^9–10^10); early-stops vanished after ep ~60; late_len=25,
    late_reward≈−1.07e6, energy 660→82. v44: same v10 schedule/shaping
    with v15 `reward_learning_scale=1e-4` restored.
    """
    return SNNConfig(
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=EVAL_MAX_STEPS,
        alpha_beta_threshold=BIOMARKER_THRESHOLD,
        subthreshold_steps_required=2,
        epsilon_decay_steps=3_500,
        epsilon_end=0.15,
        learning_rate=5e-4,
        frequency_sensitivity=15.0,
        pulse_width_sensitivity=0.1,
        threshold_reward=300.0,
        energy_penalty=0.0,
        alpha_beta_progress_coef=2000.0,
        alpha_beta_progress_cap_per_step=10_000.0,
        warm_zone_upper=220.0,
        warm_zone_bonus_coef=150.0,
        truncation_penalty=600_000.0,
        reward_learning_scale=1e-4,
        stimulated_neurons=1,
        log_episodes=True,
    )
