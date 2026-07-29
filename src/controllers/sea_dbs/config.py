"""SEA-DBS hyperparameters (Ravivarapu et al., docs/controllers/sea_dbs/replication.md)."""

from __future__ import annotations

from dataclasses import dataclass, replace

# Paper §V.A / Table I (fixed RL cadence metadata).
STEP_DURATION_MS: float = 2.0
# Minimum plant segment for multitaper P_beta (Kumaravelu GPi spikes); 2 ms yields zero PSD.
BIOMARKER_WINDOW_S: float = 0.1
MAX_EPISODE_STEPS: int = 30
TRAIN_EPISODES: int = 150
BETA_THRESHOLD: float = 0.35
REWARD_SCALE: float = 10.0
OBSERVATION_SCALE: float = 1000.0
BUFFER_CAPACITY: int = 8192
BATCH_SIZE: int = 32
GAMMA: float = 0.99
ACTOR_LR: float = 5e-4
CRITIC_LR: float = 1e-3
DEFAULT_CARRIER_HZ: float = 50.0
INFERENCE_CARRIER_30HZ: float = 30.0
INFERENCE_CARRIER_50HZ: float = 50.0
ABLATION_EVAL_STEPS: int = 10


@dataclass(frozen=True)
class SEADBSConfig:
    """Ravivarapu SEA-DBS defaults; open hyperparameters documented beside fields."""

    # RL timing (§5)
    step_duration_ms: float = STEP_DURATION_MS
    biomarker_window_s: float = BIOMARKER_WINDOW_S
    max_episode_steps: int = MAX_EPISODE_STEPS
    num_episodes: int = TRAIN_EPISODES

    # Reward / observation (§6, §14.1)
    beta_threshold: float = BETA_THRESHOLD
    reward_scale: float = REWARD_SCALE
    observation_scale: float = OBSERVATION_SCALE
    n_obs: int = 5  # window for mean P_beta (Eq. 4–5); open in Table I
    state_dim: int = 1  # default: mean P_bar only (replication.md §14.1)

    # Table I
    actor_lr: float = ACTOR_LR
    critic_lr: float = CRITIC_LR
    pred_lr: float = ACTOR_LR  # open — same as actor by default
    buffer_capacity: int = BUFFER_CAPACITY
    batch_size: int = BATCH_SIZE
    gamma: float = GAMMA
    polyak_tau: float = 0.005  # open — standard DDPG default
    min_buffer_size: int = BATCH_SIZE
    update_frequency: int = 1

    # Gumbel-Softmax (§8, Eq. 14) — open schedule
    gs_tau0: float = 1.0
    gs_tau_min: float = 0.1
    gs_lambda: float = 1e-4

    # Baseline exploration (no GS)
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05

    # MLP topology (§14.9 — open)
    hidden_size: int = 64

    # Plant / DBS carrier (§14.10 — fixed eval knob, default train carrier)
    carrier_hz: float = DEFAULT_CARRIER_HZ
    plant_dt_ms: float = 0.02

    variant: str = "paper"
    seed: int = 0
    device: str = "cpu"
    log_episodes: bool = False

    @property
    def step_duration_s(self) -> float:
        return self.step_duration_ms / 1000.0

    @property
    def integration_duration_s(self) -> float:
        """Plant integrate duration per RL step (≥ biomarker_window_s for valid P_beta)."""
        return max(self.step_duration_s, self.biomarker_window_s)

    @property
    def use_predictive_model(self) -> bool:
        return self.variant in {"baseline-pm", "paper"}

    @property
    def use_gumbel_softmax(self) -> bool:
        return self.variant in {"baseline-gs", "paper"}

    @property
    def n_actions(self) -> int:
        return 2

    def with_variant_defaults(self) -> SEADBSConfig:
        if self.variant in {"baseline", "baseline-pm", "baseline-gs", "paper"}:
            return self
        msg = f"unknown sea_dbs variant {self.variant!r}"
        raise ValueError(msg)

    def for_smoke(
        self,
        *,
        episodes: int = 2,
        max_steps: int = 5,
    ) -> SEADBSConfig:
        return replace(
            self,
            num_episodes=int(episodes),
            max_episode_steps=int(max_steps),
            min_buffer_size=min(self.min_buffer_size, 8),
            batch_size=min(self.batch_size, 8),
        )


def fig4_ravivarapu_config(*, seed: int = 0, num_episodes: int = TRAIN_EPISODES) -> SEADBSConfig:
    """Shared Fig 4a/4b training defaults (150 episodes, seed 0)."""
    return SEADBSConfig(seed=seed, num_episodes=num_episodes, log_episodes=True)
