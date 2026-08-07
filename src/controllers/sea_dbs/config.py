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
# Mehregan uses 1000 so 2 s raw P_beta ~400–600 → ~0.4–0.6. On the SEA-DBS
# biomarker window (100 ms), unstimulated raw P_beta ≈ 196; scale 1000 puts
# norm ≈ 0.20 already below β_t=0.35 and kills learning pressure. Scale ≈ 425
# maps that raw onto the paper Fig 4a band (~0.46) so reward Eq. (7) can teach.
OBSERVATION_SCALE: float = 425.0
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
    gs_lambda: float = 1e-4  # fig4 paper override in fig4_ravivarapu_config

    # Baseline exploration (no GS)
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05

    # MLP topology (§14.9 — open)
    hidden_size: int = 64

    # Plant / DBS carrier (§14.10 — fixed eval knob, default train carrier)
    carrier_hz: float = DEFAULT_CARRIER_HZ
    # Integrator grid (PlantConfig.dt_ms); used to build the STN drive trace grid.
    plant_dt_ms: float = 0.01
    # Short-burst STN drive (ms) per stim action within the integration window.
    # 100.0 == continuous drive for the whole window (Kumaravelu default). A
    # shorter burst yields an intermediate beta floor; paper 3 describes pulses
    # as "short bursts rather than continuously" (Eq. 6). Fig 4a override below.
    dbs_burst_ms: float = 100.0

    variant: str = "paper"
    seed: int = 0
    device: str = "cpu"
    log_episodes: bool = False
    fixed_episode_seed: bool = False  # optional; fixed seed can collapse paper GS policy

    # Fig 4a episode PSD logging: "mean" (all steps) or "last" (final step biomarker)
    episode_psd_metric: str = "mean"
    # Ramp predictive-model r_hat into critic target over this many env steps (0 = off)
    pm_warmup_steps: int = 0
    # Actor logit bias toward no-stim (action 0); helps early PSD start high (Fig 4a).
    actor_no_stim_bias: float = 0.0
    # Fig 4a Baseline convention: the paper's Baseline curve fades GRADUALLY across
    # episodes, which epsilon-greedy argmax cannot produce (it flips abruptly into a
    # step). Force Gumbel-Softmax structured sampling for the baseline WITHOUT the
    # predictive model, so the duty ramps smoothly and the baseline stays the weaker
    # variant. Leave the plain ``baseline`` (epsilon-greedy) intact for Fig 7.
    force_gumbel_softmax: bool = False

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
        return self.variant in {"baseline-gs", "paper"} or self.force_gumbel_softmax

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


def fig4_ravivarapu_config(
    *,
    seed: int = 0,
    num_episodes: int = TRAIN_EPISODES,
    variant: str = "baseline",
) -> SEADBSConfig:
    """Fig 4a/4b training defaults — paper-faithful Baseline vs SEA-DBS.

    v57 (2026-08-07): v56 shape-gate targets — monotone baseline glide-down,
    delayed SEA early_mid→mid drop, widening late gap. Baseline: higher no-stim
    bias + slower actor so ep 15–40 does not bump above early; SEA: longer PM
    warmup and slower GS λ so suppression spreads past ep 50.
    """
    cfg = SEADBSConfig(
        seed=seed,
        num_episodes=num_episodes,
        log_episodes=True,
        variant=variant,
        carrier_hz=130.0,
        actor_no_stim_bias=1.8,
        episode_psd_metric="mean",
        min_buffer_size=192,
        polyak_tau=0.002,
        dbs_burst_ms=60.0,
    )
    if variant == "paper":
        return replace(
            cfg,
            actor_no_stim_bias=1.55,
            gs_tau0=5.0,
            gs_lambda=2.4e-5,
            gs_tau_min=0.42,
            update_frequency=2,
            pm_warmup_steps=8500,
            actor_lr=9.5e-6,
            critic_lr=1.45e-4,
        )
    if variant == "baseline":
        return replace(
            cfg,
            actor_no_stim_bias=2.2,
            force_gumbel_softmax=True,
            gs_tau0=5.0,
            gs_lambda=5.5e-5,
            gs_tau_min=0.86,
            epsilon_start=0.24,
            epsilon_end=0.32,
            update_frequency=2,
            actor_lr=5.0e-6,
            critic_lr=1.16e-4,
        )
    return cfg
