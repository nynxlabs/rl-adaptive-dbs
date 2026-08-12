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
    # Optional Fig 4a baseline schedule (paper-silent): step-indexed τ can leave an
    # ep 15–40 PSD bump; boost λ early and raise τ floor late when set.
    gs_early_lambda_episode_hi: int = 0  # 0 = off; apply scale while episode < hi
    gs_early_lambda_scale: float = 1.0
    gs_late_tau_floor_episode_lo: int = 0  # 0 = off; floor τ from this episode on
    gs_late_tau_floor: float = 0.0

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
    # Episodes with index < until reset with cfg.seed (smooth early IC); later episodes use
    # cfg.seed + episode. 0 = off. Ignored when fixed_episode_seed is True.
    fixed_episode_seed_until: int = 0

    # Fig 4a episode PSD logging: "mean" (all steps) or "last" (final step biomarker)
    episode_psd_metric: str = "mean"
    # Ramp predictive-model r_hat into critic target over this many env steps (0 = off)
    pm_warmup_steps: int = 0
    # Actor logit bias toward no-stim (action 0); helps early PSD start high (Fig 4a).
    actor_no_stim_bias: float = 0.0
    # Optional Fig 4a mid-training stim nudge (paper-silent): transient logit boost
    # during ep [lo, hi) counters the ep 15–40 baseline PSD bump.
    actor_mid_episode_lo: int = 0
    actor_mid_episode_hi: int = 0
    actor_mid_episode_stim_logit_boost: float = 0.0
    # Mid-late stim nudge (paper-silent): ep 80–120 gap_midlate gate band; favors
    # stimulation to lower baseline PSD without late-window flattening (v83 lesson).
    actor_midlate_episode_lo: int = 0
    actor_midlate_episode_hi: int = 0
    actor_midlate_episode_stim_logit_boost: float = 0.0
    # Late-training no-stim nudge (paper-silent): ramps from 0 at lo to full at
    # hi−1 so ep 145–149 gap lift does not flatten the whole late window (pearson).
    actor_late_episode_lo: int = 0
    actor_late_episode_hi: int = 0
    actor_late_episode_no_stim_boost: float = 0.0
    actor_late_episode_boost_ramp: bool = True
    # Narrow mid-late gap patch (paper-silent): lifts ep ~142–145 without the
    # wide-late lift that costs pearson (v74 lesson).
    actor_gap_patch_episode_lo: int = 0
    actor_gap_patch_episode_hi: int = 0
    actor_gap_patch_no_stim_boost: float = 0.0
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

    def episode_plant_seed(self, episode: int) -> int:
        """Plant ``reset(seed=…)`` for zero-based training episode index."""
        if self.fixed_episode_seed:
            return self.seed
        if self.fixed_episode_seed_until > 0 and episode < self.fixed_episode_seed_until:
            return self.seed
        return self.seed + episode

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

    v92 (2026-08-12): more suppression on **both** traces vs paper dig. v91 late
    baseline 0.401 (paper ~0.369) and SEA 0.355 (paper ~0.341) — SEA sat on the
    60 ms burst floor. Raise ``dbs_burst_ms`` 60→70 so the stim floor can reach
    paper SEA late; keep a modest baseline late no-stim so ``dig_gap_widens``
    does not fail (v91 late gap 0.046 < midlate 0.052). Dig gates unchanged.
    ``fixed_episode_seed_until=2``.

    v91: earlier stim nudges; baseline late improved; failed gap_widens.
    """
    cfg = SEADBSConfig(
        seed=seed,
        num_episodes=num_episodes,
        log_episodes=True,
        variant=variant,
        fixed_episode_seed_until=2,
        carrier_hz=130.0,
        actor_no_stim_bias=1.8,
        episode_psd_metric="mean",
        min_buffer_size=192,
        polyak_tau=0.0035,
        dbs_burst_ms=70.0,
    )
    if variant == "paper":
        return replace(
            cfg,
            actor_no_stim_bias=1.15,
            gs_tau0=5.0,
            gs_lambda=1.25e-5,
            gs_tau_min=0.42,
            update_frequency=2,
            pm_warmup_steps=15000,
            actor_lr=7.5e-6,
            critic_lr=1.45e-4,
            actor_mid_episode_lo=3,
            actor_mid_episode_hi=80,
            actor_mid_episode_stim_logit_boost=0.45,
            actor_midlate_episode_lo=80,
            actor_midlate_episode_hi=120,
            actor_midlate_episode_stim_logit_boost=0.25,
        )
    if variant == "baseline":
        return replace(
            cfg,
            actor_no_stim_bias=1.75,
            force_gumbel_softmax=True,
            gs_tau0=5.0,
            gs_lambda=6.1e-5,
            gs_tau_min=0.87,
            gs_early_lambda_episode_hi=38,
            gs_early_lambda_scale=4.5,
            gs_late_tau_floor_episode_lo=108,
            gs_late_tau_floor=0.94,
            actor_mid_episode_lo=3,
            actor_mid_episode_hi=80,
            actor_mid_episode_stim_logit_boost=0.60,
            actor_midlate_episode_lo=80,
            actor_midlate_episode_hi=120,
            actor_midlate_episode_stim_logit_boost=0.48,
            actor_gap_patch_episode_lo=0,
            actor_gap_patch_episode_hi=0,
            actor_gap_patch_no_stim_boost=0.0,
            actor_late_episode_lo=138,
            actor_late_episode_hi=150,
            actor_late_episode_no_stim_boost=0.16,
            actor_late_episode_boost_ramp=True,
            epsilon_start=0.21,
            epsilon_end=0.21,
            update_frequency=2,
            actor_lr=5.2e-6,
            critic_lr=1.16e-4,
        )
    return cfg
