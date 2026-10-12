"""SEA-DBS hyperparameters (Ravivarapu et al., docs/controllers/sea_dbs/replication.md).

Every Ravivarapu panel (Figs 4–7) trains and evaluates with **one** config: the
variants differ only in which paper components they switch on (predictive
model, Gumbel-Softmax). Values the paper states (Table I, §V.A, Eq. (7)) are
fixed; paper-silent values are labeled conventions below and in
``docs/controllers/sea_dbs/replication.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

# Paper §V.A / Table I (fixed RL cadence metadata).
STEP_DURATION_MS: float = 2.0
# Convention: plant segment per RL step for multitaper P_beta. 2 ms yields no
# beta estimate. 150 ms is the shortest window in which the paper's 50 Hz
# carrier suppresses GPi beta on Kumaravelu (100 ms: stim/untreated ratio
# 0.95 ± 0.16 over seeds 0–9; 150 ms: 0.69 ± 0.06). Probe:
# scripts/probes/ravivarapu_plant_levels.py.
BIOMARKER_WINDOW_S: float = 0.15
MAX_EPISODE_STEPS: int = 30
TRAIN_EPISODES: int = 150
BETA_THRESHOLD: float = 0.35
REWARD_SCALE: float = 10.0
# Convention: normalization so untreated GPi P_beta (150 ms shot, mean over
# plant seeds 0–9: raw 217.9) maps onto the paper's untreated level ~0.46
# (Figs 4–7 all start there). Eq. (7)'s beta_t = 0.35 only teaches on that scale.
OBSERVATION_SCALE: float = 472.0
BUFFER_CAPACITY: int = 8192
BATCH_SIZE: int = 32
GAMMA: float = 0.99
ACTOR_LR: float = 5e-4
CRITIC_LR: float = 1e-3
# Paper Figs 5a / 6 / 7 stimulation frequency. Training carrier is
# paper-silent; the paper names no other frequency, so training uses 50 Hz too.
DEFAULT_CARRIER_HZ: float = 50.0
INFERENCE_CARRIER_30HZ: float = 30.0
INFERENCE_CARRIER_50HZ: float = 50.0
ABLATION_EVAL_STEPS: int = 10
# Paper Figs 5–7 x-axis is steps 0–10: untreated PSD at t=0 plus 10 stim actions.
INFERENCE_PSD_SAMPLES: int = ABLATION_EVAL_STEPS + 1

PlantIntegrationMode = Literal["disconnected", "continuous"]
GumbelActionSampling = Literal["relaxed", "hard"]


@dataclass(frozen=True)
class SEADBSConfig:
    """Ravivarapu SEA-DBS defaults; open hyperparameters documented beside fields."""

    # RL timing (§5)
    step_duration_ms: float = STEP_DURATION_MS
    biomarker_window_s: float = BIOMARKER_WINDOW_S
    # ``disconnected`` (default): each RL step is an independent plant shot from
    # the episode IC (Mehregan-style env). ``continuous`` stitches one Kumaravelu
    # trajectory across the episode's steps (PythonPlant only; diagnostic).
    plant_integration_mode: PlantIntegrationMode = "disconnected"
    max_episode_steps: int = MAX_EPISODE_STEPS
    num_episodes: int = TRAIN_EPISODES

    # Reward / observation (§6, §14.1)
    beta_threshold: float = BETA_THRESHOLD
    reward_scale: float = REWARD_SCALE
    observation_scale: float = OBSERVATION_SCALE
    n_obs: int = 5  # window for mean P_beta (Eq. 4–5); open in Table I
    # Convention: Eq. (4) state is a window of *past* beta estimates; on reset the
    # window holds n_obs untreated samples (the history before the episode).
    prefill_obs_window: bool = True
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
    update_frequency: int = 1  # open — gradient steps per env step

    # Gumbel-Softmax (§IV.C, Eq. 14). tau_0, tau_min, lambda_tau are not in
    # Table I; defaults follow the cited Gumbel-Softmax paper (Jang et al. 2016:
    # tau = max(0.5, exp(-r t)), r in {1e-5, 1e-4}).
    gs_tau0: float = 1.0
    gs_tau_min: float = 0.5
    gs_lambda: float = 1e-4
    # How the *executed* binary action is drawn. ``relaxed`` (default) samples
    # a ~ Categorical(a_tilde) from the Eq. (11) relaxed probabilities — Fig. 3
    # "actions are sampled from the resulting distribution", and the only reading
    # in which tau "promotes exploration" (§IV.C). ``hard`` is Gumbel-max
    # (argmax of a_tilde), which is tau-independent.
    gs_action_sampling: GumbelActionSampling = "relaxed"

    # Baseline DDPG exploration (no GS; paper-silent): epsilon-greedy with a
    # linear decay over the whole run.
    epsilon_start: float = 0.5
    epsilon_end: float = 0.05

    # MLP topology (§14.9 — open)
    hidden_size: int = 64

    # Plant / DBS carrier (§14.10)
    carrier_hz: float = DEFAULT_CARRIER_HZ
    # Integrator grid (PlantConfig.dt_ms); used to build the STN drive trace grid.
    plant_dt_ms: float = 0.01
    # Stim action = carrier pulse train for this many ms of the step's plant
    # window. ``None`` = the whole window (default).
    dbs_burst_ms: float | None = None

    variant: str = "paper"
    seed: int = 0
    device: str = "cpu"
    log_episodes: bool = False
    # Default: episode e resets the plant with seed + e. True = same IC every episode.
    fixed_episode_seed: bool = False

    # Fig 4a episode PSD logging: "mean" (all steps) or "last" (final step biomarker)
    episode_psd_metric: str = "mean"

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

    def episode_plant_seed(self, episode: int) -> int:
        """Plant ``reset(seed=…)`` for zero-based training episode index."""
        if self.fixed_episode_seed:
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


# Knobs removed in the honest-replication pass. Old checkpoints still carry them;
# ``config_from_dict`` drops them (they only ever changed training, never the
# saved actor weights' meaning).
REMOVED_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "untreated_window_s",
        "dbs_pulse_delay_ms",
        "fixed_episode_seed_until",
        "pm_warmup_steps",
        "actor_no_stim_bias",
        "actor_mid_episode_lo",
        "actor_mid_episode_hi",
        "actor_mid_episode_stim_logit_boost",
        "actor_midlate_episode_lo",
        "actor_midlate_episode_hi",
        "actor_midlate_episode_stim_logit_boost",
        "actor_late_episode_lo",
        "actor_late_episode_hi",
        "actor_late_episode_no_stim_boost",
        "actor_late_episode_stim_logit_boost",
        "actor_late_episode_boost_ramp",
        "actor_gap_patch_episode_lo",
        "actor_gap_patch_episode_hi",
        "actor_gap_patch_no_stim_boost",
        "gs_early_lambda_episode_hi",
        "gs_early_lambda_scale",
        "gs_late_tau_floor_episode_lo",
        "gs_late_tau_floor",
        "force_gumbel_softmax",
    }
)


def config_from_dict(raw: dict[str, Any]) -> SEADBSConfig:
    """Build a config from a checkpoint dict, ignoring removed legacy knobs."""
    known = {f.name for f in fields(SEADBSConfig)}
    unknown = set(raw) - known - REMOVED_CONFIG_FIELDS
    if unknown:
        msg = f"unknown SEADBSConfig fields in checkpoint: {sorted(unknown)}"
        raise ValueError(msg)
    return SEADBSConfig(**{k: v for k, v in raw.items() if k in known})


def fig4_ravivarapu_config(
    *,
    seed: int = 0,
    num_episodes: int = TRAIN_EPISODES,
    variant: str = "baseline",
) -> SEADBSConfig:
    """Training config for every Ravivarapu panel (Figs 4–7).

    Same config for all four variants; ``variant`` only switches the paper's
    components (PM, GS). No per-variant or per-episode knobs.
    """
    return SEADBSConfig(
        seed=seed,
        num_episodes=num_episodes,
        log_episodes=True,
        variant=variant,
    )
