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
StepMode = Literal["shot", "trailing"]
StepPulseMode = Literal["carrier", "per_step"]
UpdateCadence = Literal["step", "episode"]
GumbelActionSampling = Literal["relaxed", "hard"]
CriticActionInput = Literal["one_hot", "logits"]
StateMode = Literal["mean", "window", "mean_stim"]


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
    # ``shot`` (default): each RL step integrates ``biomarker_window_s`` of plant
    # (repo convention, replication.md §5). ``trailing``: the paper's §V.A timing —
    # one continuous simulation per episode, the agent acts every
    # ``step_duration_ms`` (2 ms), and each step's P_beta comes from the GPi spikes
    # of the last ``trailing_window_s`` (paper-silent; a convention). Reset runs an
    # untreated warm-up so the trailing window and the Eq. (4) window start full.
    step_mode: StepMode = "shot"
    trailing_window_s: float = 0.15
    # What a = 1 delivers in a ``trailing`` step: ``carrier`` gates a phase-continuous
    # ``carrier_hz`` train (a pulse fires only when one of its onsets falls in a
    # stimulating step); ``per_step`` delivers one pulse at the start of the step
    # (Eq. (6): "a_t = 1 corresponds to delivering a DBS pulse"; up to 500 Hz).
    step_pulse_mode: StepPulseMode = "carrier"
    # ``trailing`` only, a reading that contradicts Algorithm 1 line 3 ("Reset
    # environment"): one simulation for the whole run. Only the first reset warms up
    # the plant; later resets keep its state, the spike history, the pulse clock and
    # the Eq. (4) window and only restart the episode's step count, so stimulation
    # can build up across 60 ms episodes. Candidate experiments only.
    carry_plant_across_episodes: bool = False
    # Replay done flag at the 30-step time limit (Algorithm 1 stores none; replication.md
    # §14.6). False bootstraps through episode ends — the natural pairing with a plant
    # carried across episodes, where an episode end doesn't end the simulation.
    mask_episode_end: bool = True
    max_episode_steps: int = MAX_EPISODE_STEPS
    num_episodes: int = TRAIN_EPISODES

    # Reward / observation (§6, §14.1)
    beta_threshold: float = BETA_THRESHOLD
    reward_scale: float = REWARD_SCALE
    # Candidate reading beyond Eq. (7) (§V.A says the reward penalizes overstimulation
    # and promotes energy efficiency, but Eq. (7) has no stimulation term): the
    # training reward subtracts ``stim_cost`` on every stimulating step. Logged
    # episode rewards (Fig 4b) stay Eq. (7) of P̄, as the paper reports them.
    stim_cost: float = 0.0
    observation_scale: float = OBSERVATION_SCALE
    # Window for mean P_beta (Eq. 4–5); paper-silent. Convention: 10, the paper's
    # inference horizon (Figs 5–7 plot 10 stimulation steps). With independent
    # plant shots the window mean reaches its floor exactly at step n_obs; the
    # digitized Fig 5a curves keep falling through step 10 (n_obs = 5 floored at
    # step 5). Same value in training and eval (docs/figures/ravivarapu/5a.md).
    n_obs: int = 10
    # Convention: Eq. (4) state is a window of *past* beta estimates; on reset the
    # window holds n_obs untreated samples (the history before the episode).
    prefill_obs_window: bool = True
    state_dim: int = 1  # default: mean P_bar only (replication.md §14.1)
    # Candidate readings of the Eq. (4) state (replication.md §14.1; default ``mean``):
    # ``window`` feeds the n_obs normalized P_beta estimates (oldest first) instead of
    # their mean; ``mean_stim`` appends the last n_obs actions (0 / 1, oldest first;
    # zeros after reset) to P_bar. Network input size is ``obs_dim``.
    state_mode: StateMode = "mean"

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
    # Candidate reading that contradicts Algorithm 1 (which updates inside the step
    # loop): ``episode`` runs ``updates_per_episode`` gradient steps after each episode
    # instead of ``update_frequency`` per env step.
    update_cadence: UpdateCadence = "step"
    updates_per_episode: int = 1

    # Gumbel-Softmax (§IV.C, Eq. 14). tau_0, tau_min, lambda_tau are not in
    # Table I. Convention: anneal over exactly the 4,500 training steps from a
    # near-uniform relaxed sample to tau = 1 (lambda = ln(200) / 4500). The actor's
    # stim logit margin reaches ~15–20 within ~15 episodes at Table I learning
    # rates, so tau must start ~10x that to explore at all; Jang et al.'s
    # tau_0 = 1 (and tau_0 = 10, 50) leave SEA-DBS stimulating 70–90 % of
    # steps from the first episodes (docs/figures/ravivarapu/4a.md).
    gs_tau0: float = 200.0
    gs_tau_min: float = 1.0
    gs_lambda: float = 1.18e-3
    # How the *executed* binary action is drawn. ``relaxed`` (default) samples
    # a ~ Categorical(a_tilde) from the Eq. (11) relaxed probabilities — Fig. 3
    # "actions are sampled from the resulting distribution", and the only reading
    # in which tau "promotes exploration" (§IV.C). ``hard`` is Gumbel-max
    # (argmax of a_tilde), which is tau-independent.
    gs_action_sampling: GumbelActionSampling = "relaxed"

    # Baseline DDPG exploration (no GS; paper-silent): epsilon-greedy, linear
    # decay from fully random to 0.05 over the whole run.
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    # What the critic scores as "the action" (Algorithm 1 stores a_logits next
    # to a_t; the paper does not say which one Q_phi reads). ``one_hot``: the
    # executed binary action; the non-GS actor maximizes the softmax expectation
    # and explores epsilon-greedy. ``logits``: the DDPG backbone of Mehregan et
    # al. (same group, same Table I values), whose critic scores the actor's
    # continuous logit vector — non-GS variants add Gaussian noise
    # (``logit_noise_std``) to the logits, play argmax of the noisy vector, and
    # store that vector; GS variants store and score their Eq. (11) relaxed
    # sample. The actor maximizes Q(s, mu(s)) (or Q(s, GS(mu(s)))).
    critic_action_input: CriticActionInput = "one_hot"
    logit_noise_std: float = 1.0

    # MLP topology (§14.9 — open). ``hidden_layers`` hidden ReLU layers of
    # ``hidden_size`` units in the actor, critic and predictive model.
    hidden_size: int = 64
    hidden_layers: int = 2
    # Candidate (paper-silent; no entropy term in Algorithm 1): add
    # -coef * H(softmax(actor logits)) to the actor loss, which bounds the stim
    # logit margin instead of letting Adam grow it without limit.
    actor_entropy_coef: float = 0.0

    # Plant / DBS carrier (§14.10)
    carrier_hz: float = DEFAULT_CARRIER_HZ
    # Plant integration step (PlantConfig.dt_ms) and STN drive grid. §V.A: "integrated at
    # a resolution of 0.02 ms". Checkpoints from before oct 17 2026 stored 0.01 (the
    # Kumaravelu reference step) and evaluate at that step.
    plant_dt_ms: float = 0.02
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
    def obs_dim(self) -> int:
        """Network input size for ``state_mode``."""
        if self.state_mode == "window":
            return int(self.n_obs)
        if self.state_mode == "mean_stim":
            return 1 + int(self.n_obs)
        return int(self.state_dim)

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
        if self.critic_action_input not in {"one_hot", "logits"}:
            msg = f"unknown critic_action_input {self.critic_action_input!r}"
            raise ValueError(msg)
        if self.state_mode not in {"mean", "window", "mean_stim"}:
            msg = f"unknown state_mode {self.state_mode!r}"
            raise ValueError(msg)
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
