"""SNN / DSQN hyperparameters (Nguyen et al., docs/controllers/snn/method.md)."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

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
    # reproduces Figs 4-6 and the fixed network does not (docs/controllers/snn/figures/4.md).
    plant_fixed_network: bool = False
    # Candidate between the two (in neither paper nor the authors' code): keep one wiring
    # for this many 100 ms segments, then redraw. 0 = off; ignored with plant_fixed_network.
    plant_network_hold_steps: int = 0
    # STN DBS recruitment by distance (``PlantConfig.stn_dbs_spread``); 1.0 = Kumaravelu.
    # A plant deviation, not in the paper (docs/controllers/snn/figures/5.md).
    stn_dbs_spread: float = 1.0
    # Conventions from the authors' published environment (github.com/howyoubinh/CL-DBS-RL,
    # docs/controllers/snn/figures/4.md § Authors' published code). All off by default.
    # Frequency of the reset window (the 100 ms simulated before the first action). 0 = the
    # episode's initial frequency (40 Hz). The authors' reset() runs it at 100 Hz / 300 /
    # 0.3 ms, then sets 40 Hz / 0.3 ms / 300 for the first action.
    reset_window_frequency_hz: float = 0.0
    # With plant_carry: restart the STN pulse train at t = 0 of every 100 ms segment (each
    # segment opens with a pulse) instead of keeping it phase-continuous.
    plant_pulse_restart: bool = False
    # With plant_carry: zero synaptic gating and the spike-delay buffers at the start of
    # every segment; voltages, channel gates and calcium still carry.
    plant_restart_synapses: bool = False
    # Training episodes all reset with ``seed`` instead of ``seed + episode`` (the authors'
    # env draws its seed once and reuses it). Evaluation keeps a new seed per episode.
    train_fixed_episode_seed: bool = False
    max_episode_steps: int = EVAL_MAX_STEPS
    num_episodes: int = TRAIN_EPISODES

    # Observation layout (intentionally open — fixed across train/eval once chosen)
    sequence_steps: int = 10
    neurons_per_region: int = 10
    n_regions: int = 1  # scaffold: GPi only; expand when encoder covers full CBGT

    # Biomarker / termination
    alpha_beta_threshold: float = BIOMARKER_THRESHOLD
    subthreshold_steps_required: int = 3  # t_u — open in paper
    # Paper §III: an episode terminates after t_u sub-threshold steps. False keeps
    # training episodes running to the horizon (Eq. 7 non-terminated reward every step)
    # and only flags where they would have stopped. A deviation from the paper, kept
    # for the Fig 7 probe (docs/controllers/snn/figures/7.md); not adopted without approval.
    terminate_on_subthreshold: bool = True

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
    # Average-pool each row of the spike matrix over groups of this many consecutive
    # neurons before the first layer (sequence mode; 0 or 1 = off). The authors' published
    # trainer pools the 80 neurons to 16 channels (``AvgPool1d(kernel_size=5, stride=5)``,
    # CL-DBS-RL ``scripts/train_rl_rockpool_16ch.py``) for its Xylo target. Candidate only.
    snn_input_avgpool: int = 0
    # Linear+LIF stages of hidden_size neurons before the output stage. §III.B: "a
    # three-layer spiking neural network with a single hidden layer of 128 LIF neurons"
    # (input → 128 → 9) is 1. The default 2 (input → 128 → 128 → 9) is what every run
    # before Oct 17 2026 trained; kept so their checkpoints load.
    hidden_layers: int = 2
    # LIF reset. "eq2": Eq. (2) as written, U[t] = βU[t−1] + WX[t] − S[t−1]θ, and the
    # membrane read out (the Q-value) is U[t] before the subtraction (snnTorch "subtract").
    # "post" (legacy default): the stored membrane is U − Sθ and the next step leaks it,
    # so the reset is βθ, and the readout is the post-reset membrane.
    lif_reset: str = "post"

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
    # Store horizon truncations as terminal (no bootstrap). False bootstraps from the
    # next state at the horizon, as the authors' published trainer does (only true
    # termination ends the return).
    replay_truncation_terminal: bool = True
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
    # Spike-count ties (paper silent, §III.B): ``membrane`` gives a tied head to the higher
    # output membrane potential; ``first`` takes the first tied index (−1, decrease), which is
    # what ``torch.argmax`` does in the spike-count branch of the authors' published
    # ``get_action_dict`` (CL-DBS-RL ``src/utils/action_utils_rockpool.py``). Candidate only.
    action_tie_break: str = "membrane"

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
    """Nguyen Fig. 4–6 training config (docs/controllers/snn/figures/4.md, 6.md).

    Paper-stated: Eq. (7) reward with unnormalized d, θ = 150, init 300 nA/cm² / 40 Hz /
    0.3 ms, 25-step episodes, replay update every 128 transitions, binary spikes, and the
    DSQN of §III.B: one hidden layer of 128 LIF neurons, β = 0.95, Eq. (2) reset by
    subtraction, the n × N spike matrix fed one row per SNN step, actions by spike-count
    argmax and Q from the output membrane.
    From the authors' published code (CL-DBS-RL, arXiv 2606.28600), where the paper is
    silent: t_u = 5, δ = 0.01, amplitude step 5 nA/cm², soft target τ 0.005, AdamW,
    SmoothL1, value clipping 100, γ 0.99, lr 1e-3, buffer 100k, batch 128,
    surrogate-gradient training, per-episode continuous simulation, all eight CBGT
    populations observed. Step sizes 5 Hz / 0.1 ms read off Fig 6.
    Paper-silent conventions, chosen against the digitized figures (4.md § Network
    faithfulness audit): τ = 1,000 (the paper's late Fig 4a reward level), 10 bins of
    10 ms per step (100 one-ms bins train less stably), linear ε 0.9 → 0.05 over 2,000
    steps (the exponential schedule lands the length drop ~40 episodes late), 64
    minibatches per update, 1,650-transition replay warm-up, rewards scaled by 1e-4 for
    learning only. Plant: Kumaravelu's coupling, every STN neuron gets the full DBS current
    (ρ = 1, as in the authors' published simulator; the Oct 11 2026 ρ = 3 deviation was
    dropped Oct 18 2026 with the maintainer's OK). Training redraws the wiring every 100 ms step;
    the Fig 7 eval keeps one wiring per episode (docs/controllers/snn/figures/4.md § Wiring as a
    setting).
    """
    return SNNConfig(
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=EVAL_MAX_STEPS,
        alpha_beta_threshold=BIOMARKER_THRESHOLD,
        subthreshold_steps_required=5,
        energy_penalty=0.01,
        threshold_reward=1000.0,
        reward_learning_scale=1e-4,
        amplitude_sensitivity=5.0,
        frequency_sensitivity=5.0,
        pulse_width_sensitivity=0.1,
        plant_carry=True,
        n_regions=8,
        neurons_per_region=10,
        sequence_steps=10,
        surrogate_gradient="atan",
        snn_input_mode="sequence",
        hidden_layers=1,
        lif_reset="eq2",
        replay_update_cadence=128,
        replay_update_steps=64,
        replay_warmup_transitions=1_650,
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
        stn_dbs_spread=1.0,
        log_episodes=True,
    )


def parse_config_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse repeatable ``--set field=value`` into typed ``SNNConfig`` overrides."""
    known = {f.name for f in fields(SNNConfig)}
    defaults = SNNConfig()
    out: dict[str, Any] = {}
    for item in items or []:
        name, sep, raw = item.partition("=")
        name = name.strip().replace("-", "_")
        if not sep or name not in known:
            raise SystemExit(f"--set expects FIELD=VALUE with an SNNConfig field, got {item!r}")
        current = getattr(defaults, name)
        if isinstance(current, bool):
            if raw.lower() not in {"true", "false", "1", "0"}:
                raise SystemExit(f"--set {name}: expected true/false, got {raw!r}")
            out[name] = raw.lower() in {"true", "1"}
        elif isinstance(current, int):
            out[name] = int(raw)
        elif isinstance(current, float) or current is None:
            out[name] = float(raw)
        else:
            out[name] = raw
    return out
