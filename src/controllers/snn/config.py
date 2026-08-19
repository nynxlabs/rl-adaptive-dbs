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
    # Bellman TD loss: ``mse`` or ``huber`` (smooth L1 — dampens timeout Q spikes).
    q_loss_fn: str = "mse"
    # Hard-copy target network every N gradient updates (paper silent — convention).
    target_update_period: int = 100

    # Exploration (ε-greedy on spike-count argmax)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2_500
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
    v46 FAIL: t_u=3 + 500ep; first-100 length rose (80–100 ≈24.6 vs paper ~10).
    v48 FAIL: decay=1900 from step 0 locked a weak greedy policy.
    v49 FAIL: delay=1100 held ε=1 through ep 50. v50 FAIL: delay=500
    flattened length ~18.5. v51: restore v47's slow slope (decay=3200,
    no hold) then dump after ~70 ep (accelerate_after=1400, dump=500)
    so 50–70 can still glide and 80–100 can show greedy length.
    v51 FAIL: mid-glide true (~17.2) but dump too late (ε=0.46 at ep 80);
    80–100 stayed ~17. v52 FAIL: after=1000 + freq_sens=15 hit ε floor by
    ep 70 and greedy timed out (80–100 ≈19.4, lost glide). v53: freq=20
    again; dump after ~60 ep (1200) over 350 steps so floor by ~ep 80.
    v53 FAIL: ε floor by ep 80 as intended; 80–100 still ~17 (greedy ~16–17
    steps, α–β ~220). Shape not ready for 500ep. v54: keep v53 ε schedule;
    raise alpha_beta_progress_coef 2500→4000 so greedy drives α–β down
    faster and stops earlier in the episode.
    v54 FAIL: identical first-100 to v53 (0–50 ≈20). Raw median is already
    25; 16/50 lucky tu=2 stops pull the smooth start down. v55: keep v53 ε
    dump; revert prog=2500; frequency_sensitivity=10 so random +freq is
    less likely to hit 80 Hz in the first 50 episodes.
    v55: ep1 length=25 but reward ≈−1.40e6 (paper start ≈−0.66e6). The
    extra million is truncation_penalty on timeout. v56: truncation=0 so
    a 25-step first episode is Eq. (7) only (~−0.4e6, near paper −0.65e6).
    v56 ep1: length 25, reward −3.98e5 (too high vs paper −6.6e5). v57:
    truncation_penalty=250k so ep1 ≈ −0.65e6 with length still 25.
    v57 FAIL: ep1 matched; greedy timed out after ε floor (80–100 len ≈24).
    v58 FAIL: cadence=16 made 0–50 length 22.6 (horizon false) and still no
    glide (80–100 ≈22.6). One ES at ep 59 did not stick.
    v59 FAIL: ep1 OK; 0–50 len 22.75 (horizon false); 80–100 ≈22.4;
    greedy still timeouts. v60: frequency_sensitivity=15, same dump@1400 /
    replay=128 / trunc=250k.
    v60 FAIL: start collapsed (0–50 len ≈21.4). Do not raise freq. v61:
    freq_sens=10 (v57 start) + epsilon_end=0.15 (v10 late exploration).
    v61 FAIL: 80–100 len ≈23.2 (rose). v62: slower dump (decay=800) so ε
    stays higher through 80–100; keep freq=10 / ε_end=0.15 / trunc=250k.
    v62 FAIL: 80–100 len=25 (all timeouts). Revert dump=400. v63: ε_end=0.20.
    v63 FAIL: 80–100 ≈24.1. ε-floor family exhausted. v64: pulse_width_sensitivity=0.2
    so greedy can reach paper ~1 ms without raising freq.
    v64 FAIL: start held (~22.3) but 80–100 len=25 (all timeouts). v65:
    pulse_width_sensitivity=0.3 (still no freq raise).
    v65 FAIL: pw family exhausted; constant freq=10 starves 80 Hz replay.
    v66: frequency_sensitivity_explore=20 (schedule vs ε) so high-ε random
    walks reach ~80 Hz early-stops while ε-floor greedy keeps exploit=10.
    v66 FAIL: linear schedule at ε≈1 used explore=20 → ep1 len=15 and
    20/50 lucky stops (smooth 0–50 ≈19). v67: explore only in mid-anneal
    band (ε≤0.7 and >ε_end); hold exploit=10 at ε>0.7 and at floor.
    v67 FAIL: ep1 OK but greedy still 30 Hz / α–β 309 at ep 99; exploit=10
    cannot reach ~80 Hz in one episode. v68: episode curriculum — freq=10 for
    ep 0–34 (paper start), then freq=20 so greedy can suppress; v53 ε dump
    (1200/350, ε_end=0.05); replay cadence 32 for ~75 SGD steps / 100 ep.
    v68 FAIL: bimodal — good eps at 100–120 Hz (len 4–14) vs F=0 collapse
    timeouts; smoothed 80–100 still 24.     v69: frequency_min=10, amplitude_min=50;
    early curriculum through ep 39.
    v69 FAIL (100ep + 500ep): F=10 collapse timeouts drive spikes; 500ep
    late_len≈13.9, 80–100≈23.     v71 (same knobs as v70 branch): frequency_min
    at paper init (40 Hz), amplitude_min=200; early curriculum through ep 49
    so 0–50 keeps freq=10/step but greedy cannot dive below init frequency.
    v71 FAIL: reward gates pass but ep0 lucky stop (len 8); 0–50 smooth ≈21.2
    (need ≥23). v72: keep v71 floors; frequency_sensitivity_early=1 so ε≈1
    random walks stay near paper init 40 Hz (v55 spirit) while exploit=20 after
    ep 49 for late suppression. v72 FAIL: early gates pass (0–50 len=25) but
    80–100 len≈20.7 and reward_by_100 fail — 50 eps at early=1 starved mid
    freq ramp. v73: early=1 for ep 0–24 only, then exploit=20; slower ε
    (decay=4200, accelerate@2000, prog=2000) for paper mid-glide by ep 80.
    v73 FAIL: shortening early broke 0–50 smooth (22.0) without fixing 80–100
    (20.75); ep500 timeout.     v74: v72 early lock (1 Hz, 50 ep) + v22 slower ε
    only — isolate epsilon from v73's shortened curriculum mistake. v74 FAIL:
    early OK (0–50=25) but 80–100≈21.3 (worse than v72); slower ε alone no help.
    v75: early=3 for 50 ep (v71=10 vs v72=1 compromise) + v72 fast ε
    (decay=3200, accelerate@1200, prog=2500). v75 FAIL length only: reward
    shape+full PASS; 0–50=25, mid-glide true, but 80–100≈16.0, late_len≈14.9,
    post100 ptp=8.7 (need ≤4.5). v76: early=5 (toward v71 mid-curve) keeping
    v75 ε schedule; lock floors and exploit=20 after ep 49. v76 PASS (500ep):
    shape_pass+pass; early_hz=5, early_eps=50, freq_min=40, amp_min=200.
    Visual gap: bimodal timeout vs early-stop drives spikier raw/smoothed than
    paper (length ptp 100–200 ≈7.9 vs paper ≈1.8).     v77: ε_end=0.02 and
    target_update_period=50 for stabler late greedy policy. v77 PASS gates
    but smoothness regressed (late timeout 55%); ship v76 not v77.
    v78: v76 + replay_update_cadence=16 (2× SGD per env step for smoother
    Q fit) and target_update_period=50 with ε_end=0.05 kept — isolate v77's
    target-sync knob without starving late exploration. v78 FAIL: collapse
    to 100% timeouts by ep ~200 (last ES ep 177); cadence=16 over-trains.
    Ship v76. v80: learning_rate=3e-4 (paper-silent) to cut late Q churn /
    timeout flips; keep v76 replay cadence 32 and target_update 100. v81:
    v80 + batch_size=64 — worse late_len; do not ship. v82: v80 +
    q_loss_fn=huber — FAIL shape (mid-glide too fast: smooth 50–100≈11.4
    vs paper 17.8); do not retry huber fresh train. v83: v80 mse +
    target_update_period=200 (slower hard targets; not v77's tu=50+ε_end=0.02).
    v83 shape_pass, best late (timeout 13% ep350–500); checkpoint_v83.pt.
    v84: v83 + replay_update_cadence=48 — FAIL late (timeout 40%, late_len≈15.5);
    do not retry cadence>32. v85: v83 + epsilon_end=0.06 (slightly more late ε vs
    v77's 0.02 trap; keep tu=200, cadence=32).
    """
    return SNNConfig(
        seed=seed,
        num_episodes=num_episodes,
        max_episode_steps=EVAL_MAX_STEPS,
        alpha_beta_threshold=BIOMARKER_THRESHOLD,
        subthreshold_steps_required=2,
        epsilon_decay_steps=3_200,
        epsilon_decay_delay_steps=0,
        epsilon_accelerate_after_steps=1_200,
        epsilon_accelerate_decay_steps=350,
        epsilon_end=0.06,
        learning_rate=3e-4,
        batch_size=32,
        q_loss_fn="mse",
        target_update_period=200,
        frequency_sensitivity=20.0,
        frequency_sensitivity_early=5.0,
        frequency_sensitivity_early_episodes=50,
        frequency_sensitivity_explore=0.0,
        frequency_min=INIT_FREQUENCY_HZ,
        amplitude_min=200.0,
        pulse_width_sensitivity=0.3,
        threshold_reward=300.0,
        energy_penalty=0.0,
        alpha_beta_progress_coef=2500.0,
        alpha_beta_progress_cap_per_step=10_000.0,
        warm_zone_upper=220.0,
        warm_zone_bonus_coef=150.0,
        truncation_penalty=250_000.0,
        replay_update_cadence=32,
        reward_learning_scale=1e-4,
        stimulated_neurons=1,
        log_episodes=True,
    )
