# RL environment specification (Mehregan et al., adaptive DBS / quantization)

This document specifies the **computational reinforcement-learning environment** described in *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning* (Mehregan et al.). It is meant to align a Gymnasium-style implementation under `envs/` (repository root) with the published setup. The **DDPG actor–critic controller** (networks, replay, losses, quantization) is specified separately in [controllers/ddpg/replication.md](controllers/ddpg/replication.md).

The **plant** is the shared **Kumaravelu et al. (2016)** parkinsonian CBGT model with **STN DBS**—topology, integration, actuation, and biomarker primitives are specified in **[plant.md](plant.md)**. This document covers the **Mehregan et al. Gymnasium-style RL environment** built on that plant.

---

## 1. Scope

| In scope | Out of scope (same paper, different system) |
|----------|---------------------------------------------|
| **Simulator-in-the-loop RL** (Algorithm 1: environment steps with transitions stored in a replay buffer) against the **MATLAB / computational** network | **In vivo** rat optogenetics pipeline (trial timing, behavior, recording hardware) |
| **Single biomarker** control using **beta-band power** in GPi | **Error index (EI)** as a *driving* observation (used historically in related work; this study motivates **Pβ only** for practicality) |

---

## 2. Plant (dynamics model)

Use the shared biophysical plant per **[plant.md](plant.md)** (Kumaravelu et al., 2016 CBGT, STN DBS, $n = 10$ per region, parkinsonian default). Mehregan §IV.A.1 reports integration step **0.02 ms**; the bundled reference defaults to **0.01 ms**—see [plant.md](plant.md) §5. The `envs/` Mehregan API does not duplicate dynamics; it wraps the plant with **2 s** RL steps, biomarker windows, and reward Eq. (8) below.

---

## 3. Biomarker and observation

### 3.1 Beta power in GPi (primary signal)

Integrated **beta (13–35 Hz)** power over GPi spiking, averaged across the **n = 10** GPi neurons (paper Equation (1)). The paper’s introduction cites literature using **12–35 Hz** in places; **Equation (1) and the computational biomarker use 13–35 Hz**—implementations should match Eq. (1) for replication. The reference script integrates **7–35 Hz** by default; Mehregan replication must use **13–35 Hz** ([plant.md](plant.md) §6).

$$
P_\beta = \frac{1}{n} \sum_{j=1}^{n} \int_{\omega = 2\pi \cdot 13\,\mathrm{Hz}}^{2\pi \cdot 35\,\mathrm{Hz}} P_j^{\mathrm{GPi}}(\omega)\, d\omega
$$

where $P_j^{\mathrm{GPi}}$ is the **power spectral density** of the **action potentials** of neuron $j$ in GPi.

### 3.2 State for the RL agent

- The **actor** uses a **temporal window** of biomarker samples (paper §III.B: CNN over time; “Brain” class with **state length** / window parameters).
- **Algorithm 1** in the paper: at each RL step, form state **s** from **$P_\beta$** (after running the plant for the step duration).
- For the **computational experiments** in §IV.A.1, the **biomarker window** for $P_\\beta$ is described as spanning the **full length of the simulation segment** used for that step (see §5).

**Implementation (Jul 2026, TASK-67):** The paper's "window spanning the full simulation segment" means **`state_length=1`** — a single $P_\\beta$ value computed over the 2 s step. This is **not** a multi-step history window. With `state_length > 1` (e.g. 15), only the most recent element changes per step while the rest are stale history, causing the CNN to see near-identical states and collapse to a constant policy. **Use `state_length=1` as default.**

**Optional context (not used as the main RL input in this study):** **Error index (EI)** for thalamic reliability (paper Equation (2)) appears in §II.B for background and comparison with prior work; the **stated RL objective** uses **$P_\beta$ alone**.

---

## 4. Action space

- **Semantic:** A **stimulation pattern** in STN (not only a scalar frequency): the actor emits **logits** over a **discrete** pattern set; **softmax + argmax** (or stored logits for the critic) yields the pattern applied to the plant (paper §III.B). The paper motivates **discrete patterns** for exploration but §IV.A.1 does **not** state the **cardinality** of the pattern alphabet or the exact encoding; define that in code and keep it stable across training / evaluation / quantization comparisons.
- **Initialization:** Policy / pattern family is initialized from **regular pulses** at a target **mean frequency** (e.g. **45 Hz** or **30 Hz** in §IV) to speed learning.
- **Baselines for comparison:** Periodic **45 Hz** stimulation and conventional **130 Hz** cDBS, same random seed where applicable.

### 4.1 Implementation vs paper: scalar frequency vs fixed-budget pattern (TASK-81)

**Finding (TASK-81, root-cause of the TASK-67 constant-policy collapse):** the current action space (`envs/mehregan/patterns.py`) maps each discrete action to a **single STN pulse frequency** on the Kumaravelu grid (`freqs = 0:5:200`, action `a` → `a × 5` Hz). This diverges from Mehregan et al.'s action space in a way that fully explains the collapse:

- **Paper:** the action is a **temporal pulse pattern at a fixed mean frequency** — "average frequency of action $f$" is an **input** to Algorithm 1, and the action space is **initialized with regular pulses at 45 Hz (or 30 Hz)**. Mean stimulation rate (energy) is **held constant by construction**; the agent optimizes only the **temporal arrangement** of pulses within that budget. The learned policy is an **irregular / aperiodic** pattern at the fixed mean rate (§IV.A.2, Fig. 5b; "RL-designed temporal pattern", Fig. 8), **not** a state-dependent frequency schedule.
- **This repo:** the action is a **scalar frequency** the agent may set freely (0–200 Hz). Because reward Eq. (8) has **no stimulation-cost term** (faithful to the paper), an action's value is essentially state-independent and monotone in beta suppression, so the optimal scalar action is a **single fixed frequency**. **Constant policy is the expected, correct optimum for this action space** — not an exploration or plant-response bug.

**Consequence:** a scalar frequency has no room to be an "irregular pattern," so the paper's *adaptive temporal pattern* cannot be expressed in the current action space. Resolving this requires a **design decision** (see [replication.md](controllers/ddpg/replication.md) §5, TASK-81): **(A)** accept a fixed scalar policy (diverges from the paper's pattern claim); **(B)** add an explicit energy/cost term to the reward (an **extension** — the paper has none); or **(C)** redesign the action space as a **fixed-mean-frequency pulse pattern** to match the paper (faithful, but a larger change to the pattern alphabet and the plant DBS drive).

### 4.2 Fixed-mean-frequency pattern alphabet (Option C, TASK-83)

**Decision (master-approved, jul 7 2026):** Option C — redesign the action space to match Mehregan et al.'s fixed-mean-frequency pulse pattern formulation. Implemented in `envs/mehregan/fixed_mean_patterns.py`.

**What changes:** The actor no longer selects a stimulation *frequency* (0–200 Hz). Instead, it selects a *temporal pulse pattern* from a discrete alphabet, all sharing the same mean stimulation rate (45 Hz default, 30 Hz for ablation). Mean frequency is an **input** to Algorithm 1, not an output — the agent shapes *when* pulses occur, not *how many*.

**Pattern alphabet:**

| Property | Value | Rationale |
|----------|-------|-----------|
| `n_patterns` | **41** | Matches the scalar-frequency alphabet size / actor head, so no DDPG topology change needed |
| Pattern 0 | Regular periodic train at `mean_hz` | Byte-identical to `create_dbs_current(mean_hz)` — the paper's initialization target |
| Patterns 1–40 | Deterministic irregular trains | Same pulse count as pattern 0 (mean rate preserved exactly). Interior onsets jittered by ±1/3 ISI; first/last onsets pinned so total span is unchanged |
| Jitter PRNG | Seeded by `(mean_hz, pattern_index)` | Reproducible across training, evaluation, and quantization |
| Resolution | Plant time grid (`dt_ms = 0.01`) | Same grid as the integrator; precomputed traces cached via `lru_cache` |

**Paper-silent choices (documented here per repo convention):**

- Alphabet size 41 is an implementation convenience, not paper-specified. The paper does not state the cardinality of the pattern set.
- Jitter fraction 1/3 is chosen to keep consecutive pulses well separated (≫ pulse width) while producing visibly irregular trains.
- First/last onset pinning preserves the total temporal span (and thus the sum of inter-spike intervals) across all patterns in the alphabet.

**Plant integration:** `DbsSpec` carries an optional `idbs` field — a precomputed STN drive trace on the plant time grid. When set, `integrate_network` applies it directly instead of synthesizing a regular train from `frequency_hz`. The MATLAB backend ignores `idbs`; pattern mode is Python-plant only.

**Usage:** Instantiate `MehreganEnv(alphabet=FixedMeanPatternAlphabet(mean_hz=45.0))`. The env's `action_space` and `step()` are unchanged — only the alphabet-to-DbsSpec mapping differs.

**Baselines in pattern mode:** Pattern 0 (regular train at `mean_hz`) replaces the old periodic-45Hz baseline. The 130 Hz cDBS and no-stimulation baselines are not representable in pattern mode (different mean frequency); use the scalar-frequency alphabet for those comparisons.

---

## 5. Timing and transitions

Values from **§IV.A.1 (computational setup)** unless noted.

| Quantity | Value |
|----------|--------|
| **RL step duration** (`dt_rl` / “duration $l$” in Alg. 1) | **2 s** of simulated time per RL transition |
| **Plant integration step** | **0.02 ms** in Mehregan §IV.A.1; reference default **0.01 ms** — [plant.md](plant.md) §5 |
| **Steps per training episode** | **30** |
| **Episode reset** | New **initial conditions** for the plant; collect $P_\beta$ and reward over each step (Alg. 1) |

**Suggested environment API:**

- `reset()` → new parkinsonian initial conditions (and optional noise seed); integrate for duration **$l$** (2 s) per Algorithm 1 step 7, compute **$P_\beta$** over the biomarker window, form initial observation **$s_0$**, and return it (plus optional initial **$R$** and info). The first `step` then applies an action for another **$l$**.
- `step(action)` → apply selected **pattern** to STN for **2 s** simulated time, integrate the network, compute **$P_\beta$** over the window policy, return observation, reward, terminated, truncated, info.

---

## 6. Reward

Instantaneous reward **$R$** (paper Equation (8)) depends on the **average** of the observed state entries over the observation window:

$$
s_{\mathrm{sum}} = \frac{1}{n_{\mathrm{obs}}} \sum_{i=1}^{n_{\mathrm{obs}}} s(i)
$$

$$
R =
\begin{cases}
\left(s_{\mathrm{sum}} - \beta_t\right) \cdot 10, & s_{\mathrm{sum}} < \beta_t \\
-\left(\left(s_{\mathrm{sum}} - \beta_t\right) \cdot 10\right)^2, & s_{\mathrm{sum}} \ge \beta_t
\end{cases}
$$

- **Threshold:** $\beta_t = 0.35$ (paper §III.C).
- **Intent:** Reward **increases** (is more favorable) as the average biomarker-derived state moves **below** $\beta_t$; **quadratic penalty** when at or above. Paper §III.C and Figure 3c state **positive reward below threshold**; the printed Eq. (8) linear branch uses $(s_{\mathrm{sum}} - \beta_t)$, which is negative when $s_{\mathrm{sum}} < \beta_t$. **Implementation (TASK-78):** negate the linear branch so below-threshold transitions yield $R = (\beta_t - s_{\mathrm{sum}}) \cdot 10 > 0$, matching prose and Fig. 3c. The $P_\beta \rightarrow s(i)$ mapping remains $s(i) = P_\beta / 1000$ (`observation_scale`); with unstimulated parkinsonian $s \approx 0.4$–$0.5$, most transitions sit in the **quadratic penalty** branch until stimulation reduces beta.

**Relationship to the paper’s efficiency goal:** The problem statement motivates **both** symptom reduction (beta) and **energy-aware** stimulation (e.g. lower mean frequency vs **130 Hz** cDBS). The **published reward in Eq. (8) depends only on beta-band state vs $\beta_t$**—there is **no separate reward term** for pulse count or instantaneous frequency. Mean-frequency shaping enters through **initialization** (e.g. **45 Hz** / **30 Hz** target), the **learned discrete pattern family**, and **baseline comparisons** in §IV, not through an explicit energy penalty in $R$.

**Implementation note:** The paper’s Figure 4 discusses **raw beta power** on the order of **hundreds** in the uncontrolled model; the published $\beta_t = 0.35$ implies **$s(i)$** are **normalized or preprocessed** consistently between the **observation** pipeline and **reward**. The implementation should apply a single, documented normalization (e.g. scaling $P_\beta$ into the range where $\beta_t = 0.35$ is meaningful) so that reward and logged metrics match the paper’s intent.

---

## 7. Episode termination

- **Training (paper Alg. 1):** After **30** steps, end episode.
- **`dw` (done flag) for DDPG targets:** 1 if episode finished, else 0 (paper Equation (3)).

**Implemented (`MehreganEnv`):** no dynamical terminal state — episodes end with **`truncated=True`** at `max_episode_steps` (default **30**); `terminated=False`. Set `info` flags for replay consumers in Phase 3.

---

## 8. Training loop parameters (computational study)

From **§IV.A.1** (for replication / defaults):

| Hyperparameter | Value |
|----------------|--------|
| Actor learning rate | **5×10⁻⁴** |
| Critic learning rate | **1×10⁻³** |
| Replay buffer capacity | **8192** |
| Minibatch size | **32** |
| Training episodes (reported run) | **10** |
| Mean stimulation frequency at init | **45 Hz** (also **30 Hz** experiment, other settings unchanged) |

**Not fixed numerically in §IV.A.1 (still required for DDPG replication):** Algorithm 1 also initializes **discount** $\gamma$, **target-network soft-update** coefficient $\tau$, and an inner-loop **update frequency** (gradient steps per environment step). The computational-setup paragraph does **not** report numeric values for these; match released code if available, or document chosen defaults explicitly in the implementation.

**Evaluation protocol (paper §IV.A.2):** After training, fix **random seed**; run a **10 s** simulation with **2 s** for reset / baseline, then **5** repeated applications of the stimulation **step** for comparison across models and quantization variants. The paper also describes a **2 s** GPi baseline before applying actor actions (Figure 5). If each eval segment uses the training step duration **$l = 2$ s**, five segments plus a 2 s reset exceed **10 s** total—**intentionally open** until released code or a project convention fixes segment length vs. the reported **10 s** wall time. When released code is unavailable, a leading hypothesis is **1.6 s per eval step** ($2 + 5 \times 1.6 = 10$ s), which reconciles the paper’s three stated durations. Cross-controller use of this protocol (baselines, metrics, run identity) is defined in [benchmarking.md](benchmarking.md).

---

## 9. Optional extensions (same publication)

- **Quantization:** PTQ / QAT on actor–critic networks affects **inference-time** policy outputs; the **environment interface** (plant, $P_\beta$, timing, reward) is unchanged. Implementation: [controllers/ddpg/replication.md](controllers/ddpg/replication.md) §6, [controllers/ddpg/quantization.py](../../controllers/ddpg/quantization.py).
- **Animal validation:** Not part of the computational `Env` API; documented separately if this repo later adds experiment logs or replay buffers for **in vivo** trials.

---

## 10. Minimal consistency checklist

- [x] STN-injected patterns match discrete action set used during training (`PatternAlphabet` → `DbsSpec`).
- [x] **2 s** simulated time per RL step, **30** steps per episode (`MehreganEnvConfig` defaults).
- [x] **$P_\beta$** from **GPi** spikes, **13–35 Hz**, mean over **10** neurons (`envs.plant.biomarkers`).
- [x] Reward uses **$s_{\mathrm{sum}}$** vs **$\beta_t = 0.35$** with documented scaling of **$s$** (`observation_scale=1000`).
- [x] Baselines: **no stimulation**, **130 Hz** cDBS, periodic **45 Hz** (and **30 Hz** where relevant), same seeds (`run_baseline_rollout`).
- [x] DDPG: **$\gamma$, $\tau$,** and **updates per env step** documented (Phase 3 — [controllers/ddpg/replication.md](controllers/ddpg/replication.md)).

---

## 11. Open questions / TBD

### 1. DBS pulse waveform details

Mehregan §IV.A.1 does not override pulse timing, amplitude, or other STN waveform parameters. **Fixed:** STN DBS per Kumaravelu et al. (2016). **Open:** exact waveform encoding. **Decide in** plant bridge — [plant.md](plant.md) §4, §10.

### 2. Plant integration step ($\Delta t$)

See [plant.md](plant.md) §5, §10. **Decide in** plant config; validate biomarkers if keeping reference **0.01 ms**.

### 3. Discrete pattern alphabet

The actor emits logits over a discrete STN pattern set, but §IV.A.1 does not state **cardinality** or **waveform encoding**. **Fixed:** discrete patterns via softmax + argmax. Alphabet size and per-pattern STN drive semantics are now **decided** (TASK-83, Option C).

**Scalar mode (`envs/mehregan/patterns.py`):** 41 actions → Kumaravelu `pick_dbs_freq` 1…41 (`freqs = 0:5:200`); action `0` is no DBS (`pick_dbs_freq == 1`).

**Pattern mode (`envs/mehregan/fixed_mean_patterns.py`):** 41 irregular pulse patterns at fixed mean frequency (45 Hz / 30 Hz). See §4.2 for full specification.

### 4. Observation normalization for reward Eq. (8)

Eq. (8) uses threshold $\beta_t = 0.35$ on averaged state entries $s(i)$, but the paper never defines the **monotone mapping** from raw $P_\beta$ (order hundreds in Figure 4) to $s(i)$. **Fixed:** reward shape (linear below $\beta_t$, quadratic at/above). **Open:** normalization pipeline shared by observation and reward. **Decide in** the environment implementation with a single documented transform.

**Implemented (`envs/mehregan/`):** $s(i) = P_\beta / 1000$ (`MehreganEnvConfig.observation_scale`). Unstimulated parkinsonian segments (~400–500 raw) map to ~0.4–0.5, aligning with Mehregan Fig. 3c–4 prose. Override `observation_scale` if released code differs.

### 5. DDPG hyperparameters not in §IV.A.1

Algorithm 1 requires discount $\gamma$, target soft-update $\tau$, and inner-loop **update frequency**, but §IV.A.1 reports no numeric values. **Fixed:** actor/critic learning rates, buffer **8192**, batch **32**. **Open:** $\gamma$, $\tau$, updates per env step. **Decide in** `controllers/ddpg/` config; match released code if available (see [controllers/ddpg/replication.md](controllers/ddpg/replication.md)).

**Implemented (`controllers/ddpg/config.py`):** $\gamma = 0.99$, $\tau = 0.005$, `update_frequency = 1`.

### 6. Post-training evaluation segment timing

§IV.A.2 describes a **10 s** eval run with **2 s** reset and **5** stimulation segments at training step duration $l = 2$ s, which sums to more than 10 s. **Fixed:** 10 s wall time, 2 s baseline, five repeated eval segments. **Open:** per-segment duration vs. total budget. **Decide in** project eval convention; leading hypothesis is **1.6 s** per segment ($2 + 5 \times 1.6 = 10$ s) unless released code resolves it.

### 7. Episode termination flag semantics

Training runs **30** steps per episode with no true dynamical terminal state described. **Fixed:** horizon of 30 steps; $dw = 1$ when the episode finishes (Eq. (3)). **Open:** Gymnasium `terminated` vs `truncated` assignment. **Decide in** `envs/` API; typically **truncation at horizon**.

**Implemented:** `MehreganEnv` sets `truncated=True` at horizon; `terminated=False` ([§7](#7-episode-termination)).

### 8. Kumaravelu reference beta band vs Eq. (1)

See [plant.md](plant.md) §6. **Decide in** biomarker pipeline; use **13–35 Hz** for Mehregan fidelity.

---

## 12. Future direction: native Python plant (optional)

See [plant.md](plant.md) §9.

---

## 13. References

- Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*.
- Kumaravelu et al. (2016) — plant: [plant.md](plant.md); bundled MATLAB: [`reference-material/KumaraveluEtAl2016/`](../reference-material/KumaraveluEtAl2016/) ([`readme.txt`](../reference-material/KumaraveluEtAl2016/readme.txt)).
