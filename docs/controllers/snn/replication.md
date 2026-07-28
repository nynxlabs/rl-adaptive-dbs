# SNN controller specification (Nguyen et al., closed-loop neuromorphic DBS)

This document specifies the **deep spiking Q-network (DSQN)** controller from *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks* (Nguyen et al.). It is meant to align `controllers/snn/` (and training scripts) with the published method.

**Companion spec:** The shared **Kumaravelu et al. (2016)** plant — [plant.md](../../plant.md). **Mehregan et al.** Gymnasium API — [environment.md](../../environment.md). **This document is authoritative for the Nguyen controller**—observation, action, reward, timing, and DSQN training—unless the others explicitly describe the same quantity.

**Scheduling:** Nguyen replication is **Phase 5** and is **active in parallel** with Mehregan ([roadmap.md](../../development/roadmap.md)). Panel tracker: [figures/nguyen/replications.md](../../figures/nguyen/replications.md). Prefer adapter-local conventions over changing Mehregan env defaults.

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| **DSQN** (three-layer **LIF** network, **DQN**-style value learning, §III.B) | Neuromorphic **hardware** deployment and on-chip learning rules |
| **Closed-loop** modulation of **amplitude**, **frequency**, and **pulse width** via ternary $\{-1,0,1\}$ adjustments | Mehregan-style **discrete STN pattern logits** and **DDPG** actor–critic (see [controllers/ddpg/replication.md](../ddpg/replication.md)) |
| **Spike-matrix** observations, **GPi $\alpha$–$\beta$** feedback, **energy-aware** reward (Eq. (7)), **$\epsilon$‑greedy** exploration | In vivo validation, patient-specific clinical programming workflows |
| **Adapter** from the shared plant wrapper to Nguyen I/O when running on the repo’s unified `envs/` stack | Exact **scalar step sizes** for DBS parameter updates (paper gives semantics only) |

---

## 2. Controller role in the closed loop

At each RL step of duration **100 ms** simulated time (§IV):

1. The **plant** integrates the Kumaravelu CBGT network under the current STN DBS settings (amplitude, frequency, pulse width).
2. The environment exposes a **binary spike matrix**—binary spike presence (0/1) per neuron/region in the observation tensor (§III.B Eq. (4))—and computes **GPi $\alpha$–$\beta$ oscillation power** for feedback and termination.
3. The **DSQN** consumes the spike observation, maintains **LIF membrane potentials** over internal time steps, and produces **nine action outputs**. **Applied control** is derived from **spike counts** (argmax over actions) while **Q‑targets** use the **final layer membrane potentials** as continuous value estimates (§III.B).
4. **Reward** balances sub-threshold $\alpha$–$\beta$ power and **DBS energy** (Eqs. (6)–(7)).

The controller does **not** own plant integration; it owns **spike encoding**, **DSQN forward/training**, and **mapping** ternary actions to updated DBS parameters.

---

## 3. Plant and biomarker (shared dynamics, paper-specific feedback)

- **Plant:** Same validated **6-OHDA rat CBGT** network as Nguyen §II.A (Kumaravelu et al., 2016)—**10 neurons per region**, STN DBS actuation ([plant.md](../../plant.md)). Match the reference neuron models (Izhikevich cortex, HH-type BG/thalamus) per [plant.md](../../plant.md) §3.
- **Feedback signal:** **GPi $\alpha$–$\beta$ power**—oscillation power spanning **$\alpha$ (7–13 Hz)** and **$\beta$ (13–35 Hz)** (§II.A; treat as the **7–35 Hz** band unless released code specifies separate $\alpha$ and $\beta$ integrals). Unlike Mehregan Eq. (1), Nguyen does **not** give a closed-form PSD integral for this quantity—**intentionally open** how it is computed from GPi spikes. It is **not** the Mehregan **$P_\beta$ (13–35 Hz)–only** biomarker in [environment.md](../../environment.md) §3.
- **Control threshold:** $\theta = 150$ on the **raw** $\alpha$–$\beta$ scale used in §IV (chosen from the PD-state distribution first quartile; Fig. 3).
- **RL step duration:** **100 ms** simulated time per transition (§IV)—distinct from the **2 s** Mehregan step in [environment.md](../../environment.md) §5.

**Adapter note:** When `controllers/snn/` trains against the shared `envs/` package, the **snn adapter** must (a) run or subsample the plant at Nguyen step duration, (b) build the **spike observation** (Eq. (4)), (c) compute **$\alpha$–$\beta$** for reward/termination, and (d) apply **ternary parameter deltas** instead of discrete STN patterns.

---

## 4. Observation and action spaces

### 4.1 Observation (Eq. (4))

$$
\text{Observation} = \text{Spikes} \in [0,1]^{n \times N}
$$

- **$N$:** Number of neurons represented (paper: spikes **per brain region** across sequence steps; region list follows the CBGT model in §II.A).
- **$n$:** Number of time steps in the observation sequence fed to the DSQN per RL transition.
- **Encoding:** **Binary**—$1$ if a spike occurred in the window, $0$ otherwise (§III.B).

The DSQN is described as processing **batches of 128 inputs** (§III.B)—treat this as the **flattened or sequenced spike feature dimension per forward pass** (implementation detail). The paper also uses **128** for **replay update cadence** (§6.2)—these are distinct quantities; do not conflate input feature dimension with minibatch or replay size. **Which regions and how $n$ aligns with the 100 ms step** are **not** fully fixed in the manuscript — **intentionally open**; document tensor shapes in code and keep them fixed across train/eval.

### 4.2 Action (Eq. (5))

For **each** of the three DBS parameters (**amplitude**, **frequency**, **pulse width**):

$$
a \in \{-1, 0, 1\}
$$

| Value | Semantics |
|-------|-----------|
| $-1$ | Decrease parameter |
| $0$ | Maintain |
| $+1$ | Increase parameter |

Each ternary draw is **multiplied by a scalar sensitivity** (per-parameter step size) to move in the underlying continuous space (§III.B). **Numeric sensitivities are not given** — **intentionally open**; choose values that keep parameters in biologically plausible ranges and document them next to the adapter.

**Network outputs:** **Nine** action outputs (§III.B)—consistent with **three parameters $\times$ three choices** scored independently (not the Mehregan **pattern alphabet**). **Joint vs. factored Q‑heads** (single argmax over 9 vs. three argmaxes) are **not** spelled out — **intentionally open**; pick one scheme, document it, and use it consistently for training and evaluation.

### 4.3 Initial DBS parameters (training)

Per §IV and Fig. 4–6, each episode starts from:

| Parameter | Initial value |
|-----------|----------------|
| Frequency | **40 Hz** |
| Pulse width | **0.3 ms** |
| Amplitude | **300 nA/cm²** |

---

## 5. DSQN architecture and neuron model

### 5.1 LIF dynamics (Eqs. (2)–(3))

Membrane update and spike rule (§II.C):

$$
U[t] = \beta U[t-1] + W X[t] - S_{\mathrm{out}}[t-1]\,\theta_{\mathrm{th}}
$$

$$
S_{\mathrm{out}}[t] = \begin{cases} 1, & U[t] > \theta_{\mathrm{th}} \\ 0, & \text{otherwise} \end{cases}
$$

- **$\beta$:** Membrane leak; **$\beta = 0.95$** in §III.B.
- **$\theta_{\mathrm{th}}$:** Firing threshold (paper symbol $\theta$; distinct from biomarker threshold $\theta = 150$ in §IV).

### 5.2 Network topology (§III.B)

| Layer | Description |
|-------|-------------|
| Input | Linear + LIF on spike features |
| Hidden | **128** LIF neurons |
| Output | **9** LIF units (action heads) |

Each stage: **linear transform** then **LIF** with leak $\beta = 0.95$. The network unrolls over **multiple internal timesteps**, carrying membrane state forward.

### 5.3 Action selection vs. Q-values

- **Behavior (control):** Select the action with the **largest spike count** over the unrolled window (§III.B).
- **Learning (values):** Use the **final-layer membrane potentials** as **continuous Q-value estimates** for Bellman targets and policy improvement (§III.B).

This split is characteristic of **spiking DQN** variants; implementations should not swap spike counts for membrane values in the loss without documenting the change.

---

## 6. Training algorithm (DQN + replay)

### 6.1 Algorithm family

Standard **Deep Q-Network** bootstrapping (§II.B Eq. (1)):

$$
Q^*(s,a) = \mathbb{E}_{s' \sim \mathcal{E}}\bigl[r + \gamma \max_{a'} Q^*(s', a') \mid s, a\bigr]
$$

Train $Q(s,a;\theta) \approx Q^*(s,a)$ by minimizing error to a **target** $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ (or the spiking analogue using membrane potentials at $s'$). The paper does **not** name **Double DQN**, **dueling**, or **target-network period** — **intentionally open**; use a conventional DQN stabilizer set and document it beside the code.

### 6.2 Interaction schedule (§III.B, §IV)

| Item | Value / rule |
|------|----------------|
| Training episodes | **500** |
| Replay cadence | Update weights after every **128** stored transitions |
| Exploration | **$\epsilon$‑greedy** with **decreasing $\epsilon$** each time step (§III.B) |
| Exploration phase | Fig. 4: strong exploration early; transition toward exploitation around episode **~100** (qualitative; exact $\epsilon$ schedule **not** tabulated — **intentionally open**) |

### 6.3 Episode termination (success)

If **$\alpha$–$\beta < \theta$** for **$t_u$ consecutive** time steps, **terminate** the episode early and add a **terminal bonus** using remaining horizon $t_r$ (Eq. (7), §III.B). **$t_u$** is **not** numerically specified — **intentionally open**.

### 6.4 Evaluation protocol (§IV, Fig. 7)

After training, evaluate on **50** test episodes with **per-episode random seeds** (patient-specific variability). Report metrics over **25** time steps per episode unless extending the horizon for ablations (§V notes longer horizons as future work).

---

## 7. DBS energy (Eq. (6))

Per-step energy index:

$$
E_t = N \times \sqrt{\frac{\sum_{i=1}^{n} I_{\mathrm{DBS},i}^2}{n}}
$$

- **$N$:** Number of stimulated neurons.
- **$I_{\mathrm{DBS}}$:** Stimulation current samples in the step (function of amplitude, frequency, pulse width).
- **$n$:** Pulses per time step (§III.C).

Energy enters the **reward** (§8) and reported **~22%** reduction vs. open-loop **130 Hz** DBS after training (§IV).

---

## 8. Reward (Eq. (7))

Let $\theta_u = 1$ if $\alpha$–$\beta < \theta$ (**sub-threshold**, favorable), else $0$. Let $d$ be the **squared distance** of $\alpha$–$\beta$ from $\theta$. Let $E$ be DBS energy (§7). Then:

$$
R = \begin{cases}
-\delta E + \tau \theta_u + (1 - \theta_u)\, d & \text{if not terminated} \\
\tau (t_r + 1) - \delta E & \text{if terminated (early success)}
\end{cases}
$$

| Symbol | Role |
|--------|------|
| $\delta$ | Energy penalty weight |
| $\tau$ | Reward for staying **under** threshold ($\theta_u = 1$); also scales the **terminal** bonus |
| $t_r$ | Remaining steps in episode when terminated early |
| $d$ | Squared gap to $\theta$ when above threshold |

**$\theta_u$ wording in the paper** is ambiguous (“greater than or less than the threshold”), but §III.D defines **$\tau$ as the reward for being under the threshold**, so $\theta_u = 1$ on the **sub-threshold** branch is the reading consistent with Eq. (7).

**Coefficients $\delta$, $\tau$, and the exact $d$ normalization** are **not** given numerically in the excerpted methods — **intentionally open**; align with released code if available, else tune for stable learning and document values.

**Distinction from shared env reward:** [environment.md](../../environment.md) §6 uses Mehregan Eq. (8) on **normalized** beta state with $\beta_t = 0.35$. The **snn** trainer must use **Eq. (7)** above, not the shared-env reward, when replicating Nguyen et al.

---

## 9. Reported outcomes (sanity targets)

After **500** training episodes (§IV), qualitative replication targets:

- **Learned parameters (approx.):** amplitude **~262 nA/cm²**, frequency **~78.65 Hz**, pulse width **~1 ms** (Fig. 6).
- **$\alpha$–$\beta$:** Sustained reduction **below** $\theta = 150$ relative to PD distribution (Fig. 3, 6).
- **Energy:** **~22%** lower **DBS stimulation energy** in simulation than conventional **130 Hz** open-loop DBS under Eq. (6)—not demonstrated neuromorphic hardware or SNN inference power reduction.

These are **evaluation anchors**, not hard CI gates, until the adapter and hyperparameters are fixed in code.

---

## 10. Suggested module interface (`controllers/snn/`)

**Intentionally minimal** so code can evolve without contradicting the paper:

- **`SpikeObservationEncoder`:** Plant traces $\rightarrow$ tensor in $[0,1]^{n \times N}$.
- **`DBSParameterState`:** Holds $(A, f, w)$ with `apply_delta(ternary_actions, sensitivities)`.
- **`LIFLayer` / `DSQN`:** Forward with membrane carry; returns spike counts **and** output membrane potentials.
- **`select_action`:** Argmax on spike counts ($\epsilon$‑greedy overlay).
- **`ReplayBuffer`:** Store $(s, a, r, s', \mathrm{done})$; flush training every **128** transitions.
- **`DSQNTrainer`:** DQN loss on membrane-potential Q estimates; decreasing $\epsilon$.
- **`NguyenEnvAdapter`:** Wraps shared plant `step`/`reset` to **100 ms**, spike obs, $\alpha$–$\beta$, Eq. (7) reward, and early termination.

Hyperparameters with **fixed** values in §IV (episode count, threshold **150**, initial DBS triple, **100 ms** step, replay cadence **128**, hidden **128**, $\beta=0.95$) should be **defaults**; open values ($\gamma$, learning rate, $\epsilon$ schedule, sensitivities, $t_u$, DQN stabilizers) should be **config fields** pointing to this spec.

---

## 11. Consistency checklist

- [ ] Plant is Kumaravelu CBGT with STN DBS; feedback uses **GPi $\alpha$–$\beta$**, threshold **150**.
- [ ] RL step is **100 ms** (adapter), not Mehregan **2 s**, unless running an explicit cross-paper ablation documented elsewhere.
- [ ] Observations are **binary spike matrices** (Eq. (4)); actions are **ternary per-parameter** deltas (Eq. (5)), **not** Mehregan pattern indices.
- [ ] DSQN: **128** hidden LIF, **9** outputs, leak **$\beta = 0.95$**; control from **spike counts**, Q from **membrane potentials**.
- [ ] Training: **500** episodes; replay update every **128** transitions; **decreasing $\epsilon$‑greedy**.
- [ ] Reward follows **Eq. (7)** with energy **Eq. (6)**; **$\theta_u = 1$** when $\alpha$–$\beta < \theta$ (sub-threshold); early stop when sub-threshold for **$t_u$** consecutive steps.
- [ ] Init DBS: **40 Hz**, **0.3 ms**, **300 nA/cm²**; eval: **50** episodes, **25** steps, seeded variability.
- [ ] Adapter documented where shared `envs/` API differs from this spec.

---

## 12. Open questions / TBD

### 1. Spike observation layout ($n \times N$)

Eq. (4) defines a binary spike matrix; §III.B mentions **128** inputs per forward pass (distinct from replay cadence **128**). **Fixed:** binary encoding; **100 ms** RL step. **Chosen (v1):** GPi-only ($N=10$), sequence length $n=10$ (`SNNConfig.n_regions=1`, `sequence_steps=10`). Expand regions when the encoder covers full CBGT.

### 2. Per-parameter DBS delta sensitivities

Ternary actions $\{-1,0,1\}$ scale amplitude, frequency, and pulse width by unspecified scalar step sizes. **Fixed:** three parameters, three ternary choices each (nine outputs). **Chosen (v1):** amplitude **10** nA/cm², frequency **5** Hz, pulse width **0.05** ms per $+1$ (`SNNConfig.*_sensitivity`), with adapter clamps.

### 3. Nine-way vs factored action selection

Nine output LIF units are consistent with three parameters × three choices, but the paper does not say whether control uses one **joint** argmax over nine actions or **three independent** argmaxes. **Fixed:** spike-count selection for behavior, membrane potentials for Q-learning. **Chosen (v1):** `action_scheme="factored"` — three group argmaxes; Q is the sum of the three selected group membrane values (replay index 0–26).

### 4. DQN stabilizers and target-network policy

The paper uses DQN bootstrapping but does not name Double DQN, dueling heads, or target-network **update period**. **Fixed:** replay flush every **128** stored transitions; Q-targets from output membrane potentials at $s'$. **Chosen (v1):** $\gamma=0.99$, Adam lr $10^{-3}$, hard target copy every **100** gradient updates (`SNNConfig.target_update_period`).

### 5. $\epsilon$-greedy exploration schedule

§III.B specifies decreasing $\epsilon$ each time step; Fig. 4 qualitatively shifts toward exploitation around episode **~100**, but no numeric schedule is tabulated. **Fixed:** $\epsilon$-greedy overlay on spike-count argmax. **Chosen (v1):** $\epsilon: 1.0 \rightarrow 0.05$ over **2500** env steps (~100 episodes × 25 steps).

### 6. Early-termination persistence $t_u$

Episodes terminate early when GPi $\alpha$–$\beta < \theta = 150$ for $t_u$ **consecutive** steps, with terminal bonus using remaining horizon $t_r$ (Eq. (7)). **Fixed:** threshold **150**; bonus uses $t_r$. **Chosen (v1):** $t_u = 3$ (`SNNConfig.subthreshold_steps_required`).

### 7. Reward coefficients and distance metric $d$

Eq. (7) combines energy penalty $\delta E$, threshold indicator $\theta_u$, squared gap $d$, and terminal scale $\tau$, but numeric $\delta$, $\tau$, and the exact normalization of $d$ are not given. **Fixed:** reward structure and energy index Eq. (6); **$\theta_u = 1$ when $\alpha$–$\beta < \theta$** (consistent with $\tau$ as “reward for being under the threshold”). **Chosen (v1):** $\delta=0.01$, $\tau=1.0$ (`SNNConfig.energy_penalty`, `threshold_reward`).

### 8. LIF firing threshold $\theta_{\mathrm{th}}$

§III.B fixes leak $\beta = 0.95$ and distinguishes biomarker threshold $\theta = 150$ from the LIF spike threshold symbol $\theta_{\mathrm{th}}$. **Fixed:** three-layer LIF DSQN with **128** hidden and **9** output units. **Chosen (v1):** $\theta_{\mathrm{th}} = 1.0$ (`SNNConfig.lif_threshold`).

### 9. Shared `envs/` adapter vs Nguyen timing

The unified Gym API follows Mehregan **2 s** steps and Eq. (8) reward; Nguyen requires **100 ms** steps, spike observations, $\alpha$–$\beta$ feedback, and Eq. (7) reward. **Fixed:** **100 ms** step via `NguyenEnvAdapter` calling `PythonPlant.integrate(duration_s=0.1)` each transition; Eq. (7) reward; ternary parameter deltas; early termination on $\alpha$–$\beta$. **Chosen (v1):** GPi spike trains from each 100 ms segment are binned into a $(10 \times 10)$ matrix (`sequence_steps=10`, `neurons_per_region=10`); unstable DBS triples that crash the Kumaravelu integrator roll back to the previous triple (`plant_guard` in step info).

### 10. Evaluation horizon extensions

§IV eval reports **25** steps over **50** seeded episodes; §V notes longer horizons as future work. **Fixed:** default eval protocol above. **Open:** whether ablations extend episode length beyond 25 steps. **Decide in** `nguyen_eval` suite config when running horizon ablations.

---

## 13. References

- Nguyen et al., *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks*.
- Kumaravelu et al. (2016) CBGT model (plant); see [plant.md](../../plant.md) for repo integration notes.

For **benchmarking**, use the `nguyen_eval` suite per [benchmarking.md](../../benchmarking.md) §3.2; for optional cross-paper plant-level comparison, see [benchmarking.md](../../benchmarking.md) §3.3.
