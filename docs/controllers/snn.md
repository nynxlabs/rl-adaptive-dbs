# SNN controller specification (Nguyen et al., closed-loop neuromorphic DBS)

This document specifies the **deep spiking Q-network (DSQN)** controller from *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks* (Nguyen et al.). It is meant to align `controllers/snn/` (and training scripts) with the published method.

**Companion spec:** The **Kumaravelu et al. (2016)** parkinsonian CBGT plant is shared with other controllers; dynamics and provenance are summarized in [environment.md](../environment.md) §2. That document is authoritative for the **Mehregan et al.** Gymnasium API (2 s steps, $P_\beta$-only state, pattern actions, Eq. (8) reward). **This document is authoritative for the Nguyen controller**—observation, action, reward, timing, and DSQN training—unless the two explicitly describe the same quantity.

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| **DSQN** (three-layer **LIF** network, **DQN**-style value learning, §III.B) | Neuromorphic **hardware** deployment and on-chip learning rules |
| **Closed-loop** modulation of **amplitude**, **frequency**, and **pulse width** via ternary $\{-1,0,1\}$ adjustments | Mehregan-style **discrete STN pattern logits** and **DDPG** actor–critic (see [controllers/ddpg.md](ddpg.md)) |
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

- **Plant:** Same validated **6-OHDA rat CBGT** Hodgkin–Huxley network as Nguyen §II.A (Kumaravelu et al., 2016)—**10 neurons per region**, STN DBS actuation ([environment.md](../environment.md) §2).
- **Feedback signal:** **GPi $\alpha$–$\beta$ power**—combined **$\alpha$ (7–13 Hz)** and **$\beta$ (13–35 Hz)** band power (§II.A), **not** the Mehregan **$P_\beta$ (13–35 Hz)–only** biomarker used in [environment.md](../environment.md) §3.
- **Control threshold:** $\theta = 150$ on the **raw** $\alpha$–$\beta$ scale used in §IV (chosen from the PD-state distribution first quartile; Fig. 3).
- **RL step duration:** **100 ms** simulated time per transition (§IV)—distinct from the **2 s** Mehregan step in [environment.md](../environment.md) §5.

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

Let $\theta_u = 1$ if $\alpha$–$\beta > \theta$ (biomarker threshold), else $0$. Let $d$ be the **squared distance** of $\alpha$–$\beta$ from $\theta$. Let $E$ be DBS energy (§7). Then:

$$
R = \begin{cases}
-\delta E + \tau \theta_u + (1 - \theta_u)\, d & \text{if not terminated} \\
\tau (t_r + 1) - \delta E & \text{if terminated (early success)}
\end{cases}
$$

| Symbol | Role |
|--------|------|
| $\delta$ | Energy penalty weight |
| $\tau$ | Reward for staying under threshold / terminal bonus scale |
| $t_r$ | Remaining steps in episode when terminated early |
| $d$ | Squared gap to $\theta$ when above threshold |

**Coefficients $\delta$, $\tau$, and the exact $d$ normalization** are **not** given numerically in the excerpted methods — **intentionally open**; align with released code if available, else tune for stable learning and document values.

**Distinction from shared env reward:** [environment.md](../environment.md) §6 uses Mehregan Eq. (8) on **normalized** beta state with $\beta_t = 0.35$. The **snn** trainer must use **Eq. (7)** above, not the shared-env reward, when replicating Nguyen et al.

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
- [ ] Reward follows **Eq. (7)** with energy **Eq. (6)**; early stop when $\alpha$–$\beta < \theta$ for **$t_u$** consecutive steps.
- [ ] Init DBS: **40 Hz**, **0.3 ms**, **300 nA/cm²**; eval: **50** episodes, **25** steps, seeded variability.
- [ ] Adapter documented where shared `envs/` API differs from this spec.

---

## 12. Reference

- Nguyen et al., *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks*.
- Kumaravelu et al. (2016) CBGT model (plant); see [environment.md](../environment.md) §2 for repo integration notes.

For **benchmarking**, use the `nguyen_eval` suite (§3.2); for optional cross-paper plant-level comparison, see [benchmarking.md](../benchmarking.md) §3.3.
