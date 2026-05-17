# RL controller specification (Mehregan et al., adaptive DBS / quantization)

This document specifies the **learning-based controller** (actor–critic training loop, losses, targets, and optional quantization) from *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning* (Mehregan et al.). It is meant to align `controllers/` (and training scripts) with the published method.

**Companion spec:** Plant dynamics, biomarker $P_\beta$, step timing, reward $R$, and baselines are defined in [environment.md](environment.md). That document is authoritative for **environment** I/O; this document is authoritative for **policy optimization** details unless the two explicitly cross-reference the same quantity.

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| Offline DDPG-style **actor–critic** training with **replay buffer**, **target networks**, and losses as in the paper’s §III and Algorithm 1 | **In vivo** optogenetics protocol, trial structure, and recording pipeline (§IV.B) |
| **CNN** actor and critic over a **temporal** biomarker state; **discrete** stimulation patterns via **logits → softmax → argmax** | Exact **pattern alphabet** size and waveform encoding beyond “discrete STN patterns” (intentionally open; fix in code) |
| **PTQ** and **QAT** as described in §III.D | Hardware-specific latency, power budgets, or certified medical deployment |

---

## 2. Controller role in the closed loop

At each RL step of duration $l$ (see [environment.md](environment.md) §5: **2 s** simulated time):

1. The **environment** integrates the parkinsonian network under the chosen STN stimulation pattern and returns biomarker-derived observations (primarily $P_\beta$ and its windowed form as state $s$).
2. The **actor** $\mu(\cdot \mid \theta_\mu)$ maps the current state to **action logits**; **softmax + argmax** yields the discrete pattern applied until the next step.
3. The **critic** $Q(\cdot, \cdot \mid \theta_c)$ scores state–**logit** pairs used in the published losses (Algorithm 1 stores $a_{\mathrm{logit}}$ for the critic).

The controller does **not** compute $P_\beta$ inside the policy network; it consumes whatever vector (or tensor) the environment defines as $s$, consistent with §III.B and Algorithm 1.

---

## 3. Networks and representations

### 3.1 Naming note (DDPG vs discrete actions)

The paper frames the method as **Deep Deterministic Policy Gradient (DDPG)** with an actor–critic architecture. Operationally, the **actor outputs logits over a finite set of stimulation patterns**, and the **applied action** is the **argmax** pattern. Implementations should treat the **stored trainable output** as **logits** (or one-hot–equivalent input to the critic), not as unconstrained continuous stimulation parameters.

### 3.2 Actor $\mu(s \mid \theta_\mu)$

- **Input:** State $s$ formed from the biomarker trajectory (paper §III.B: **CNN** suited to temporal $P_\beta$ structure). For the computational study, the biomarker **window matches the full simulation segment** per step ([environment.md](environment.md) §3.2, §5).
- **Backbone:** Convolutional layers, **average pooling**, then **linear** layers (paper §III.B, Figure 3a). Channel counts, kernel sizes, and **“shrink dimension”** are **not** numerically specified in §IV.A.1 — **intentionally open**; choose a compact CNN, document shapes in code, and keep them fixed across training / evaluation / quantization comparisons.
- **Output:** **Logits** over the discrete pattern set. Training stores these as $a_{\mathrm{logit}}$.
- **Action execution:** $\mathrm{softmax}(\mathrm{logits})$ then **argmax** selects the pattern index sent to the plant (paper §III.B).

### 3.3 Critic $Q(s, a_{\mathrm{logit}} \mid \theta_c)$

- **Input:** Same state representation as the actor, concatenated or otherwise fused with the **action logits** (paper §III.B, Figure 3b; losses use $Q(s^{\tilde{B}}, a_{\mathrm{logit}}^{\tilde{B}})$).
- **Output:** Scalar **state–action value** estimate.
- **Target network:** $Q_{\mathrm{target}}$ and $\mu_{\mathrm{target}}$ for bootstrapping (Algorithm 1, Eq. (3)).

### 3.4 “Brain” configuration (paper wording)

The paper’s **Brain** class bundles **pattern / frequency semantics**, biomarker **window shape**, and **action cardinality** (§III.B). A minimal implementation should expose those as explicit config fields so `envs/` and `controllers/` stay aligned.

---

## 4. Training algorithm (Algorithm 1)

### 4.1 Initialization

Initialize:

- Actor $\mu(\cdot \mid \theta_\mu)$, critic $Q(\cdot,\cdot \mid \theta_c)$, and matching **target** copies $\mu_{\mathrm{target}}$, $Q_{\mathrm{target}}$.
- **Replay buffer** $B$ (capacity **8192** in §IV.A.1).
- **Discount** $\gamma$, **soft update rate** $\tau$, and inner-loop **update frequency** (gradient steps per env step). **§IV.A.1 does not give numeric $\gamma$, $\tau$, or update frequency** — **intentionally open**; match released code if available, else pick standard DDPG defaults and document them next to the implementation.

### 4.2 Per-step interaction

For each environment step of duration $l$:

1. **Select action:** $u \leftarrow \mu(s \mid \theta_\mu)$ (forward pass yields logits; argmax selects discrete pattern $a$ applied to STN).
2. **Step environment:** obtain next state $s'$, reward $R$, and episode-done flag $dw \in \{0,1\}$ ($dw=1$ iff the episode finished after this transition, for $Q$ targets; see [environment.md](environment.md) §7).
3. **Store transition** in $B$. Algorithm 1 lists the tuple as $(s', a, a_{\mathrm{logit}}, R, s, dw)$; a practical layout is equivalent if it preserves $(s, a_{\mathrm{logit}}, R, s', dw)$ for standard replay sampling.

### 4.3 Minibatch update (repeated `update_frequency` times per step)

Sample minibatch $\tilde{B}$ from $B$ (batch size **32** in §IV.A.1).

**Bootstrap target (paper Eq. (3)):**

$$
Q_{\mathrm{target}} \leftarrow R^{\tilde{B}} + \gamma (1 - dw^{\tilde{B}})\, Q_{\mathrm{target}}\bigl(s'^{\tilde{B}},\, \mu_{\mathrm{target}}(s'^{\tilde{B}})\bigr)
$$

Here $\mu_{\mathrm{target}}(s'^{\tilde{B}})$ denotes the **logits** produced by the target actor on $s'$ (used inside $Q_{\mathrm{target}}$ as in standard actor–critic bootstrapping).

**Critic loss (paper Eq. (4)):**

$$
J_{\mathrm{critic}} = \mathrm{MSE}\Bigl(Q_{\mathrm{target}},\, Q\bigl(s^{\tilde{B}}, a_{\mathrm{logit}}^{\tilde{B}}\bigr)\Bigr)
$$

Optimize $J_{\mathrm{critic}}$ with **Adam** (critic learning rate **$10^{-3}$** in §IV.A.1).

**Actor update (paper §III.B, Eq. (5)):**

$$
J_{\mathrm{Actor}} = \frac{1}{|\tilde{B}|} \sum_{t \in \tilde{B}} Q\bigl(s(t), a_{\mathrm{logit}}(t)\bigr)
$$

Maximize $J_{\mathrm{Actor}}$ (implementation: minimize $-J_{\mathrm{Actor}}$) with **Adam** (actor learning rate **$5 \times 10^{-4}$** in §IV.A.1).

**Critic freeze during actor step:** The paper’s Algorithm 1 **freezes the critic** while updating the actor, then **unfreezes** before the next micro-batch iteration. Mirror that ordering to match the described compute-saving behavior.

**Soft target updates (paper Eqs. (6)–(7)):** After both optimizers step, Polyak-average both targets:

$$
\theta_{c,\mathrm{target}} \leftarrow \tau \theta_c + (1-\tau)\theta_{c,\mathrm{target}}, \qquad
\theta_{\mu,\mathrm{target}} \leftarrow \tau \theta_\mu + (1-\tau)\theta_{\mu,\mathrm{target}}
$$

(Use the actor’s weights $\theta_\mu$ for $\theta_{\mu,\mathrm{target}}$; the manuscript’s “$\theta_{u}$” denotes the actor.)

### 4.4 Episode schedule

- **30** steps per episode, **10** training episodes for the reported computational run ([environment.md](environment.md) §5, §8).
- **Policy initialization:** Regular pulses at **mean** frequency **45 Hz** for the main experiment; **30 Hz** variant with other hyperparameters unchanged (paper §IV.A.1–2).

### 4.5 Output artifact

After training, the **actor network** (full precision unless using QAT) is the **deployed policy** $\pi^* \approx \mu(\cdot \mid \theta_\mu)$ mapping biomarker state to pattern logits.

---

## 5. Reward and critic alignment

The **instantaneous reward** $R$ is defined in paper Eq. (8) from the averaged observation ([environment.md](environment.md) §6). The **critic** predicts discounted return under that reward; **normalization** of $s(i)$ must be **shared** between the observation pipeline and $R$ so that $\beta_t = 0.35$ remains meaningful ([environment.md](environment.md) §6 implementation note).

---

## 6. Quantization (§III.D)

Optional for replication of §IV.A.3:

| Mode | When | Notes |
|------|------|------|
| **PTQ** | After full-precision training; **inference** only | PyTorch-style **dynamic** weight quantization mentioned in the paper; activations mapped with Eqs. (9)–(11). Evaluated at **FP16** and **INT8** in §IV.A.3. |
| **QAT** | During training | **Fake quantization** on selected layers; actor: quant stub on **input**, dequant stub on **logits** before action selection (§III.D). Paper reports **weaker** beta suppression than PTQ under **10 episodes**; longer training may be needed. |

**Environment contract:** Quantization changes **network weights/activations**, not plant timing or biomarker definition. The **env** step API stays as in [environment.md](environment.md).

---

## 7. Suggested module interface (`controllers/`)

**Intentionally minimal** so code can evolve without contradicting the paper:

- **`Actor`:** `forward(state) -> logits`; helper `select_action(logits) -> pattern_id` (softmax + argmax) for env interaction.
- **`Critic`:** `forward(state, action_logits) -> scalar`.
- **`TargetNet`:** Polyak copy of actor/critic parameters each update.
- **`ReplayBuffer`:** Store at least $(s, a_{\mathrm{logit}}, R, s', dw)$ with capacity 8192.
- **`MehreganTrainer` (or equivalent):** Implements Algorithm 1 ordering (env step → buffer → `update_frequency` × minibatch updates).

Hyperparameters with **fixed** values in §IV.A.1 should be **defaults**; open values ($\gamma$, $\tau$, update frequency, CNN topology, pattern count) should be **constructor or config fields** with comments pointing to this spec.

---

## 8. Consistency checklist

- [ ] Actor is **CNN-over-time** on biomarker state; critic uses **same state geometry** plus **logits**.
- [ ] Applied control is **discrete pattern index** from **argmax**; replay stores **logits** for the critic.
- [ ] Target value uses **$Q_{\mathrm{target}}(s', \mu_{\mathrm{target}}(s'))$** with $(1-dw)$ masking.
- [ ] Critic **MSE** to bootstrap target; actor maximizes **$Q(s, a_{\mathrm{logit}})$** with critic **frozen** during actor Adam step.
- [ ] Soft updates use shared **$\tau$** for actor and critic targets.
- [ ] Learning rates **$5\times 10^{-4}$** (actor), **$10^{-3}$** (critic); buffer **8192**; batch **32**; **10** episodes × **30** steps; step **2 s**; init mean **45 Hz** (and **30 Hz** ablation).
- [ ] Quantization experiments: document **FP16 / INT8 PTQ** vs **QAT** training budget separately.

---

## 9. Reference

- Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*.

For the **computational plant and evaluation protocol**, see [environment.md](environment.md).
