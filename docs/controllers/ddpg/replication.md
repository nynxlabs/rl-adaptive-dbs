# RL controller specification (Mehregan et al., adaptive DBS / quantization)

This document specifies the **learning-based controller** (actor–critic training loop, losses, targets, and optional quantization) from *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning* (Mehregan et al.). It is meant to align `controllers/ddpg/` (and training scripts) with the published method.

**Companion spec:** Plant dynamics and biomarker primitives — [plant.md](../../plant.md). Mehregan env ($P_\beta$ window, step timing, reward $R$, baselines) — [environment.md](../../environment.md). Benchmark suites and variant slugs — [benchmarking.md](../../benchmarking.md); CLI — [cli.md](../../cli.md). This document is authoritative for **policy optimization** unless the others explicitly describe the same quantity.

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| **DDPG-style actor–critic** training (Algorithm 1: **online** simulator interaction, **replay buffer**, **target networks**, losses as in §III) | **In vivo** optogenetics protocol, trial structure, and recording pipeline (§IV.B) |
| **CNN** actor and critic over a **temporal** biomarker state; **discrete** stimulation patterns via **logits → softmax → argmax** | Exact **pattern alphabet** size and waveform encoding beyond “discrete STN patterns” (intentionally open; fix in code) |
| **PTQ** and **QAT** as described in §III.D | Hardware-specific latency, power budgets, or certified medical deployment |

---

## 2. Controller role in the closed loop

At each RL step of duration $l$ (see [environment.md](../../environment.md) §5: **2 s** simulated time):

1. The **environment** integrates the parkinsonian network under the chosen STN stimulation pattern and returns biomarker-derived observations (primarily $P_\beta$ and its windowed form as state $s$).
2. The **actor** $\mu(\cdot \mid \theta_\mu)$ maps the current state to **action logits**; **softmax + argmax** yields the discrete pattern applied until the next step.
3. The **critic** $Q(\cdot, \cdot \mid \theta_c)$ scores state–**logit** pairs used in the published losses (Algorithm 1 stores $a_{\mathrm{logit}}$ for the critic).

The controller does **not** compute $P_\beta$ inside the policy network; it consumes whatever vector (or tensor) the environment defines as $s$, consistent with §III.B and Algorithm 1.

---

## 3. Networks and representations

### 3.1 Naming note (DDPG vs discrete actions)

The paper frames the method as **Deep Deterministic Policy Gradient (DDPG)** with an actor–critic architecture. Operationally, the **actor outputs logits over a finite set of stimulation patterns**, and the **applied action** is the **argmax** pattern. Implementations should treat the **stored trainable output** as **logits** (or one-hot–equivalent input to the critic), not as unconstrained continuous stimulation parameters.

### 3.2 Actor $\mu(s \mid \theta_\mu)$

- **Input:** State $s$ formed from the biomarker trajectory (paper §III.B: **CNN** suited to temporal $P_\beta$ structure). For the computational study, the biomarker **window matches the full simulation segment** per step ([environment.md](../../environment.md) §3.2, §5).
- **Backbone:** Convolutional layers, **average pooling**, then **linear** layers (paper §III.B, Figure 3a). Channel counts, kernel sizes, and **“shrink dimension”** are **not** numerically specified in §IV.A.1 — **intentionally open**; choose a compact CNN, document shapes in code, and keep them fixed across training / evaluation / quantization comparisons.
- **Output:** **Logits** over the discrete pattern set. Training stores these as $a_{\mathrm{logit}}$.
- **Action execution:** $\mathrm{softmax}(\mathrm{logits})$ then **argmax** selects the pattern index sent to the plant (paper §III.B).

### 3.3 Critic $Q(s, a_{\mathrm{logit}} \mid \theta_c)$

- **Input:** Same state representation as the actor, concatenated or otherwise fused with the **action logits** (paper §III.B, Figure 3b; losses use $Q(s^{\tilde{B}}, a_{\mathrm{logit}}^{\tilde{B}})$).
- **Output:** Scalar **state–action value** estimate.
- **Target network:** $Q_{\mathrm{target}}$ and $\mu_{\mathrm{target}}$ for bootstrapping (Algorithm 1, Eq. (3)).

### 3.4 “Brain” configuration (paper wording)

The paper’s **Brain** class bundles **pattern / frequency semantics**, biomarker **window shape**, and **action cardinality** (§III.B). A minimal implementation should expose those as explicit config fields so `envs/` and `controllers/ddpg/` stay aligned.

---

## 4. Training algorithm (Algorithm 1)

### 4.1 Initialization

Initialize:

- Actor $\mu(\cdot \mid \theta_\mu)$, critic $Q(\cdot,\cdot \mid \theta_c)$, and matching **target** copies $\mu_{\mathrm{target}}$, $Q_{\mathrm{target}}$. Algorithm 1 does not specify initializing target networks as copies of the online networks; that copy-at-init convention is standard in DDPG implementations but is not stated in the paper.
- **Replay buffer** $B$ (capacity **8192** in §IV.A.1).
- **Discount** $\gamma$, **soft update rate** $\tau$, and inner-loop **update frequency** (gradient steps per env step). **§IV.A.1 does not give numeric $\gamma$, $\tau$, or update frequency** — **intentionally open**; match released code if available, else pick standard DDPG defaults and document them next to the implementation.

### 4.2 Per-step interaction

For each environment step of duration $l$:

1. **Select action:** $u \leftarrow \mu(s \mid \theta_\mu)$ (forward pass yields logits; **greedy** softmax + argmax selects discrete pattern $a$ applied to STN at **evaluation**). During **training**, the implementation supports two exploration modes (`DDPGConfig.exploration_mode`):
   - **`epsilon` (default):** $\epsilon$-greedy — with probability $\epsilon_t$ sample a uniform random pattern, else argmax on logits. $\epsilon_t$ linearly decays from **0.5** to **0.1** over the full training schedule.
   - **`softmax`:** sample from $\mathrm{Categorical}(\mathrm{softmax}(\mathrm{logits}/\tau_t))$ with temperature $\tau_t$ linearly annealed from **2.0** to **0.5**.
   Replay stores the actor logits $a_{\mathrm{logit}}$ from the forward pass (not the random/sampled override). The paper does not specify online exploration; these conventions break constant-policy collapse when `state_length > 1` (TASK-67).
   - **Exploration vs greedy (TASK-67 finding):** Exploration **does not help** with this reward landscape. The reward range across all 41 actions is only ~1.13 (~0.03 per adjacent action pair). Exploration noise (epsilon/softmax) adds variance that **drowns out** the weak reward signal, causing the critic to learn a flat Q-landscape and the actor to collapse to a constant action. The paper's **greedy argmax** approach concentrates data around the current policy's actions, giving the critic a clearer (though still weak) signal. Additional exploration knobs (`logit_noise_std`, `entropy_coeff`, `random_warmup_steps`, `obs_normalize`) are implemented in `DDPGConfig` but do not resolve the collapse — the bottleneck is the reward signal strength, not the exploration strategy.

**Critical finding (Jul 2026):** With `state_length=15`, only the last element of the 15-length observation window changes per step (14/15 are stale history). The CNN cannot distinguish action effects — constant-policy collapse is **inevitable** regardless of exploration strategy. State diversity diagnostic confirms the plant IS responsive (P$_\beta$ spans 0.150–0.559 across actions), but the signal-to-noise ratio in the 15-length window is too low. **The paper uses `state_length=1`** (single P$_\beta$ per step = direct signal). `state_length=1` is the correct default; `state_length > 1` requires architectural changes (obs preprocessing or attention) to work. See TASK-67 comments for full evidence.
2. **Step environment:** obtain next state $s'$, reward $R$, and episode-done flag $dw \in \{0,1\}$ ($dw=1$ iff the episode finished after this transition, for $Q$ targets; see [environment.md](../../environment.md) §7).
3. **Store transition** in $B$. Algorithm 1 lists the tuple as $(s', a, a_{\mathrm{logit}}, R, s, dw)$; a practical layout is equivalent if it preserves $(s, a, a_{\mathrm{logit}}, R, s', dw)$ for standard replay sampling. Store the discrete pattern index $a$ alongside $a_{\mathrm{logit}}$ (derivable from argmax but included in the paper tuple) for paper-aligned replay dumps.

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

**Critic action input (`DDPGConfig.critic_action_input`):** The paper tuple stores actor logits $a_{\mathrm{logit}}$, but when **training exploration** overrides argmax (§4.2), reward $R$ depends on the **executed** discrete index $a$, not on $\arg\max a_{\mathrm{logit}}$. Default **`one_hot`** feeds the critic a one-hot vector for the executed $a$ from replay; bootstrap uses $\arg\max \mu_{\mathrm{target}}(s')$ as a one-hot vector. Legacy **`logits`** mode uses stored $a_{\mathrm{logit}}$ for both critic and bootstrap (paper tuple layout; valid only when interaction is greedy).

**Actor update (paper §III.B, Eq. (5)):**

$$
J_{\mathrm{Actor}} = \frac{1}{|\tilde{B}|} \sum_{t \in \tilde{B}} Q\bigl(s(t), a_{\mathrm{logit}}(t)\bigr)
$$

Maximize $J_{\mathrm{Actor}}$ (implementation: minimize $-J_{\mathrm{Actor}}$) with **Adam** (actor learning rate **$5 \times 10^{-4}$** in §IV.A.1). With **`one_hot`** critic input, the implementation maximizes the **softmax expectation** $\sum_a \mathrm{softmax}(\mu(s))_a\, Q(s, a)$ over discrete actions (differentiable discrete policy improvement). With **`logits`**, the actor maximizes $Q(s, \mu(s))$ on fresh logits.

**Critic freeze during actor step:** The paper’s Algorithm 1 **freezes the critic** while updating the actor, then **unfreezes** before the next micro-batch iteration. Mirror that ordering to match the described compute-saving behavior.

**Soft target updates (paper Eqs. (6)–(7)):** After both optimizers step, Polyak-average both targets:

$$
\theta_{c,\mathrm{target}} \leftarrow \tau \theta_c + (1-\tau)\theta_{c,\mathrm{target}}, \qquad
\theta_{\mu,\mathrm{target}} \leftarrow \tau \theta_\mu + (1-\tau)\theta_{\mu,\mathrm{target}}
$$

(Use the actor’s weights $\theta_\mu$ for $\theta_{\mu,\mathrm{target}}$; the manuscript’s “$\theta_{u}$” denotes the actor.)

### 4.4 Episode schedule

- **30** steps per episode, **10** training episodes for the reported computational run ([environment.md](../../environment.md) §5, §8).
- **Policy initialization:** Regular pulses at **mean** frequency **45 Hz** for the main experiment; **30 Hz** variant with other hyperparameters unchanged (paper §IV.A.1–2).

### 4.5 Output artifact

After training, the **actor network** (full precision unless using QAT) is the **deployed policy** $\pi^* \approx \mu(\cdot \mid \theta_\mu)$ mapping biomarker state to pattern logits.

### 4.6 Variants (benchmark slugs)

Repo **variant** slugs for `controller: ddpg` map to Mehregan et al. §IV experiments as follows (full taxonomy: [benchmarking.md](../../benchmarking.md) §2):

| Variant slug | Paper experiment |
|--------------|------------------|
| `paper` | §IV.A.1 — full-precision training with **45 Hz** mean init |
| `init-30hz` | §IV.A.2 — **30 Hz** init ablation |
| `ptq-fp16`, `ptq-int8` | §IV.A.3 — **PTQ** (post-training quantization) |
| `qat` | §IV.A.3 — **QAT** (quantization-aware training) |

---

## 5. Reward and critic alignment

The **instantaneous reward** $R$ is defined in paper Eq. (8) from the averaged observation ([environment.md](../../environment.md) §6). The **critic** predicts discounted return under that reward; **normalization** of $s(i)$ must be **shared** between the observation pipeline and $R$ so that $\beta_t = 0.35$ remains meaningful ([environment.md](../../environment.md) §6 implementation note).

**Reward sign bug (found 2026-07-07, TASK-67):** The paper's Eq. (8) as printed yields a **negative** value in the below-threshold branch ($\delta < 0$ when $s(i) < \beta_t$), but the prose and Figure 3c specify **positive reward below threshold**. The original code (`return delta`) penalized the agent for keeping the biomarker suppressed — the opposite of training intent. Fix: `return -delta` in the below-threshold branch. This bug caused all DDPG training runs (epsilon-greedy, softmax, greedy argmax, all state lengths) to collapse to constant actions, because the reward signal was inverted. After the fix, episode-1 reward jumps from ~-26 to ~-1.7 at sl=1.

---

## 6. Quantization (§III.D)

Optional for replication of §IV.A.3:

| Mode | When | Notes |
|------|------|------|
| **PTQ** | After full-precision training; **inference** only | PyTorch-style **dynamic** weight quantization mentioned in the paper; activations mapped with Eqs. (9)–(11). Evaluated at **FP16** and **INT8** in §IV.A.3. |
| **QAT** | During training | **Fake quantization** on selected layers; actor: quant stub on **input**, dequant stub on **logits** before action selection (§III.D). Implemented via PyTorch eager-mode `prepare_qat` on `Conv1d` / `Linear` plus input/output stubs (`controllers/ddpg/quantization.py`). Checkpoints store FP actor weights plus `qat_state_dict` (observer / fake-quant buffers) for eval. Paper reports **weaker** beta suppression than PTQ under **10 episodes**; longer training may be needed. |

**Implementation notes:** QAT uses the platform default eager backend (`fbgemm` on x86, `qnnpack` on ARM). Training device defaults to CPU (`DDPGConfig.device`). PTQ variants never train — they load a `paper` checkpoint and quantize at eval only.

**Environment contract:** Quantization changes **network weights/activations**, not plant timing or biomarker definition. The **env** step API stays as in [environment.md](../../environment.md).

---

## 7. Suggested module interface (`controllers/ddpg/`)

**Intentionally minimal** so code can evolve without contradicting the paper:

- **`Actor`:** `forward(state) -> logits`; helper `select_action(logits) -> pattern_id` (softmax + argmax) for env interaction.
- **`Critic`:** `forward(state, action_logits) -> scalar`.
- **`TargetNet`:** Polyak copy of actor/critic parameters each update.
- **`ReplayBuffer`:** Store at least $(s, a, a_{\mathrm{logit}}, R, s', dw)$ with capacity 8192.
- **`DDPGTrainer` (or equivalent):** Implements Algorithm 1 ordering (env step → buffer → `update_frequency` × minibatch updates).
- **`train` / `evaluate`:** Module entry points for training and post-training eval (`controllers/ddpg/__init__.py`); CLI delegates here in a later phase.
- **`save_checkpoint` / `load_actor`:** Persist and restore actor weights + `DDPGConfig` (`controllers/ddpg/checkpoint.py`).
- **`run_replication` / `write_replication_summary`:** Paper-scale train → `mehregan_eval` → baseline comparison (`controllers/ddpg/replication.py`); MATLAB entry point: `scripts/replicate_mehregan_ddpg.py`.

Hyperparameters with **fixed** values in §IV.A.1 should be **defaults**; open values ($\gamma$, $\tau$, update frequency, CNN topology, pattern count) should be **constructor or config fields** with comments pointing to this spec.

---

## 8. Consistency checklist

- [x] Actor is **CNN-over-time** on biomarker state; critic uses **same state geometry** plus **logits**.
- [x] Applied control is **discrete pattern index** $a$ from **argmax**; replay stores **$a$** and **logits** $a_{\mathrm{logit}}$ for the critic.
- [x] Target value uses **$Q_{\mathrm{target}}(s', \mu_{\mathrm{target}}(s'))$** with $(1-dw)$ masking.
- [x] Critic **MSE** to bootstrap target; actor maximizes **$Q(s, \mu(s))$** with critic **frozen** during actor Adam step.
- [x] Soft updates use shared **$\tau$** for actor and critic targets.
- [x] Learning rates **$5\times 10^{-4}$** (actor), **$10^{-3}$** (critic); buffer **8192**; batch **32**; **10** episodes × **30** steps; step **2 s**; init mean **45 Hz** (and **30 Hz** ablation via `init-30hz` variant).
- [x] Quantization experiments (**FP16 / INT8 PTQ**, **QAT**): `controllers/ddpg/quantization.py`; validate via `mehregan_eval` slugs `ptq-fp16`, `ptq-int8`, `qat` ([benchmarking.md](../../benchmarking.md)).

---

## 9. Open questions / TBD

### 1. Discrete pattern alphabet size and encoding

The actor outputs logits over a finite STN pattern set, but the paper does not fix **cardinality** or per-pattern waveform semantics. **Fixed:** discrete control via softmax + argmax; replay stores pattern index $a$ and logits $a_{\mathrm{logit}}$. **Open:** alphabet definition. **Decide in** shared env / Brain config and keep fixed across train, eval, and quantization (see [environment.md](../../environment.md) §4).

### 2. CNN actor–critic topology

§III.B specifies a CNN with average pooling and linear heads but §IV.A.1 gives no channel counts, kernel sizes, or **shrink dimension**. **Fixed:** temporal CNN over biomarker state; critic fuses the same state with action logits. **Open:** layer shapes. **Decide in** `controllers/ddpg/` and document tensor geometry in code.

**Implemented (`controllers/ddpg/networks.py`):** `Conv1d(1→16, k=3)` → `AdaptiveAvgPool1d(shrink_dim=4)` → `Conv1d(16→32, k=3)` → flatten → linear head; critic concatenates encoded state with action logits before MLP. `shrink_dim` and `conv_channels` are `DDPGConfig` fields.

### 3. Discount, Polyak $\tau$, and update frequency

Algorithm 1 requires $\gamma$, soft-update $\tau$, and **update_frequency** (gradient steps per env step); §IV.A.1 lists none of these numerically. **Fixed:** Adam rates, buffer **8192**, batch **32**, **10** episodes × **30** steps. **Open:** $\gamma$, $\tau$, update cadence. **Decide in** trainer config; match released code if available, else standard DDPG defaults with explicit documentation.

**Implemented (`controllers/ddpg/config.py`):** $\gamma = 0.99$, $\tau = 0.005$, `update_frequency = 1` (standard DDPG defaults when the paper is silent).

### 4. Target network initialization

Standard DDPG copies online weights into $\mu_{\mathrm{target}}$ and $Q_{\mathrm{target}}$ at init, but Algorithm 1 does not state this. **Fixed:** Polyak updates after each optimizer step (Eqs. (6)–(7)). **Open:** initial target weights. **Decide in** implementation; copy-at-init is the expected convention unless released code differs.

### 5. Observation normalization alignment

Reward Eq. (8) assumes $s(i)$ and $\beta_t = 0.35$ live on the same scale, but the paper does not define the raw-$P_\beta$ → $s$ mapping. **Fixed:** critic predicts return under environment reward. **Open:** normalization shared with [environment.md](../../environment.md) §6. **Decide in** the env observation pipeline; controller consumes whatever $s$ the env exposes.

### 6. QAT training budget

§IV.A.3 reports weaker beta suppression under QAT with **10 episodes** and notes longer training may help. **Fixed:** fake-quant stubs on actor input and logits (§III.D); PTQ at FP16/INT8 is post-training only. **Open:** QAT episode count and layer selection beyond the paper’s stubs. **Decide in** quantization experiment config and document separately from full-precision runs.

### 7. Post-training eval segment duration

The **10 s** eval protocol in §IV.A.2 is ambiguous when each segment uses training $l = 2$ s (see [environment.md](../../environment.md) §8). **Fixed:** `mehregan_eval` cross-controller protocol. **Open:** per-segment simulated time within the 10 s budget. **Decide in** benchmark harness; align with project convention once env eval timing is fixed.

---

## 10. References

- Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*.

For the **plant**, see [plant.md](../../plant.md). For the **Mehregan environment and evaluation protocol**, see [environment.md](../../environment.md). For eval suites, use `mehregan_eval` per [benchmarking.md](../../benchmarking.md) §3.2.
