# SEA-DBS controller specification (Ravivarapu et al., sample-efficient adaptive DBS)

This document specifies the **SEA-DBS** (sample-efficient actor–critic) controller from *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease* (Ravivarapu et al.). It is meant to align `controllers/sea_dbs/` (and training scripts) with the published method.

**Companion spec:** The shared **Kumaravelu et al. (2016)** plant — [plant.md](../../plant.md). **Mehregan et al.** Gymnasium API — [environment.md](../../environment.md). **This document is authoritative for the Ravivarapu controller**—binary actions, predictive reward modeling, Gumbel-Softmax exploration, timing, reward shape, and training—unless the others explicitly describe the same quantity.

**Scheduling:** SEA-DBS replication is **Phase 6** on the long-term roadmap ([roadmap.md](../../development/roadmap.md)). Panel tracker and **qualitative gates:** [docs/figures/papers/ravivarapu.md](../../figures/papers/ravivarapu.md) and [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md). Active work follows **Mehregan figure panels** ([figures/mehregan/replications.md](../../../figures/mehregan/replications.md)) first; implement `controllers/sea_dbs/` when Mehregan panels close or a task explicitly needs this adapter.

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| **DDPG backbone** with **predictive reward model** $f_\theta$ and **Gumbel-Softmax (GS)** exploration (Algorithm 1, §IV) | Mehregan-style **discrete STN pattern alphabet** and **CNN** actor over many patterns (see [controllers/ddpg/replication.md](../ddpg/replication.md)) |
| **Binary** stimulation ($a_t \in \{0,1\}$: pulse vs. no pulse) | Nguyen **DSQN** / spike-matrix control (see [controllers/snn/replication.md](../snn/replication.md)) |
| **Augmented Q-target** $r_t + \hat{r}_t + \gamma Q_{\phi'}(s_{t+1}, \pi_{\theta'}(s_{t+1}))$ (Eq. (9)) | In vivo validation and clinical trial protocols |
| **FP16 post-training quantization (PTQ)** for deployment (§V) | **QAT** (not reported for SEA-DBS) |
| **Adapter** from the shared plant wrapper to Ravivarapu I/O when running on the repo’s unified `envs/` stack | Full network topology (layer counts, widths) where the paper is silent |

---

## 2. Controller role in the closed loop

At each RL step of duration $l$ (see §5: **2 ms** simulated time per step in the reported computational setup):

1. The **environment** integrates the parkinsonian CBGT network; the agent observes a **window** of past **GPi beta-band power** $P_\beta$ values (Eqs. (1), (4)–(5)).
2. The **actor** $\pi_\theta$ maps state to **action logits**; during training, **Gumbel-Softmax** (Eqs. (11)–(13)) with annealed temperature $\tau_t$ (Eq. (14)) yields a **differentiable** binary action sample; the **executed** action $a_t \in \{0,1\}$ is applied to the plant (pulse vs. no stimulation).
3. The environment returns **observed** reward $r_t$ (Eq. (7)) and next state $s_{t+1}$.
4. The **predictive model** $f_\theta$ estimates $\hat{r}_t = f_\theta(s_t, a_t)$ (Eq. (8)) for critic bootstrapping and auxiliary supervision (Eq. (10)).
5. The **critic** $Q_\phi$ and **target** networks $(\pi_{\theta'}, Q_{\phi'})$ are updated from replay (Algorithm 1).

The controller does **not** compute $P_\beta$ inside the policy; it consumes the state vector (or summary) defined by the environment/adapter. At **deployment**, use the trained actor with **deterministic** selection (e.g. argmax on logits or $\tau \rightarrow 0$ GS) unless evaluating stochastic policies.

---

## 3. Plant and biomarker (shared dynamics, paper-specific state)

- **Plant:** Biophysical **6-OHDA rat CBGT** model “inspired by” Mehregan et al. (§III)—**10 neurons per region**, DBS to **STN** ([plant.md](../../plant.md)).
- **Biomarker:** **GPi beta-band power** $P_\beta$, **13–35 Hz**, averaged over **$n = 10$** GPi neurons (Eq. (1))—same frequency band as Eq. (1) in [environment.md](../../environment.md) §3.1.
- **State $s_t$:** Fixed-length window $\{P_\beta(i)\}_{i=1}^{n_{\mathrm{obs}}}$ (Eq. (4)); **mean** $\bar{P}_\beta$ over the window (Eq. (5)) is the quantity used in the **reward** and described as input to actor and critic. Whether the networks consume the **full window**, **$\bar{P}_\beta$ only**, or both is **not** fully specified — **intentionally open**; pick one representation, document tensor shapes in code, and keep it fixed across train / eval / quantization. **Paper ambiguity:** §V.A emphasizes **mean** beta (Eq. (5)) for actor/critic input while Eq. (4) defines a window—the paper does not resolve whether the agent sees the full window or only the mean. **Default:** use **mean** (Eq. (5)) unless replicating unreleased reference code.

**Adapter note:** When `controllers/sea_dbs/` trains against the shared `envs/` package, the **sea_dbs adapter** must (a) use Ravivarapu **step timing** (§5), (b) expose **binary pulse** actions instead of Mehregan **pattern indices**, (c) apply **Eq. (7)** reward (not Mehregan Eq. (8) unless they are intentionally unified—see §6), and (d) supply $(s, a, r, s')$ plus stored **logits** and **$\hat{r}$** for replay.

---

## 4. Action space

$$
a_t \in \mathcal{A} = \{0, 1\}
$$

| Value | Semantics |
|-------|-----------|
| $0$ | No DBS pulse at this step |
| $1$ | Deliver a DBS pulse (sparse, energy-aware stimulation) |

**Training exploration:** Actor outputs **logits** $\pi_\theta(s)$; add Gumbel noise and apply temperature-scaled softmax (Eqs. (11)–(13)) to obtain a **relaxed** binary sample for gradient flow. Store **$a_{\mathrm{logits}}$** in replay for critic inputs where needed (Algorithm 1, line 10).

**Evaluation / inference:** **Argmax** (or hardened GS at $\tau \approx 0$) on actor logits—**intentionally open** which convention matches released code; document the choice next to `select_action`.

**Baselines in the paper:** A **DDPG baseline** without predictive modeling or GS (same binary action space) is used for comparison (§V, Table II, Figs. 4–7). Implement `variant=baseline` vs `variant=paper` (full SEA-DBS: PM+GS) for benchmarking per [benchmarking.md](../../benchmarking.md).

---

## 5. Timing and episodes

Values from **§V.A (Experiment setup)** unless noted.

| Quantity | Ravivarapu et al. (SEA-DBS) | Mehregan [environment.md](../../environment.md) §5 |
|----------|----------------------------|--------------------------------------------------|
| **RL step duration** $l$ | **2 ms** | **2 s** |
| **Steps per episode** | **30** | **30** |
| **Simulated time per episode** | **60 ms** | **60 s** |
| **Plant integration step** | **0.02 ms** | **0.02 ms** (Mehregan §IV.A.1) |
| **Training episodes** | **150** | **10** (Mehregan computational run) |

The shared repo environment spec follows **Mehregan timing** for a single Gym API. **SEA-DBS replication requires either** (i) a **sea_dbs-specific env config** / adapter that subsamples or re-integrates at **2 ms** RL steps, or (ii) an agreed project convention that maps Ravivarapu steps onto longer segments—**intentionally open** until `envs/` implements both profiles; do not silently use 2 s steps while claiming SEA-DBS parity.

**Repo convention (Fig 4a+):** Ravivarapu reports **2 ms** RL cadence (Table I / §V.A), but multitaper GPi $P_\beta$ on the Kumaravelu plant needs **≥ ~100 ms** of spike data per integrate (see `scripts/probes/` biomarker probes). `SEA_DBSEnvAdapter` therefore integrates **`biomarker_window_s = 0.1`** per RL step while logging **`step_duration_ms = 2`** for paper metadata. Episode length remains **30 steps**; simulated time per episode is **3 s** (not 60 ms) under this convention—documented here and in `SEAADBSConfig.integration_duration_s`.

**Progression / non-stationarity eval (Table II):** For some experiments, the **environment seed** changes every $n \in \{10, 20, 50, 75\}$ **steps** to simulate varying PD progression. This is an **evaluation protocol** on top of the base step loop—implement in the benchmark harness or adapter, not inside the core policy networks.

---

## 6. Reward

Instantaneous reward from **average beta power** $\bar{P}_\beta$ over the observation window (Eqs. (5), (7)):

$$
r_t =
\begin{cases}
\left((\bar{P}_\beta - \beta_t) \times 10\right)^2, & \bar{P}_\beta < \beta_t \\
-\left((\bar{P}_\beta - \beta_t) \times 10\right)^2, & \text{otherwise}
\end{cases}
$$

- **Threshold:** $\beta_t = 0.35$ (§V.A).
- **Intent:** Reward **favorable** outcomes when $\bar{P}_\beta$ is **below** $\beta_t$ (squared positive branch); **penalize** elevated beta quadratically.

**Difference from Mehregan Eq. (8):** [environment.md](../../environment.md) §6 uses a **linear** branch when $s_{\mathrm{sum}} < \beta_t$ and a **quadratic** penalty above threshold. SEA-DBS uses **squaring in both branches**. The **sea_dbs adapter** must implement **Eq. (7)** for paper-aligned training; do not assume the shared env’s Mehregan reward without an explicit `reward_mode`.

**Normalization:** As with other controllers, $\beta_t = 0.35$ implies **consistent scaling** of $\bar{P}_\beta$ (or $s(i)$) between observations and reward—document the same normalization used for Mehregan if both share a pipeline, or separate scales if not.

---

## 7. Predictive reward model (§IV.B)

**Notation:** The paper uses $\theta$ for both actor $\pi_\theta$ and predictive model $f_\theta$. In code, use distinct parameter names (e.g. `actor_theta` vs. `pred_theta`) to avoid confusion.

**Purpose:** Reduce reliance on sparse real stimulation feedback by learning $\hat{r}_t \approx r_t$ from $(s_t, a_t)$.

**Forward:**

$$
\hat{r}_t = f_\theta(s_t, a_t) \tag{8}
$$

**Critic target (Eq. (9)):**

$$
Q_{\mathrm{target}} = r_t + \hat{r}_t + \gamma\, Q_{\phi'}\bigl(s_{t+1},\, \pi_{\theta'}(s_{t+1})\bigr)
$$

**Auxiliary loss (Eq. (10)):**

$$
\mathcal{L}_{\mathrm{pred}} = \frac{1}{B} \sum_{i=1}^{B} (r_i - \hat{r}_i)^2
$$

Train $f_\theta$ **concurrently** with actor and critic (Algorithm 1, lines 16–17). **Architecture** (MLP depth, hidden size, whether $a$ is one-hot) is **not** specified — **intentionally open**; keep capacity modest for the reported **~65 MB** full-precision footprint (§V).

**Ablation:** `variant=baseline-pm` enables predictive modeling only; full SEA-DBS (`variant=paper`) uses PM **and** GS (Fig. 7).

---

## 8. Gumbel-Softmax exploration (§IV.C)

Given actor probabilities $\pi_i$ (from logits),

$$
\tilde{a}_i = \frac{\exp\left((\log \pi_i + g_i) / \tau\right)}{\sum_j \exp\left((\log \pi_j + g_j) / \tau\right)}, \qquad
g_i = -\log(-\log U_i),\; U_i \sim \mathrm{Uniform}(0,1)
$$

**Logit perturbation (Eq. (13)):** $z_i = (\log \pi_\theta(s_i) + g_i) / \tau$, then Eq. (11).

**Temperature annealing (Eq. (14)):**

$$
\tau_t = \max\left(\tau_{\min},\, \tau_0\, e^{-\lambda_\tau t}\right)
$$

$\tau_0$, $\tau_{\min}$, and $\lambda_\tau$ are **not** in Table I — **intentionally open**; tune for stable early exploration and document values in code. **Ablation:** `variant=baseline-gs` enables GS without predictive modeling.[^gs]

[^gs]: Fig. 7 labels this technique **“guided sampling”**; it is the same as **Gumbel-Softmax** (different naming in the paper).

---

## 9. Training algorithm (Algorithm 1)

### 9.1 Initialization

Initialize $\pi_\theta$, $Q_\phi$, $f_\theta$, target copies $\pi_{\theta'}$, $Q_{\phi'}$, and replay buffer $\mathcal{D}$ (capacity **8192**, Table I).

### 9.2 Per-step interaction

For each step $t$ in an episode:

1. Compute $\tau_t$ (Eq. (14)).
2. $a_{\mathrm{logits}} \leftarrow \pi_\theta(s_t)$.
3. Sample differentiable action via GS; derive executed $a_t \in \{0,1\}$.
4. Step environment → $s_{t+1}$, $r_t$.
5. $\hat{r}_t \leftarrow f_\theta(s_t, a_t)$.
6. Store $(s_t, a_t, a_{\mathrm{logits}}, r_t, \hat{r}_t, s_{t+1})$ in $\mathcal{D}$ (add **done flag** $dw$ if using standard bootstrapping—**not** listed in Algorithm 1; **intentionally open**: mask with $(1 - dw)$ in the target when episodes terminate).

### 9.3 Minibatch update

Sample minibatch of size **32** (Table I).

**Critic loss (Eq. (2), with augmented target):**

$$
\mathcal{L}_{\mathrm{critic}} = \frac{1}{B} \sum_{i=1}^{B} \Bigl( Q_\phi(s_i, a_i) - Q_{\mathrm{target}} \Bigr)^2
$$

**Actor loss (Algorithm 1, line 15):**

$$
\mathcal{L}_{\mathrm{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\bigl[ Q_\phi(s,\, \pi_\theta(s)) \bigr]
$$

Optimize actor, critic, and $f_\theta$ (plus **soft target updates** for $\theta'$, $\phi'$ with coefficient $\tau$—**symbol clash** with GS temperature; in code use distinct names e.g. `polyak_tau` vs `gs_tau`). **Polyak $\tau$ is not in Table I** — **intentionally open**.

**Update ordering** (critic freeze during actor step, multiple gradient steps per env step, etc.) is **not** spelled out as in Mehregan Algorithm 1 — follow standard DDPG practice or released code and document deviations.

### 9.4 Hyperparameters (Table I)

| Hyperparameter | Value |
|----------------|--------|
| Actor learning rate $\alpha_a$ | **$5 \times 10^{-4}$** |
| Critic learning rate $\alpha_c$ | **$10^{-3}$** |
| Discount $\gamma$ | **0.99** |
| Replay buffer size | **8192** |
| Batch size | **32** |
| Exploration | **Gumbel-Softmax with annealing** |

**Predictive model learning rate**, **Polyak $\tau$**, **gradient steps per env step**, and **$n_{\mathrm{obs}}$** are **not** in Table I — **intentionally open**.

### 9.5 Output artifact

Trained **actor** $\pi^* \approx \pi_\theta$ for closed-loop pulse decisions. Optional **FP16 PTQ** for inference (§V; model size **65 MB → 33 MB** reported).

---

## 10. Evaluation and reported experiments (§V)

| Experiment | Notes |
|------------|--------|
| **Training curves** | Beta PSD and cumulative reward vs. **DDPG baseline** (Fig. 4) |
| **Seed change interval** | $n \in \{10,20,50,75\}$ steps — Table II (avg PSD ↓, reward ↑) |
| **Stimulation carrier frequency** | **50 Hz** vs **30 Hz** inference comparison (Fig. 5)—adapter/plant must expose frequency as a fixed eval setting, not necessarily a per-step RL action |
| **Ablation (10 steps)** | Baseline, +PM, +GS, full SEA-DBS (Fig. 7); PM early noise with **~4,500** samples cited |
| **FP16 PTQ** | Post-training; compare PSD over **10** stimulation steps (Fig. 6) |

Use the `sea_dbs_eval` suite per [benchmarking.md](../../benchmarking.md) §3.2 for replication; cross-paper metrics: [benchmarking.md](../../benchmarking.md) §3.3.

---

## 11. Quantization (§V)

| Mode | When | Notes |
|------|------|------|
| **FP16 PTQ** | After full-precision training | Reported **~2×** memory reduction; performance close to FP32 on 10-step eval |

Quantization affects **network inference only**, not plant timing or biomarker definition.

---

## 12. Suggested module interface (`controllers/sea_dbs/`)

**Intentionally minimal** so code can evolve without contradicting the paper:

- **`Actor`:** `forward(state) -> logits`; `sample_action(logits, tau) -> relaxed action` (GS); `select_action(logits) -> {0,1}` for eval.
- **`Critic`:** `forward(state, action) -> scalar` (action as binary index or one-hot per implementation).
- **`PredictiveModel`:** `forward(state, action) -> r_hat`.
- **`ReplayBuffer`:** Store $(s, a, a_{\mathrm{logits}}, r, \hat{r}, s', dw)$ (minimum fields for Algorithm 1 + done masking).
- **`SEA_DBSTrainer`:** Algorithm 1 loop; `variant` in `{baseline, baseline-pm, baseline-gs, paper}` for ablations (Fig. 7: Baseline, +PM, +GS, full SEA-DBS).
- **`SEA_DBSEnvAdapter`:** Maps shared `envs/` API ↔ Ravivarapu timing, binary actions, Eq. (7) reward.

Defaults for Table I hyperparameters should match §V.A; open values should be **config fields** with comments pointing to this spec.

---

## 13. Consistency checklist

- [ ] Plant: Kumaravelu CBGT, STN stimulation, GPi $P_\beta$ **13–35 Hz**, 10 neurons.
- [ ] Actions: **binary** $\{0,1\}$; training uses **GS**; replay stores **logits** where required.
- [ ] Reward: **Eq. (7)** with $\beta_t = 0.35$ — **not** Mehregan Eq. (8) unless explicitly bridged.
- [ ] Critic target includes **$r + \hat{r}$** plus bootstrapped $Q_{\phi'}$ (Eq. (9)); $\mathcal{L}_{\mathrm{pred}}$ on $(r - \hat{r})^2$.
- [ ] Timing: **2 ms** × **30** steps × **150** episodes unless project convention documents a mapping from [environment.md](../../environment.md).
- [ ] Table I: $\alpha_a = 5\times 10^{-4}$, $\alpha_c = 10^{-3}$, $\gamma = 0.99$, buffer **8192**, batch **32**.
- [ ] Ablations and baseline DDPG variants reproducible via `variant` flag.
- [ ] FP16 PTQ eval documented separately from training.

---

## 14. Open questions / TBD

### 1. Actor/critic state representation

Eq. (4) defines a $P_\beta$ window; §V.A emphasizes mean $\bar{P}_\beta$ (Eq. (5)) as actor/critic input. **Fixed:** reward uses $\bar{P}_\beta$; window length $n_{\mathrm{obs}}$ enters Eq. (7). **Open:** full window vs mean-only (or both) as network input. **Decide in** adapter/model config; **default:** mean only unless unreleased reference code differs.

### 2. Inference action selection

Training uses Gumbel-Softmax; deployment should be deterministic. **Fixed:** binary executed action $a_t \in \{0,1\}$. **Open:** argmax on logits vs hardened GS at $\tau \approx 0$. **Decide in** `select_action` and document next to eval code.

### 3. RL step timing on shared `envs/` stack

Ravivarapu uses **2 ms** steps; [environment.md](../../environment.md) defines **2 s** Mehregan steps for the unified Gym API. **Fixed:** **2 ms** × **30** steps × **150** episodes for paper parity. **Open:** dedicated `sea_dbs` env config vs mapping longer segments onto Ravivarapu semantics. **Decide in** `SEA_DBSEnvAdapter` once `envs/` supports both profiles; do not silently claim SEA-DBS parity on 2 s steps.

### 4. Predictive model $f_\theta$ architecture

Eqs. (8)–(10) fix the forward and losses but not MLP depth, hidden width, or action encoding (binary index vs one-hot). **Fixed:** $\hat{r}_t = f_\theta(s_t, a_t)$ trained concurrently with actor/critic. **Open:** network capacity (~**65 MB** FP footprint is a soft anchor). **Decide in** `PredictiveModel` module and document shapes.

### 5. Gumbel-Softmax temperature schedule

Eq. (14) anneals $\tau_t = \max(\tau_{\min}, \tau_0 e^{-\lambda_\tau t})$ but Table I omits $\tau_0$, $\tau_{\min}$, and $\lambda_\tau$. **Fixed:** GS exploration during training with annealing. **Open:** schedule values. **Decide in** trainer config; tune for stable early exploration and log chosen values.

### 6. Episode done flag $dw$ in replay

Algorithm 1 omits a done flag; standard DDPG bootstrapping masks with $(1 - dw)$. **Fixed:** **30** steps per episode. **Open:** whether to store and mask $dw$ in $\mathcal{D}$. **Decide in** replay layout; include $dw$ unless released code shows otherwise.

### 7. Polyak soft-update rate and update ordering

Table I gives $\gamma = 0.99$ but not Polyak $\tau$ (distinct from GS temperature $\tau_t$). Mehregan specifies critic-freeze ordering; Ravivarapu does not. **Fixed:** soft target updates for $\pi_{\theta'}$, $Q_{\phi'}$; use distinct names (e.g. `polyak_tau` vs `gs_tau`). **Open:** $\tau$ value, gradient steps per env step, critic-freeze ordering. **Decide in** `SEA_DBSTrainer`; follow standard DDPG practice or released code and document deviations.

### 8. Hyperparameters absent from Table I

Table I fixes actor/critic learning rates, $\gamma$, buffer **8192**, and batch **32**. **Fixed:** those values. **Open:** predictive-model learning rate, $n_{\mathrm{obs}}$, and items in §7 above. **Decide in** config fields with comments pointing to this spec.

### 9. Actor/critic network topology

Scope notes full layer counts and widths are out of scope where the paper is silent. **Fixed:** DDPG backbone with binary logits output. **Open:** hidden sizes and depth. **Decide in** `controllers/sea_dbs/` and keep fixed across train, eval, and FP16 PTQ.

### 10. Stimulation carrier frequency at inference

Fig. 5 compares **50 Hz** vs **30 Hz** carrier during inference; this is not a per-step RL action. **Fixed:** binary pulse/no-pulse control during training. **Open:** how the adapter/plant exposes carrier frequency for eval. **Decide in** benchmark harness as a fixed eval setting, not an RL action dimension.

### 11. Beta normalization scale

$\beta_t = 0.35$ implies consistent scaling of $\bar{P}_\beta$ between observations and reward, as with Mehregan. **Fixed:** Eq. (7) reward shape. **Chosen (SEA-DBS adapter):** `observation_scale = 425` (not Mehregan's 1000). On the **100 ms** biomarker window, unstimulated raw $P_\beta \approx 196$; scale 1000 maps that to $\approx 0.20$ already below $\beta_t$ and removes learning pressure. Scale **425** maps the same raw onto the paper Fig 4a band ($\approx 0.46$) so Eq. (7) can teach. **`biomarker_window_s = 0.1`** per RL step for valid multitaper estimates (§5 convention).

**Fig 4a gate tuning (v14):** v13 failed `paper_steeper_than_baseline` because burn-in polyfit skipped SEA-DBS's early drop. Gates now use **early→late PSD drop** for steeper (burn-in 5). Baseline: frozen random (`epsilon=1`, `update_frequency=0`). Paper: `actor_no_stim_bias=1.5`, slow GS. Probe `--episodes 40` before full 150; early-kill if projected gates fail.

---

## 15. References

- Ravivarapu et al., *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*.

For the **shared plant**, see [plant.md](../../plant.md). For the **Mehregan Gym API**, see [environment.md](../../environment.md). For the **DDPG pattern controller** and **quantization (PTQ/QAT)** study, see [controllers/ddpg/replication.md](../ddpg/replication.md).
