# SEA-DBS controller specification (Ravivarapu et al., sample-efficient adaptive DBS)

This document specifies the **SEA-DBS** (sample-efficient actor–critic) controller from *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease* (Ravivarapu et al.). It is meant to align `controllers/sea_dbs/` (and training scripts) with the published method.

**Companion spec:** The shared **Kumaravelu et al. (2016)** plant — [plant.md](../../plant.md). **Mehregan et al.** Gymnasium API — [environment.md](../../environment.md). **This document is authoritative for the Ravivarapu controller**—binary actions, predictive reward modeling, Gumbel-Softmax exploration, timing, reward shape, and training—unless the others explicitly describe the same quantity.

**Scheduling:** SEA-DBS replication is **Phase 6** and is **active in parallel** with Nguyen ([roadmap.md](../../development/roadmap.md)). Panel tracker and **qualitative gates:** [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md) (all panels Fail with documented gaps after the oct 2026 honest-replication pass, §15). Gate design notes: [docs/figures/papers/ravivarapu.md](../../figures/papers/ravivarapu.md). Mehregan panel set is Pass — [figures/mehregan/replications.md](../../../figures/mehregan/replications.md).

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

**Baselines in the paper:** A **DDPG baseline** without predictive modeling or GS (same binary action space) is used for comparison (§V, Table II, Figs. 4–7). Implement `variant=baseline` vs `variant=paper` (full SEA-DBS: PM+GS) for benchmarking per [benchmarking.md](../../benchmarking.md). **Chosen (Baseline exploration, paper-silent):** ε-greedy on the actor's argmax, ε linear from 1.0 to 0.05 over the whole run; its actor loss feeds $\mathrm{softmax}(\text{logits})$ to the critic (the deterministic-policy-gradient analogue for two actions). All four variants share every other setting; `variant` only switches PM and GS. **Screened and rejected:** the DDPG backbone of Mehregan et al. (same group, same Table I values) — critic scores the actor's logit vector, exploration is Gaussian noise ($\sigma = 1$) on the logits, the actor maximizes $Q(s, \mu(s))$ (`critic_action_input="logits"`; GS variants then score their relaxed sample). The Baseline then stimulates on 78 % of steps from the first 15 episodes and never declines (Fig 4a Pearson 0.07, 15 of 27 gates as before), and at inference it still stimulates every step. Default stays `one_hot`.

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

**Repo convention (all Ravivarapu panels):** Ravivarapu reports **2 ms** RL cadence (Table I / §V.A), but multitaper GPi $P_\beta$ on the Kumaravelu plant needs ≳100 ms of spike data per integrate. `SEA_DBSEnvAdapter` therefore integrates **`biomarker_window_s = 0.15`** per RL step while logging **`step_duration_ms = 2`** for paper metadata. Episode length remains **30 steps**; simulated time per episode is **4.5 s** (not 60 ms). Each step is an **independent plant shot** from the episode IC (`plant_integration_mode="disconnected"`, the Mehregan-style env); a stimulating step runs the carrier for the whole window. The plant integrates at the paper's **0.02 ms** (`plant_dt_ms = 0.02`, passed to the plant's `PlantConfig.dt_ms`; oct 17 2026). Until then it ran at the Kumaravelu reference step of 0.01 ms; the two give the same levels over plant seeds 0–9 (untreated raw $P_\beta$ 219.5 vs 217.9, 50 Hz stim/untreated $0.675 \pm 0.092$ vs $0.687 \pm 0.062$, 30 Hz $1.133$ vs $1.134$; `scripts/probes/ravivarapu_open_questions.py --experiments dt`), so `observation_scale = 472` stays. Checkpoints trained before the switch store 0.01 and evaluate at it.

Why 150 ms: it is the shortest window in which the paper's 50 Hz stimulation suppresses GPi beta on Kumaravelu. Stim / untreated $P_\beta$ over plant seeds 0–9 (`scripts/probes/ravivarapu_plant_levels.py`):

| Shot | 50 Hz | 30 Hz | 130 Hz |
|------|-------|-------|--------|
| 100 ms | $0.95 \pm 0.16$ | $1.05 \pm 0.08$ | $0.39 \pm 0.14$ |
| 150 ms | $0.69 \pm 0.06$ | $1.13 \pm 0.20$ | — |
| 200 ms | $0.65 \pm 0.14$ | $1.21 \pm 0.26$ | — |

The paper's Fig 5a SEA-DBS end/start ratio is $0.67$. 30 Hz raises beta at every window, so Fig 5b's declining curves are out of reach of any policy on this plant ([docs/figures/ravivarapu/5b.md](../../figures/ravivarapu/5b.md)).

**Continuous plant (diagnostic only):** `plant_integration_mode="continuous"` carries the network state across the episode's segments (PythonPlant `carry=True`). Open-loop (`scripts/probes/ravivarapu_openloop_traces.py`, seeds 0–4): untreated beta drifts up from 0.43 to ~0.53 over 10 segments, and always-on 50 Hz falls to 0.32 by step 5 and rebounds to ~0.36. Neither matches the paper's steady 50 Hz decline better than independent shots, so training and evals keep `disconnected`.

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

$\tau_0$, $\tau_{\min}$, and $\lambda_\tau$ are **not** in Table I — **intentionally open**. **Chosen:** $\tau_0 = 200$, $\tau_{\min} = 1$, $\lambda_\tau = \ln(200)/4500 \approx 1.18 \times 10^{-3}$ per env step ($t$ counts env steps), so $\tau$ anneals from a near-uniform relaxed sample to $\tau = 1$ over exactly the 4,500 training steps — "initially a high $\tau$ promotes exploration; over time, decreasing $\tau$ sharpens the policy" (§IV.C). "High" is relative to the actor's logits: at Table I learning rates the stim logit margin reaches ~15–20 within ~15 episodes, so the cited Gumbel-Softmax defaults (Jang et al. 2016: $\tau = \max(0.5, e^{-rt})$, i.e. $\tau_0 = 1$) and $\tau_0 \in \{10, 50\}$ leave SEA-DBS stimulating 70–90 % of steps from the first episodes. Candidate table: [docs/figures/ravivarapu/4a.md](../../figures/ravivarapu/4a.md) § Exploration conventions.

**Executed action (chosen):** the binary action is **sampled from the relaxed probabilities** $\tilde a$ of Eq. (11), $a_t \sim \mathrm{Categorical}(\tilde a)$ (`gs_action_sampling="relaxed"`). Fig. 3 says actions "are sampled from the resulting distribution", and §IV.C says a high $\tau$ promotes exploration. Gumbel-max ($\arg\max \tilde a$) gives the same action for every $\tau$, so with it Eq. (14) would not affect exploration at all. `gs_action_sampling="hard"` keeps Gumbel-max for comparison; screened with $n_{\mathrm{obs}} = 10$ it is worse (SEA-DBS stimulates 90 % of steps over episodes 0–15, 4a 14 of 27). A slower anneal ($\lambda_\tau = 3 \times 10^{-4}$, $\tau$ still ~52 at the end) does not slow SEA-DBS either: stim duty is 1.0 by episode 40–50, because Adam scales the actor's steps independently of the $1/\tau$ gradient factor and the stim logit margin outgrows any $\tau$. The actor loss uses the relaxed sample (straight-through is not needed). **Ablation:** `variant=baseline-gs` enables GS without predictive modeling.[^gs]

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
| **Seed change interval** | $n \in \{10,20,50,75\}$ steps — Table II (avg PSD ↓, reward ↑); protocol convention §14.13, panel `scripts/figures/papers/ravivarapu/table2/plot.py` |
| **Stimulation carrier frequency** | **50 Hz** vs **30 Hz** inference comparison (Fig. 5)—adapter/plant must expose frequency as a fixed eval setting, not necessarily a per-step RL action |
| **Ablation (10 steps)** | Baseline, +PM, +GS, full SEA-DBS (Fig. 7); PM early noise with **~4,500** samples cited |
| **FP16 PTQ** | Post-training; compare PSD over **10** stimulation steps (Fig. 6) |

Use the `sea_dbs_eval` suite per [benchmarking.md](../../benchmarking.md) §3.2 for replication; cross-paper metrics: [benchmarking.md](../../benchmarking.md) §3.3.

---

## 11. Quantization (§V)

| Mode | When | Notes |
|------|------|------|
| **FP16 PTQ** | After full-precision training | Reported **~2×** memory reduction; performance close to FP32 on 10-step eval. Implemented as a plain `.half()` cast of the trained actor (`apply_fp16_ptq`); inference casts the state to fp16. No weight perturbation. Fig 6 reports parameter bytes (fp32 vs fp16), not checkpoint file size. |

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

Fig. 5 compares **50 Hz** vs **30 Hz** carrier during inference; this is not a per-step RL action. **Fixed:** binary pulse/no-pulse control during training. **Chosen (training carrier):** 50 Hz — the only frequency the paper names for its main results (Figs 5a, 6, 7); the training carrier itself is paper-silent. **Chosen (eval):** Figs 5–7 roll out each actor on exactly the plant / observation setup it was trained on (`evaluate()` reads window, burst, `n_obs`, prefill, and normalization from the checkpoint). The only override is the paper-stated carrier, applied through `SEA_DBSEnvAdapter.set_carrier_hz`, which survives `reset()`. Plant reset uses the panel seed (0). **Chosen (eval actions):** deterministic **argmax** on the actor logits (§2: deployment is deterministic). `action_mode="gumbel"` (hard Gumbel-max, torch seed = panel seed, never tuned) remains for diagnostics. Earlier eval-only windows, `n_obs`, untreated windows, pulse delays, per-step schedules, and hand-picked Gumbel seed offsets are removed (§15).

### 11. Beta normalization scale

$\beta_t = 0.35$ implies consistent scaling of $\bar{P}_\beta$ between observations and reward, as with Mehregan. **Fixed:** Eq. (7) reward shape. **Chosen (SEA-DBS adapter):** `observation_scale = 472` (not Mehregan's 1000): untreated raw $P_\beta$ on a 150 ms shot averages 217.9 over plant seeds 0–9, and 472 maps that onto the paper's untreated level (~0.46, the start of Figs 4a and 5–7). At scale 1000 untreated beta would already sit below $\beta_t$ and Eq. (7) would reward doing nothing.

### 12. Observation window on reset

Eq. (4) makes the state a window of the last $n_{\mathrm{obs}}$ beta estimates. **Chosen:** $n_{\mathrm{obs}} = 10$ and, on reset, the window holds $n_{\mathrm{obs}}$ untreated samples (the history before the episode; `prefill_obs_window=True`), in training and eval alike. Ten is the paper's inference horizon (Figs 5–7 plot 10 stimulation steps). Each step is an independent plant shot, so the window mean reaches its floor exactly at step $n_{\mathrm{obs}}$: with $n_{\mathrm{obs}} = 5$ every inference trace was flat after step 5, while the digitized Figs 5–7 keep falling through step 10. Screen (always-on 50 Hz, seed 0, Fig 5a gates): $n_{\mathrm{obs}} = 5 / 6 / 8 / 10 / 12$ pass $8 / 11 / 12 / 12 / 12$ of 17; 10 is the only value that passes both early-MAE gates of Figs 6 and 7 too. Fig 4 training with 5 and 10 gives the same gate count ([docs/figures/ravivarapu/4a.md](../../figures/ravivarapu/4a.md) § Convention screen).

### 13. Table II protocol

§V.B says only that the environment seed changes every $n \in \{10, 20, 50, 75\}$ steps and that Table II reports average PSD and average reward. **Fixed:** the four intervals; Avg Reward is Eq. (7) of Avg PSD / 1000 (true of all eight published cells), so only PSD is gated. **Chosen (oct 17 2026):** `evaluate_reseeded` rolls each Fig 4 actor out at **50 Hz** on its training plant setup for **30 blocks of $n$ argmax steps**, resetting the plant with the next seed at each block (`seed`, …, `seed + 29`), so every interval visits the same 30 plant seeds. A reset re-prefills the Eq. (5) window with that seed's untreated level, so short intervals spend more steps with untreated samples in the window. Avg PSD is the mean Eq. (5) observation × 1000. Until oct 17 the rollout was a fixed 300 steps, which visited seeds 0–3 for $n = 75$ but 0–29 for $n = 10$; the full-stimulation floor ranges 221–479 across those seeds (mean 289 for seeds 0–3, 341 for 0–29), so the old $n$ trend was mostly a seed-pool effect. **Paper arithmetic:** the paper's SEA-DBS row fits $\text{avg}(n) = F + A/n$ (fit $F = 298$, $A = 273$ PSD·steps), so each reseed costs a fixed transient. Its $n = 10$ cells (326 / 364) sit well below the 10-step means of its own Fig 5a curves (~359 SEA-DBS / ~404 Baseline), so a Table II reseed does not restart from Fig 5a's untreated start. **Open:** the paper's horizon, checkpoint, action mode, and what the "environment seed" controls.

---

## 15. Honest-replication audit (oct 2026)

Before this pass every Ravivarapu panel was marked Pass, but the passes rested on mechanisms the paper does not describe, tuned version by version against the digitized curves. All were removed; no shipped config can reach them (`config_from_dict` drops the old fields from legacy checkpoints).

| Shortcut | Where | What it did |
|----------|-------|-------------|
| Episode-scheduled actor logit boosts | Fig 4 trainer | Added hand-tuned stim / no-stim logit offsets by episode range (mid, mid-late, ramped late brake up to +2.4, gap patch) per variant |
| No-stim actor init bias | Fig 4 trainer | Shifted the final-layer bias toward no-stim (1.25 SEA / 1.38 Baseline) so early PSD started at the paper's untreated level; the paper initializes randomly |
| Per-variant plant | Fig 4 config | Baseline 58 ms bursts, SEA-DBS 62 ms bursts at 130 Hz |
| Per-variant learning rates | Fig 4 config | $\alpha_a$ $5.2$–$7.5 \times 10^{-6}$, $\alpha_c$ $1.2$–$1.5 \times 10^{-4}$ instead of Table I's $5 \times 10^{-4}$ / $10^{-3}$ |
| Forced Gumbel-Softmax on Baseline | Fig 4 config | Turned the DDPG Baseline into a GS agent so its curve faded gradually |
| Episode-dependent $\tau$ schedule | Fig 4 trainer | Early $\lambda$ ×4.5 before ep 38 and a late $\tau$ floor after ep 108 (Baseline) |
| Predictive-model warmup | Fig 4 trainer | Ramped $\hat r$ into Eq. (9) over 15,000 steps; Eq. (9) adds it directly |
| Fixed seed for episodes 0–1 | Fig 4 config | Removed one episode's IC dip |
| Eval-only plant windows and `n_obs` | Figs 5a / 6 / 7 | 150–160 ms stim windows, 100 ms untreated window (two different estimators in one trace), `n_obs` 6 / 8 chosen per panel |
| Hand-scheduled pulse delays and bursts | Fig 5b | Per-step pulse delays `(2, 2, 0, 5, …)` and per-variant bursts written to reproduce the paper's 30 Hz shape open-loop |
| Cherry-picked Gumbel seeds | Figs 5a / 5b / 6 / 7 | Eval RNG offsets (5, 6, 34, 141, 225, 586, …) picked per panel and per variant for the action sequences that matched the paper |
| PTQ weight noise | Fig 6 | Gaussian noise (σ 0.08–0.28, retried over seeds) added before `.half()` so PTQ traces split from fp32; checkpoint file size halved as "model size" |
| Different training config for +PM / +GS | Fig 7 | +PM / +GS trained on library defaults (other plant, other learning rates) while Baseline / SEA-DBS came from the tuned Fig 4 config |
| Loosened gates | `ravivarapu_gates.py` | Fig 4a gap bands and level tolerances widened to admit tuned runs; Fig 5a steps 3–5 MAE 0.015 → 0.020; Fig 7 GS gate with a −5.0 margin on ~0.4-scale data (always true) |

Kept as labeled conventions (paper-silent): 150 ms shot, 50 Hz training carrier, scale 472, $n_{\mathrm{obs}} = 10$ with prefill (5 until the oct 12 convention screen), independent shots, Gumbel schedule defaults, Baseline ε-greedy, Polyak 0.005, one update per env step, 64-unit MLPs, seed 0, and the display-only 10-episode rolling mean on Figs 4a/4b (gates use raw episodes). Panel results: [figures/ravivarapu/replications.md](../../../figures/ravivarapu/replications.md).

## 16. References

- Ravivarapu et al., *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*.

For the **shared plant**, see [plant.md](../../plant.md). For the **Mehregan Gym API**, see [environment.md](../../environment.md). For the **DDPG pattern controller** and **quantization (PTQ/QAT)** study, see [controllers/ddpg/replication.md](../ddpg/replication.md).
