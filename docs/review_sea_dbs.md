# SEA-DBS controller spec review (faithfulness to Ravivarapu et al.)

**Reviewed documents**

| Document | Path |
|----------|------|
| Controller spec | `docs/controllers/sea_dbs.md` |
| Paper notes | `/home/devat/knowledge-base/neuroengineering/brain-stimulation-engineering/effort/papers/paper_3.md` (Ravivarapu et al., *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*) |

**Legend:** **CONFIRMED** — spec matches the paper; **DISCREPANCY** — spec conflicts with or misstates the paper; **OPEN** — paper silent, ambiguous, or repo-specific (not falsifiable from the paper alone).

---

## Executive summary

`sea_dbs.md` is **largely faithful** to Ravivarapu et al. Core methodology—DDPG actor–critic backbone, binary actions, Eq. (7) reward, predictive reward model with augmented Q-target (Eqs. (8)–(10)), Gumbel-Softmax exploration (Eqs. (11)–(14)), Algorithm 1, Table I hyperparameters, timing (2 ms × 30 steps × 150 episodes), baselines/ablations, and FP16 PTQ—is **confirmed** against the paper notes.

**Notable gaps (expected):** window length $n_{\mathrm{obs}}$, GS temperature schedule ($\tau_0$, $\tau_{\min}$, $\lambda_\tau$), Polyak coefficient for target networks, predictive-model learning rate, and network architectures are **not** in the paper; the spec correctly marks these **OPEN**.

**Items to treat carefully when implementing:** (1) **state tensor shape**—the paper defines a beta window (Eq. (4)) but states that **mean** $\bar{P}_\beta$ (Eq. (5)) feeds actor and critic; §V.A also emphasizes “average” beta power; (2) **adapter** is repo-only but its requirements align with paper timing and I/O; (3) one **introductory** paper phrase (“future outcomes”) does not match the formal predictive model (current-step $\hat{r}_t$).

No **material DISCREPANCY** was found between the spec and the paper notes for Eqs. (6)–(10), timing, Table I, or baseline comparisons.

---

## 1. Architecture: actor–critic, Gumbel-Softmax exploration

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| DDPG **actor–critic** backbone | §IV, Eq. (2); baseline target Eq. (3) | **CONFIRMED** |
| **Predictive reward model** $f_\theta$ concurrent with actor/critic | §IV.B, Algorithm 1 lines 9–10, 16–17 | **CONFIRMED** |
| **Gumbel-Softmax** for differentiable binary exploration | §IV.C, Eqs. (11)–(14), Algorithm 1 lines 5–7 | **CONFIRMED** |
| Critic target uses **Eq. (9)** ($r + \hat{r} + \gamma Q'$) | Eq. (9), Algorithm 1 line 13 | **CONFIRMED** |
| Algorithm 1 replay stores $(s, a, a_{\mathrm{logits}}, r, \hat{r}, s')$ | Algorithm 1 line 10 | **CONFIRMED** |
| Soft target updates for $\theta'$, $\phi'$ | Algorithm 1 lines 18–20 | **CONFIRMED** (coefficient value not given—see §9) |
| Spec scopes **FP16 PTQ**, not QAT | §V, Fig. 6; no QAT reported | **CONFIRMED** |

**Notes (not spec errors):**

- Paper Eq. (3) labels the baseline Bellman target `$Q_{\mathrm{actor}}$` and omits $\hat{r}$; full SEA-DBS target is Eq. (9). The spec correctly uses Eq. (9) for SEA-DBS and implies baseline omits PM (§4, §10).
- Paper abstract/intro say the predictive model “estimates **future** outcomes”; Eqs. (8)–(9) define $\hat{r}_t = f_\theta(s_t, a_t)$ as an estimate of the **current** reward used in the same-step Q-target. The spec’s wording (“$\hat{r}_t \approx r_t$”) matches the **equations**, not the intro prose.
- Paper reuses symbol $\theta$ for both actor $\pi_\theta$ and predictive model $f_\theta$; spec does not flag this (paper notation issue).

**Verdict:** **CONFIRMED**

---

## 2. Observation space

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| Biomarker: **GPi beta-band power** $P_\beta$, **13–35 Hz** | Eq. (1), §III | **CONFIRMED** |
| $P_\beta$ averaged over **$n$ GPi neurons** (10 neurons per population in model) | Eq. (1), §III (“ten neurons per population”) | **CONFIRMED** |
| State window $s_t = \{P_\beta(i)\}_{i=1}^{n_{\mathrm{obs}}}$ | Eq. (4) | **CONFIRMED** |
| Mean $\bar{P}_\beta$ over window used in **reward** and described as **actor/critic input** | Eqs. (5), (7); §IV.A after Eq. (5) | **CONFIRMED** |
| Spec leaves **full window vs. $\bar{P}_\beta$ only** as implementation choice | Paper mixes formulations (see below) | **OPEN** (appropriate) |

**Ambiguity in paper (spec handles correctly):**

- §IV.A (after Eq. (5)): “$\bar{P}_\beta$ … is used as input to both actor and critic networks.”
- §V.A: “The agent observes neural dynamics through the **average** beta-band power” but also “state $s_t$ consists of a fixed-length **window** of past beta power values.”

The spec’s “intentionally open” stance matches the paper; implementers should pick one representation and keep it fixed (as the spec states).

**Value of $n_{\mathrm{obs}}$:** Not stated in paper notes → spec §9.4 **OPEN** is correct.

**Verdict:** **CONFIRMED**, with **OPEN** on exact observation tensor (faithful to paper ambiguity).

---

## 3. Action space: binary pulse actions

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| $a_t \in \{0, 1\}$ | Eq. (6), §IV.A, §V.A | **CONFIRMED** |
| $0$ = no pulse, $1$ = DBS pulse | Eq. (6), §V.A | **CONFIRMED** |
| Sparse / energy-aware stimulation narrative | §IV.A (action space paragraph) | **CONFIRMED** |
| Training: logits + GS; eval: argmax / $\tau \to 0$ | §IV.C, Algorithm 1 | **CONFIRMED** (eval convention **OPEN** in spec—paper does not spell out hard vs. relaxed action at deploy time) |

**Verdict:** **CONFIRMED**

---

## 4. Step timing: 2 ms steps

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| RL step duration $l =$ **2 ms** | §V.A: “30 environment steps of **2 ms** each” | **CONFIRMED** |
| **30** steps per episode | §V.A | **CONFIRMED** |
| **60 ms** simulated time per episode | §V.A | **CONFIRMED** |
| Plant integration **0.02 ms** | §V.A | **CONFIRMED** |
| **150** training episodes | §V.A | **CONFIRMED** |

**Verdict:** **CONFIRMED**

---

## 5. Reward function: Eq. (7)

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| Piecewise squared form with factor **×10** | Eq. (7) | **CONFIRMED** |
| Positive branch when $\bar{P}_\beta < \beta_t$; negative otherwise | Eq. (7) | **CONFIRMED** |
| $\beta_t = **0.35**$ | Eq. (7), §V.A | **CONFIRMED** |
| Uses $\bar{P}_\beta$ from observation window | Eqs. (5), (7) | **CONFIRMED** |
| Distinction from Mehregan Eq. (8) (linear low branch) | Not in Ravivarapu paper; spec cross-ref to `environment.md` | **CONFIRMED** (repo cross-doc; Ravivarapu uses Eq. (7) only) |

**Verdict:** **CONFIRMED**

---

## 6. Sample efficiency: claims and methods

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| Sample efficiency via **predictive modeling** + **Gumbel-Softmax** | Abstract, §I, §IV | **CONFIRMED** |
| Faster convergence / higher reward vs. baseline | §V.B, Fig. 4(b) | **CONFIRMED** |
| Stronger beta PSD suppression vs. baseline | §V.B, Fig. 4(a) | **CONFIRMED** |
| Ablation: PM helps early training; **~4,500 samples** cited for noisy early PM | §V.B, Fig. 7 caption | **CONFIRMED** ($150 \times 30 = 4500$ env steps total in reported training run) |
| Reduced reliance on sparse stimulation feedback (PM) | §IV.B | **CONFIRMED** |

**Verdict:** **CONFIRMED**

---

## 7. Predictive reward model

| Claim in spec | Paper source | Verdict |
|---------------|--------------|---------|
| $\hat{r}_t = f_\theta(s_t, a_t)$ | Eq. (8) | **CONFIRMED** |
| Q-target: $r_t + \hat{r}_t + \gamma Q_{\phi'}(s_{t+1}, \pi_{\theta'}(s_{t+1}))$ | Eq. (9) | **CONFIRMED** |
| Auxiliary loss $\mathcal{L}_{\mathrm{pred}} = \frac{1}{B}\sum (r_i - \hat{r}_i)^2$ | Eq. (10) | **CONFIRMED** |
| Trained concurrently with actor/critic | Algorithm 1, §IV.B | **CONFIRMED** |
| Purpose: supplement sparse real feedback for critic bootstrapping | §IV.B | **CONFIRMED** |
| Architecture (MLP depth, one-hot action, etc.) not specified | Paper silent | **OPEN** (spec correctly) |
| Ablation `baseline+pm` | Fig. 7: “Baseline+PM” | **CONFIRMED** |

**Clarification:** The model predicts **immediate** reward from $(s_t, a_t)$, not a separate “next-step” or “future” reward variable. The spec is consistent with Eqs. (8)–(9).

**Verdict:** **CONFIRMED** (with **OPEN** on architecture, as in spec)

---

## 8. Adapter design (shared plant → 2 ms SEA-DBS I/O)

The paper does **not** define a software adapter; it assumes a single simulation environment with the stated timing and I/O. The spec’s `SEA_DBSEnvAdapter` is **repo architecture**.

| Adapter requirement in spec | Grounding in paper | Verdict |
|-----------------------------|-------------------|---------|
| **2 ms** RL steps (not Mehregan 2 s) | §V.A | **CONFIRMED** (requirement correct) |
| **Binary** pulse actions vs. pattern alphabet | Eq. (6) | **CONFIRMED** |
| **Eq. (7)** reward, not Mehregan Eq. (8) | Eq. (7), §V.A | **CONFIRMED** |
| Expose $(s, a, r, s')$ plus logits and $\hat{r}$ for replay | Algorithm 1 | **CONFIRMED** |
| How to subsample/re-integrate from shared `envs/` stack | Paper silent | **OPEN** (spec §5 explicitly) |
| Plant: CBGT, STN stimulation, 10 neurons/region | §III, §V.A | **CONFIRMED** |
| Spec cites **Kumaravelu** plant via `environment.md` | Paper: model “**inspired by** Mehregan et al. [29]” (no “Kumaravelu” in paper notes) | **OPEN** (provenance via Mehregan/Kumaravelu chain is repo convention, not stated in Ravivarapu text) |

**Verdict:** **OPEN** for adapter mechanism; **CONFIRMED** for functional requirements derived from the paper.

---

## 9. Training hyperparameters

| Hyperparameter | Spec (Table / §9.4) | Paper Table I / §V.A | Verdict |
|----------------|---------------------|----------------------|---------|
| Actor LR $\alpha_a = 5\times 10^{-4}$ | ✓ | 0.0005 | **CONFIRMED** |
| Critic LR $\alpha_c = 10^{-3}$ | ✓ | 0.001 | **CONFIRMED** |
| Discount $\gamma = 0.99$ | ✓ | 0.99 | **CONFIRMED** |
| Replay buffer **8192** | ✓ | 8192 | **CONFIRMED** |
| Batch size **32** | ✓ | 32 | **CONFIRMED** |
| Exploration: GS with annealing | ✓ | Table I | **CONFIRMED** |
| $\tau_0$, $\tau_{\min}$, $\lambda_\tau$ (Eq. 14) | OPEN | Not in Table I | **OPEN** (correct) |
| Polyak $\tau$ for target nets | OPEN | Algorithm 1 uses $\tau$ but no numeric value | **OPEN** (correct) |
| Predictive model LR | OPEN | Not in Table I | **OPEN** (correct) |
| $n_{\mathrm{obs}}$ | OPEN | Not in paper notes | **OPEN** (correct) |
| Gradient steps per env step, update ordering | OPEN | Not detailed like Mehregan | **OPEN** (correct) |
| Done flag $dw$ in replay | OPEN | Not in Algorithm 1 | **OPEN** (reasonable extension) |

**Verdict:** **CONFIRMED** for Table I; **OPEN** for undocumented values (appropriately flagged in spec).

---

## 10. Comparison baselines (paper experiments)

| Experiment / baseline | Spec (§4, §10) | Paper | Verdict |
|----------------------|----------------|-------|---------|
| **Baseline**: DDPG without PM or GS | `variant=baseline` | §V.B “Baseline (DDPG)”; Fig. 4 | **CONFIRMED** |
| **Baseline+PM** | `variant=baseline_pm` | Fig. 7 “Baseline+PM” | **CONFIRMED** |
| **Baseline+GS** (“guided sampling” in Fig. 7 caption) | `variant=baseline_gs` | Fig. 7 “Baseline+GS” | **CONFIRMED** |
| **SEA-DBS** (PM + GS) | `variant=sea_dbs` | Full method, Fig. 7 | **CONFIRMED** |
| Training curves vs. baseline | §10 | Fig. 4 | **CONFIRMED** |
| **Seed change** every $n \in \{10,20,50,75\}$ **steps** (PD progression protocol) | §5, §10 | §V.B, Table II | **CONFIRMED** |
| Table II: avg PSD ↓, reward ↑ for SEA-DBS | §10 | Table II | **CONFIRMED** |
| Inference **50 Hz vs 30 Hz** carrier | §10 | Fig. 5, §V.B | **CONFIRMED** |
| Ablation over **10 stimulation steps** | §10 | Fig. 7 | **CONFIRMED** |
| FP16 PTQ: **65 MB → 33 MB**, 10-step eval | §11 | §V.B, Fig. 6 | **CONFIRMED** |

**Minor wording note:** Table II header in the paper says “stimulation **update intervals**” while the text describes changing the **environment seed** every $n$ steps. The spec’s “seed change interval” matches the prose, not the table header.

**Verdict:** **CONFIRMED**

---

## Cross-cutting checklist (`sea_dbs.md` §13)

| Checklist item | Verdict |
|----------------|---------|
| Plant, STN, GPi $P_\beta$ 13–35 Hz, 10 neurons | **CONFIRMED** |
| Binary actions + GS + logits in replay | **CONFIRMED** |
| Eq. (7) reward, $\beta_t=0.35$ | **CONFIRMED** |
| Critic target $r+\hat{r}+\gamma Q'$; $\mathcal{L}_{\mathrm{pred}}$ | **CONFIRMED** |
| 2 ms × 30 × 150 episodes | **CONFIRMED** |
| Table I hyperparameters | **CONFIRMED** |
| Baseline / ablation variants | **CONFIRMED** |
| FP16 PTQ separate from training | **CONFIRMED** |

---

## Discrepancies summary

| ID | Severity | Issue |
|----|----------|--------|
| — | — | **No material discrepancies** between `sea_dbs.md` and `paper_3.md` on the ten review axes. |

**Non-discrepancy clarifications** (paper internal or repo-only):

1. **State representation:** window (Eq. 4) vs. mean input (Eq. 5) / §V.A “average”—spec **OPEN** is the right response.
2. **Plant citation:** Ravivarapu cites Mehregan-inspired model; spec points to Kumaravelu via shared `environment.md`—align in code/docs, not a conflict with Ravivarapu equations.
3. **“Future outcomes”** in paper intro vs. current-step $\hat{r}_t$ in math—spec follows math.

---

## Recommended spec tweaks (optional, not required for faithfulness)

These would reduce implementer confusion but are outside strict paper fidelity:

1. Add a one-line note that paper §V.A emphasizes **mean** beta for observations while Eq. (4) defines a **window**—default recommendation could cite Eq. (5) unless replicating unreleased code.
2. Footnote that Fig. 7 “guided sampling” = Gumbel-Softmax (paper naming).
3. Note paper’s shared $\theta$ symbol for actor and predictive model when naming parameters in code (`actor_theta` vs. `pred_theta`).

---

## Review metadata

| Field | Value |
|-------|--------|
| Review date | 2026-05-18 |
| Spec version reviewed | `docs/controllers/sea_dbs.md` (current workspace) |
| Paper source | Ravivarapu et al. notes (`paper_3.md`) |
