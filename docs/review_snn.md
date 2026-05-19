# SNN controller spec review: faithfulness to Nguyen et al.

**Reviewed:** `docs/controllers/snn.md`  
**Source:** `paper_2.md` (Nguyen et al., *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks*)  
**Date:** 2026-05-18

This report compares every major claim in the SNN controller specification against the paper notes. Verdicts are **CONFIRMED**, **DISCREPANCY**, or **OPEN** (not specified in the paper; spec correctly defers or must choose an implementation).

---

## Executive summary

The spec is **largely faithful** to the paper. Architecture, timing, action/reward formulations, training schedule, initial conditions, and reported outcomes match the source text and equations. Gaps are mostly **intentionally open** items the paper does not fix (hyperparameters, observation tensor layout, adapter bridging to the shared repo plant).

**Notable findings:**

| Severity | Item |
|----------|------|
| Minor | Spec uses “binary spike **counts**” in §2; paper specifies **binary** (0/1) presence encoding, not count aggregation. |
| Minor | Paper abbreviates the network as **DQSN** once (§III.B) but **DSQN** elsewhere; spec standardizes on DSQN. |
| Open | “Batches of 128 inputs” (architecture) vs “128 state transitions” (replay) are two distinct uses of 128; spec separates them but input-dimension interpretation remains ambiguous in the paper. |
| Open | Adapter design (100 ms steps, α–β biomarker, ternary DBS deltas) is **repo inference**, not described in the paper—the paper assumes a native Nguyen Gymnasium environment. |

---

## 1. Network architecture (DSQN layers, neuron model, spike encoding)

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| Three-layer **DSQN** with one hidden layer | §III.B: “three-layer spiking neural network with a single hidden layer of 128 LIF neurons” | **CONFIRMED** |
| Hidden layer: **128 LIF neurons** | §III.B | **CONFIRMED** |
| Output layer: **9 LIF units** | §III.B: “produce 9 action outputs” | **CONFIRMED** |
| Each stage: linear transform + **LIF** | §III.B: “Each layer combines a linear transformation with a LIF neuron model” | **CONFIRMED** |
| Membrane leak **β = 0.95** | §III.B | **CONFIRMED** |
| LIF dynamics Eqs. (2)–(3) | §II.C | **CONFIRMED** |
| Neuron model is **LIF**, not IF | §II.C title and equations | **CONFIRMED** |
| Network unrolls over **multiple internal timesteps**; membrane state carried forward | §III.B: “Operating across multiple timesteps, the network maintains membrane potentials” | **CONFIRMED** |
| **Control:** action = argmax **spike counts** over unrolled window | §III.B: “actions are selected based on maximum spike counts” | **CONFIRMED** |
| **Learning:** **final-layer membrane potentials** as continuous Q-estimates | §III.B: “final layer’s membrane potential provides continuous Q-value estimates” | **CONFIRMED** |
| Spike encoding: **binary** (1 = spike, 0 = no spike) | §III.B after Eq. (4) | **CONFIRMED** |
| “Processing **batches of 128 inputs**” | §III.B (same sentence as hidden 128) | **OPEN** — Paper does not clarify whether 128 is input feature dimension, temporal sequence length, or conflated with the replay batch size (also 128). Spec’s interpretation as flattened/sequenced spike features is reasonable but not verifiable from the manuscript alone. |
| Input layer size / connectivity | Not specified | **OPEN** — Spec does not over-claim; correctly leaves tensor layout open in §4.1. |

**Notes:**

- Paper uses **DQSN** in §III.B and **DSQN** in the abstract. Spec’s DSQN label is consistent with the paper title and abstract.
- Spec’s use of **θ_th** for LIF firing threshold (distinct from biomarker θ = 150) is a helpful disambiguation; paper reuses θ for both LIF threshold (§II.C) and control threshold (§IV).

---

## 2. Observation space

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| Observation = Spikes ∈ **[0, 1]^{n × N}** (Eq. (4)) | §III.B Eq. (4) | **CONFIRMED** |
| **Binary** encoding: 1 if spike occurred, 0 otherwise | §III.B: “Neural spikes are represented in binary format” | **CONFIRMED** |
| **n** = number of steps in observation sequence | §III.B: “n is the number of steps” | **CONFIRMED** |
| Spikes measured **per brain region** across sequence steps | §III.B: “total number of spikes measured from each brain region across sequence steps” | **CONFIRMED** (wording) |
| **N** = number of neurons | §III.B: “N is the number of neurons” | **CONFIRMED** (symbol) |
| Region list and alignment of **n** with the 100 ms RL step | Not fully specified | **OPEN** — Spec correctly marks this. Paper text mixes “per brain region” with “N = number of neurons,” leaving aggregation semantics ambiguous. |
| “Binary spike **counts**” per neuron (§2 step 2) | Paper says binary presence, not counts | **DISCREPANCY (minor)** — Wording in spec §2 suggests counting; paper encoding is strictly 0/1 presence per window. Recommend changing “spike counts” to “binary spike matrix” or “spike presence flags” for fidelity. |

---

## 3. Action space

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| Three DBS parameters: **amplitude, frequency, pulse width** | §I, §III.A, §IV | **CONFIRMED** |
| Per-parameter action **a ∈ {−1, 0, 1}** (Eq. (5)) | §III.B Eq. (5) | **CONFIRMED** |
| Semantics: decrease / maintain / increase | §III.B | **CONFIRMED** |
| Each ternary value × **scalar sensitivity** (step size) | §III.B: “Each parameter is multiplied by a scalar value that adjusts the sensitivity” | **CONFIRMED** |
| Numeric sensitivities not given | Not in paper | **OPEN** — Spec correctly defers. |
| **Nine** network outputs (3 params × 3 choices) | §III.B: “9 action outputs” + Eq. (5) per-parameter ternary | **CONFIRMED** (inference is sound) |
| Joint vs. factored Q-heads (single argmax over 9 vs. three argmaxes) | Not specified | **OPEN** — Spec correctly flags. |
| Mehregan-style discrete STN **pattern** actions excluded | Paper modulates continuous DBS parameters, not spatial patterns | **CONFIRMED** (scope statement) |

---

## 4. Step timing (100 ms)

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| RL step duration = **100 ms** simulated time | §IV: “each episode simulates **100 ms per time step**” | **CONFIRMED** |
| Eval: **25 time steps** per episode | §IV, Fig. 7 caption: “averaging 50 test episodes across **25 time steps**” | **CONFIRMED** |
| Training episode **horizon** (max steps before truncation) | Not explicitly stated for training | **OPEN** — Paper describes early termination (§III.B) but not a fixed max episode length during the 500 training episodes. |
| Distinction from Mehregan **2 s** steps | Not in Nguyen paper (repo cross-reference) | **N/A (repo)** — Correct contextual note for adapter work; not a paper claim. |

---

## 5. Reward function (α–β feedback)

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| Feedback biomarker: **GPi α–β oscillation power** (α 7–13 Hz + β 13–35 Hz) | §II.A, §III.A | **CONFIRMED** |
| Control threshold **θ = 150** | §IV: “threshold is selected to be **150**” | **CONFIRMED** |
| Threshold rationale: ~first quartile of PD distribution (Fig. 3) | §IV | **CONFIRMED** |
| **θ_u = 1** if α–β **>** θ, else 0 | §III.D: “θ_u … whether the α-β oscillation power is **greater than or less than** the threshold” | **CONFIRMED** |
| **d** = squared distance of α–β from θ | §III.D | **CONFIRMED** |
| Non-terminal reward: **−δE + τθ_u + (1 − θ_u)d** | §III.D Eq. (7) | **CONFIRMED** |
| Terminal (early success) reward: **τ(t_r + 1) − δE** | §III.D Eq. (7) | **CONFIRMED** |
| Coefficients **δ, τ** and **d** normalization | Not numerically specified | **OPEN** — Spec correctly defers. |
| Early termination when **α–β < θ** for **t_u** consecutive steps | §III.B | **CONFIRMED** (rule); **t_u** value is **OPEN** |
| Distinction from Mehregan Eq. (8) on normalized P_β | Not in Nguyen paper | **N/A (repo)** — Appropriate cross-controller note. |

---

## 6. Q-learning update, target network, exploration

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| Algorithm family: **DQN** / Bellman Eq. (1) | §II.B Eq. (1), §III.B | **CONFIRMED** |
| Target **y = r + γ max_{a′} Q(s′, a′; θ⁻)** (or spiking analogue) | §II.B (target described; notation garbled as “θ\|s,a”) | **CONFIRMED** (conceptually) |
| **Target network** update period | Not named | **OPEN** — Spec correctly does not invent a period. |
| **Double DQN**, dueling, etc. | Not mentioned | **OPEN** — Correctly excluded. |
| **ε-greedy** exploration | §III.B: “random sample based on the ε-Greedy approach” | **CONFIRMED** |
| **Decreasing ε** each time step | §III.B: “decreasing-ε method … for every time step” | **CONFIRMED** |
| Exploration → exploitation around episode **~100** (qualitative) | §IV, Fig. 4 narrative | **CONFIRMED** (qualitative) |
| Exact **ε** schedule, **γ**, learning rate, optimizer, loss form | Not tabulated | **OPEN** — Spec correctly defers. |
| Replay: update weights every **128** stored transitions | §III.B: “updating its weights after collecting every **128 state transitions** in its replay memory” | **CONFIRMED** |
| **500** training episodes | §III.B, §IV | **CONFIRMED** |
| Q-values from **membrane potentials** (not spike counts) for Bellman targets | §III.B | **CONFIRMED** |

---

## 7. Adapter design (shared plant → Nguyen I/O)

The paper integrates the Kumaravelu CBGT model **directly** into a Gymnasium RL environment (§III.B). It does **not** describe bridging from a Mehregan-style wrapper (2 s steps, P_β-only state, pattern actions). Adapter claims are **repo architecture**, evaluated for consistency with paper requirements.

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| Shared plant: Kumaravelu **CBGT**, 10 neurons/region, STN DBS | §II.A (Kumaravelu et al. [7]) | **CONFIRMED** |
| Spec adds “**6-OHDA** rat” label | Kumaravelu [7] title (6-OHDA lesioned rat); not spelled out in Nguyen §II.A | **CONFIRMED** (via citation; reasonable) |
| Feedback must use **α–β**, not P_β-only | §II.A, §III.A vs Mehregan (external) | **CONFIRMED** (paper); **N/A** (Mehregan contrast) |
| Adapter: **100 ms** plant steps / subsampling | §IV | **CONFIRMED** (requirement); adapter mechanism is **OPEN** (implementation) |
| Adapter: build **spike observation** Eq. (4) | §III.B | **CONFIRMED** (requirement) |
| Adapter: compute **α–β** for reward/termination | §III.A, §III.D | **CONFIRMED** (requirement) |
| Adapter: **ternary parameter deltas** instead of STN patterns | §III.B Eq. (5) | **CONFIRMED** (requirement) |
| Adapter module **`NguyenEnvAdapter`** | Not in paper | **N/A (repo)** — Reasonable design; not verifiable against paper. |

**Assessment:** Adapter responsibilities are a **correct decomposition** of Nguyen I/O requirements for a shared-plant codebase. No paper contradiction, but the adapter itself is **implementation guidance**, not a reproduced paper claim.

---

## 8. Training hyperparameters

| Parameter | Spec value | Paper | Verdict |
|-----------|------------|-------|---------|
| Training episodes | 500 | §III.B, §IV | **CONFIRMED** |
| Replay update cadence | every 128 transitions | §III.B | **CONFIRMED** |
| Hidden LIF units | 128 | §III.B | **CONFIRMED** |
| LIF leak β | 0.95 | §III.B | **CONFIRMED** |
| Output units | 9 | §III.B | **CONFIRMED** |
| Biomarker threshold θ | 150 | §IV | **CONFIRMED** |
| Initial frequency | 40 Hz | §IV | **CONFIRMED** |
| Initial pulse width | 0.3 ms | §IV | **CONFIRMED** |
| Initial amplitude | 300 nA/cm² | §IV | **CONFIRMED** |
| RL step | 100 ms | §IV | **CONFIRMED** |
| Eval episodes | 50, random seed each | §IV | **CONFIRMED** |
| Eval steps per episode | 25 | §IV, Fig. 7 | **CONFIRMED** |
| γ (discount) | config field | Not given | **OPEN** |
| Learning rate / optimizer | config field | Not given | **OPEN** |
| ε schedule (numeric) | config field | Not given | **OPEN** |
| t_u (consecutive sub-threshold steps) | config field | Mentioned, not valued | **OPEN** |
| DBS parameter sensitivities | config field | Not given | **OPEN** |
| δ, τ reward weights | config field | Not given | **OPEN** |
| Target network period | config field | Not given | **OPEN** |

---

## 9. Energy efficiency and neuromorphic claims

| Claim (spec) | Paper source | Verdict |
|--------------|--------------|---------|
| DBS energy index Eq. (6): **E_t = N × √(Σ I²_DBS / n)** | §III.C Eq. (6) | **CONFIRMED** |
| **n** = pulses per time step; **I_DBS** function of amplitude, frequency, pulse width | §III.C | **CONFIRMED** |
| **~22%** energy reduction vs open-loop **130 Hz** DBS after training | §IV: “reduces DBS energy consumption by ≈22% compared to standard open-loop DBS (130 Hz)” | **CONFIRMED** |
| Reward balances therapeutic α–β control and **energy** (Eq. (7)) | §III.D, Fig. 4–5 | **CONFIRMED** |
| SNNs as **energy-efficient** alternative to conventional ANNs (motivation) | §I, §VI | **CONFIRMED** (motivational claim in paper) |
| **Neuromorphic hardware** deployment / on-chip learning | §VI: “Future directions include **integration with neuromorphic hardware**” | **CONFIRMED** that spec correctly scopes this **out of scope** — paper presents simulation only; hardware efficiency is aspirational, not experimentally demonstrated. |
| End-to-end spike processing (sensory spikes → SNN → stimulation) | §I abstract, §VI contribution (3) | **CONFIRMED** (framework claim) |
| Learned parameters ~**262 nA/cm²**, **78.65 Hz**, **~1 ms** pulse width | §IV, Fig. 6 | **CONFIRMED** |
| α–β sustained below θ = 150 | §IV, Fig. 3, 6 | **CONFIRMED** (qualitative replication target) |

**Caution:** The **22% DBS energy reduction** is a **simulation result** comparing learned closed-loop parameters to a **130 Hz open-loop baseline**, not measured neuromorphic chip power. Spec’s “energy-aware reward” and “~22%” anchor are faithful; readers should not conflate DBS stimulation energy with SNN inference energy unless explicitly measured (paper does not report the latter).

---

## 10. Additional cross-checks

| Item | Verdict |
|------|---------|
| Spec cites §III.B for early termination + terminal bonus | **CONFIRMED** — termination rule in §III.B; bonus form in Eq. (7) §III.D. |
| Spec §6.4 “unless extending horizon for ablations (§V notes longer horizons as future work)” | **CONFIRMED** — §V: “test performance beyond 25 time steps remains to be an area of research.” |
| Gymnasium as RL platform | **CONFIRMED** — §III.B cites Gymnasium [24]; spec does not contradict. |
| Paper reference to Actor-Critic reward inspiration (Wang et al. [13]) | **CONFIRMED** — §III.D; spec’s Eq. (7) matches Nguyen’s modified form, not Mehregan Eq. (8). |

---

## Recommended spec edits (optional)

These are **not blockers** for implementation; they would tighten paper fidelity:

1. **§2, step 2:** Replace “binary spike counts” with “binary spike matrix” or “binary spike presence per neuron/region.”
2. **§4.1:** Add one sentence noting the paper’s ambiguous “128 inputs” vs “128 replay transitions” distinction so implementers do not conflate input dimension with minibatch size.
3. **§9 or §1:** Clarify that **22% energy savings** refers to **DBS stimulation energy in simulation**, not demonstrated neuromorphic hardware power reduction.

---

## Overall verdict

**`docs/controllers/snn.md` is faithful to Nguyen et al.** for all paper-specified mechanics: LIF-based DSQN topology, binary spike observations, ternary DBS parameter actions, 100 ms steps, GPi α–β feedback with θ = 150, Eqs. (6)–(7) reward/energy, DQN + ε-greedy training with 500 episodes and 128-transition replay updates, and reported outcome anchors.

Remaining gaps are appropriately labeled **intentionally open** or belong to **repo adapter design** outside the paper’s scope. One minor terminology discrepancy (“spike counts” vs binary presence) should be corrected when the spec is next edited.
