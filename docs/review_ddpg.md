# DDPG controller spec review — faithfulness to Mehregan et al.

**Reviewed:** `docs/controllers/ddpg.md`  
**Reference:** `paper_1.md` (Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*)  
**Date:** 2026-05-18

**Legend**

| Verdict | Meaning |
|---------|---------|
| **CONFIRMED** | Spec matches the paper (or correctly defers to a companion spec that matches). |
| **DISCREPANCY** | Spec states something inconsistent with, or materially narrower than, the paper. |
| **OPEN** | Paper is silent or ambiguous; spec correctly marks as implementation choice, or spec should label it open but does not. |

---

## Executive summary

`docs/controllers/ddpg.md` is **largely faithful** to Mehregan et al. §III (actor–critic, Algorithm 1, losses, quantization) and §IV.A.1 (computational hyperparameters). The spec improves on the manuscript in three valuable ways: (1) clarifying that “DDPG” here means **discrete pattern selection via logits + softmax + argmax**, (2) resolving **transition tuple semantics** ($s$ = pre-action, $s'$ = post-action) where Algorithm 1’s step ordering is confusing, and (3) explicitly marking **unspecified numerics** ($\gamma$, $\tau$, `update_frequency`, CNN sizes, pattern count) as open.

**Notable gaps (not wrong, but incomplete):** no dedicated **variant slug** table in `ddpg.md` (taxonomy lives in `benchmarking.md` / `development.md`); **replay** omits stored discrete pattern index `a` that Algorithm 1 lists; **target-network initialization** is not in the paper’s Algorithm 1; **critic–action fusion** and **QAT on the critic** are unspecified in both paper and spec.

---

## 1. Actor network architecture (CNN, temporal window, softmax output)

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| CNN backbone for temporal biomarker state | §III.B: “CNNs were chosen due to the temporal nature of the biomarker”; conv + average pooling + linear (Fig. 3a) | **CONFIRMED** | |
| Forward pass yields **action logits**; **softmax + argmax** for applied pattern | §III.B: logits stored; “softmax and argmax operations generate the full action” | **CONFIRMED** | |
| Channel counts, kernels, **shrink dimension** not numerically fixed | §III.B lists parameters but §IV.A.1 gives no architecture table | **OPEN** | Spec §3.2 correctly marks topology as intentionally open. |
| Biomarker window = full simulation segment per RL step | §IV.A.1: “window size was set to the length of the simulation” (2 s steps) | **CONFIRMED** | Spec points to `environment.md` §3.2, §5 — consistent. |
| Discrete **pattern alphabet** size not fixed | §III.B “action size”; §IV.A.1 does not state cardinality | **OPEN** | Spec §1 scope table correctly defers pattern count to code. |

**Paper vs. spec framing (not a spec error):** §III.A and the abstract describe DDPG as suited to **continuous** actions; the implemented actor is **discrete over patterns**. Spec §3.1 documents this faithfully; the manuscript’s “continuous actions” wording is internally inconsistent with §III.B.

---

## 2. Critic network architecture

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| Same CNN-style backbone (conv, avg pool, linear) | §III.B, Fig. 3b | **CONFIRMED** | |
| Input: state + **action logits** $a_{\mathrm{logit}}$; scalar $Q$ output | Eq. (4), (5); $Q(s^{\tilde{B}}, a_{\mathrm{logit}}^{\tilde{B}})$ | **CONFIRMED** | |
| State fused with logits via “concatenated or otherwise fused” | Fig. 3b (structure not readable in notes); text does not say “concatenate” | **OPEN** | Spec correctly does not over-specify fusion. |
| Target critic $Q_{\mathrm{target}}$ for bootstrapping | Algorithm 1, Eq. (3) | **CONFIRMED** | |
| Critic uses **stored** $a_{\mathrm{logit}}$ from replay in actor/critic losses | Eq. (4)–(5) use batch logits, not re-sampled actor output | **CONFIRMED** | |

**Manuscript typo (spec handles better):** Eq. (7) prose says $\theta_{u,i}$ is “in the **critic** network”; it should be the **actor**. Spec §4.3 notes $\theta_u$ denotes the actor.

---

## 3. Replay buffer design and minibatch sampling

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| Capacity **8192** | §IV.A.1 | **CONFIRMED** | |
| Minibatch size **32** | §IV.A.1 | **CONFIRMED** | |
| Sample $\tilde{B}$ each inner update | Algorithm 1 lines 15–16 | **CONFIRMED** | |
| Store transition with $s$, $s'$, $a_{\mathrm{logit}}$, $R$, $dw$ | Algorithm 1 line 13: $(s', a, a_{\mathrm{logit}}, R, s, dw)$; §III.B “states before and after” | **CONFIRMED** | Spec’s $(s, a_{\mathrm{logit}}, R, s', dw)$ matches Eq. (4) semantics. |
| Store discrete pattern index **`a`** | Algorithm 1 and buffer narrative include **`a`** alongside logits | **DISCREPANCY** (minor) | Spec §4.2 / §7 list only $a_{\mathrm{logit}}$; index is derivable from argmax but the paper’s tuple explicitly includes **`a`**. Recommend storing `a` for paper-aligned replay dumps. |
| `update_frequency` inner loop count | Algorithm 1 line 15: “for update frequency” | **OPEN** | No numeric value in §IV.A.1; spec §4.1 / §4.3 correctly open. |

**Algorithm 1 ordering ambiguity (paper):** After `step`, line 12 says “Form state $s$ from $P_\beta$” while line 13 stores both $s$ and $s'$. §III.B clarifies pre/post states. Spec §4.2 step 2–3 aligns with Eq. (4), not the ambiguous line order — **CONFIRMED** as the right interpretation.

---

## 4. Target network soft-update ($\tau$)

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| Polyak update Eqs. (6)–(7) after critic + actor steps | Algorithm 1 lines 25–26; Eqs. (6)–(7) | **CONFIRMED** | |
| Same $\tau$ for actor and critic targets | Single $\tau$ initialized in Algorithm 1 step 1 | **CONFIRMED** | |
| Numeric **$\tau$** | Initialized, never given in §IV.A.1 | **OPEN** | Spec §4.1 correctly open. |
| $\theta_{\mu,\mathrm{target}} \leftarrow \tau \theta_\mu + (1-\tau)\theta_{\mu,\mathrm{target}}$ | Algorithm 1 line 26 in notes has typo: RHS uses $\theta_\mu$ twice | **CONFIRMED** | Spec fixes the typo; matches Eq. (7). |
| Initialize $\mu_{\mathrm{target}}$, $Q_{\mathrm{target}}$ at start | Not in Algorithm 1 steps 1–5 (only online nets + buffer) | **OPEN** | Spec §4.1 assumes standard DDPG target copies — reasonable, not paper-specified. |

---

## 5. Discount factor $\gamma$

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| $\gamma$ used in bootstrap $(1 - dw)$ mask | Eq. (3), Algorithm 1 line 18 | **CONFIRMED** | |
| Numeric $\gamma$ | Algorithm 1 step 2; **not** in §IV.A.1 | **OPEN** | Spec §4.1 correctly open. |

---

## 6. Algorithm 1 — training loop (step-by-step)

| Step / behavior | Spec section | Paper (Alg. 1 + §III.B) | Verdict |
|-----------------|--------------|-------------------------|---------|
| Init $\tau$, $\gamma$, $\mu$, $Q$, buffer $B$ | §4.1 | Lines 1–5 | **CONFIRMED** |
| Episode loop; reset env; $s_0$ from $P_\beta$ | §4.2 + env cross-ref | Lines 6–8 | **CONFIRMED** |
| $u \leftarrow \mu(s)$; argmax pattern to plant | §4.2 | Lines 10–11 | **CONFIRMED** |
| Step duration $l$; collect $P_\beta$, $R$, $dw$ | §2, §4.2 | Lines 11–12 | **CONFIRMED** ($l = 2$ s in §IV.A.1) |
| Store transition; $s \leftarrow s'$ | §4.2 | Lines 13–14 | **CONFIRMED** (semantics clarified in spec) |
| Inner loop `update_frequency` × minibatch | §4.3 | Lines 15–27 | **CONFIRMED** |
| $Q' = Q_{\mathrm{target}}(s', \mu_{\mathrm{target}}(s'))$; target $= R + \gamma(1-dw)Q'$ | §4.3 | Lines 17–18, Eq. (3) | **CONFIRMED** |
| Critic loss MSE; Adam | §4.3 | Lines 19–20, Eq. (4) | **CONFIRMED** |
| Freeze critic → actor loss → Adam → unfreeze | §4.3 | Lines 21–24, Eq. (5) | **CONFIRMED** |
| Soft-update targets | §4.3 | Lines 25–26 | **CONFIRMED** |
| Output $\pi^* \approx \mu(\cdot \mid \theta_\mu)$ | §4.5 | Lines 30–31 | **CONFIRMED** |
| Actor LR $5\times10^{-4}$, critic LR $10^{-3}$ | §4.3 | §IV.A.1 | **CONFIRMED** |
| 30 steps/episode, 10 episodes | §4.4 | §IV.A.1 | **CONFIRMED** |
| Init regular pulses **45 Hz**; **30 Hz** ablation | §4.4 | §IV.A.1, §IV.A.2 | **CONFIRMED** |
| Exploration noise / $\epsilon$-greedy | — | Not described | **OPEN** | Spec does not add noise; paper uses deterministic argmax — aligned with text. |

**Bootstrap detail:** Spec states $\mu_{\mathrm{target}}(s')$ supplies **logits** into $Q_{\mathrm{target}}$ — matches “target actor network” in Eq. (3) and standard actor–critic bootstrapping with logit-valued critic inputs — **CONFIRMED**.

---

## 7. Quantization — PTQ and QAT

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| **PTQ** after full-precision training; inference only | §III.D | **CONFIRMED** | |
| PyTorch **`quantize_dynamic`** on weights | §III.D cites [38, 39] | **CONFIRMED** | |
| Activations via Eqs. (9)–(11) ($Q(r)$, $S$, $Z$) | §III.D | **CONFIRMED** | |
| Evaluated at **FP16** and **INT8** | §III.D, §IV.A.3 | **CONFIRMED** | |
| **QAT**: fake quantization; forward/backward emulate int8 | §III.D | **CONFIRMED** | |
| Actor: **quant stub** on input; **dequant stub** on logits before action | §III.D (Actor module only) | **CONFIRMED** | |
| QAT under **10 episodes** underperforms PTQ; may need longer training | §IV.A.3 | **CONFIRMED** | Spec §6 matches paper discussion. |
| Quantization does not change plant / env API | Implied (network-only) | **CONFIRMED** | |
| **QAT on critic** (stubs, which layers) | §III.D details Actor only | **OPEN** | Spec mirrors paper scope; full four-network QAT layout not specified in either. |
| PTQ calibration procedure | “may need to be calibrated” | **OPEN** | Spec does not claim a fixed calibration protocol — appropriate. |

---

## 8. Variants (`paper`, `init-30hz`, `ptq-int8`, etc.)

| Variant / experiment | In paper | In `ddpg.md` | Verdict |
|----------------------|----------|--------------|---------|
| Default training: **45 Hz** mean init | §IV.A.1 | §4.4 | **CONFIRMED** |
| **30 Hz** init, other params fixed | §IV.A.2 | §4.4 | **CONFIRMED** |
| **PTQ** FP16 / INT8 at 45 & 30 Hz | §IV.A.3, Fig. 6 | §6, checklist | **CONFIRMED** |
| **QAT** same run settings as FP trainer | §IV.A.3 | §6 | **CONFIRMED** |
| Slugs: `paper`, `init-30hz`, `ptq-int8`, … | — | Not defined in `ddpg.md` | **OPEN** | Repo convention in `benchmarking.md` §2, `development.md` — scientifically fine, but `ddpg.md` does not map slugs → paper figures. |
| Baselines `cdbs-130hz`, `periodic-45hz`, `periodic-30hz` | §IV.A.2 | Out of scope (env/benchmarking) | **CONFIRMED** | Correctly not duplicated as controller variants. |

**Suggestion (documentation only):** Add a short § “Benchmark variants” cross-linking `paper` = 45 Hz FP train, `init-30hz` = §IV.A.2, `ptq-fp16` / `ptq-int8` = §IV.A.3 PTQ, `qat` = §IV.A.3 QAT.

---

## 9. Hyperparameters — fixed vs open labeling

| Parameter | Paper §IV.A.1 | Spec labeling | Verdict |
|-----------|---------------|---------------|---------|
| Actor LR $5\times10^{-4}$ | Given | Fixed default | **CONFIRMED** |
| Critic LR $10^{-3}$ | Given | Fixed default | **CONFIRMED** |
| Buffer 8192, batch 32 | Given | Fixed default | **CONFIRMED** |
| RL step 2 s, plant dt 0.02 ms | Given | Via `environment.md` | **CONFIRMED** |
| 30 steps × 10 episodes | Given | §4.4 | **CONFIRMED** |
| Init mean 45 Hz (30 Hz ablation) | Given | §4.4 | **CONFIRMED** |
| $\gamma$ | Not numeric | Open §4.1 | **CONFIRMED** |
| $\tau$ | Not numeric | Open §4.1 | **CONFIRMED** |
| `update_frequency` | Not numeric | Open §4.1 | **CONFIRMED** |
| CNN topology / shrink dimension | Not numeric | Open §3.2 | **CONFIRMED** |
| Pattern count / encoding | Not numeric | Open §1, §3.4 | **CONFIRMED** |
| Target net init copy | Unspecified | Implied in §4.1 | **OPEN** | Could add one line: “not specified in paper; use standard copy-at-init.” |
| Reward $\beta_t = 0.35$ | Eq. (8) | `environment.md` §6 | **CONFIRMED** | Appropriately env-side; spec §5 cross-references. |

---

## 10. Environment observations vs actor inputs

| Claim in spec | Paper source | Verdict | Notes |
|---------------|--------------|---------|-------|
| Policy does **not** compute $P_\beta$ internally | §III.B: actor “observes the biomarkers”; plant simulates brain | **CONFIRMED** | Spec §2 bullet 3. |
| State $s$ formed from $P_\beta$ trajectory / window | Algorithm 1 lines 8, 12; §III.B “Brain” window | **CONFIRMED** | |
| CNN input = temporal biomarker structure | §III.B | **CONFIRMED** | |
| Window = length of simulation segment (2 s step) | §IV.A.1 | **CONFIRMED** | Spec defers detail to `environment.md`. |
| Reward Eq. (8) uses same normalized $s(i)$ as observation | §III.C; $\beta_t = 0.35$ | **CONFIRMED** | Spec §5 + env cross-ref. |
| EI biomarker **not** in RL state | §II.C objective (single biomarker) | **CONFIRMED** | Env spec §1. |
| Exact tensor shape (1×T vs multi-channel) | Not in paper | **OPEN** | Spec does not over-specify — correct. |

---

## Cross-cutting discrepancies and documentation debt

| ID | Severity | Issue | Recommendation |
|----|----------|-------|----------------|
| D1 | Low | Replay tuple in spec omits discrete **`a`** | Add `a` (pattern id) to §4.2 / §7 buffer fields for Algorithm 1 parity. |
| D2 | Low | Variant slugs only in `benchmarking.md` | Optional table in `ddpg.md` linking slug ↔ §IV experiment. |
| D3 | Info | Paper markets “continuous” DDPG; implementation is discrete | Spec §3.1 already documents; no change required. |
| D4 | Info | Algorithm 1 / Eq. (7) typos in manuscript | Spec already corrects target-actor update. |

---

## Section verdict summary

| # | Section | Overall verdict |
|---|---------|-----------------|
| 1 | Actor architecture | **CONFIRMED** (topology numeric **OPEN**) |
| 2 | Critic architecture | **CONFIRMED** (fusion mechanism **OPEN**) |
| 3 | Replay buffer / sampling | **CONFIRMED** (minor **DISCREPANCY** on storing `a`; `update_frequency` **OPEN**) |
| 4 | Soft-update $\tau$ | **CONFIRMED** ($\tau$ value **OPEN**; target init **OPEN**) |
| 5 | Discount $\gamma$ | **CONFIRMED** ($\gamma$ value **OPEN**) |
| 6 | Algorithm 1 loop | **CONFIRMED** |
| 7 | Quantization PTQ/QAT | **CONFIRMED** (critic QAT scope **OPEN**) |
| 8 | Variants | **CONFIRMED** scientifically; slug table **OPEN** in `ddpg.md` |
| 9 | Hyperparameter labeling | **CONFIRMED** |
| 10 | Env obs ↔ actor inputs | **CONFIRMED** (tensor layout **OPEN**) |

---

## Conclusion

`docs/controllers/ddpg.md` is **suitable as the controller authority** for Mehregan et al. replication: core algorithm, losses, hyperparameters reported in §IV.A.1, initialization frequencies, and quantization narrative match the paper. Gaps are mostly **intentional openness** where the manuscript is silent, plus one **minor replay-field omission** (`a`) and **variant naming** deferred to benchmarking docs.

**Recommended follow-ups (spec edits, optional):**

1. Include optional field `a` (discrete pattern index) in replay buffer spec.  
2. Add a “Variants” subsection mapping `paper` / `init-30hz` / `ptq-int8` / `qat` to paper §IV figures.  
3. One sentence on target-network initialization as implementation convention (paper-silent).
