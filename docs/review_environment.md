# Environment spec review: `docs/environment.md` vs Mehregan et al.

**Review date:** 2026-05-18  
**Sources compared:**
- Spec: [`docs/environment.md`](environment.md)
- Paper notes: `/home/devat/knowledge-base/neuroengineering/brain-stimulation-engineering/effort/papers/paper_1.md` (Mehregan et al.)
- Reference plant: [`reference-material/KumaraveluEtAl2016/`](../reference-material/KumaraveluEtAl2016/) (`simulate_network_model.m`, `readme.txt`)

**Legend:** **CONFIRMED** — spec matches the paper (and reference where applicable). **DISCREPANCY** — spec differs from paper or reference; detail noted. **OPEN** — paper is silent or ambiguous; spec’s “intentionally open” label assessed.

---

## Executive summary

`docs/environment.md` is **largely faithful** to Mehregan et al. for the computational RL environment. Numeric training hyperparameters, timing, biomarker definition (Eq. 1), reward (Eq. 8), and action semantics (discrete patterns via softmax + argmax) all match §III–§IV.A.

The spec **correctly surfaces** several genuine ambiguities in the paper (normalization of \(s(i)\) vs raw \(P_\beta\), eval segment arithmetic, pattern alphabet size, \(\gamma\)/\(\tau\)/update frequency). It also **correctly documents** one important plant detail the paper does not mention: Mehregan §IV.A.1 uses **0.02 ms** integration step while the bundled Kumaravelu MATLAB defaults to **0.01 ms**.

Notable gaps or nuances **not fully called out** in the spec:
1. The Kumaravelu reference integrates beta power over **7–35 Hz**, not **13–35 Hz** as in Mehregan Eq. (1).
2. The paper’s §II.A “Hodgkin–Huxley type” description is a simplification: the reference uses **Izhikevich-style** cortex dynamics, not full HH everywhere.
3. §IV.A.1’s “**10 neurons**” (whole model) contradicts §II.A’s “**10 per region**”; the spec resolves this sensibly but the paper itself is inconsistent.
4. The paper’s verbal reward description (“positive below threshold”) is **in tension** with Eq. (8) unless \(s(i)\) uses an unstated invert/normalize transform—the spec notes normalization but could state more explicitly that the paper never defines the mapping.

---

## §1 Scope

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Computational simulator-in-the-loop RL; not in vivo | **CONFIRMED** | Paper §IV.A vs §IV.B; Alg. 1 |
| Single biomarker \(P_\beta\); EI out of scope as RL input | **CONFIRMED** | §II.B–C: EI background only; RL uses beta alone |
| Kumaravelu et al. (2016) as plant reference | **CONFIRMED** | Paper [32], §II.A; `readme.txt` citation |

---

## §2 Plant (dynamics model)

### Topology and pathology

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Cortex (exc/inh), direct/indirect striatum, STN, GPe, GPi, thalamus | **CONFIRMED** | Paper §II.A; `CTX_BG_TH_network` outputs match |
| Stochastic and fixed excitatory/inhibitory inter-region connections | **CONFIRMED** | Paper §II.A; reference uses stochastic connection draws (e.g. `gsngen`, `gsngea`) and fixed synaptic kernels |
| 6-OHDA parkinsonian regime with exaggerated beta | **CONFIRMED** | Paper §II.A; `pd=1` in reference |
| DBS injected in STN | **CONFIRMED** | Paper §II.A, Fig. 1a; `Idbs` added to STN equation (line 622) |
| ~130 Hz cDBS as conventional baseline | **CONFIRMED** | Paper §II.B, Fig. 1b, §IV.A.2 |

### Neuron count and model type

| Claim | Verdict | Evidence |
|-------|---------|----------|
| **10 neurons per region** (replication choice) | **CONFIRMED** (with paper internal caveat) | §II.A: “Each region is simulated by **10** single-compartment Hodgkin–Huxley type neurons.” Reference: `n = 10; % number of neurons in each nucleus` |
| §IV.A.1 “10 neurons” for whole model | **DISCREPANCY (in paper, not spec)** | §IV.A.1: “The biophysical model simulates **10 neurons**.” Likely editorial slip; spec §2 correctly prefers §II.A + reference |
| Hodgkin–Huxley–type neurons per structure | **DISCREPANCY (paper vs reference)** | Paper §II.A says HH-type for all regions. Reference: STN/GPe/GPi/thalamus use multi-compartment HH-style gating; **cortex uses Izhikevich** (`0.04*v^2+5v+140`, lines 759–793); striatum uses reduced HH-like kinetics. Spec does not mention this mixed-model detail |

### Integration step and DBS waveform

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Mehregan plant step **0.02 ms** | **CONFIRMED** | §IV.A.1: “step size of **0.02 ms**” |
| Kumaravelu default **0.01 ms** | **CONFIRMED** | `simulate_network_model.m` line 12: `dt = 0.01` |
| Spec recommends 0.02 ms for paper replication | **CONFIRMED** (reasonable) | Paper explicit; reference is upstream default—spec §2 note is appropriate |
| Pulse timing/amplitude follow Kumaravelu unless overridden | **OPEN** | Paper §IV.A.1 silent on PW/amplitude. Reference: `PW = 0.3` ms, `amplitude = 300` nA/cm². Spec correctly defers to reference |

---

## §3 Biomarker and observation

### §3.1 Beta power (Eq. 1)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| GPi beta-band power, **13–35 Hz** | **CONFIRMED** | Paper §II.B Eq. (1), §II.B text “beta oscillations (13–35 Hz)” |
| Mean over **n = 10** GPi neurons | **CONFIRMED** | Eq. (1): “n is the total number of neurons in the GPi region”; §II.A: 10 per region |
| PSD of **action potentials** (spikes) | **CONFIRMED** | Eq. (1): “power spectral density of action potentials”; reference uses `find_spike_times` → multitaper spectrum |
| Intro literature **12–35 Hz** vs Eq. (1) **13–35 Hz** | **CONFIRMED** (spec handles correctly) | Paper §I: “12–35 Hz”; Eq. (1): 13–35 Hz. Spec §3.1 directs implementers to Eq. (1) |

**Reference vs paper band:**

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Kumaravelu bundled script uses **7–35 Hz** for integrated GPi power | **DISCREPANCY (reference vs paper; spec partially noted)** | `make_Spectrum`: `beta = S(f>7 & f<35)` (lines 1017–1019). Spec mentions 13–35 for replication but does **not** explicitly flag the 7–35 Hz default in the reference biomarker pipeline |

### §3.2 State for RL agent

| Claim | Verdict | Evidence |
|-------|---------|----------|
| CNN temporal window over biomarker samples | **CONFIRMED** | §III.B: CNN chosen for temporal biomarker; “state length” parameter |
| State **s** formed from \(P_\beta\) each step (Alg. 1) | **CONFIRMED** | Alg. 1 steps 7–8, 11–12 |
| Computational biomarker window = **full simulation segment** per step | **CONFIRMED** | §IV.A.1: “window size was set to the **length of the simulation**” (2 s per step) |
| Numeric **state length** / CNN dimensions | **OPEN** | §III.B names parameters but §IV.A.1 gives no numbers. Spec does not claim a value—appropriate |
| EI (Eq. 2) optional context only | **CONFIRMED** | §II.B–C |

**Abstract “discretization of state … spaces”:** Paper abstract mentions state discretization; **no computational detail** in §III–IV.A. **OPEN** — genuinely ambiguous; spec does not over-specify.

---

## §4 Action space

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Discrete **stimulation patterns** in STN (not scalar frequency alone) | **CONFIRMED** | §III.B; abstract “stimulation patterns”; §IV discusses irregular learned patterns |
| Actor outputs **logits**; **softmax + argmax** selects pattern | **CONFIRMED** | §III.B: “action logits … softmax and argmax operations generate the full action” |
| Logits stored for critic (\(a_{\mathrm{logit}}\) in Alg. 1) | **CONFIRMED** | Alg. 1 step 13; Eq. (4) uses \(a_{\mathrm{logit}}\) |
| Pattern alphabet **cardinality / encoding** unspecified | **OPEN** (genuinely ambiguous) | §III.B mentions “action size”; §IV.A.1 silent. Spec §4 correctly marks open |
| Init from **regular pulses** at mean frequency (45 Hz primary; 30 Hz ablation) | **CONFIRMED** | Alg. 1 input “Average frequency of action f”; §IV.A.1 init 45 Hz; §IV.A.2 30 Hz experiment |
| Baselines: periodic **45 Hz**, **130 Hz** cDBS, same seed | **CONFIRMED** | §IV.A.2 |

**Note:** Paper names the algorithm DDPG but applies **discrete** argmax actions—a naming quirk in the source, not an env-spec error (handled in `controllers/ddpg.md`).

---

## §5 Timing and transitions

| Claim | Verdict | Evidence |
|-------|---------|----------|
| RL step duration **2 s** (\(l\) in Alg. 1) | **CONFIRMED** | §IV.A.1: “**2 seconds** worth of data per step”; Alg. 1 steps 7, 11 |
| Plant integration step **0.02 ms** (Mehregan) | **CONFIRMED** | §IV.A.1 |
| **30 steps per training episode** | **CONFIRMED** | §IV.A.1: “**30 steps per episode**” |
| Episode reset → new initial conditions | **CONFIRMED** | Alg. 1 step 7; §III.B end of episode reset |
| Suggested `reset()` / `step()` API | **OPEN** (implementation convention) | Reasonable reading of Alg. 1; paper does not define a Gym API. Not a faithfulness issue |

**Training episode length:** 30 × 2 s = **60 s** simulated time per episode—derivable from paper; spec implies but does not state total explicitly (minor omission, not a discrepancy).

---

## §6 Reward (Eq. 8)

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Piecewise Eq. (8) with \(s_{\mathrm{sum}} = \frac{1}{n_{\mathrm{obs}}}\sum s(i)\) | **CONFIRMED** | §III.C Eq. (8)—spec formula matches (using \(s_{\mathrm{sum}}\) notation consistently) |
| Threshold **\(\beta_t = 0.35\)** | **CONFIRMED** | §III.C: “The value \(\beta_t\) was set to **0.35**.” |
| Linear branch for \(s_{\mathrm{sum}} < \beta_t\); quadratic penalty for \(s_{\mathrm{sum}} \ge \beta_t\) | **CONFIRMED** | Eq. (8); Fig. 3c |
| No separate energy/frequency term in \(R\) | **CONFIRMED** | Eq. (8) beta-only; energy efficiency via patterns/init/baselines (§II.C, §IV)—spec §6 “Relationship to efficiency goal” is accurate |
| Raw \(P_\beta\) ~400–500 uncontrolled (Fig. 4) vs \(\beta_t = 0.35\) | **OPEN** (genuinely ambiguous) | Paper never defines normalization \(s(i) = f(P_\beta)\). Spec §6 implementation note is **required** and correctly flagged |
| Verbal “positive reward below threshold” vs Eq. (8) sign | **OPEN / internal paper tension** | §III.C text vs formula: if \(s(i)\) were raw beta power, \((s_{\mathrm{sum}}-\beta_t)\cdot 10 < 0\) when below threshold. Fig. 4 shows reward **increasing** as beta **decreases** → \(s\) must be monotone-decreasing in beta or inverted. **Paper silent on transform**—spec should treat normalization as blocking for replication |

---

## §7 Episode termination

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Training ends after **30** steps | **CONFIRMED** | §IV.A.1; Alg. 1 loop |
| **`dw`** done flag for DDPG targets | **CONFIRMED** | Alg. 1 step 11; Eq. (3) |

---

## §8 Training hyperparameters and eval protocol

### §IV.A.1 training table

| Hyperparameter | Spec value | Verdict | Paper source |
|----------------|------------|---------|--------------|
| Actor LR | 5×10⁻⁴ | **CONFIRMED** | §IV.A.1: “actor learning rate at **0.0005**” |
| Critic LR | 1×10⁻³ | **CONFIRMED** | “critic learning rate at **0.001**” |
| Replay buffer | 8192 | **CONFIRMED** | “buffer size is set to **8,192**” |
| Minibatch | 32 | **CONFIRMED** | “batch size is set to **32**” |
| Training episodes | 10 | **CONFIRMED** | “training occurs over **10 episodes**” |
| Init mean frequency | 45 Hz (30 Hz ablation) | **CONFIRMED** | §IV.A.1; §IV.A.2 / Fig. 5b |

### Not reported in §IV.A.1

| Item | Verdict | Evidence |
|------|---------|----------|
| Discount **\(\gamma\)** | **OPEN** (genuinely ambiguous) | Alg. 1 step 2 initializes \(\gamma\); no numeric value in §IV.A.1 |
| Target soft-update **\(\tau\)** | **OPEN** | Alg. 1 step 1; Eqs. (6)–(7); no numeric value |
| Inner **update frequency** | **OPEN** | Alg. 1 step 15; no numeric value |
| CNN architecture (“shrink dimension”, etc.) | **OPEN** | §III.B only |

Spec §8 “Not fixed numerically” list is **accurate**—these are not “not yet resolved by the project”; they are **absent from the paper**.

### §IV.A.2 evaluation protocol

| Claim | Verdict | Evidence |
|-------|---------|----------|
| Fixed random seed after training | **CONFIRMED** | §IV.A.2 |
| **10 s** total simulation | **CONFIRMED** | §IV.A.2: “The simulation lasts **10 seconds**” |
| **2 s** for reset / data reset | **CONFIRMED** | §IV.A.2: “**2 seconds** for data reset” |
| **5** repetitions of stimulation step | **CONFIRMED** | §IV.A.2: “**five repetitions** of the step function” |
| **2 s** GPi baseline before actor actions (Fig. 5) | **CONFIRMED** | §IV.A.2: “Initially, the power is calculated … for **two seconds**. Subsequently, actions … are applied.” |
| Compare RL vs **45 Hz** periodic and **130 Hz** cDBS | **CONFIRMED** | §IV.A.2 |

### Eval timing arithmetic (spec “intentionally open”)

| Issue | Verdict | Analysis |
|-------|---------|----------|
| 2 s reset + 5 × 2 s steps = **12 s** vs reported **10 s** | **OPEN** (genuine paper ambiguity) | Paper gives all three numbers without reconciling. Possible resolutions: (a) eval steps ≠ 2 s → **1.6 s** each fits \(2 + 5×1.6 = 10\) s; (b) 2 s baseline is **inside** the 10 s window with fewer than five full RL steps; (c) typo in “five” or “ten”. **Spec is correct** to flag this—not a project oversight |
| Whether “2 s reset” equals Fig. 5 “2 s baseline before actions” | **OPEN** | Plausible same segment; paper does not state explicitly |

---

## §9–§12 (extensions, checklist, references, future Python plant)

| Section | Verdict | Notes |
|---------|---------|-------|
| Quantization unchanged env interface | **CONFIRMED** | §III.D; §IV.A.3 |
| Animal validation out of scope | **CONFIRMED** | §IV.B |
| Checklist items | **CONFIRMED** | Align with §IV.A.1 and Eq. (1)/(8) |
| Kumaravelu citation / path | **CONFIRMED** | Paper [32]; repo layout |
| Native Python plant as future work | N/A | Project roadmap; not a paper claim |

---

## Intentionally open items — assessment

| Spec item | Genuinely ambiguous in paper? | Assessment |
|-----------|-------------------------------|------------|
| Pattern alphabet size / encoding | **Yes** | Only “action size” in Brain class; no cardinality in §IV.A.1 |
| \(s(i)\) normalization vs raw \(P_\beta\) | **Yes** | Fig. 4 vs \(\beta_t=0.35\); no formula for \(s\) |
| \(\gamma\), \(\tau\), update frequency | **Yes** | Initialized in Alg. 1, never numerically reported |
| Eval 10 s vs 2 s + 5×2 s | **Yes** | Arithmetic conflict in §IV.A.2 |
| CNN / state-length hyperparameters | **Yes** | §III.B qualitative only |
| DBS pulse PW/amplitude for Mehregan runs | **Yes** | Deferred to Kumaravelu in practice |
| 0.01 ms (reference) vs 0.02 ms (Mehregan) | **Partially** | Mehregan states 0.02 ms; does not say they changed Kumaravelu defaults—spec correctly treats 0.02 as paper authority |
| Gym `reset()`/`step()` API | **N/A (project convention)** | Reasonable; not claimed as paper text |

**Verdict:** All spec “intentionally open” markers reviewed are **justified**—they reflect real gaps or conflicts in the source paper, not merely unfinished repo work.

---

## Reference MATLAB cross-check summary

| Quantity | Kumaravelu reference | Mehregan paper | Spec |
|----------|---------------------|----------------|------|
| Neurons per nucleus | 10 | 10 per region (§II.A); “10” total (§IV.A.1 slip) | 10 per region ✓ |
| `dt` | 0.01 ms | 0.02 ms | Notes both ✓ |
| Beta integration band | 7–35 Hz | 13–35 Hz (Eq. 1) | 13–35 Hz ✓ (reference band not highlighted) |
| DBS site | STN | STN | ✓ |
| Default sim length in script | 2000 ms (2 s) | 2 s per RL step | ✓ |
| Biomarker | GPi spike spectrum | GPi AP spectrum Eq. (1) | ✓ |

---

## Recommended spec adjustments (optional)

These are **review findings**, not applied edits to `environment.md`:

1. **§3.1:** Add explicit note that Kumaravelu `make_Spectrum` integrates **7–35 Hz**, so reimplementations wrapping the reference must **re-band** to 13–35 Hz for Mehregan fidelity.
2. **§2:** One sentence that the reference mixes **HH (BG/thalamus)** and **Izhikevich (cortex)** dynamics despite the paper’s uniform “HH-type” wording.
3. **§6:** State clearly that the paper’s prose (“positive reward below threshold”) requires an **unspecified** monotone transform from raw \(P_\beta\) to \(s(i)\); Eq. (8) alone is underdetermined for replication.
4. **§8 eval:** Mention the **1.6 s-per-step** resolution (\(2 + 5×1.6 = 10\) s) as a leading hypothesis when released code is unavailable.

---

## Section verdict rollup

| Spec section | Overall |
|--------------|---------|
| §1 Scope | **CONFIRMED** |
| §2 Plant | **CONFIRMED** (minor: mixed neuron models; reference beta band) |
| §3 Biomarker / observation | **CONFIRMED** (reference 7–35 Hz worth noting) |
| §4 Action space | **CONFIRMED** + justified **OPEN** items |
| §5 Timing | **CONFIRMED** |
| §6 Reward | **CONFIRMED** formula/threshold; **OPEN** normalization |
| §7 Termination | **CONFIRMED** |
| §8 Training + eval | **CONFIRMED** numerics; **OPEN** eval timing conflict |
| §9–§12 | **CONFIRMED** / N/A |

**Bottom line:** `docs/environment.md` is a **faithful and appropriately cautious** translation of Mehregan et al.’s computational environment. Remaining replication risk sits where the paper is incomplete (normalization, pattern set, eval segment length, DDPG hyperparameters \(\gamma\)/\(\tau\)) and where the Kumaravelu bundle **differs in integration step and beta band** from Mehregan’s stated setup.
