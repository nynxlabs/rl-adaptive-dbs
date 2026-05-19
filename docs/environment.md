# RL environment specification (Mehregan et al., adaptive DBS / quantization)

This document specifies the **computational reinforcement-learning environment** described in *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning* (Mehregan et al.). It is meant to align a Gymnasium-style implementation under `envs/` (repository root) with the published setup. The **DDPG actor–critic controller** (networks, replay, losses, quantization) is specified separately in [controllers/ddpg.md](controllers/ddpg.md).

The **plant** is the validated **cortex–basal ganglia–thalamus** biophysical network for **6-OHDA–lesioned (parkinsonian) rat**, with **DBS delivered in the STN**. Use the **Kumaravelu, Brocker, and Grill (2016)** biophysical network model (the MATLAB distribution under [`reference-material/KumaraveluEtAl2016/`](../reference-material/KumaraveluEtAl2016/); see [`readme.txt`](../reference-material/KumaraveluEtAl2016/readme.txt) for citation and provenance).

---

## 1. Scope

| In scope | Out of scope (same paper, different system) |
|----------|---------------------------------------------|
| **Simulator-in-the-loop RL** (Algorithm 1: environment steps with transitions stored in a replay buffer) against the **MATLAB / computational** network | **In vivo** rat optogenetics pipeline (trial timing, behavior, recording hardware) |
| **Single biomarker** control using **beta-band power** in GPi | **Error index (EI)** as a *driving* observation (used historically in related work; this study motivates **Pβ only** for practicality) |

---

## 2. Plant (dynamics model)

- **Topology:** Cortical (excitatory / inhibitory), direct and indirect striatum, **STN**, **GPe**, **GPi**, thalamus; stochastic and fixed excitatory/inhibitory inter-region connections as in the Kumaravelu et al. (2016) publication cited in the paper.
- **Units per region:** **10** single-compartment **Hodgkin–Huxley–type** neurons per structure (paper §II.A). §IV.A.1 briefly says “10 neurons” for the whole model; replication uses **10 per region** as in §II.A and the Kumaravelu reference implementation.
- **Pathology:** Parkinsonian (6-OHDA lesioned) regime with exaggerated **beta** oscillations versus healthy controls.
- **Actuation:** DBS **injected in the STN** (paper Figure 1a); compare to conventional **~130 Hz** periodic high-frequency STN DBS as a baseline. **Pulse timing, amplitude, and other waveform details** are not overridden in §IV.A.1; follow the **Kumaravelu et al. (2016)** reference implementation unless a later publication or released code specifies otherwise.

**Implementation note:** Wrapping the reference MATLAB model in a Python RL loop is the expected path until a native reimplementation exists. Mehregan et al. §IV.A.1 reports a plant integration step of **0.02 ms** (see §5). The bundled Kumaravelu et al. (2016) script in `reference-material/KumaraveluEtAl2016/` defaults to **0.01 ms** (`dt` in `simulate_network_model.m`). For paper replication, use **0.02 ms** unless Mehregan’s released training code specifies otherwise; document any deliberate deviation and validate biomarker statistics if you keep the reference default.

---

## 3. Biomarker and observation

### 3.1 Beta power in GPi (primary signal)

Integrated **beta (13–35 Hz)** power over GPi spiking, averaged across the **n = 10** GPi neurons (paper Equation (1)). The paper’s introduction cites literature using **12–35 Hz** in places; **Equation (1) and the computational biomarker use 13–35 Hz**—implementations should match Eq. (1) for replication.

$$
P_\beta = \frac{1}{n} \sum_{j=1}^{n} \int_{\omega = 2\pi \cdot 13\,\mathrm{Hz}}^{2\pi \cdot 35\,\mathrm{Hz}} P_j^{\mathrm{GPi}}(\omega)\, d\omega
$$

where $P_j^{\mathrm{GPi}}$ is the **power spectral density** of the **action potentials** of neuron $j$ in GPi.

### 3.2 State for the RL agent

- The **actor** uses a **temporal window** of biomarker samples (paper §III.B: CNN over time; “Brain” class with **state length** / window parameters).
- **Algorithm 1** in the paper: at each RL step, form state **s** from **$P_\beta$** (after running the plant for the step duration).
- For the **computational experiments** in §IV.A.1, the **biomarker window** for $P_\beta$ is described as spanning the **full length of the simulation segment** used for that step (see §5).

**Optional context (not used as the main RL input in this study):** **Error index (EI)** for thalamic reliability (paper Equation (2)) appears in §II.B for background and comparison with prior work; the **stated RL objective** uses **$P_\beta$ alone**.

---

## 4. Action space

- **Semantic:** A **stimulation pattern** in STN (not only a scalar frequency): the actor emits **logits** over a **discrete** pattern set; **softmax + argmax** (or stored logits for the critic) yields the pattern applied to the plant (paper §III.B). The paper motivates **discrete patterns** for exploration but §IV.A.1 does **not** state the **cardinality** of the pattern alphabet or the exact encoding; define that in code and keep it stable across training / evaluation / quantization comparisons.
- **Initialization:** Policy / pattern family is initialized from **regular pulses** at a target **mean frequency** (e.g. **45 Hz** or **30 Hz** in §IV) to speed learning.
- **Baselines for comparison:** Periodic **45 Hz** stimulation and conventional **130 Hz** cDBS, same random seed where applicable.

---

## 5. Timing and transitions

Values from **§IV.A.1 (computational setup)** unless noted.

| Quantity | Value |
|----------|--------|
| **RL step duration** (`dt_rl` / “duration $l$” in Alg. 1) | **2 s** of simulated time per RL transition |
| **Plant integration step** | **0.02 ms** in Mehregan §IV.A.1 (“step size of 0.02 ms”); bundled Kumaravelu MATLAB defaults to **0.01 ms** — see §2 implementation note |
| **Steps per training episode** | **30** |
| **Episode reset** | New **initial conditions** for the plant; collect $P_\beta$ and reward over each step (Alg. 1) |

**Suggested environment API:**

- `reset()` → new parkinsonian initial conditions (and optional noise seed); integrate for duration **$l$** (2 s) per Algorithm 1 step 7, compute **$P_\beta$** over the biomarker window, form initial observation **$s_0$**, and return it (plus optional initial **$R$** and info). The first `step` then applies an action for another **$l$**.
- `step(action)` → apply selected **pattern** to STN for **2 s** simulated time, integrate the network, compute **$P_\beta$** over the window policy, return observation, reward, terminated, truncated, info.

---

## 6. Reward

Instantaneous reward **$R$** (paper Equation (8)) depends on the **average** of the observed state entries over the observation window:

$$
s_{\mathrm{sum}} = \frac{1}{n_{\mathrm{obs}}} \sum_{i=1}^{n_{\mathrm{obs}}} s(i)
$$

$$
R =
\begin{cases}
\left(s_{\mathrm{sum}} - \beta_t\right) \cdot 10, & s_{\mathrm{sum}} < \beta_t \\
-\left(\left(s_{\mathrm{sum}} - \beta_t\right) \cdot 10\right)^2, & s_{\mathrm{sum}} \ge \beta_t
\end{cases}
$$

- **Threshold:** $\beta_t = 0.35$ (paper §III.C).
- **Intent:** Reward **increases** (is more favorable) as the average biomarker-derived state moves **below** $\beta_t$; **quadratic penalty** when at or above. With raw $P_\beta$ on the order of hundreds (paper Figure 4), Eq. (8) is only meaningful after the **normalization** in the note below—do not assume every favorable transition has $R > 0$.

**Relationship to the paper’s efficiency goal:** The problem statement motivates **both** symptom reduction (beta) and **energy-aware** stimulation (e.g. lower mean frequency vs **130 Hz** cDBS). The **published reward in Eq. (8) depends only on beta-band state vs $\beta_t$**—there is **no separate reward term** for pulse count or instantaneous frequency. Mean-frequency shaping enters through **initialization** (e.g. **45 Hz** / **30 Hz** target), the **learned discrete pattern family**, and **baseline comparisons** in §IV, not through an explicit energy penalty in $R$.

**Implementation note:** The paper’s Figure 4 discusses **raw beta power** on the order of **hundreds** in the uncontrolled model; the published $\beta_t = 0.35$ implies **$s(i)$** are **normalized or preprocessed** consistently between the **observation** pipeline and **reward**. The implementation should apply a single, documented normalization (e.g. scaling $P_\beta$ into the range where $\beta_t = 0.35$ is meaningful) so that reward and logged metrics match the paper’s intent.

---

## 7. Episode termination

- **Training (paper Alg. 1):** After **30** steps, end episode (`terminated=True` or `truncated=True` depending on whether the plant can reach a true terminal dynamical state; typically **truncation at horizon**).
- **`dw` (done flag) for DDPG targets:** 1 if episode finished, else 0 (paper Equation (3)).

---

## 8. Training loop parameters (computational study)

From **§IV.A.1** (for replication / defaults):

| Hyperparameter | Value |
|----------------|--------|
| Actor learning rate | **5×10⁻⁴** |
| Critic learning rate | **1×10⁻³** |
| Replay buffer capacity | **8192** |
| Minibatch size | **32** |
| Training episodes (reported run) | **10** |
| Mean stimulation frequency at init | **45 Hz** (also **30 Hz** experiment, other settings unchanged) |

**Not fixed numerically in §IV.A.1 (still required for DDPG replication):** Algorithm 1 also initializes **discount** $\gamma$, **target-network soft-update** coefficient $\tau$, and an inner-loop **update frequency** (gradient steps per environment step). The computational-setup paragraph does **not** report numeric values for these; match released code if available, or document chosen defaults explicitly in the implementation.

**Evaluation protocol (paper §IV.A.2):** After training, fix **random seed**; run a **10 s** simulation with **2 s** for reset / baseline, then **5** repeated applications of the stimulation **step** for comparison across models and quantization variants. The paper also describes a **2 s** GPi baseline before applying actor actions (Figure 5). If each eval segment uses the training step duration **$l = 2$ s**, five segments plus a 2 s reset exceed **10 s** total—**intentionally open** until released code or a project convention fixes segment length vs. the reported **10 s** wall time. Cross-controller use of this protocol (baselines, metrics, run identity) is defined in [benchmarking.md](benchmarking.md).

---

## 9. Optional extensions (same publication)

- **Quantization:** PTQ / QAT on actor–critic networks affects **inference-time** policy outputs; the **environment interface** (plant, $P_\beta$, timing, reward) is unchanged.
- **Animal validation:** Not part of the computational `Env` API; documented separately if this repo later adds experiment logs or replay buffers for **in vivo** trials.

---

## 10. Minimal consistency checklist

- [ ] STN-injected patterns match discrete action set used during training.
- [ ] **2 s** simulated time per RL step, **30** steps per episode (training config).
- [ ] **$P_\beta$** from **GPi** spikes, **13–35 Hz**, mean over **10** neurons.
- [ ] Reward uses **$s_{\mathrm{sum}}$** vs **$\beta_t = 0.35$** with documented scaling of **$s$**.
- [ ] Baselines: **no stimulation**, **130 Hz** cDBS, periodic **45 Hz** (and **30 Hz** where relevant), same seeds for fair comparison.
- [ ] DDPG: **$\gamma$, $\tau$,** and **updates per env step** documented (not numerically fixed in §IV.A.1 of the paper).

---

## 11. References

- Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*.
- Kumaravelu K, Brocker DT, Grill WM (2016), *A biophysical model of the cortex–basal ganglia–thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease*, *J Comput Neurosci* 40:207–29 — bundled MATLAB model: [`reference-material/KumaraveluEtAl2016/`](../reference-material/KumaraveluEtAl2016/) ([`readme.txt`](../reference-material/KumaraveluEtAl2016/readme.txt)).

---

## 12. Future direction: native Python plant (optional)

The project may **eventually replace the MATLAB wrapper** with a **native Python** reimplementation of the same biophysical network (or a subset validated against it), for example to remove the MATLAB dependency, simplify CI, or improve batching and deployment. If that happens, treat the reference MATLAB simulator as the **source of truth** until a port passes **equivalence checks** (time step, dynamics, and biomarker statistics under the same protocols).
