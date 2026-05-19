# Fusion — synergistic synthesis of SEA-DBS and DSQN

*Post-replication research direction. This page defines the long-term goal of combining Ravivarapu et al.'s sample-efficient techniques with Nguyen et al.'s neuromorphic spiking architecture. Neither system is a prerequisite for the other; both should be replicated individually before fusion work begins.*

---

## 1. Motivation

SEA-DBS and DSQN each address a different bottleneck in adaptive DBS:

| System | Key strength | Key limitation |
|--------|-------------|----------------|
| **SEA-DBS** (Ravivarapu et al.) | Sample efficiency — predictive reward model + Gumbel-Softmax exploration | Conventional (non-spiking) networks; higher inference energy |
| **DSQN** (Nguyen et al.) | Energy efficiency — spiking neural network with neuromorphic deployment potential | Sample-hungry tabular DQN-style learning; no predictive model |

Fusion aims to unify both strengths: **sample-efficient learning** (few real CBGT interactions) within an **energy-efficient spiking architecture** (low-power inference for implantable deployment).

---

## 2. Hierarchical neuromorphic controller

The primary fusion architecture:

### 2.1 Fast spiking gatekeeper

- A lightweight **SNN binary gate** that decides whether to stimulate ($a \in \{0, 1\}$) at each step.
- Trained with the SEA-DBS **Gumbel-Softmax** exploration strategy adapted for spiking outputs (spike-count argmax for execution, membrane potentials for gradient flow).
- Minimal energy cost — a small network making a single binary decision.

### 2.2 Spiking parameter-tuning network

- Invoked **only when stimulation is active** (gatekeeper outputs 1).
- A deeper **DSQN-style SNN** that jointly optimizes amplitude, frequency, and pulse width via the 9-action discretized output space (3 parameters × 3 ternary choices).
- Amortizes energy cost — parameter tuning only runs when needed, not every step.

### 2.3 Predictive reward model for SNN training

- Adapt the SEA-DBS **predictive model** $f_\theta$ to generate synthetic training signals for the spiking networks.
- Reduces the number of real CBGT interactions needed to train the SNN policy — addressing DSQN's sample-hungry nature.
- The predictive model itself can be a conventional (non-spiking) network since it runs offline during training, not at inference time.

---

## 3. Reward structure

The fusion reward should unify the strengths of both systems:

- **Oscillation control** — gatekeeper tracks **GPi beta-band power** (13–35 Hz) as in SEA-DBS; parameter tuning uses **GPi $\alpha$–$\beta$ power** (7–35 Hz) as in Nguyen et al. Do not treat these as interchangeable without an explicit mapping in the adapter.
- **DBS energy** — explicit stimulation energy term from Nguyen et al. Eq. (6), rewarding the gatekeeper for keeping stimulation off when unnecessary.
- **SNN inference energy** — penalize spike count in the parameter-tuning network to encourage sparse, efficient computation.
- **Augmented Q-target** — use the predictive model to bootstrap $r + \hat{r} + \gamma Q'$ (SEA-DBS Eq. (9)) for faster SNN policy learning.

---

## 4. Observation and action spaces

| Component | Observation | Action |
|-----------|-------------|--------|
| **Gatekeeper** | GPi **beta-band** power window (Ravivarapu et al. Eqs. (4)–(5); **13–35 Hz**, same $P_\beta$ as Eq. (1)) | Binary $\{0, 1\}$ — pulse vs. no pulse (Ravivarapu Eq. (6)) |
| **Parameter network** | Spike matrix from CBGT regions (Nguyen et al. Eq. (4)) | 9-way ternary per-parameter deltas (Nguyen et al. Eq. (5)) |

The gatekeeper uses beta-band power (efficient, low-dimensional) for fast decisions. The parameter network uses full spike observations (rich, high-dimensional) only when stimulation is active.

---

## 5. Training algorithm

1. **Phase 1 — Gatekeeper pretraining:** Train the binary gate SNN with GS exploration on the beta-band state, using the predictive reward model for augmented targets.
2. **Phase 2 — Parameter network training:** With the gatekeeper fixed, train the parameter-tuning SNN on spike observations, conditioned on the gatekeeper's on/off decisions.
3. **Phase 3 — Joint fine-tuning:** Unfreeze both networks and train end-to-end with the unified reward, allowing the gatekeeper and parameter network to co-adapt.

---

## 6. Evaluation targets

- **Beta suppression** — comparable to or better than either system alone.
- **Stimulation energy** — lower than SEA-DBS (gatekeeper keeps stimulation off more often).
- **Sample efficiency** — fewer real CBGT interactions than standalone DSQN (predictive model accelerates SNN learning).
- **Inference energy** — lower than standalone SEA-DBS (spiking networks at inference time).

---

## 7. Broader directions

Beyond the hierarchical architecture:

- **Generalized adaptive neuromodulation framework** — design the fusion as a modular system that can integrate additional RL-based DBS controllers beyond SEA-DBS and DSQN.
- **Extension to other oscillatory conditions** — adapt the framework to essential tremor, epilepsy, or dystonia by changing the target biomarker and reward structure.
- **On-device learning** — explore continual on-chip learning rules for the SNN components as neuromorphic hardware matures.
- **Rechargeable implantable systems** — study the system-level implications of spiking inference on battery life and recharge cycles.

---

*Fusion work should be tracked as a separate benchmarking suite (e.g. `fusion_eval`) once both individual replications are complete. See [../benchmarking.md](../benchmarking.md) §2 for variant naming conventions.*
