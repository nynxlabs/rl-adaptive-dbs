# SNN extensions — post-replication directions

*Exploratory work beyond Nguyen et al. paper replication. See [replication.md](replication.md) for the paper-accurate spec.*

---

## 1. Deeper spiking network architectures

The replication uses a three-layer LIF network (input → 128 hidden → 9 output). Extensions:

- **Deeper SNNs** — add hidden layers to increase representational capacity; study how depth affects spike efficiency and learning stability.
- **Recurrent spiking layers** (e.g. SRNN) to capture temporal dynamics in spike trains without explicit observation windowing.
- **Surrogate gradient methods** — replace the simple LIF membrane potential Q-estimates with surrogate-gradient-trained SNNs for potentially sharper credit assignment.

## 2. Continuous action spaces

Nguyen et al. uses 9 discrete outputs (3 parameters × 3 ternary choices). Extensions:

- **Continuous amplitude/frequency/pulse-width control** via spiking policy gradient methods, enabling finer DBS parameter adjustments than ternary deltas.
- **Hybrid discrete-continuous** — discrete on/off gate (like SEA-DBS) combined with continuous parameter output when stimulation is active.

## 3. Learned attention over brain regions

The current spec treats all CBGT regions in the spike observation equally. Extensions:

- **Attention mechanisms** that learn which brain regions carry the most informative spike patterns for the control task.
- **Region dropout** — randomly mask subsets of regions during training to encourage robustness and reveal which regions the policy depends on.
- **Hierarchical encoding** — separate encoders per brain region with a learned aggregation layer.

## 4. Transfer learning across PD severity

- **Pretrain on one severity level, fine-tune on another** — study whether learned SNN policies transfer across different 6-OHDA lesion intensities or beta oscillation profiles.
- **Meta-learning** (e.g. MAML-style) for rapid adaptation to new patient-specific dynamics with few gradient steps.
- **Domain randomization** during training — vary PD severity parameters episode-to-episode to build robustness.

## 5. Energy-aware reward refinement

Nguyen et al. Eq. (7) includes an energy term but the coefficients are unspecified. Extensions:

- **Explicit neuromorphic energy budget** — constrain SNN inference energy (spike count × per-spike cost) alongside DBS stimulation energy.
- **Pareto-optimal policies** across symptom reduction and total energy (stimulation + computation).
- **Hardware-in-the-loop energy measurement** — pair with neuromorphic chip simulators (e.g. Lava, Norse) to get realistic energy numbers.

---

*Each direction should be scoped as a named **variant** in the benchmarking framework (see [benchmarking.md](../../benchmarking.md) §2) when implemented.*
