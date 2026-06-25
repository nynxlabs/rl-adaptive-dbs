# DDPG extensions — post-replication directions

*Exploratory work beyond Mehregan et al. paper replication. See [replication.md](replication.md) for the paper-accurate spec.*

---

## 1. Quantization deeper dives

Paper-aligned **PTQ** and **QAT** for Mehregan et al. are implemented in [replication.md](replication.md) §6 (`controllers/ddpg/quantization.py`). This section covers extensions **beyond** §IV.A.3:

- **Mixed-precision quantization** — different bitwidths per layer based on sensitivity analysis.
- **Quantization-aware fine-tuning** — start from a PTQ checkpoint and fine-tune with fake quantization for fewer training episodes than full QAT.
- **Deployment profiling** — measure actual inference latency and memory footprint on representative hardware (e.g. ARM Cortex-M, neuromorphic chips) to ground the paper's efficiency claims.

## 2. Reward shaping and multi-objective control

Mehregan et al. Eq. (8) uses beta-only reward with threshold $\beta_t = 0.35$. Possible extensions:

- **Explicit energy penalty** — add a term for stimulation frequency or pulse count to the reward, addressing the paper's stated motivation of energy-aware stimulation that is currently only implicit via initialization.
- **Multi-objective Pareto front** — trade off symptom reduction (beta power) against stimulation energy and side-effect proxies.
- **Adaptive thresholds** — learn or schedule $\beta_t$ rather than fixing it, accommodating patient-specific baselines.

## 3. Policy architecture exploration

The paper specifies a CNN-over-temporal-state actor but gives no layer dimensions. Beyond replication:

- **Recurrent policies** (LSTM/GRU) to capture longer temporal dependencies in the biomarker trajectory.
- **Transformer-based attention** over the observation window so the policy learns which time steps matter most.
- **Wider pattern alphabets** with structured encoding (e.g. parametric patterns rather than a fixed discrete set).

## 4. Training regime improvements

- **Longer training runs** — the paper reports only 10 episodes; systematic sweeps over episode count to find convergence behavior.
- **Curriculum learning** — start with easier PD dynamics (lower beta amplitude) and progressively increase severity.
- **Transfer across PD severity levels** — pretrain on one severity and fine-tune on another without full retraining.

---

*Each direction should be scoped as a named **variant** in the benchmarking framework (see [benchmarking.md](../../benchmarking.md) §2) when implemented.*
