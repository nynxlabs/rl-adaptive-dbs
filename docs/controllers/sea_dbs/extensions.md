# SEA-DBS extensions — post-replication directions

*Exploratory work beyond Ravivarapu et al. paper replication. See [replication.md](replication.md) for the paper-accurate spec.*

---

## 1. Robustness across patient-specific dynamics

Ravivarapu et al. trains and evaluates on a single CBGT parameterization. Extensions:

- **Patient-specific fine-tuning** — pretrain on a canonical PD model, then fine-tune on individualized neural dynamics with minimal data.
- **Domain randomization** — vary CBGT parameters (coupling strengths, lesion severity, baseline beta) across episodes to learn a policy that generalizes.
- **Robust control** — adversarial training or distributional RL to handle worst-case neural dynamics.

## 2. Adaptation to biomarker drift

Real neural signals drift over time due to medication cycles, disease progression, and circadian rhythms. Extensions:

- **Online adaptation** — continue updating the policy post-deployment with a small replay buffer of recent real interactions.
- **Change-point detection** — monitor the predictive reward model's error as a drift detector; trigger retraining or adaptation when error spikes.
- **Non-stationary evaluation** — extend Ravivarapu's seed-change experiments (Table II) with more realistic drift models (gradual rather than abrupt).

## 3. Predictive reward model for model-based planning

Ravivarapu et al. uses the predictive model $f_\theta$ only as a learning accelerator (augmented Q-target). Extensions:

- **Model-predictive control (MPC)** — use $f_\theta$ to simulate lookahead trajectories and select actions that minimize predicted cost over a planning horizon.
- **Dyna-style integration** — generate synthetic experience from $f_\theta$ to augment the real replay buffer, reducing required CBGT interactions.
- **Uncertainty-aware planning** — ensemble the predictive model or add calibrated uncertainty estimates to avoid overcommitting to inaccurate predictions.

## 4. Episodic memory for rare pathological events

Standard replay buffers decay and forget infrequent but clinically important events (e.g. sudden beta bursts). Extensions:

- **Prioritized experience replay** — weight sampling by TD error or surprise to retain rare high-error transitions.
- **Episodic memory module** — a separate, smaller buffer that permanently stores rare substantial pathological events and resamples them during training.
- **Hindsight experience replay (HER)** — relabel failed episodes (where beta remained high) with alternative goals to extract learning signal from otherwise wasted transitions.

## 5. Carrier frequency and waveform optimization

Ravivarapu evaluates fixed carrier frequencies (30 Hz, 50 Hz) at inference. Extensions:

- **Learned carrier frequency** — extend the binary pulse action to include carrier frequency as a continuous or discretized RL action dimension.
- **Waveform shape exploration** — beyond rectangular pulses, study how different waveform morphologies (biphasic, charge-balanced) affect the learned policy.

---

*Each direction should be scoped as a named **variant** in the benchmarking framework (see [benchmarking.md](../../benchmarking.md) §2) when implemented.*
