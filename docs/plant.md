# Plant specification (Kumaravelu et al., 2016 CBGT model)

This document is the **authoritative spec for the shared biophysical plant**: the **cortex–basal ganglia–thalamus (CBGT)** network for the **6-OHDA–lesioned (parkinsonian) rat**, with **DBS delivered in the STN**. All controllers and adapters drive **one** instance of this dynamics model—no duplicated CBGT code under `controllers/`.

**Related specs:**

- **Mehregan Gymnasium API** (2 s RL steps, pattern actions, Eq. (8) reward, baselines) — [environment.md](environment.md).
- **Per-controller RL interfaces** (timing, observations, rewards) — [controllers/](controllers/).
- **Equivalence testing** — [development/testing.md](development/testing.md) §3 (`@pytest.mark.matlab`).

---

## 1. Scope

| In scope | Out of scope |
|----------|----------------|
| Kumaravelu et al. (2016) **network topology**, neuron models, pathology flag, **STN DBS** waveform parameters in the reference script | Mehregan **RL** step duration, reward Eq. (8), discrete pattern alphabet cardinality ([environment.md](environment.md)) |
| **Plant integration** ($\Delta t$), simulated-time segments, IC reset | DDPG / DSQN / SEA-DBS **training loops** ([controllers/](controllers/)) |
| **Biomarker primitives** from GPi spiking (PSD, band integrals); which band applies is **suite- or adapter-specific** | Benchmark **run manifests** and cross-paper tables ([benchmarking.md](benchmarking.md)) |

The plant layer exposes **dynamics + actuation + spike/biomarker outputs**. The `envs/` package wraps it with the Mehregan RL contract; Nguyen and SEA-DBS adapters subsample or re-window the same plant without reimplementing dynamics.

---

## 2. Reference and provenance

- **Publication:** Kumaravelu, K., Brocker, D. T., Grill, W. M. (2016). *A biophysical model of the cortex–basal ganglia–thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease.* *Journal of Computational Neuroscience*, 40, 207–229.
- **Bundled MATLAB:** [`reference-material/KumaraveluEtAl2016/`](../reference-material/KumaraveluEtAl2016/) — entry script `simulate_network_model.m`, network routine `CTX_BG_TH_network` in the same file. Citation and provenance: [`readme.txt`](../reference-material/KumaraveluEtAl2016/readme.txt).
- **Upstream:** [ModelDBRepository/206232](https://github.com/ModelDBRepository/206232).

**Replication default:** **Parkinsonian** runs use `pd = 1` in the reference script (`pd = 0` is healthy). RL work in this repo targets the **parkinsonian** regime unless a benchmark explicitly includes healthy controls.

---

## 3. Network topology and dynamics

- **Regions:** Cortex (excitatory / inhibitory populations), direct and indirect striatum, **STN**, **GPe**, **GPi**, thalamus; inter-region connections as in Kumaravelu et al. (2016).
- **Units per region:** **$n = 10$** neurons per structure in the reference implementation (`n = 10` in `simulate_network_model.m`).
- **Neuron models:** The reference uses **Izhikevich** dynamics for cortical populations and **Hodgkin–Huxley–type** dynamics for basal ganglia and thalamus. Adaptive DBS papers often describe the model uniformly as “HH-type”; **match the reference implementation** for fidelity, not the prose shorthand alone.
- **Pathology:** `pd = 1` selects parkinsonian (6-OHDA lesioned) parameters with exaggerated **beta** oscillations versus `pd = 0` (healthy).
- **Outputs used downstream:** At minimum, **GPi action-potential trains** for biomarker computation; the reference also saves STN, GPe, striatal, cortical, and thalamic APs.

---

## 4. STN DBS actuation

The reference drives DBS as a **current injected in STN** (`Idbs` passed to `CTX_BG_TH_network`).

| Parameter | Reference default (`simulate_network_model.m`) | Notes |
|-----------|-----------------------------------------------|--------|
| **Pulse width** `PW` | **0.3 ms** | Rectangular pulse |
| **Amplitude** | **300 nA/cm²** | Constant within pulse |
| **Carrier pattern** | Scalar **frequency in Hz** via `pick_dbs_freq` indexing `freqs = 0:5:200` | `pick_dbs_freq == 1` → **no DBS** (`Idbs = 0`); otherwise `creatdbs` builds a pulse train at `pattern` Hz |
| **Optional cortical stimulus** | `corstim` | Off (`0`) for standard DBS-only runs |

**Pulse train construction (`creatdbs`):** For frequency `pattern` Hz, pulses of width `PW` at amplitude `amplitude` are placed with inter-pulse interval `isi = 1000 / pattern` ms on the integration grid.

**Mehregan discrete patterns:** Mehregan et al. apply a **discrete STN pattern alphabet** at the RL layer ([environment.md](environment.md) §4). The plant must accept a **drive specification** per simulated segment (frequency, pulse train, or precomputed `Idbs` waveform). **Open:** exact mapping from Mehregan pattern indices to STN current—**decide in** the MATLAB/Python bridge; keep stable across training, eval, and quantization.

**Baselines (plant-level):** **No stimulation**, conventional **~130 Hz** cDBS, and periodic **45 Hz** (and **30 Hz** where papers compare)—implemented as fixed `pattern` / `Idbs` settings with documented seeds.

---

## 5. Integration and simulated time

| Quantity | Reference (`simulate_network_model.m`) | Mehregan replication target |
|----------|----------------------------------------|-----------------------------|
| **Integration step** $\Delta t$ | **0.01 ms** (`dt = 0.01`) | Mehregan §IV.A.1 reports **0.02 ms** for the computational plant |
| **Default segment in reference script** | **2000 ms** (`tmax = 2000`) per call | RL **segment length** is set by the env or adapter (e.g. **2 s** Mehregan, **100 ms** Nguyen, **2 ms** SEA-DBS)—the plant integrates whatever duration the caller requests |

**Implementation note:** For Mehregan-aligned replication, default plant config should use **$\Delta t = 0.02$ ms** unless released Mehregan training code specifies otherwise. If the bridge keeps the reference **0.01 ms** step, **document the deviation** and validate biomarker statistics ($P_\beta$, baseline traces) under the same protocols.

**Episode / IC reset:** A new RL episode draws **new initial conditions** (reference: randomized membrane voltages per population, e.g. `v1 = -62 + randn(n,1)*5`). The plant `reset` (or equivalent) must support reproducible seeds for benchmarking.

---

## 6. Biomarkers from GPi spiking

Biomarker **definitions differ by paper**; the plant and wrapper should compute spectra from **GPi spike trains** and expose band integrals consistently.

### 6.1 Reference script (Kumaravelu bundle)

After simulation, `make_Spectrum(GPi_APs, params)` computes a multitaper PSD (`mtspectrumpt`, Chronux-style parameters in-script: `Fs` from `dt`, `fpass = [1 100]`, `tapers = [3 5]`). The reference integrates power over **7–35 Hz**:

```matlab
beta = S(f>7 & f<35);
area = trapz(betaf, beta);  % gpi_alpha_beta_area
```

This **`gpi_alpha_beta_area`** is the reference’s “alpha–beta” scalar for GPi.

### 6.2 Mehregan $P_\beta$ (13–35 Hz)

Mehregan Eq. (1) integrates GPi spike PSD over **13–35 Hz**, averaged over **$n = 10$** GPi neurons:

$$
P_\beta = \frac{1}{n} \sum_{j=1}^{n} \int_{\omega = 2\pi \cdot 13\,\mathrm{Hz}}^{2\pi \cdot 35\,\mathrm{Hz}} P_j^{\mathrm{GPi}}(\omega)\, d\omega
$$

**Replication:** Use **13–35 Hz** for Mehregan fidelity even when wrapping the reference script—**re-band** or post-process; do not silently use the reference’s **7–35 Hz** integral for Mehregan metrics ([environment.md](environment.md) §3.1).

### 6.3 Nguyen $\alpha$–$\beta$ feedback

Nguyen uses **7–35 Hz** GPi oscillation power for reward and early termination ([controllers/snn/replication.md](controllers/snn/replication.md)). That aligns with the **reference integral band**, not Mehregan’s **13–35 Hz** $P_\beta$ alone.

### 6.4 Logging for cross-controller comparison

For optional **plant-level** suites ([benchmarking.md](benchmarking.md) §3.3), log **raw GPi traces** and/or **multiple band integrals** ($P_\beta$, 7–35 Hz, duty cycle) so adapters are not conflated in tables.

---

## 7. Target plant wrapper API

The plant lives under `envs/plant/` as a **non-Gym** service the Mehregan `Env` and adapters call.

### Implemented (Phase 2, partial)

| Symbol | Module | Notes |
|--------|--------|--------|
| `MatlabPlant` | `envs.plant` | `matlab.engine` bridge to `simulate_network_model(..., dynamics_only=true)` |
| `PythonPlant` | `envs.plant` | Native NumPy port of Kumaravelu CBGT network ([native-plant-port.md](development/native-plant-port.md)); **GPi / $P_\beta$ parity with MATLAB** when init draws come from `plant_init_export` fixtures (see below) |
| `PlantConfig` | `envs.plant` | `pd`, `dt_ms` (reference **0.01 ms**), `corstim` |
| `DbsSpec` | `envs.plant` | `pick_dbs_freq` index; `DbsSpec.none()`, `from_frequency_hz(hz)` |
| `IntegrateResult` | `envs.plant` | `gpi_spikes` (10 neurons, times in **seconds**), `info` |

**Backend selection:** `.rl-dbs.yaml` → `plant.backend: matlab` (default) or `python`; env override `RL_DBS_PLANT_BACKEND`. `rl-dbs info plant` shows the resolved backend. Default stays **`matlab`** until the **≥10×** integrate speedup gate passes ([native-plant-port.md](development/native-plant-port.md) §5); parity gates are met (2026-07-03).

**Python init draws:** MATLAB’s `randperm(n, k)` for heterogeneous STN→GPe/GPi conductances is not reproducible from NumPy’s `Generator` alone. `PythonPlant.reset(seed=N)` loads `tests/fixtures/plant_init_seed{N}.npz` (or `~/.cache/rl-adaptive-dbs/plant_init/`) exported via `scripts/export_plant_init_draws.py`. With matching init draws, 2 s GPi spike trains and $P_\beta$ match MATLAB within documented tolerances (`tests/envs/python_integrator_fixed_ic_test.py`, `tests/envs/plant_backend_equivalence_test.py`).

**Performance (2026-07-03, WSL):** 2 s `integrate` (no DBS, seed=42) — MATLAB **≈58–63 s**, PythonPlant **≈190 s** (~3× slower; improved from ~219 s via scalar state + convolver fast path). **≥10× gate open** — Numba inner-loop JIT is the next step before flipping the default backend.

```python
from envs.plant import DbsSpec, MatlabPlant, PythonPlant

# MATLAB backend (default)
with MatlabPlant() as plant:
    result = plant.reset(seed=42).integrate(2.0, DbsSpec.none())

# Native Python (opt-in via config or explicit construction)
with PythonPlant() as plant:
    result = plant.reset(seed=42).integrate(2.0, DbsSpec.none())
    # result.gpi_spikes[i] — spike times for GPi neuron i
```

MATLAB vendor args and return packing: [kumaravelu_vendor_patches.md](reference-material/kumaravelu_vendor_patches.md). Tests: `tests/envs/matlab_plant_test.py` ([development/testing.md](development/testing.md)).

**Biomarkers (implemented):** `envs.plant.biomarkers.p_beta` — multitaper GPi PSD (Chronux-style; Kumaravelu `Fs` / tapers), **13–35 Hz** per-neuron integral then mean (Mehregan Eq. (1)). `MatlabPlant.integrate` sets `IntegrateResult.p_beta`.

**Mehregan Gym (implemented):** `envs.mehregan.MehreganEnv` — see [environment.md](environment.md); baselines via `run_baseline_rollout`.

### Target surface

| Method | Behavior |
|--------|----------|
| `reset(seed=None)` | Parkinsonian ICs (`pd = 1`); optional RNG seed |
| `integrate(duration_s, dbs_spec, *, record_spikes=True)` | Advance dynamics for `duration_s` simulated seconds with STN drive `dbs_spec`; return GPi (and optionally other) APs, last biomarker samples, info |
| `config` | $\Delta t$, `pd`, default DBS waveform parameters, biomarker bands |

**Not in the plant API:** Gymnasium `step`/`reset` return shapes, Mehregan reward, or controller actions—the **environment** and **adapters** map RL actions to `dbs_spec` and call `integrate` for the correct **segment duration**.

---

## 8. Equivalence and validation

Before trusting the Python/MATLAB bridge for training:

1. **Fixed seed, `pd = 1`, no DBS:** compare GPi traces or $P_\beta$ / 7–35 Hz integral to a reference `.mat` output from `simulate_network_model.m`.
2. **Fixed DBS frequency** (e.g. 130 Hz, 45 Hz): match pulse timing and biomarker level within documented tolerance.
3. **$\Delta t$:** if using 0.02 ms vs reference 0.01 ms, document and run (1)–(2) under the chosen step.
4. **Band:** confirm Mehregan metrics use **13–35 Hz** even when the reference script returns **7–35 Hz**.

**PythonPlant status (2026-07-03):** Gates (1)–(2) pass for seeds with exported init fixtures (`plant_init_seed{N}.npz`). Spike times match on the shared **0.01 ms** grid (0 ms atol); $P_\beta$ relative error **< 1%**. Cross-backend suite: `tests/envs/plant_backend_equivalence_test.py`. Remaining exit criterion for default-backend flip: **≥10×** speedup vs MATLAB on 2 s integrate ([native-plant-port.md](development/native-plant-port.md) §5).

Mark heavy checks `@pytest.mark.matlab` ([development/testing.md](development/testing.md)). CI defaults should skip MATLAB.

---

## 9. Native Python plant (`PythonPlant`)

**Status (TASK-17, 2026-07-03):** Implemented in `envs/plant/python_backend.py` with the same `integrate` / `reset` contract as `MatlabPlant`. Dynamics port: `envs/plant/network/` ([native-plant-port.md](development/native-plant-port.md)).

| Gate | Status |
|------|--------|
| GPi spike parity (2 s, fixed init draws) | **Pass** |
| $P_\beta$ parity (< 1% rel error) | **Pass** |
| DBS ordering (130 Hz / 45 Hz lowers beta vs none) | **Pass** (`plant_backend_equivalence_test.py`) |
| ≥10× speedup vs MATLAB (2 s integrate) | **Open** (~3× slower on WSL after integrator opts; Numba inner-loop follow-up) |
| Default backend flip to `python` | **Blocked** on speedup gate |

Opt in via `plant.backend: python` or `RL_DBS_PLANT_BACKEND=python`. Until the speedup gate closes, treat **`reference-material/KumaraveluEtAl2016/`** as the dynamics reference for audits; Python is validated against it via exported init fixtures and parametrized `@pytest.mark.matlab` tests.

---

## 10. Open questions / TBD

### 1. Integration step $\Delta t$

**Fixed:** Mehregan replication target **0.02 ms**; reference default **0.01 ms**. **Open:** released Mehregan code value. **Decide in** plant config; validate biomarkers if keeping 0.01 ms.

### 2. Mehregan pattern → STN current

**Fixed:** actuation is STN-injected DBS per Kumaravelu et al. (2016). **Open:** encoding of discrete Mehregan patterns. **Decide in** env bridge ([environment.md](environment.md) §4).

**Implemented:** `envs/mehregan/patterns.py` — action index → `pick_dbs_freq` (see [environment.md](environment.md) §11.3).

### 3. Multitaper / PSD parameters

Reference uses specific `Fs`, `fpass`, and tapers. **Open:** whether adapters may share one PSD implementation for all bands. **Decide in** `envs/` biomarker module; document if diverging from reference `make_Spectrum`.

### 4. Healthy vs parkinsonian eval

**Fixed:** training targets **parkinsonian** (`pd = 1`). **Open:** whether benchmarks include `pd = 0` controls. **Decide in** suite manifests.

---

## 11. Consistency checklist

- [x] Single plant backend; **no** CBGT dynamics under `controllers/`.
- [x] **$n = 10$** neurons per region; reference Kumaravelu topology via `MatlabPlant`.
- [x] STN DBS via Kumaravelu `PW`, amplitude, and `pick_dbs_freq` / `Idbs` waveform.
- [x] Mehregan metrics: GPi **13–35 Hz** $P_\beta$ (`envs.plant.biomarkers`); Nguyen **7–35 Hz** deferred to adapter.
- [x] $\Delta t$ **0.01 ms** (reference default) documented in `PlantConfig`; Mehregan **0.02 ms** target noted in §10.
- [x] Equivalence tests: GPi spike reproducibility, $P_\beta$, `MehreganEnv` rollouts (`tests/envs/*_test.py`, `@pytest.mark.matlab`).
- [x] `PythonPlant` GPi / $P_\beta$ parity vs MATLAB (init fixtures; default backend still `matlab` pending speedup).

---

## 12. References

- Kumaravelu et al. (2016) — plant dynamics; bundled MATLAB under [`reference-material/KumaraveluEtAl2016/`](../reference-material/KumaraveluEtAl2016/).
- Mehregan et al. — RL environment using this plant; [environment.md](environment.md).
- Nguyen et al., Ravivarapu et al. — adapters on the same plant; [controllers/snn/replication.md](controllers/snn/replication.md), [controllers/sea_dbs/replication.md](controllers/sea_dbs/replication.md).
