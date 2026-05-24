# rl-adaptive-dbs

Replication of published **adaptive DBS** reinforcement-learning work on one shared **parkinsonian plant** (Kumaravelu et al., 2016): a Mehregan et al. Gymnasium-style environment in `envs/`, then **separate controller implementations** from each source paper (with adapters where a paper’s RL interface differs), benchmarking, cross-controller comparison, and later fusion and modularity.

## Scope

Work is delivered in **phases** (see [docs/development.md](docs/development.md)): rough specs → environment for the first controller → `ddpg` → Mehregan benchmarking → SNN and SEA-DBS with adapters → cross-controller comparison → fusion → native Python plant and framework hardening. Architecturally, the repo has three layers:

1. **Environment (single source of truth)** — Replicate the **computational RL environment** from Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*: Kumaravelu et al. (2016) **cortex–basal ganglia–thalamus** dynamics, GPi **beta-band** biomarker, 2 s steps, reward Eq. (8), and a Gymnasium-style API that `ddpg` uses directly. Other papers connect through **adapters** on the same plant. Spec: [docs/environment.md](docs/environment.md). **Current focus:** Phase 2 (environment before controllers).

2. **Controllers (one per paper)** — Replicate each paper’s **learning-based controller** under `controllers/`, all driving the **same plant** (no duplicated CBGT dynamics):

   | Package | Paper | Uses `envs/` directly? | Phase (roadmap) |
   |---------|--------|-------------------------|-----------------|
   | `controllers/ddpg/` | Mehregan et al. — DDPG actor–critic with optional PTQ/QAT | Yes (Mehregan API) | 3 |
   | `controllers/snn/` | Nguyen et al. — closed-loop neuromorphic DBS (deep spiking Q-network) | No — **adapter** (100 ms steps, spike obs, α–β) | 5 |
   | `controllers/sea_dbs/` | Ravivarapu et al. — SEA-DBS (sample-efficient actor–critic) | No — **adapter** (2 ms steps, binary pulse, Eq. (7) reward) | 5 |

   Per-paper specs: [docs/controllers/](docs/controllers/) (`replication.md`, `extensions.md` for post-replication ideas). **Fusion** (SEA-DBS + DSQN synthesis) is Phase 7: [docs/controllers/fusion.md](docs/controllers/fusion.md).

3. **Benchmarking** — **Per-paper suites** first (`mehregan_eval`, then `nguyen_eval`, `sea_dbs_eval`); then optional **cross-paper** comparison on **plant-level** metrics only (`cross_controller_plant`). Spec: [docs/benchmarking.md](docs/benchmarking.md). Runs are keyed by `controller` + `variant` + `run_id`; outputs go under `results/` (local, gitignored).

**Later (Phase 8+):** validated native Python plant (drop MATLAB after equivalence checks), modular plant/controller/benchmark layout, CI, and training/eval CLIs—see [docs/development.md](docs/development.md).

## Platform support

This repository is intended to work on **Windows**, **macOS**, and **Linux** (including WSL on Windows). Python setup uses [uv](https://docs.astral.sh/uv/) with a shared lockfile; see [docs/venv.md](docs/venv.md). Scripts and docs should stay portable unless a step is explicitly OS-specific.

## Layout

Python code lives at the **repository root** as two installable top-level packages (after `uv sync`, editable):

- **`envs/`** — Shared Gymnasium-style RL environment (Mehregan et al. computational setup).
- **`controllers/`** — Per-paper controllers: `ddpg`, `snn`, `sea_dbs` (stubs until implemented).

- `docs/` — [getting_started.md](docs/getting_started.md) (setup & use), [development.md](docs/development.md) (roadmap & status), [plant.md](docs/plant.md) (CBGT dynamics), [environment.md](docs/environment.md) (Mehregan Gym API), [controllers/](docs/controllers/) (per-controller specs + fusion), [benchmarking.md](docs/benchmarking.md), [venv.md](docs/venv.md).
- `results/` — benchmark outputs (created by future eval runs; not committed).
- `reference-material/` — Third-party models and scripts. Kumaravelu et al. (2016) MATLAB network: `reference-material/KumaraveluEtAl2016/` ([`readme.txt`](reference-material/KumaraveluEtAl2016/readme.txt) for citation and provenance).

Import example: `from envs.foo import Bar` once modules exist.

## Getting started

**[docs/getting_started.md](docs/getting_started.md)** — install, verify, and day-to-day use.

**[docs/development.md](docs/development.md)** — roadmap, conventions, implementation status.

Specs: [plant.md](docs/plant.md), [environment.md](docs/environment.md), [controllers/](docs/controllers/), [benchmarking.md](docs/benchmarking.md). Tooling: [venv.md](docs/venv.md).

## Benchmarking

Cross-controller comparison uses **per-paper eval suites** plus an optional **same-plant** suite; see [docs/benchmarking.md](docs/benchmarking.md) §3 and [docs/development.md](docs/development.md#cross-controller-benchmarking).

## References

### Computational environment and DDPG controller

- **Mehregan, J., et al.** *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning.* Defines the shared computational RL environment (parkinsonian plant interface, GPi beta-band biomarker, 2 s steps, reward Eq. (8)) and the DDPG actor–critic with optional PTQ/QAT. Specs: [docs/environment.md](docs/environment.md), [docs/controllers/ddpg/replication.md](docs/controllers/ddpg/replication.md).

### Additional RL controllers (same plant, paper-specific adapters)

- **Nguyen, B., et al.** *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks.* Deep spiking Q-network (DSQN); 100 ms steps, spike observations, α–β feedback. Spec: [docs/controllers/snn/replication.md](docs/controllers/snn/replication.md).
- **Ravivarapu, H., et al.** *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson's Disease* (SEA-DBS). Binary pulse actions, predictive reward model, Gumbel-Softmax exploration. Spec: [docs/controllers/sea_dbs/replication.md](docs/controllers/sea_dbs/replication.md).

### Biophysical plant model

- **Kumaravelu, K., Brocker, D. T., Grill, W. M.** (2016). *A biophysical model of the cortex–basal ganglia–thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease.* *Journal of Computational Neuroscience*, 40, 207–229. Spec: [docs/plant.md](docs/plant.md). Upstream MATLAB: [ModelDBRepository/206232](https://github.com/ModelDBRepository/206232) (vendored: [reference-material/KumaraveluEtAl2016/](reference-material/KumaraveluEtAl2016/); see `readme.txt` for citation and provenance).


