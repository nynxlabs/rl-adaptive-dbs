# rl-adaptive-dbs

Replication of published **adaptive DBS** reinforcement-learning work: one shared **parkinsonian plant** (Kumaravelu et al., 2016), a **Mehregan et al.** Gymnasium-style environment in `envs/`, and **separate controller implementations** from each source paper (with adapters where a paper’s RL interface differs).

## Scope

This repository has three layers:

1. **Environment (single source of truth)** — Replicate the **computational RL environment** from Mehregan et al., *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning*: the Kumaravelu et al. (2016) **cortex–basal ganglia–thalamus** parkinsonian plant, GPi **beta-band** biomarker, step timing, reward, and Gymnasium-style API. Spec: [docs/environment.md](docs/environment.md).

2. **Controllers (one per paper)** — Replicate each paper’s **learning-based controller** as its own module under `controllers/`, all driving the **same plant** (not separate networks per method):

   | Package | Paper | Uses `envs/` directly? |
   |---------|--------|-------------------------|
   | `controllers/ddpg/` | Mehregan et al. — DDPG actor–critic with optional PTQ/QAT | Yes (Mehregan API) |
   | `controllers/snn/` | Nguyen et al. — closed-loop neuromorphic DBS (deep spiking Q-network) | No — **adapter** (100 ms steps, spike obs, α–β) |
   | `controllers/sea_dbs/` | Ravivarapu et al. — SEA-DBS (sample-efficient actor–critic) | No — **adapter** (2 ms steps, binary pulse, Eq. (7) reward) |

Controller-specific training details live in [docs/controllers/](docs/controllers/) (`ddpg.md`, `snn.md`, `sea_dbs.md` — one spec per package, aligned with `controllers/`). The **biophysical plant** is shared; **RL step timing, observations, actions, and reward** follow each paper via `envs/` or that package’s **adapter**.

3. **Benchmarking** — Compare **variants** within a paper (`mehregan_eval`, `nguyen_eval`, `sea_dbs_eval`) and optionally across papers on **plant-level** metrics only (`cross_controller_plant`). Spec: [docs/benchmarking.md](docs/benchmarking.md). Each run is identified by `controller` + `variant` + `run_id`; results land under `results/` (local, gitignored).

**Out of scope for the shared environment:** in vivo animal pipelines, hardware-in-the-loop implants, and paper-specific evaluation protocols that do not use the computational plant (documented separately if added later).

## Platform support

This repository is intended to work on **Windows**, **macOS**, and **Linux** (including WSL on Windows). Python setup uses [uv](https://docs.astral.sh/uv/) with a shared lockfile; see [docs/venv.md](docs/venv.md). Scripts and docs should stay portable unless a step is explicitly OS-specific.

## Layout

Python code lives at the **repository root** as two installable top-level packages (after `uv sync`, editable):

- **`envs/`** — Shared Gymnasium-style RL environment (Mehregan et al. computational setup).
- **`controllers/`** — Per-paper controllers: `ddpg`, `snn`, `sea_dbs` (stubs until implemented).

The **distribution** name in `pyproject.toml` is `rl-adaptive-dbs` (hyphen, normal for PyPI-style metadata). **Import** names follow Python rules (no hyphen), so you use `import envs`, `from controllers.ddpg import ...`, etc.

- `docs/` — [getting_started.md](docs/getting_started.md) (setup & use), [development.md](docs/development.md) (roadmap & status), specs under [environment.md](docs/environment.md), [controllers/](docs/controllers/), [benchmarking.md](docs/benchmarking.md), [venv.md](docs/venv.md).
- `results/` — benchmark outputs (created by future eval runs; not committed).
- `reference-material/` — Third-party models and scripts (directory name uses a **hyphen**, not `reference_material`). Kumaravelu et al. (2016) MATLAB network: `reference-material/KumaraveluEtAl2016/` ([`readme.txt`](reference-material/KumaraveluEtAl2016/readme.txt) for citation and provenance).

Import example: `from envs.foo import Bar` once modules exist.

## Getting started

**[docs/getting_started.md](docs/getting_started.md)** — install, verify, and day-to-day use.

**[docs/development.md](docs/development.md)** — roadmap, conventions, implementation status.

Specs: [environment.md](docs/environment.md), [controllers/](docs/controllers/), [benchmarking.md](docs/benchmarking.md). Tooling: [venv.md](docs/venv.md).

## References

### Computational environment and DDPG controller

- **Mehregan, J., et al.** *Enhancing Adaptive Deep Brain Stimulation via Efficient Reinforcement Learning.* Defines the shared computational RL environment (parkinsonian plant interface, GPi beta-band biomarker, 2 s steps, reward Eq. (8)) and the DDPG actor–critic with optional PTQ/QAT. Specs: [docs/environment.md](docs/environment.md), [docs/controllers/ddpg.md](docs/controllers/ddpg.md).

### Additional RL controllers (same plant, paper-specific adapters)

- **Nguyen, B., et al.** *Closed-Loop Neuromorphic Deep Brain Stimulation using Deep Spiking Q-Networks.* Deep spiking Q-network (DSQN); 100 ms steps, spike observations, α–β feedback. Spec: [docs/controllers/snn.md](docs/controllers/snn.md).
- **Ravivarapu, H., et al.** *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease* (SEA-DBS). Binary pulse actions, predictive reward model, Gumbel-Softmax exploration. Spec: [docs/controllers/sea_dbs.md](docs/controllers/sea_dbs.md).

### Biophysical plant model

- **Kumaravelu, K., Brocker, D. T., Grill, W. M.** (2016). *A biophysical model of the cortex–basal ganglia–thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease.* *Journal of Computational Neuroscience*, 40, 207–229. Third-party MATLAB distribution: [reference-material/KumaraveluEtAl2016/](reference-material/KumaraveluEtAl2016/) (`readme.txt` for citation and provenance).

### Benchmarking

Cross-controller comparison uses **per-paper eval suites** plus an optional **same-plant** suite; see [docs/benchmarking.md](docs/benchmarking.md) §3 and [docs/development.md](docs/development.md) (Cross-controller benchmarking).
