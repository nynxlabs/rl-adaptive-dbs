# Development

Roadmap, conventions, and implementation status for **rl-adaptive-dbs**. For clone, install, and day-to-day commands, see [getting_started.md](getting_started.md). Paper-aligned behavior lives in the **specs**, not here.

---

## 1. Roadmap

Work proceeds in layers: one shared environment, then controllers, then cross-controller benchmarking.

### Phase 0 — Specifications (complete)

| Deliverable | Spec | Code |
|-------------|------|------|
| Shared plant / Gym API | [environment.md](environment.md) | — |
| DDPG controller | [controllers/ddpg.md](controllers/ddpg.md) | — |
| SNN controller | [controllers/snn.md](controllers/snn.md) | — |
| SEA-DBS controller | [controllers/sea_dbs.md](controllers/sea_dbs.md) | — |
| Comparison protocol | [benchmarking.md](benchmarking.md) | — |

### Phase 1 — Shared environment (current)

- Wrap Kumaravelu et al. (2016) MATLAB model (`reference-material/KumaraveluEtAl2016/`) or begin validated native Python port.
- Gymnasium `reset` / `step`: parkinsonian ICs, STN DBS, GPi $P_\beta$, 2 s RL steps, reward Eq. (8).
- Equivalence checks vs reference (integration step, biomarker band, baseline traces).
- **Exit criteria:** reproducible rollouts; baselines (`none`, `cdbs-130hz`, `periodic-45hz`) runnable from `envs/`.

### Phase 2 — First controller (`ddpg`)

- Actor–critic per [controllers/ddpg.md](controllers/ddpg.md); training loop (Algorithm 1).
- Variants: `paper`, `init-30hz`, optional PTQ/QAT (`ptq-int8`, etc.).
- **Exit criteria:** training run completes; eval roll-out matches spec checklist.

### Phase 3 — Additional controllers

| Package | Paper | Notes |
|---------|--------|--------|
| `controllers/snn/` | Nguyen et al. | Spec: [controllers/snn.md](controllers/snn.md); adapter for Nguyen I/O |
| `controllers/sea_dbs/` | Ravivarapu et al. | Spec: [controllers/sea_dbs.md](controllers/sea_dbs.md); adapter for SEA-DBS I/O |

- **Exit criteria:** each controller trains/evals on the **same plant wrapper** (no duplicated CBGT dynamics); `ddpg` uses `envs/` directly, others via adapters documented in their specs.

### Phase 4 — Benchmarking

- Suite definitions (YAML or equivalent) per [benchmarking.md](benchmarking.md).
- **Per-paper suites** for replication (`mehregan_eval`, `nguyen_eval`, `sea_dbs_eval`); optional **cross-paper** suite on the shared plant only (see [benchmarking.md](benchmarking.md) §3).
- Runner: baselines + controllers × variants × seeds → `results/`.
- Summary tables / plots over core metrics ($P_\beta$, reward, stim frequency).

### Phase 5 — Optional

- Native Python plant ([environment.md](environment.md) §12); drop MATLAB dependency after equivalence checks.
- CI on smoke tests + selected benchmark suite.
- Training/eval CLIs documented in [getting_started.md](getting_started.md).

---

## 2. Conventions

### Spec-driven changes

1. Read the relevant spec before non-trivial implementation.
2. Code under `envs/` or `controllers/<name>/`.
3. Update the spec in the **same change** if behavior or interfaces change.
4. Resolve “intentionally open” items in spec or code—not only in issues or chat.

| Layer | Spec path | Code path |
|-------|-----------|-----------|
| Environment | [environment.md](environment.md) | `envs/` |
| Controller | [controllers/ddpg.md](controllers/ddpg.md), [snn.md](controllers/snn.md), [sea_dbs.md](controllers/sea_dbs.md) | `controllers/<name>/` |
| Benchmarks | [benchmarking.md](benchmarking.md) | runner TBD |

### Naming and layout

- Python packages match directories: `ddpg`, `snn`, `sea_dbs`.
- Controller docs mirror packages: `docs/controllers/ddpg.md`, etc.
- Default paper-replication benchmark variant: `paper`.
- Distribution name `rl-adaptive-dbs` (hyphen); imports `envs`, `controllers` (underscore, no hyphen).

### Cross-platform code

- Use `pathlib`, `uv run`; avoid hard-coded POSIX-only paths in library code.
- Document OS-specific steps only in [venv.md](venv.md) or [getting_started.md](getting_started.md).

### Documentation

- Cite papers by author/title in prose—not opaque ids like `paper_1`.
- Math in markdown: `$...$` inline, `$$...$$` display ([AGENTS.md](../AGENTS.md)).

### Controllers vs environment

- **One plant** for all controllers; variants do not fork `envs/`.
- Paper-specific observation/action mismatches → **adapter** in `controllers/<name>/`.
- Do not duplicate plant dynamics inside controller packages.

### Cross-controller benchmarking

All controllers share the **Kumaravelu et al. (2016)** parkinsonian plant, but each paper defines its own **RL interface** (step length, observation, action, reward). The shared `envs/` package implements the **Mehregan et al.** Gym API ([environment.md](environment.md)); `ddpg` can train on it directly. **Nguyen** and **Ravivarapu** need adapters (see [controllers/snn.md](controllers/snn.md), [controllers/sea_dbs.md](controllers/sea_dbs.md)).

| Comparison type | What is held fixed | What differs | When to use |
|-----------------|-------------------|--------------|-------------|
| **Within-paper** (`*_eval` suite) | Plant + that paper’s timing, obs, action, reward | Variants (PTQ, ablations, hyperparameters) | Replication and ablations |
| **Cross-paper** (optional suite) | Plant, seeds, integration step; often **$P_\beta$** logged for all | Per-controller training/eval protocol via adapters | High-level “same plant” tables—not claimable as identical to any single paper’s Table/Figure |

Do **not** put `snn` and `ddpg` in one suite and assume the same **reward_sum** or **episode_length** are comparable without reading the suite manifest: Nguyen uses **100 ms** steps and **α–β** feedback; SEA-DBS uses **2 ms** steps and **Eq. (7)** reward; Mehregan uses **2 s** steps and **Eq. (8)**. Cross-paper runs should log **plant-level** metrics (e.g. raw $P_\beta$, stim duty cycle) plus `adapter: true` and `suite` name. Details: [benchmarking.md](benchmarking.md) §3, §5.

### Version control

Do **not** commit:

- `.venv/`, `results/`, MATLAB cache files
- Secrets, machine-local paths, large checkpoints (until an artifact policy exists)

---

## 3. Implementation status

*Update this section when milestones land.*

| Component | Status | Notes |
|-----------|--------|--------|
| [environment.md](environment.md) | Draft | — |
| [controllers/ddpg.md](controllers/ddpg.md) | Draft | — |
| [controllers/snn.md](controllers/snn.md) | Draft | Adapter + DSQN timing in spec |
| [controllers/sea_dbs.md](controllers/sea_dbs.md) | Draft | Adapter + binary action / Eq. (7) in spec |
| [benchmarking.md](benchmarking.md) | Draft | Per-paper vs cross-paper suites in spec; runner not implemented |
| `envs/` | Placeholder | No Gym env yet |
| `controllers/ddpg/` | Placeholder | — |
| `controllers/snn/` | Placeholder | — |
| `controllers/sea_dbs/` | Placeholder | — |
| MATLAB plant bridge | Not started | Reference in `reference-material/` |
| Benchmark suite runner | Not started | — |
| Training / eval CLI | Not started | — |

**Current phase:** 1 (Phase 0 specs complete; shared environment implementation in progress).

---

## 4. Documentation index

| Doc | Role |
|-----|------|
| [getting_started.md](getting_started.md) | Setup and how to use the repo |
| [development.md](development.md) | Roadmap, conventions, status (this file) |
| [venv.md](venv.md) | `uv` and dependencies |
| [environment.md](environment.md) | Shared plant / Gym API |
| [controllers/](controllers/) | Per-controller specs |
| [benchmarking.md](benchmarking.md) | Cross-controller comparison |
| [README.md](../README.md) | Project scope and References |
