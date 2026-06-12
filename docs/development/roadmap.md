# Roadmap & status

Phases and implementation status for **rl-adaptive-dbs**. Contributor rules: [conventions.md](conventions.md). For clone, install, and day-to-day commands, see [getting_started.md](../getting_started.md). Paper-aligned behavior lives in the **specs**, not here.

---

## 1. Roadmap

Work proceeds in layers: rough specs, then environment and controllers in paper order (DDPG first), then benchmarking and comparison, then fusion, then long-term modularity and a native plant.

### Phase 1 — Rough specifications (complete)

Draft specs that define scope, interfaces, and paper-aligned intent—not final implementation detail.

| Deliverable | Spec | Code |
|-------------|------|------|
| Shared plant (CBGT) | [plant.md](../plant.md) | — |
| Mehregan Gym API | [environment.md](../environment.md) | — |
| DDPG controller | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) | — |
| SNN controller | [controllers/snn/replication.md](../controllers/snn/replication.md) | — |
| SEA-DBS controller | [controllers/sea_dbs/replication.md](../controllers/sea_dbs/replication.md) | — |
| Comparison protocol | [benchmarking.md](../benchmarking.md) | — |

### Phase 2 — Environment for the first controller (current)

Replicate the shared plant and Mehregan et al. Gym API before any controller work.

- Wrap Kumaravelu et al. (2016) MATLAB model (`reference-material/KumaraveluEtAl2016/`) per [plant.md](../plant.md); expose Mehregan Gym API per [environment.md](../environment.md).
- Gymnasium `reset` / `step`: parkinsonian ICs, STN DBS, GPi $P_\beta$, 2 s RL steps, reward Eq. (8)—the interface `ddpg` will use directly.
- Equivalence checks vs reference (integration step, biomarker band, baseline traces).
- **Exit criteria:** reproducible rollouts; baselines (`none`, `cdbs-130hz`, `periodic-45hz`) runnable from `envs/`; ready for Phase 3 training.

### Phase 3 — First controller (`ddpg`)

- Actor–critic per [controllers/ddpg/replication.md](../controllers/ddpg/replication.md); training loop (Algorithm 1).
- Variants: `paper`, `init-30hz`, optional PTQ/QAT (`ptq-int8`, etc.).
- **Exit criteria:** training run completes; eval roll-out matches spec checklist on `envs/` without adapters.

### Phase 4 — Benchmarking the first controller

- Suite definitions (YAML or equivalent) per [benchmarking.md](../benchmarking.md).
- **Per-paper suite** for Mehregan replication (`mehregan_eval`): baselines + `ddpg` variants × seeds → `results/`.
- Runner and summary tables / plots over core metrics ($P_\beta$, reward, stim frequency); CLI `benchmark` and TUI per [cli.md](../cli.md), [tui.md](../tui.md).
- **Exit criteria:** repeatable `mehregan_eval` runs; replication checklist passable for `ddpg`.

### Phase 5 — Other controllers, adapters, and per-paper benchmarking

| Package | Paper | Notes |
|---------|--------|--------|
| `controllers/snn/` | Nguyen et al. | Spec: [controllers/snn/replication.md](../controllers/snn/replication.md); adapter for Nguyen I/O |
| `controllers/sea_dbs/` | Ravivarapu et al. | Spec: [controllers/sea_dbs/replication.md](../controllers/sea_dbs/replication.md); adapter for SEA-DBS I/O |

- Replicate each controller; extend or wrap `envs/` only via **adapters** (no duplicated CBGT dynamics).
- **Per-paper suites** for Nguyen and SEA-DBS (`nguyen_eval`, `sea_dbs_eval`); same runner as Phase 4.
- **Exit criteria:** each controller trains/evals on the **same plant** through its documented adapter; per-paper benchmark runs complete.

### Phase 6 — Cross-controller comparison and benchmarkability

- Optional **cross-paper** suite on the shared plant only (see [benchmarking.md](../benchmarking.md) §3).
- Clarify which metrics are comparable across adapters vs plant-level only ($P_\beta$, stim duty cycle, etc.).
- Harden the runner, manifests, and reporting so all three controllers can be compared fairly at the plant level.
- **Exit criteria:** documented comparison protocol; cross-paper (or equivalent) runs reproducible with `adapter: true` and suite metadata logged.

### Phase 7 — Fusion

- Synthesize SEA-DBS's predictive reward model with DSQN's spiking architecture: [controllers/fusion.md](../controllers/fusion.md).
- Hierarchical neuromorphic controller (fast gatekeeper + parameter-tuning SNN) and any fusion-specific benchmarking.

### Phase 8 and beyond — Native plant, modularity, extensions

- Native Python plant ([plant.md](../plant.md) §9); drop MATLAB dependency after equivalence checks.
- Modular layout: swappable plant backends, controller packages, adapters, and benchmark runner; CI on smoke tests + selected suites.
- Training/eval CLIs per [cli.md](../cli.md); monitoring TUI per [tui.md](../tui.md); day-to-day setup in [getting_started.md](../getting_started.md).
- Optional per-controller work beyond paper replication: [controllers/ddpg/extensions.md](../controllers/ddpg/extensions.md), [controllers/snn/extensions.md](../controllers/snn/extensions.md), [controllers/sea_dbs/extensions.md](../controllers/sea_dbs/extensions.md), and broader framework ideas (other oscillatory conditions, patient-specific robustness, etc.).

Phases 8+ are intentionally open; prioritize equivalence and replication paths before large refactors.

---

## 2. Implementation status

*Update this section when milestones land.*

| Component | Status | Notes |
|-----------|--------|--------|
| [plant.md](../plant.md) | Draft | — |
| [environment.md](../environment.md) | Draft | — |
| [controllers/ddpg/replication.md](../controllers/ddpg/replication.md) | Draft | — |
| [controllers/snn/replication.md](../controllers/snn/replication.md) | Draft | Adapter + DSQN timing in spec |
| [controllers/sea_dbs/replication.md](../controllers/sea_dbs/replication.md) | Draft | Adapter + binary action / Eq. (7) in spec |
| [controllers/ddpg/extensions.md](../controllers/ddpg/extensions.md) | Outline | CV post-replication directions |
| [controllers/snn/extensions.md](../controllers/snn/extensions.md) | Outline | CV post-replication directions |
| [controllers/sea_dbs/extensions.md](../controllers/sea_dbs/extensions.md) | Outline | CV post-replication directions |
| [controllers/fusion.md](../controllers/fusion.md) | Outline | SEA-DBS + DSQN synthesis |
| [benchmarking.md](../benchmarking.md) | Draft | Per-paper vs cross-paper suites in spec; runner not implemented |
| [cli.md](../cli.md) | Draft | `rl-dbs` entry point spec; not implemented |
| [tui.md](../tui.md) | Draft | `rl-dbs-tui` read-only monitor; not implemented |
| `envs/` | Placeholder | No Gym env yet |
| `controllers/ddpg/` | Placeholder | — |
| `controllers/snn/` | Placeholder | — |
| `controllers/sea_dbs/` | Placeholder | — |
| [matlab.md](../matlab.md) + `scripts/matlab/` | Done | Install, connect, verify; cross-platform |
| MATLAB plant bridge | Not started | Reference in `reference-material/`; patches in [`changes.md`](../../reference-material/KumaraveluEtAl2016/changes.md) |
| Benchmark suite runner | Not started | — |
| Training / eval CLI | Not started | — |

**Current phase:** 2 (Phase 1 rough specs complete; environment replication for `ddpg` in progress; Phases 3–8+ outlined).

---

## 3. Documentation index

| Doc | Role |
|-----|------|
| [getting_started.md](../getting_started.md) | Setup and how to use the repo |
| [testing.md](../testing.md) | pytest layout, markers, what to test |
| [development/](README.md) | Dev docs hub ([roadmap.md](roadmap.md), [conventions.md](conventions.md)) |
| [conventions.md](conventions.md) | Contributor rules and layout |
| [matlab.md](../matlab.md) | MATLAB install, connect, Python engine |
| [venv.md](../venv.md) | `uv` and dependencies |
| [plant.md](../plant.md) | Shared CBGT plant (Kumaravelu et al.) |
| [environment.md](../environment.md) | Mehregan Gym API on the plant |
| [controllers/](../controllers/) | Per-controller specs (replication + extensions + fusion) |
| [benchmarking.md](../benchmarking.md) | Cross-controller comparison |
| [cli.md](../cli.md) | `rl-dbs` command-line interface (train, eval, benchmark, info, config) |
| [tui.md](../tui.md) | `rl-dbs-tui` terminal UI (monitor training and results) |
| [README.md](../../README.md) | Project scope and References |
