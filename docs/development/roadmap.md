# Roadmap & status

Phases and implementation status for **rl-adaptive-dbs**. Contributor rules: [conventions.md](conventions.md). For clone, install, and day-to-day commands, see [setup.md](../setup.md). Paper-aligned behavior lives in the **specs**, not here.

---

## 1. Roadmap

Work proceeds in layers: rough specs, then environment and controllers in paper order (DDPG, then SNN, then SEA-DBS), each with per-paper benchmarking, then cross-controller comparison, fusion, then long-term modularity and a native plant.

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

### Phase 2 — Environment for the first controller (complete)

Replicate the shared plant and Mehregan et al. Gym API before any controller work.

- Wrap Kumaravelu et al. (2016) MATLAB model (`reference-material/KumaraveluEtAl2016/`) per [plant.md](../plant.md); expose Mehregan Gym API per [environment.md](../environment.md).
- Gymnasium `reset` / `step`: parkinsonian ICs, STN DBS, GPi $P_\beta$, 2 s RL steps, reward Eq. (8)—the interface `ddpg` will use directly.
- Equivalence checks vs reference (integration step, biomarker band, baseline traces).
- **Exit criteria:** reproducible rollouts; baselines (`none`, `cdbs-130hz`, `periodic-45hz`) runnable from `envs/`; ready for Phase 3 training.

### Phase 3 — First controller (`ddpg`) (complete)

- Actor–critic per [controllers/ddpg/replication.md](../controllers/ddpg/replication.md); training loop (Algorithm 1).
- Variants: `paper`, `init-30hz` (full-precision, complete); **PTQ** (`ptq-fp16`, `ptq-int8`) and **QAT** (`qat`) — Phase 4 ([controllers/ddpg/replication.md](../controllers/ddpg/replication.md) §6).
- **Exit criteria:** training run completes; eval roll-out matches spec checklist on `envs/` without adapters — met via `run_replication`, `scripts/replicate_mehregan_ddpg.py`, and mock/MATLAB tests.

### Phase 4 — Benchmarking the first controller (current)

- Suite definitions (YAML or equivalent) per [benchmarking.md](../benchmarking.md).
- **Per-paper suite** for Mehregan replication (`mehregan_eval`): baselines + `ddpg` variants × seeds → `results/`.
- **DDPG quantization** (Mehregan §III.D / §IV.A.3): **PTQ** (`ptq-fp16`, `ptq-int8`) and **QAT** (`qat`) in `controllers/ddpg/` — used to exercise variant logging and the benchmark runner alongside full-precision `paper` / `init-30hz`.
- **Benchmark runner** and summary tables / plots over core metrics ($P_\beta$, reward, stim frequency).
- **CLI** ([cli.md](../cli.md)) — **start** `rl-dbs`: entry point, `benchmark` (primary), `info`, and Mehregan-focused `train` / `eval` wrappers for `ddpg` (including quantized variants).
- **TUI** ([tui.md](../tui.md)) — **start** `rl-dbs-tui`: read-only **Benchmarks** tab over `results/` (loader + fixture tests); training/logs tabs remain later.
- **Setup scripts** ([setup.md](../setup.md), [matlab.md](../matlab.md)) — **`scripts/setup.sh`** (Python + optional MATLAB); harden **`scripts/matlab/`** cross-platform; **validate on fresh VMs** (clean Linux, macOS, Windows via Git Bash/WSL) so clone → setup → verify works on other machines.
- **Fresh-machine validation** — exercise the full path on VMs or clean hosts with only git + uv (+ optional MATLAB license): `bash scripts/setup.sh`, docs match prompts, `pytest -m "not matlab"` passes; document OS-specific gaps in [setup.md](../setup.md) / [matlab.md](../matlab.md).
- **Exit criteria:** repeatable `mehregan_eval` runs across full-precision and quantized `ddpg` variants; replication checklist passable for `ddpg` (including §IV.A.3 quantization); `uv run rl-dbs benchmark` and `uv run rl-dbs-tui` usable for Phase 4 workflows; **setup scripts pass on fresh VMs** on each supported OS (or gaps documented with repro steps).
- **Results doc:** [phase4-results.md](phase4-results.md) — §8 implementation audit **done** (2026-07-01); full-suite benchmark tables **pending** TASK-9 `mehregan_eval` run.

### Phase 5 — SNN controller (`snn`), adapter, and per-paper benchmarking

| Package | Paper | Notes |
|---------|--------|--------|
| `controllers/snn/` | Nguyen et al. | Spec: [controllers/snn/replication.md](../controllers/snn/replication.md); adapter for Nguyen I/O |

- Replicate `controllers/snn/`; extend or wrap `envs/` only via the **adapter** (no duplicated CBGT dynamics).
- Extend `rl-dbs train` / `eval` to `snn` (Phase 4 covers `ddpg` only).
- **Per-paper suite** `nguyen_eval`; same runner and `rl-dbs` patterns as Phase 4.
- **Exit criteria:** `snn` trains/evals on the **same plant** through its documented adapter; `nguyen_eval` runs complete.

### Phase 6 — SEA-DBS controller (`sea_dbs`), adapter, and per-paper benchmarking

| Package | Paper | Notes |
|---------|--------|--------|
| `controllers/sea_dbs/` | Ravivarapu et al. | Spec: [controllers/sea_dbs/replication.md](../controllers/sea_dbs/replication.md); adapter for SEA-DBS I/O |

- Replicate `controllers/sea_dbs/`; extend or wrap `envs/` only via the **adapter** (no duplicated CBGT dynamics).
- Extend `rl-dbs train` / `eval` to `sea_dbs`.
- **Per-paper suite** `sea_dbs_eval`; same runner and `rl-dbs` patterns as Phase 4.
- **Exit criteria:** `sea_dbs` trains/evals on the **same plant** through its documented adapter; `sea_dbs_eval` runs complete.

### Phase 7 — Cross-controller comparison and benchmarkability

- Optional **cross-paper** suite on the shared plant only (see [benchmarking.md](../benchmarking.md) §3).
- Clarify which metrics are comparable across adapters vs plant-level only ($P_\beta$, stim duty cycle, etc.).
- Harden the runner, manifests, and reporting so all three controllers can be compared fairly at the plant level.
- **Exit criteria:** documented comparison protocol; cross-paper (or equivalent) runs reproducible with `adapter: true` and suite metadata logged.

### Phase 8 — Fusion

- Synthesize SEA-DBS's predictive reward model with DSQN's spiking architecture: [controllers/fusion.md](../controllers/fusion.md).
- Hierarchical neuromorphic controller (fast gatekeeper + parameter-tuning SNN) and any fusion-specific benchmarking.

### Phase 9 and beyond — Native plant, modularity, extensions

- Native Python plant ([plant.md](../plant.md) §9); drop MATLAB dependency after equivalence **and speedup** gates. **In progress (TASK-17):** parity + **≈10×** speed pass (Numba JIT); default flip blocked on `mehregan_eval` baseline — [native-plant-port.md](native-plant-port.md).
- Modular layout: swappable plant backends, controller packages, adapters, and benchmark runner; CI on smoke tests + selected suites.
- **Expand** `rl-dbs` / `rl-dbs-tui` (all controllers, training monitor, logs)—initial shells land in Phase 4 ([cli.md](../cli.md), [tui.md](../tui.md)).
- Optional per-controller work beyond paper replication: [controllers/ddpg/extensions.md](../controllers/ddpg/extensions.md), [controllers/snn/extensions.md](../controllers/snn/extensions.md), [controllers/sea_dbs/extensions.md](../controllers/sea_dbs/extensions.md), and broader framework ideas (other oscillatory conditions, patient-specific robustness, etc.).

Phases 9+ are intentionally open; prioritize equivalence and replication paths before large refactors.

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
| [benchmarking.md](../benchmarking.md) | Draft | Suites + `results/` layout; runner + CLI integration — Phase 4 |
| [cli.md](../cli.md) | Draft | Phase 4 — start `rl-dbs` (`benchmark`, `info`, `ddpg` train/eval) |
| [tui.md](../tui.md) | Draft | Phase 4 — start `rl-dbs-tui` (Benchmarks tab over `results/`) |
| `envs/` | Done | `MatlabPlant`, `MehreganEnv`, $P_\beta$, baselines (`run_baseline_rollout`) |
| `controllers/ddpg/` | Done (FP + PTQ/QAT) | Full-precision `paper` / `init-30hz`; PTQ/QAT in `quantization.py` |
| `controllers/snn/` | Placeholder | Phase 5 |
| `controllers/sea_dbs/` | Placeholder | Phase 6 |
| [matlab.md](../matlab.md) + `scripts/matlab/` | Done (WSL) | Fresh validation: Multipass (Linux) pending; **Sandbox (Windows) passed** 2026-06-30 — [fresh-validation.md](fresh-validation.md) |
| Project setup script | Done | `scripts/setup.sh`, `scripts/validate-fresh.sh`, Multipass + Sandbox scripts — Phase 4; see [fresh-validation.md](fresh-validation.md) |
| MATLAB plant bridge | Done | `envs/plant/` + `envs/mehregan/`; `@pytest.mark.matlab` equivalence suite |
| `PythonPlant` (native port) | In progress | Parity + **≈10×** speed **pass** (Numba JIT, 2026-07-03); default flip **blocked** on `mehregan_eval` baseline — [native-plant-port.md](native-plant-port.md) |
| Benchmark suite runner | Done | `benchmarks/` + `suites/mehregan_eval*.yaml`; baselines + `ddpg` eval |
| `rl-dbs` CLI | Partial | `benchmark`, `summary`, `info`, `config show`, `train`/`eval` (`ddpg`); `snn` Phase 5; `sea_dbs` Phase 6 |
| `rl-dbs-tui` | Partial | Benchmarks tab ([Textual](https://textual.textualize.io/)); Training/Eval/Logs later |

**Current phase:** 4 (benchmark runner, Mehregan quantization, CLI/TUI start, setup scripts + fresh-VM validation).

---

## 3. Documentation index

| Doc | Role |
|-----|------|
| [setup.md](../setup.md) | Setup and how to use the repo |
| [testing.md](testing.md) | pytest layout, markers, what to test |
| [development/](README.md) | Dev docs hub ([roadmap.md](roadmap.md), [conventions.md](conventions.md)) |
| [conventions.md](conventions.md) | Contributor rules and layout |
| [matlab.md](../matlab.md) | MATLAB install, connect, Python engine |
| [venv.md](venv.md) | `uv` and dependencies |
| [plant.md](../plant.md) | Shared CBGT plant (Kumaravelu et al.) |
| [environment.md](../environment.md) | Mehregan Gym API on the plant |
| [controllers/](../controllers/) | Per-controller specs (replication + extensions + fusion) |
| [benchmarking.md](../benchmarking.md) | Cross-controller comparison |
| [cli.md](../cli.md) | `rl-dbs` command-line interface (train, eval, benchmark, info, config) |
| [tui.md](../tui.md) | `rl-dbs-tui` terminal UI (monitor training and results) |
| [README.md](../../README.md) | Project scope and References |
