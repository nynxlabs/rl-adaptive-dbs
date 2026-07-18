# Roadmap & status

Phases and implementation status for **rl-adaptive-dbs**. Contributor rules: [conventions.md](conventions.md). For clone, install, and day-to-day commands, see [setup.md](../setup.md). Paper-aligned behavior lives in the **specs**, not here.

---

## Current priorities — figure replication

**Day-to-day work is figure-first**, not “complete Phase N, then start Phase N+1.” The practical exit criterion for Mehregan replication is **qualitative match on named paper panels** — ordering, shape, and shared baselines — documented panel-by-panel in [figures/paper_1.md](../figures/paper_1.md). Each panel has a committed script under `scripts/figures/papers/<paper>/<panel>/plot.py`, qualitative gates in the spec, and side-by-side PNGs under `figures/papers/`.

| Mehregan panel | Status | Blocks / notes |
|----------------|--------|----------------|
| Fig 1b, 2a, 2b | **Pass** | Plant / biomarker gates — [plant.md](../plant.md) |
| Fig 4a, 4b, 5a | **Pass** | Training + 45 Hz efficacy — [environment.md](../environment.md), [ddpg/replication.md](../controllers/ddpg/replication.md) |
| Fig 5b | **Open** | 30 Hz post-train efficacy; pattern-mode retrain |
| Fig 6a, 6b | **Open** | PTQ/QAT panels — [ddpg/replication.md](../controllers/ddpg/replication.md) §6 |

**How this relates to phases:** Phases 1–3 (specs, environment, DDPG) and most Phase 4 **infrastructure** (benchmark runner, `rl-dbs`, `rl-dbs-tui`, PTQ/QAT hooks) are **done**. **Setup scripts** were expanded for Phase 4 but **have not been re-verified end-to-end** on clean hosts since that expansion — see [setup.md](../setup.md) § Setup script verification status. Remaining Mehregan work is **closing open panels** — which may require plant conventions, training protocol, or quantization fixes documented in [replication-fidelity.md](replication-fidelity.md), not ticking a phase box. Phases 5–9 (SNN, SEA-DBS, cross-controller comparison, fusion, native plant) stay on the long-term plan and start when Mehregan figure replication is in good shape (or when a panel explicitly needs them).

**Contributors:** read the panel checklist before large sweeps; promote stable findings into the panel `plot.py` and update [figures/paper_1.md](../figures/paper_1.md) in the same pass. See [conventions.md](conventions.md) § Figure replication.

---

## 1. Roadmap (long-term layers)

The phase list below is **architectural history and future structure** — useful for scoping packages and specs, but **not** the primary scheduling axis while Mehregan panels remain open.

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

### Phase 4 — Benchmarking the first controller (infrastructure complete; figure panels drive remaining work)

- Suite definitions (YAML or equivalent) per [benchmarking.md](../benchmarking.md).
- **Per-paper suite** for Mehregan replication (`mehregan_eval`): baselines + `ddpg` variants × seeds → `results/`.
- **DDPG quantization** (Mehregan §III.D / §IV.A.3): **PTQ** (`ptq-fp16`, `ptq-int8`) and **QAT** (`qat`) in `controllers/ddpg/` — used to exercise variant logging and the benchmark runner alongside full-precision `paper` / `init-30hz`.
- **Benchmark runner** and summary tables / plots over core metrics ($P_\beta$, reward, stim frequency).
- **CLI** ([cli.md](../cli.md)) — **start** `rl-dbs`: entry point, `benchmark` (primary), `info`, and Mehregan-focused `train` / `eval` wrappers for `ddpg` (including quantized variants).
- **TUI** ([tui.md](../tui.md)) — **`rl-dbs-tui`**: six Textual tabs (Run, Training, Eval, Benchmarks, Logs, Settings) over `artifacts/` and `results/`; detached launch via Run tab; dev mode (`--dev`, Ctrl+R).
- **Setup scripts** ([setup.md](../setup.md), [matlab.md](../matlab.md)) — **`scripts/setup.sh`** (Python + optional MATLAB); harden **`scripts/matlab/`** cross-platform; **validate on fresh VMs** (clean Linux, macOS, Windows via Git Bash/WSL) so clone → setup → verify works on other machines. **Status:** scripts landed during Phase 4 but **not re-verified end-to-end** since CLI/TUI/`--validate` expansion — active pass tracked in [setup.md](../setup.md) § Setup script verification status.
- **Fresh-machine validation** — exercise the full path on VMs or clean hosts with only git + uv (+ optional MATLAB license): `bash scripts/setup.sh`, docs match prompts, `pytest -m "not matlab"` passes; document OS-specific gaps in [setup.md](../setup.md) / [matlab.md](../matlab.md).
- **Exit criteria:** repeatable `mehregan_eval` runs across full-precision and quantized `ddpg` variants; replication checklist passable for `ddpg` (including §IV.A.3 quantization); `uv run rl-dbs benchmark` and `uv run rl-dbs-tui` usable for Phase 4 workflows; **setup scripts pass on fresh VMs** on each supported OS (or gaps documented with repro steps).
- **Results doc:** [phase4-results.md](phase4-results.md) — §8 implementation audit **done** (2026-07-01); full-suite `mehregan_eval` **done** (TASK-9, 2026-07-03). Remaining Mehregan validation is **figure-panel** driven ([figures/paper_1.md](../figures/paper_1.md)), not another suite pass.

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
- **Expand** `rl-dbs` for `snn` / `sea_dbs` train/eval (Phase 5–6); TUI v1 is **done** ([cli.md](../cli.md), [tui.md](../tui.md)).
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
| [tui.md](../tui.md) | Draft | `rl-dbs-tui` — six tabs (Run, Training, Eval, Benchmarks, Logs, Settings) |
| `envs/` (`src/envs/`) | Done | `MatlabPlant`, `MehreganEnv`, $P_\beta$, baselines (`run_baseline_rollout`) |
| `controllers/ddpg/` (`src/controllers/ddpg/`) | Done (FP + PTQ/QAT) | Full-precision `paper` / `init-30hz`; PTQ/QAT in `quantization.py` |
| `controllers/snn/` (`src/controllers/snn/`) | Scaffold | Phase 5 — package stub + shape tests |
| `controllers/sea_dbs/` | Placeholder | Phase 6 |
| [matlab.md](../matlab.md) + `scripts/matlab/` | Done (WSL) | Fresh validation: Multipass (Linux) pending; **Sandbox (Windows) passed** 2026-06-30 — [fresh-validation.md](fresh-validation.md) |
| Project setup script | **Verification pending** | Scripts landed in Phase 4 (`setup.sh`, `validate-fresh.sh`, Multipass + Sandbox); **not re-run on clean hosts** since CLI/TUI/`--validate` expansion — [setup.md](../setup.md) § Setup script verification status |
| MATLAB plant bridge | Done | `envs/plant/` + `envs/mehregan/`; `@pytest.mark.matlab` equivalence suite |
| `PythonPlant` (native port) | In progress | Parity + **≈10×** speed **pass** (Numba JIT, 2026-07-03); default flip **blocked** on `mehregan_eval` baseline — [native-plant-port.md](native-plant-port.md) |
| Benchmark suite runner | Done | `benchmarks/` + `suites/mehregan_eval*.yaml`; baselines + `ddpg` eval |
| `rl-dbs` CLI | Partial | `benchmark`, `summary`, `info`, `config show`, `train`/`eval` (`ddpg`); `snn` Phase 5; `sea_dbs` Phase 6 |
| `rl-dbs-tui` | Done (v1) | Six tabs ([Textual](https://textual.textualize.io/)); Run tab launches detached jobs; Settings persistence |
| Thread limits (`thread_limits.py`, `run.py`) | Done | Default 3-thread cap for plant-heavy CLI + `python -m rl_adaptive_dbs.run` |
| Figure panel scripts (`scripts/figures/papers/`) | Partial | 1b–5a pass; 5b, 6a, 6b open — [figures/paper_1.md](../figures/paper_1.md) |

**Scheduling axis:** figure replication ([figures/paper_1.md](../figures/paper_1.md)) — open: Fig 5b, 6a, 6b.

**Phase map (reference):** Phases 1–4 infrastructure largely **complete**; Phase 5+ deferred until Mehregan panels close or a task explicitly needs them. **Setup scripts:** verification pass in progress — expanded for Phase 4 but not re-run on clean hosts since CLI/TUI/`--validate` additions ([setup.md](../setup.md) § Setup script verification status). Outstanding portability: Multipass Linux fresh-validation ([fresh-validation.md](fresh-validation.md)).

---

## 3. Documentation index

| Doc | Role |
|-----|------|
| [figures/paper_1.md](../figures/paper_1.md) | Mehregan panel replication — primary goal tracker |
| [replication-fidelity.md](replication-fidelity.md) | Mehregan verified / divergent / added |
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
