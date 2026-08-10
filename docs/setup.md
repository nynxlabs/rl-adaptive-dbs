# Setup

Install, verify, and use **rl-adaptive-dbs** day to day. For **priorities and status**, see [development/roadmap.md](development/roadmap.md) (figure-first) and [figures/mehregan/replications.md](../figures/mehregan/replications.md); for **conventions**, [development/conventions.md](development/conventions.md). For scope and architecture, see [README.md](../README.md).

---

## What you are setting up

**rl-adaptive-dbs** provides:

- A shared **plant** (Kumaravelu et al., 2016; MATLAB in `reference-material/`) wrapped for RL
- A shared **Gymnasium-style environment** (`src/envs/`) — Mehregan et al. API (2 s steps, $P_\beta$, Eq. (8) reward)
- **Controllers** (`src/controllers/ddpg` implemented; `src/controllers/snn` scaffolded for Phase 5; `sea_dbs` until Phase 6) — one implementation per paper; `ddpg` uses `envs/` directly, `snn` and `sea_dbs` use **adapters** for their paper’s RL interface
- **Benchmarking & CLI** (Phase 4) — `src/benchmarks/` runner, **`rl-dbs`** (`benchmark`, `summary`, `info`, `train`/`eval`), **`rl-dbs-tui`** (six tabs — Run, Training, Eval, Benchmarks, Logs, Settings); Mehregan **PTQ/QAT** in `controllers/ddpg/quantization.py` — [benchmarking.md](benchmarking.md), [cli.md](cli.md), [tui.md](tui.md)
- **Setup scripts** — **`scripts/setup.sh`** (Python + optional MATLAB); MATLAB detail in **`scripts/matlab/`** — [matlab.md](matlab.md)

Packages install in editable mode so local changes are importable immediately after `uv sync`.

---

## Setup script verification status

Phase 4 expanded the install surface (benchmark runner, `rl-dbs` / `rl-dbs-tui`, PTQ/QAT deps, `scripts/validation/validate-fresh.sh`, Multipass / Windows Sandbox host scripts, and the `--validate` flag on `scripts/setup.sh`). A **re-verification pass** started 2026-07-18 after that expansion.

| Area | Status |
|------|--------|
| **`scripts/setup.sh`** (Python-only path) | **Verified** on maintainer WSL (2026-07-18) — `bash scripts/setup.sh --python-only --non-interactive --validate`; 291 pytest passed, CLI smoke OK. Syncs **`dev` + `figures`** (matplotlib required by figure-panel tests). |
| **`scripts/setup.sh --with-matlab`** | **Unverified** in this pass — delegates to `scripts/matlab/setup.sh` |
| **`scripts/matlab/`** | **Partially verified** on WSL only ([matlab.md](matlab.md)); cross-platform prompts not recently exercised |
| **Fresh-host validation** (`validate-fresh.sh`, Multipass / Sandbox) | **Pending** — last Sandbox pass 2026-06-30 (pre-CLI smoke); re-run after WSL fixes land ([fresh-validation.md](development/fresh-validation.md)) |

**If you are setting up on a new machine:** start with `bash scripts/setup.sh --python-only --non-interactive` and the checks in §3 below. Report failures in an issue with OS, `uv --version`, and the command output.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Git** | To clone the repository |
| **uv** | Python + venv manager — [install uv](https://docs.astral.sh/uv/getting-started/installation/) ([more detail](development/venv.md#install-uv)) |
| **MATLAB** | Optional — plant bridge only; `bash scripts/setup.sh --with-matlab` or [matlab.md](matlab.md). Skip for Python-only work |

**Platforms:** Windows, macOS, Linux (including WSL). Shell scripts are **bash**; on Windows use **Git Bash** or **WSL**.

---

## 2. Install

**Recommended (from repo root):**

```bash
git clone <repository-url>
cd rl-adaptive-dbs
bash scripts/setup.sh
```

`scripts/setup.sh` runs `uv sync --group dev --group figures` (Python + pytest tooling + matplotlib for figure-panel tests). With `--with-matlab` it also syncs the **matlab** group and delegates to **`scripts/matlab/setup.sh`**. Use `uv sync --all-groups` only when you want every optional group in one step on a dev machine.

| Flag | Effect |
|------|--------|
| `--python-only` | Skip MATLAB (non-interactive default) |
| `--with-matlab` | Run full MATLAB setup flow |
| `--skip-tests` | Skip pytest after Python setup |
| `--non-interactive` | No prompts; use with `--python-only` or `--with-matlab` |

**Manual Python-only:**

```bash
uv sync --all-groups
```

**MATLAB only** (if Python deps are already installed):

```bash
bash scripts/matlab/setup.sh
```

| Command | What it does |
|---------|----------------|
| `bash scripts/setup.sh` | Python + verify; optional MATLAB ([matlab.md](matlab.md)) |
| `uv sync` | Runtime deps + editable `envs` / `controllers` (minimal / CI) |
| `uv sync --group dev --group figures` | Above + **dev** (`pytest`, TUI) + **figures** (`matplotlib` for panel tests) — default `setup.sh` Python path |
| `uv sync --all-groups` | Every optional group (`dev`, `figures`, `matlab`) — full dev machine |
| `bash scripts/matlab/setup.sh` | MATLAB install/connect, `uv sync`, `verify.sh` |

Details: activation, `uv add`, pinning Python → [development/venv.md](development/venv.md).

### Fresh machine validation (Phase 4)

Periodic **portability** check on clean Linux (Multipass) and Windows (Sandbox) — **testing only**, not for DDPG training or MATLAB plant work. Full procedure, scripts, timing, and troubleshooting: [development/fresh-validation.md](development/fresh-validation.md).

**Host install (desktop Admin):** `pwsh -File scripts/validation/install-fresh-validation-host.ps1` (`-Sandbox` / `-Multipass` optional). From WSL: `bash scripts/validation/install-fresh-validation-host.sh`. Logs: `.validation-logs/` (gitignored).

From **Windows desktop PowerShell**: `pwsh -File scripts/validation/check-windows-host.ps1`, then the Multipass or Sandbox launcher in that doc. Inside the guest: `bash scripts/validation/validate-fresh.sh`.

---

## 3. Verify install

After `bash scripts/setup.sh` (or manual `uv sync` + checks):

```bash
uv run python -c "import envs; import controllers; print('ok')"
uv run pytest -m "not matlab"
```

- Expect `ok` from the import check.
- Expect passing tests from the fast pytest subset. Details: [development/testing.md](development/testing.md).

**Phase 3 (complete):** DDPG in `controllers/ddpg/` — train, eval, replication workflow (full-precision). **Phase 4 infrastructure (complete):** benchmark runner, CLI/TUI, PTQ/QAT hooks, `mehregan_eval` suite. **Mehregan panels 1b–6b:** Pass — [figures/mehregan/replications.md](../figures/mehregan/replications.md). **Current work:** Nguyen Fig 4–7 and Ravivarapu Figs 5–7 (Phases 5–6 in parallel). Roadmap: [development/roadmap.md](development/roadmap.md).

```python
from envs import MehreganEnv, run_baseline_rollout

env = MehreganEnv()
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(0)  # no DBS
result = run_baseline_rollout(env, "cdbs-130hz", seed=0)
env.close()
```

**Phase 3 (DDPG, mock or MATLAB env):**

```python
from controllers.ddpg import DDPGConfig, evaluate, train
from envs import MehreganEnv

env = MehreganEnv()
result = train(env, DDPGConfig(variant="paper"), checkpoint_path="artifacts/ddpg/paper_train0.pt")
metrics = evaluate(env, "artifacts/ddpg/paper_train0.pt")
env.close()
```

**Full paper replication on MATLAB** (slow — hours at 10 episodes × 30 steps):

```bash
source scripts/matlab/env.sh
uv run python scripts/replicate_mehregan_ddpg.py --variant paper --train-seed 0
```

Writes `artifacts/ddpg/paper_train0.pt` and a JSON summary for baseline comparison.

**Mehregan replication workflow (train → eval → benchmark → checklist):**

```bash
source scripts/matlab/env.sh

# 1. Train full-precision DDPG (paper defaults: 10 episodes × 30 steps)
uv run rl-dbs train --controller ddpg --variant paper --seeds 0

# Or the replication script (train + mehregan_eval + baselines + checklist)
uv run python scripts/replicate_mehregan_ddpg.py --variant paper --train-seed 0

# 2. PTQ eval (uses paper_train0.pt — no retraining)
uv run python scripts/replicate_mehregan_ddpg.py --variant ptq-fp16 --train-seed 0
uv run python scripts/replicate_mehregan_ddpg.py --variant ptq-int8 --train-seed 0

# 3. QAT (trains with fake-quant actor stubs)
uv run rl-dbs train --controller ddpg --variant qat --seeds 0

# 4. Full benchmark suite → results/
uv run rl-dbs benchmark --suite-name mehregan_eval --controllers ddpg:paper,ddpg:ptq-int8

# 5. Summary table + checklist on a replication JSON
uv run rl-dbs summary --suite-name mehregan_eval
uv run python scripts/check_mehregan_replication.py artifacts/ddpg/paper_train0_summary.json
```

PTQ variants load **`paper_train{seed}.pt`** automatically ([controllers/ddpg/quantization.py](controllers/ddpg/quantization.py)). The checklist encodes qualitative §IV claims (DDPG lowers beta vs unstimulated; PTQ tracks FP).

With MATLAB set up ([matlab.md](matlab.md)):

```bash
source scripts/matlab/env.sh
uv sync --group matlab
uv run pytest -m matlab tests/envs/matlab_plant_test.py -v   # ~6 min; one shared engine
uv run pytest -m "not matlab"                                   # fast CI subset
```

---

## 4. Repository layout

```
rl-adaptive-dbs/
├── envs/                    # Plant bridge (`envs/plant/`) + Mehregan env (`envs/mehregan/`)
├── controllers/
│   ├── ddpg/                # Mehregan et al.
│   ├── snn/                 # Nguyen et al.
│   └── sea_dbs/             # Ravivarapu et al.
├── benchmarks/              # Suite runner (YAML → results/)
├── suites/                  # Benchmark manifests (e.g. mehregan_eval.yaml)
├── rl_adaptive_dbs/         # CLI + TUI entry points
├── tests/                   # pytest (mirrors envs/, controllers/, benchmarks/)
├── docs/                    # Guides and specs (this file: setup.md)
├── reference-material/      # Kumaravelu et al. (2016) MATLAB model
├── scripts/
│   ├── setup.sh             # Project setup (Python + optional MATLAB)
│   ├── validate-fresh.sh    # Fresh-host validation (Multipass / Sandbox)
│   └── matlab/              # MATLAB install, env, verify
├── results/                 # Benchmark output (`rl-dbs benchmark`; gitignored)
├── artifacts/               # Training checkpoints (optional; gitignored)
├── pyproject.toml
└── uv.lock
```

**Imports** (after `uv sync`):

```python
import envs
from envs.plant import DbsSpec, MatlabPlant   # MATLAB bridge (Phase 2)
from controllers import ddpg   # also: snn, sea_dbs
```

Distribution name in metadata: `rl-adaptive-dbs`. Import paths use underscores and match folder names.

---

## 5. Day-to-day commands

Prefer **`uv run`** so you do not need to activate `.venv/`:

```bash
uv sync --all-groups              # refresh after pull or pyproject change
uv run pytest                     # run tests
uv run pytest tests/benchmarks    # suite runner only (mock plant)
uv run python -c "import envs"    # quick import check
uv add <package>                  # add runtime dependency
uv add --dev <package>            # add dev dependency (e.g. torch later)
```

**Benchmarks** (Mehregan eval suite; needs MATLAB for real plant unless you inject a mock env in Python):

```bash
# Inspect planned runs without executing
uv run rl-dbs -v benchmark --suite-name mehregan_eval_smoke --dry-run

# Fast smoke suite (2 baselines × 2 seeds)
uv run rl-dbs benchmark --suite-name mehregan_eval_smoke

# Full suite — train DDPG first (artifacts/ddpg/{variant}_train0.pt)
uv run rl-dbs benchmark --suite-name mehregan_eval

# Filter controllers or override seeds
uv run rl-dbs benchmark --suite-name mehregan_eval \
  --controllers baseline:cdbs-130hz,ddpg:paper --seeds 0,1

# Comparison table from existing results/
uv run rl-dbs summary --suite-name mehregan_eval_smoke
uv run rl-dbs summary --results-dir results/ --csv results/summary.csv

# Train / eval (ddpg; MATLAB plant by default)
uv run rl-dbs train --controller ddpg --variant paper --seeds 0 --dry-run
uv run rl-dbs eval --controller baseline --variant cdbs-130hz --seeds 0

# Textual TUI — six tabs; monochrome by default; `--color` for theme; `--dev` for live reload
uv run rl-dbs-tui --results-dir results/
uv run rl-dbs-tui --dev --artifacts-dir artifacts/

# Introspection and user config
uv run rl-dbs info
cp .rl-dbs.example.yaml .rl-dbs.yaml   # optional local overrides (gitignored)
uv run rl-dbs config show env.dt_rl env.beta_t
```

Outputs land under `results/<suite_name>/` ([benchmarking.md](benchmarking.md) §6). Checkpoints default to `artifacts/ddpg/`.

Commit `pyproject.toml` and `uv.lock` together after dependency changes.

**Optional activation** (macOS / Linux / WSL): `source .venv/bin/activate` — see [development/venv.md](development/venv.md).

---

## 6. How to work on a change

Typical flow (details in [development/conventions.md](development/conventions.md), phases in [development/roadmap.md](development/roadmap.md)):

1. **Read the spec** for what you are touching ([environment.md](environment.md) or `docs/controllers/<name>/replication.md`).
2. **Edit code** in `src/envs/` or `src/controllers/<name>/`.
3. **Run checks** — `uv run pytest` ([development/testing.md](development/testing.md)).
4. **Update the spec** in the same PR/commit if behavior or the public API changed.
5. **Benchmark** (Phase 4) — `uv run rl-dbs benchmark --suite-name mehregan_eval_smoke` (quick) or full `mehregan_eval`; `uv run rl-dbs summary` for tables; `uv run rl-dbs-tui` to browse. Ad hoc: `rl-dbs train` / `eval` or `scripts/replicate_mehregan_ddpg.py`.

### Common tasks

**Shared environment**

- Implement under `envs/` following [environment.md](environment.md).
- Keep Kumaravelu equivalence in mind when changing dynamics or $P_\beta$.
- Controllers consume the env API; they should not reimplement the plant.

**New or updated controller**

- Code: `controllers/<name>/`
- Spec: `docs/controllers/<name>/replication.md` (add if missing; optional `extensions.md` for post-replication work).
- If the paper’s obs/action space differs from the shared env, add an **adapter** in that package.

**Controller variant** (e.g. `ptq-int8`, `init-30hz`)

- Same package as the base controller; distinguish with `variant` in configs and benchmark outputs.
- Do not copy or fork `envs/` for variants.

**Compare runs** (Phase 4 — `rl-dbs benchmark` + `rl-dbs-tui`)

- Write under `results/`; do not commit.
- **Within-paper:** use that paper’s suite (`mehregan_eval`, `nguyen_eval`, `sea_dbs_eval`) and the same seeds for all variants in a table.
- **Across papers:** only compare metrics marked **plant-level** in the suite manifest (see [benchmarking.md](benchmarking.md) §3.3); do not merge `reward_sum` across controllers without a documented shared reward.

---

## 7. Where to read next

| Goal | Document |
|------|----------|
| Roadmap and project status | [development/roadmap.md](development/roadmap.md) |
| Conventions | [development/conventions.md](development/conventions.md) |
| `uv`, lockfile, Python version | [development/venv.md](development/venv.md) |
| MATLAB install, license, Python engine | [matlab.md](matlab.md) |
| Tests, markers, layout | [development/testing.md](development/testing.md) |
| Biophysical plant (CBGT, DBS, biomarkers) | [plant.md](plant.md) |
| Mehregan env (reward, RL timing, Gym API) | [environment.md](environment.md) |
| DDPG (Mehregan) | [controllers/ddpg/replication.md](controllers/ddpg/replication.md) |
| SNN (Nguyen) | [controllers/snn/replication.md](controllers/snn/replication.md) |
| SEA-DBS (Ravivarapu) | [controllers/sea_dbs/replication.md](controllers/sea_dbs/replication.md) |
| Cross-controller eval | [benchmarking.md](benchmarking.md) |
| CLI (`rl-dbs`) | [cli.md](cli.md) |
| TUI (`rl-dbs-tui`) | [tui.md](tui.md) |
| SEA-DBS + DSQN fusion (post-replication) | [controllers/fusion.md](controllers/fusion.md) |
| Scope and citations | [README.md](../README.md) (References) |

**Specs** = what to build. **Setup** (this file) = how to install and use the repo. **Development** = roadmap and team conventions.
