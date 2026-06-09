# Getting started

Set up the repository, confirm your install, and use the project day to day. For the **roadmap**, **conventions**, and **status table**, see [development.md](development.md). For scope and architecture, see [README.md](../README.md).

---

## What you are setting up

**rl-adaptive-dbs** provides:

- A shared **plant** (Kumaravelu et al., 2016; MATLAB in `reference-material/`) wrapped for RL
- A shared **Gymnasium-style environment** (`envs/`) — Mehregan et al. API (2 s steps, $P_\beta$, Eq. (8) reward)
- **Controllers** (`controllers/ddpg`, `snn`, `sea_dbs`) — one implementation per paper; `ddpg` uses `envs/` directly, `snn` and `sea_dbs` use **adapters** for their paper’s RL interface
- **Benchmarking** (future) — per-paper eval suites and optional cross-plant comparison; see [benchmarking.md](benchmarking.md) §3

Packages install in editable mode so local changes are importable immediately after `uv sync`.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Git** | To clone the repository |
| **uv** | Python + venv manager — [install uv](https://docs.astral.sh/uv/getting-started/installation/) ([more detail](venv.md#install-uv)) |
| **MATLAB** | Required only when implementing the MATLAB plant bridge; skip for Python-only setup |

**Platforms:** Windows, macOS, Linux (including WSL).

---

## 2. Install

```bash
git clone <repository-url>
cd rl-adaptive-dbs
uv sync --all-groups
```

| Command | What it does |
|---------|----------------|
| `uv sync` | Runtime deps + editable `envs` / `controllers` (minimal / CI) |
| `uv sync --all-groups` | Above + **dev** group (`pytest`, etc.) — use locally |

Details: activation, `uv add`, pinning Python → [venv.md](venv.md).

---

## 3. Verify install

From the repository root:

```bash
uv run python -c "import envs; import controllers; print('ok')"
uv run pytest
```

- Expect `ok` from the import check.
- Expect at least one passing test from `pytest` (import smoke). Details: [testing.md](testing.md).

Imports confirm packaging only. The DBS plant and training loops are **not implemented yet** ([development.md](development.md) §3).

---

## 4. Repository layout

```
rl-adaptive-dbs/
├── envs/                    # Shared environment (implement per environment.md)
├── controllers/
│   ├── ddpg/                # Mehregan et al.
│   ├── snn/                 # Nguyen et al.
│   └── sea_dbs/             # Ravivarapu et al.
├── tests/                   # pytest (mirrors envs/ and controllers/)
├── docs/                    # Guides and specs
├── reference-material/      # Kumaravelu et al. (2016) MATLAB model
├── results/                 # Benchmark output (local, gitignored)
├── pyproject.toml
└── uv.lock
```

**Imports** (after `uv sync`):

```python
import envs
from controllers import ddpg   # also: snn, sea_dbs
```

Distribution name in metadata: `rl-adaptive-dbs`. Import paths use underscores and match folder names.

---

## 5. Day-to-day commands

Prefer **`uv run`** so you do not need to activate `.venv/`:

```bash
uv sync --all-groups              # refresh after pull or pyproject change
uv run pytest                     # run tests
uv run python -c "import envs"    # quick import check
uv add <package>                  # add runtime dependency
uv add --dev <package>            # add dev dependency (e.g. torch later)
```

Commit `pyproject.toml` and `uv.lock` together after dependency changes.

**Optional activation** (macOS / Linux / WSL): `source .venv/bin/activate` — see [venv.md](venv.md).

---

## 6. How to work on a change

Typical flow (details and rules in [development.md](development.md)):

1. **Read the spec** for what you are touching ([environment.md](environment.md) or `docs/controllers/<name>/replication.md`).
2. **Edit code** in `envs/` or `controllers/<name>/`.
3. **Run checks** — `uv run pytest` ([testing.md](testing.md)).
4. **Update the spec** in the same PR/commit if behavior or the public API changed.
5. **Benchmark later** — when the runner exists, use a named suite and log `controller`, `variant`, `seed` per [benchmarking.md](benchmarking.md).

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

**Compare runs** (when tooling exists)

- Write under `results/`; do not commit.
- **Within-paper:** use that paper’s suite (`mehregan_eval`, `nguyen_eval`, `sea_dbs_eval`) and the same seeds for all variants in a table.
- **Across papers:** only compare metrics marked **plant-level** in the suite manifest (see [benchmarking.md](benchmarking.md) §3.3); do not merge `reward_sum` across controllers without a documented shared reward.

---

## 7. Where to read next

| Goal | Document |
|------|----------|
| Roadmap and project status | [development.md](development.md) |
| `uv`, lockfile, Python version | [venv.md](venv.md) |
| Tests, markers, layout | [testing.md](testing.md) |
| Biophysical plant (CBGT, DBS, biomarkers) | [plant.md](plant.md) |
| Mehregan env (reward, RL timing, Gym API) | [environment.md](environment.md) |
| DDPG (Mehregan) | [controllers/ddpg/replication.md](controllers/ddpg/replication.md) |
| SNN (Nguyen) | [controllers/snn/replication.md](controllers/snn/replication.md) |
| SEA-DBS (Ravivarapu) | [controllers/sea_dbs/replication.md](controllers/sea_dbs/replication.md) |
| Cross-controller eval | [benchmarking.md](benchmarking.md) |
| SEA-DBS + DSQN fusion (post-replication) | [controllers/fusion.md](controllers/fusion.md) |
| Scope and citations | [README.md](../README.md) (References) |

**Specs** = what to build. **Getting started** (this file) = how to set up and use the repo. **Development** = where the project is headed and team conventions.
