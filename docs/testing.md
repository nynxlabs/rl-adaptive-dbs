# Testing

How to run and extend **pytest** checks for **rl-adaptive-dbs**. Setup and `uv` usage: [getting_started.md](getting_started.md). Roadmap: [development/roadmap.md](development/roadmap.md). Conventions: [development/conventions.md](development/conventions.md).

---

## 1. Run tests

From the repository root (dev dependencies installed):

```bash
uv sync --all-groups
uv run pytest
```

Useful variants:

```bash
uv run pytest -v                          # verbose
uv run pytest tests/envs                  # env / plant tests
uv run pytest -m "not matlab"             # fast subset (CI default; skips MATLAB)
uv run pytest -m matlab                   # plant bridge only (~6 min; needs license)
uv run pytest tests/envs/matlab_plant_test.py -v
uv run pytest -k "beta"                   # name filter
```

Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]` (`testpaths = ["tests"]`, registered markers).

---

## 2. Layout

Tests live under **`tests/`** at the repo root (not inside `envs/` or `controllers/`), so they are not shipped in the wheel but still import installed packages after `uv sync`.

```
tests/
├── conftest.py              # shared fixtures; skips @pytest.mark.matlab when unlicensed
├── imports_test.py          # smoke: editable install
├── docs_test.py             # spec link / section checks
├── envs/
│   ├── mock_plant.py        # helper (not collected)
│   ├── biomarkers_test.py
│   ├── baselines_test.py
│   ├── mehregan_reward_test.py
│   ├── mehregan_env_test.py     # MehreganEnv (mock plant)
│   ├── matlab_plant_test.py     # MatlabPlant bridge (@pytest.mark.matlab)
│   ├── matlab_biomarkers_test.py
│   └── matlab_mehregan_env_test.py
└── controllers/
    ├── ddpg/
    │   ├── buffer_test.py
    │   ├── checkpoint_test.py
    │   ├── eval_test.py
    │   ├── matlab_trainer_test.py   # train + eval on MatlabPlant (@pytest.mark.matlab)
    │   ├── networks_test.py
    │   ├── replication_test.py
    │   └── trainer_test.py
```

Mirror **`envs/`** and **`controllers/<name>/`** when adding modules (e.g. `tests/envs/gym_api_test.py`, `tests/controllers/ddpg/actor_test.py`).

Naming: files `*_test.py`, functions `test_*` (pytest default discovery).

---

## 3. What belongs in pytest vs elsewhere

| Kind | Where | Examples |
|------|--------|----------|
| **Smoke** | `tests/` | imports, config loads |
| **Unit** | `tests/` | reward math, observation windows, adapter shapes |
| **Integration** | `tests/` | `reset` / `step`, episode length, baseline policies |
| **Equivalence / regression** | `tests/` with `@pytest.mark.matlab` or `slow` | GPi spikes (`matlab_plant_test.py`); $P_\beta$ vs MATLAB (`matlab_biomarkers_test.py`, needs `dpss`) |
| **Paper replication benchmarks** | future runner + `results/` | full suites per [benchmarking.md](benchmarking.md); not the full eval matrix in pytest |

Keep CI fast: default runs should pass without MATLAB; skip or mark heavy checks.

---

## 4. Markers

Registered in `pyproject.toml`:

| Marker | Use |
|--------|-----|
| `slow` | Long rollouts, training smoke, large fixtures |
| `matlab` | Needs MATLAB and `reference-material/KumaraveluEtAl2016/` bridge |

Example:

```python
import pytest

@pytest.mark.matlab
def test_p_beta_matches_reference_segment() -> None:
    ...
```

Skip when MATLAB is unavailable: `tests/conftest.py` skips `@pytest.mark.matlab` tests when batch `license('test','MATLAB')` fails. With MATLAB, run `source scripts/matlab/env.sh` and `uv sync --group matlab` first ([matlab.md](matlab.md)).

---

## 5. Adding tests for a change

1. Implement under `envs/` or `controllers/<name>/` per the relevant spec.
2. Add or extend tests under the matching `tests/` subtree.
3. Run `uv run pytest` before opening a PR.
4. If behavior is spec-defined, update the spec in the same change ([development/conventions.md](development/conventions.md)).

Shared fixtures (env instances, seeds, reference traces) belong in `tests/conftest.py` or `tests/envs/conftest.py` as the suite grows.
