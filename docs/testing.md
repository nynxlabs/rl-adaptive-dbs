# Testing

How to run and extend **pytest** checks for **rl-adaptive-dbs**. Setup and `uv` usage: [getting_started.md](getting_started.md). Roadmap and conventions: [development.md](development.md).

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
uv run pytest tests/envs                  # one area
uv run pytest -k "beta"                   # name filter
uv run pytest -m "not slow and not matlab" # fast subset (when marked tests exist)
```

Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]` (`testpaths = ["tests"]`, registered markers).

---

## 2. Layout

Tests live under **`tests/`** at the repo root (not inside `envs/` or `controllers/`), so they are not shipped in the wheel but still import installed packages after `uv sync`.

```
tests/
├── conftest.py              # shared fixtures
├── test_imports.py          # smoke: editable install
├── envs/                    # plant, Gym API, reward, baselines
└── controllers/
    ├── ddpg/
    ├── snn/
    └── sea_dbs/
```

Mirror **`envs/`** and **`controllers/<name>/`** when adding modules (e.g. `tests/envs/test_gym_api.py`, `tests/controllers/ddpg/test_actor.py`).

Naming: files `test_*.py`, functions `test_*` (pytest default discovery).

---

## 3. What belongs in pytest vs elsewhere

| Kind | Where | Examples |
|------|--------|----------|
| **Smoke** | `tests/` | imports, config loads |
| **Unit** | `tests/` | reward math, observation windows, adapter shapes |
| **Integration** | `tests/` | `reset` / `step`, episode length, baseline policies |
| **Equivalence / regression** | `tests/` with `@pytest.mark.matlab` or `slow` | traces or $P_\beta$ vs Kumaravelu MATLAB ([plant.md](plant.md) §8) |
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

Skip when MATLAB is unavailable (add in `conftest.py` when the bridge exists):

```python
pytest.importorskip("matlab")  # or env-based skip in conftest
```

---

## 5. Adding tests for a change

1. Implement under `envs/` or `controllers/<name>/` per the relevant spec.
2. Add or extend tests under the matching `tests/` subtree.
3. Run `uv run pytest` before opening a PR.
4. If behavior is spec-defined, update the spec in the same change ([development.md](development.md) §2).

Shared fixtures (env instances, seeds, reference traces) belong in `tests/conftest.py` or `tests/envs/conftest.py` as the suite grows.
