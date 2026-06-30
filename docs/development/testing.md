# Testing

How to run and extend **pytest** checks for **rl-adaptive-dbs**. Setup and `uv` usage: [setup.md](../setup.md). Roadmap: [roadmap.md](roadmap.md). Conventions: [conventions.md](conventions.md).

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
uv run pytest tests/benchmarks          # suite runner + loader
uv run pytest tests/rl_adaptive_dbs     # CLI + TUI data layer
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
├── fixtures/
│   └── benchmark_results/   # TUI + summary fixture tree
├── benchmarks/
│   ├── suite_test.py        # YAML load, run expansion
│   ├── runner_test.py       # mock-plant suite execution
│   ├── loader_test.py       # results manifest loader
│   └── cli_test.py          # rl-dbs benchmark dry-run
├── rl_adaptive_dbs/
│   ├── cli_commands_test.py # train/eval/info/summary smoke
│   ├── train_cmd_test.py    # PTQ train guard, checkpoint paths
│   ├── user_config_test.py  # .rl-dbs.yaml merge / persist
│   └── tui_data_test.py     # Benchmarks tab data layer
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
    └── ddpg/
        ├── buffer_test.py
        ├── checkpoint_test.py
        ├── checklist_test.py    # Mehregan §IV replication checklist
        ├── eval_test.py
        ├── matlab_trainer_test.py   # train + eval on MatlabPlant (@pytest.mark.matlab)
        ├── networks_test.py
        ├── quantization_test.py     # PTQ/QAT stubs
        ├── replication_test.py
        └── trainer_test.py
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
| **Paper replication benchmarks** | `rl-dbs benchmark` + `results/` (Phase 4) | full suites per [benchmarking.md](../benchmarking.md); smoke subsets in pytest only |

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

Skip when MATLAB is unavailable: `tests/conftest.py` skips `@pytest.mark.matlab` tests when batch `license('test','MATLAB')` fails. With MATLAB, run `source scripts/matlab/env.sh` and `uv sync --group matlab` first ([matlab.md](../matlab.md)).

---

## 5. Adding tests for a change

1. Implement under `envs/` or `controllers/<name>/` per the relevant spec.
2. Add or extend tests under the matching `tests/` subtree.
3. Run `uv run pytest` before opening a PR.
4. If behavior is spec-defined, update the spec in the same change ([conventions.md](conventions.md)).

Shared fixtures (env instances, seeds, reference traces) belong in `tests/conftest.py` or `tests/envs/conftest.py` as the suite grows.

---

## 6. Fresh VMs (Phase 4 portability)

Automated tests do **not** replace a clean-machine run. After changing `scripts/setup.sh`, `scripts/matlab/`, or install docs, run the Multipass / Sandbox workflow — **validation only, not training**. Full guide: [fresh-validation.md](fresh-validation.md).

1. **Linux:** Multipass Ubuntu 24.04 on a Windows host.
2. **Windows (no WSL):** Windows Sandbox + Git Bash.
3. Run `bash scripts/validate-fresh.sh` (or `bash scripts/setup.sh --python-only --non-interactive --validate`) and save the printed report block.
4. MATLAB verify (`bash scripts/matlab/verify.sh`) stays on **WSL**, not in Sandbox.
5. File doc or script fixes when prompts, paths, or dependencies drift.

**macOS** fresh validation is deferred (no maintainer hardware). CI default (`pytest -m "not matlab"`) is the fast gate; Multipass + Sandbox are the portability gate for Linux and Windows.
