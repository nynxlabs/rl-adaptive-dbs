# Command-line interface specification

This document defines the **`rl-dbs`** CLI: training, evaluation, benchmarking, configuration, and repository introspection. **Phase 4** starts implementation: `benchmark` and `info` first, then Mehregan `ddpg` `train` / `eval` (including quantized variants). Broader controller coverage and packaging hardening continue in Phases 5–9 ([development/roadmap.md](development/roadmap.md)).

**Related specs:** [development/roadmap.md](development/roadmap.md) (phases), [benchmarking.md](benchmarking.md) (suites, results layout), [tui.md](tui.md) (read-only monitor), [environment.md](environment.md) (Mehregan Gym API), [plant.md](plant.md) (plant config), [controllers/](controllers/) (per-paper training). Tooling: [development/venv.md](development/venv.md) (`uv run`).

---

## 1. Goals

| Goal | Notes |
|------|--------|
| **Single entry point** | One command name across platforms: `rl-dbs`. |
| **Spec-driven behavior** | CLI delegates to `envs/`, `controllers/`, and suite manifests—does not duplicate plant dynamics or reward definitions. |
| **Reproducibility** | Every run logs `controller`, `variant`, `seed`, and paths to config/checkpoints per [benchmarking.md](benchmarking.md) §2. |
| **Cross-platform** | Paths via `pathlib`; no POSIX-only assumptions in library code; shell examples stay agnostic where possible. |

---

## 2. Distribution and invocation

### 2.1 Package entry point

Register a console script in `pyproject.toml`:

```toml
[project.scripts]
rl-dbs = "rl_adaptive_dbs.cli:main"
```

The implementation lives in **`rl_adaptive_dbs.cli`** (`rl_adaptive_dbs/cli.py`); the **user-facing command** is fixed: `rl-dbs`.

### 2.2 Recommended invocation

From the repository root (or any cwd with a discoverable project config):

```bash
uv run rl-dbs <subcommand> [options]
```

| Mode | When |
|------|------|
| `uv run rl-dbs …` | Default for docs and CI—uses project lockfile and venv. |
| `rl-dbs …` after `uv sync` | Same if the venv is activated ([development/venv.md](development/venv.md)). |

Do not require users to set `PYTHONPATH` manually; editable installs from `uv sync` must suffice.

### 2.3 Configuration discovery

Plant, environment, and global CLI defaults merge in this order (later wins):

1. Built-in dataclass defaults (`envs/plant/config.py`, `envs/mehregan/config.py`).
2. User file **`.rl-dbs.yaml`** (or `.rl-dbs.yml`): walk from the current working directory up to the git root. Template: **`.rl-dbs.example.yaml`** at the repo root (`cp .rl-dbs.example.yaml .rl-dbs.yaml`).
3. Environment variables: `RL_DBS_CONFIG` (explicit file path), `RL_DBS_SEED`, `RL_DBS_RESULTS_DIR`, `RL_DBS_MAX_THREADS` (thread-pool cap for Numba/OpenBLAS when no `--max-threads` flag).
4. Explicit CLI flags (`--config`, `--seed`, `--results-dir`, `--max-threads`, etc.).

`train`, `eval`, and `benchmark` construct `MehreganEnv` from the merged file settings when present. Copy **`.rl-dbs.example.yaml`** to **`.rl-dbs.yaml`** to customize (the latter is gitignored by default).

All filesystem paths accepted on the command line are normalized with `pathlib.Path` and expanded for `~` on every platform.

---

## 3. Command structure

```
rl-dbs [--verbose | --quiet] [--config PATH] [--seed SEED] [--max-threads N] <subcommand> [subcommand options]
```

| Subcommand | Phase (roadmap) | Role |
|------------|-----------------|------|
| `train` | 4 (`ddpg`), 5+ (others) | Train a controller variant. |
| `eval` | 4 (`ddpg`), 5+ (others) | Evaluate a trained checkpoint on a suite or single rollout. |
| `benchmark` | 4 | Run a full suite from a YAML manifest → `results/`. |
| `summary` | 4 | Print comparison table (and optional CSV) from existing `results/`. |
| `info` | 4 | Print available controllers, variants, suites, env summary. |
| `config` | 4 (`show`, `set --persist`) | Show or persist plant/env defaults. |

Global flags apply before the subcommand and affect logging only unless noted.

---

## 4. Global flags

| Flag | Short | Effect |
|------|-------|--------|
| `--verbose` | `-v` | Log level `DEBUG`; include spec/git metadata in run logs when writing `results/`. |
| `--quiet` | `-q` | Log level `WARNING`; suppress progress bars on stderr. |
| `--config` | | Path to `.rl-dbs.yaml` (overrides discovery walk). |
| `--seed` | | Default RNG seed when a subcommand omits `--seeds` (overrides `defaults.seed` in the config file). |
| `--max-threads` | | Cap in-process Numba/OpenBLAS thread pools (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMBA_NUM_THREADS`, etc.). Pair with `taskset` for a hard logical-CPU cap. Fallback: `RL_DBS_MAX_THREADS`. |

`--verbose` and `--quiet` are mutually exclusive; if both are passed, exit **2** (usage error).

### 4.1 Standalone scripts (`python -m rl_adaptive_dbs.run`)

Repo scripts under `scripts/` that import NumPy at module load should be launched through the runner so thread limits apply **before** those imports:

```bash
uv run python -m rl_adaptive_dbs.run scripts/probes/run_task177_continuous_freq_probe.py --continuous-only
```

By default the runner caps in-process thread pools at **3** (same budget as the TUI Run tab for plant-heavy scripts). Override with `--max-threads N` or `RL_DBS_MAX_THREADS`. Pair with `taskset` when you need a hard logical-CPU pin.

`rl-dbs train`, `eval`, and `benchmark` apply the same default at console entry (before heavy imports). `scripts.lib.train_runtime_guard.run_main` and `scripts.lib.probe_runtime.run_main` apply it when scripts defer heavy imports until `main()` — prefer the runner when NumPy loads at import time.

---

## 5. Subcommands

### 5.1 `train`

Train a **learned** controller (not baselines). Behavior must match the controller’s [replication.md](controllers/ddpg/replication.md) (and siblings) and the env/adapter profile for that controller.

```
rl-dbs train --controller NAME --variant VARIANT [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--controller` | Yes | Package id: `ddpg`, `snn`, `sea_dbs`. |
| `--variant` | Yes | Slug, e.g. `paper`, `init-30hz`, `ptq-int8`. Default replication id is `paper` ([development/conventions.md](development/conventions.md)). |
| `--seeds` | No | Comma-separated training seeds (default: global `--seed` only). |
| `--episodes` | No | Override training episode count when spec allows (Mehregan default **10** — [environment.md](environment.md) §8). |
| `--checkpoint-dir` | No | Directory for checkpoints (default: `artifacts/<controller>/`; files named `{variant}_train{seed}.pt`). |
| `--resume` | No | Path to checkpoint to resume training. |
| `--hyperparams` | No | Path to JSON/YAML hyperparameter overlay (merged over variant defaults). |
| `--adapter` | No | For `snn` / `sea_dbs`: force adapter on (default **true** for those controllers). Ignored for `ddpg` on Mehregan `envs/`. |
| `--dry-run` | No | Validate config and print resolved training plan without running plant steps. |
| `--parallel` | No | Process count for independent seeds (default **1**, sequential). Each worker owns one MATLAB engine (~1 GB RAM). |

**Delegation:** Call `controllers.<name>.train(...)` (or equivalent module API) with resolved config. Training metrics (episode return, loss, wall time) go to:

- **Stdout/stderr:** structured logs (§8).
- **Checkpoint dir:** `train_log.jsonl` (one JSON object per line) — path **intentionally open** until artifact policy is fixed.

**Examples:**

```bash
# Mehregan DDPG, paper replication, single seed
uv run rl-dbs train --controller ddpg --variant paper --seeds 42

# 30 Hz initialization ablation
uv run rl-dbs train --controller ddpg --variant init-30hz --seeds 0,1,2

# QAT (fake-quant stubs during training)
uv run rl-dbs train --controller ddpg --variant qat --seeds 0
```

**PTQ variants (`ptq-fp16`, `ptq-int8`) are eval-only** — `train` exits with an error. Train `paper` (or `init-30hz`) first, then `eval` or `benchmark` with the PTQ slug ([controllers/ddpg/replication.md](controllers/ddpg/replication.md) §6).

---

### 5.2 `eval`

Evaluate a **trained** policy or run a **baseline** controller on one or more seeds. For suite-shaped eval, prefer `benchmark`; `eval` is for ad hoc runs and smoke tests.

```
rl-dbs eval --controller NAME --variant VARIANT [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--controller` | Yes | `ddpg`, `snn`, `sea_dbs`, or `baseline`. |
| `--variant` | Yes | Controller variant or baseline id (`cdbs-130hz`, `periodic-45hz`, `periodic-30hz`, `none`). |
| `--checkpoint` | Conditional | Required for learned controllers; omitted for `baseline`. |
| `--seeds` | No | Eval seeds (default: global `--seed`). |
| `--suite` | No | Suite name, e.g. `mehregan_eval` — applies protocol from [benchmarking.md](benchmarking.md) §3. |
| `--results-dir` | No | Output root (default: `results/`). |
| `--run-id` | No | Explicit `run_id`; if omitted, generate `YYYYMMDD-HHMMSS-<4char>`. |
| `--adapter` | No | `true` / `false` for adapter use (manifest default when `--suite` set). |
| `--parallel` | No | Process count for independent seeds (default **1**, sequential). Each worker owns one MATLAB engine (~1 GB RAM). |

**Delegation:** Call `controllers.<name>.evaluate(seed, checkpoint, suite_config)` or baseline runner on `envs/` per suite protocol. Write one run directory per [benchmarking.md](benchmarking.md) §6 when `--results-dir` is set.

**Examples:**

```bash
# DDPG paper checkpoint on Mehregan eval protocol, one seed
uv run rl-dbs eval --controller ddpg --variant paper \
  --checkpoint artifacts/ddpg/paper_train0.pt --suite mehregan_eval --seeds 42

# Baseline: conventional 130 Hz cDBS
uv run rl-dbs eval --controller baseline --variant cdbs-130hz \
  --suite mehregan_eval --seeds 0,1,2,3,4

# SNN with adapter, no suite file (implicit nguyen_eval defaults)
uv run rl-dbs eval --controller snn --variant paper \
  --checkpoint artifacts/snn/paper/final.pt --adapter true --seeds 0
```

---

### 5.3 `benchmark`

Run a **full benchmark suite** from a YAML manifest. Primary interface for Phase 4 ([development/roadmap.md](development/roadmap.md)).

```
rl-dbs benchmark --suite PATH | --suite-name NAME [options]
```

| Option | Required | Description |
|--------|----------|-------------|
| `--suite` | One of suite flags | Path to suite YAML (e.g. `suites/mehregan_eval.yaml`). |
| `--suite-name` | One of suite flags | Shorthand: load `suites/<name>.yaml` from project root or config search path. |
| `--results-dir` | No | Default `results/`. |
| `--controllers` | No | Filter manifest entries: comma-separated `controller:variant` pairs. |
| `--seeds` | No | Override manifest `seeds` list. |
| `--parallel` | No | Process count for independent runs (default **1**, sequential). Each worker owns one MATLAB engine (~1 GB RAM); use **3–4** on ~17 GB RAM. |
| `--dry-run` | No | Print planned runs without executing. |

**Manifest format:** As in [benchmarking.md](benchmarking.md) §5 (`name`, `version`, `protocol`, `seeds`, `controllers`, optional `metrics`, `env_ref`). The runner must not mix protocols inside one suite without bumping `version`.

**Output layout** ([benchmarking.md](benchmarking.md) §6):

```
results/<suite_name>/
  manifest.json
  runs/<controller>_<variant>_<run_id>/
    config.json
    metrics.json
    timeseries/   # optional
```

**Examples:**

```bash
# Full Mehregan replication suite
uv run rl-dbs benchmark --suite suites/mehregan_eval.yaml

# Cross-controller plant comparison only for ddpg + snn
uv run rl-dbs benchmark --suite-name cross_controller_plant \
  --controllers ddpg:paper,snn:paper

# Dry run to inspect run matrix
uv run rl-dbs benchmark --suite suites/mehregan_eval.yaml --dry-run -v
```

**Baselines** in manifests use `controller: baseline` with variants `none`, `cdbs-130hz`, `periodic-45hz`, `periodic-30hz` ([benchmarking.md](benchmarking.md) §3.2).

---

### 5.4 `summary`

Print a comparison table from an existing `results/<suite_name>/` tree (no new plant steps). Optional CSV export.

```
rl-dbs summary [--results-dir PATH] [--suite-name NAME] [--csv PATH] [--width N]
```

| Option | Description |
|--------|-------------|
| `--results-dir` | Root containing suite subdirs (default: `results/`). |
| `--suite-name` | Suite subdir to summarize (default: latest by mtime). |
| `--csv` | Write the same rows to a CSV file. |
| `--width` | Terminal table width (default **100**). |

**Examples:**

```bash
uv run rl-dbs summary --suite-name mehregan_eval_smoke
uv run rl-dbs summary --results-dir results/ --csv results/summary.csv
```

---

### 5.5 `info`

Print repository and runtime introspection (no plant steps required beyond optional env probe).

```
rl-dbs info [--json] [topic]
```

| `topic` | Default if omitted | Output |
|---------|-------------------|--------|
| *(none)* | all sections | Controllers, variants, suites, env/plant summary. |
| `controllers` | | Registered packages and known variants per controller. |
| `variants` | | Variants for `--controller` if passed (`--controller` filter). |
| `suites` | | Discovered suite YAML names under `suites/`. |
| `env` | | Mehregan timing, $\beta_t$, bands from [environment.md](environment.md). |
| `plant` | | $\Delta t$, DBS defaults from [plant.md](plant.md). |
| `version` | | Package version, Python version, optional git commit. |

| Option | Description |
|--------|-------------|
| `--json` | Machine-readable output for scripts. |
| `--controller` | Filter `variants` topic. |

**Example:**

```bash
uv run rl-dbs info env --json
uv run rl-dbs info controllers
```

---

### 5.6 `config`

Show or set **non-secret** configuration values: plant integration step, DBS waveform defaults, biomarker bands, Mehregan RL step duration, reward threshold.

```
rl-dbs config show [KEY ...]
rl-dbs config set KEY VALUE [--persist]
```

| Key family | Spec source | Examples |
|------------|-------------|----------|
| `plant.dt` | [plant.md](plant.md) §5 | `0.02` (ms) Mehregan target |
| `plant.pd` | [plant.md](plant.md) §3 | `1` (parkinsonian) |
| `plant.corstim` | [plant.md](plant.md) | `0` |
| `plant.neurons_per_region` | [plant.md](plant.md) | `10` |
| `env.dt_rl` | [environment.md](environment.md) §5 | `2.0` (s) |
| `env.beta_t` | [environment.md](environment.md) §6 | `0.35` |
| `env.biomarker.band_hz` | [environment.md](environment.md) §3 | `13`, `35` |
| `env.episode_steps` | [environment.md](environment.md) §5 | `30` |
| `env.reward_scale` | [environment.md](environment.md) | `10.0` |
| `env.observation_scale` | [environment.md](environment.md) | `1000.0` |
| `env.state_length` | [environment.md](environment.md) §4 | `15` |
| `defaults.seed` | this document §4 | `42` |
| `defaults.results_dir` | [benchmarking.md](benchmarking.md) | `results` |
| `defaults.checkpoint_dir` | this document §5.1 | `artifacts/ddpg` (optional) |

**File format:** YAML (`.rl-dbs.yaml`). See **`.rl-dbs.example.yaml`**. `config show` prints the **effective** merged values and the discovered `config_file` path when present. `config set` without `--persist` previews one key; `--persist` writes or updates `.rl-dbs.yaml` (project root when no file is discovered).

**Examples:**

```bash
cp .rl-dbs.example.yaml .rl-dbs.yaml
uv run rl-dbs config show plant.dt env.dt_rl env.beta_t
uv run rl-dbs config set env.beta_t 0.42 --persist
```

---

## 6. Controllers, variants, and adapters

| `controller` | Direct `envs/`? | Adapter | Example variants |
|--------------|-----------------|---------|------------------|
| `ddpg` | Yes (Mehregan API) | No | `paper`, `init-30hz`, `ptq-fp16`, `ptq-int8`, `qat` |
| `snn` | No | Yes (Nguyen) | `paper` |
| `sea_dbs` | No | Yes (SEA-DBS) | `paper`, `baseline`, `baseline-pm`, `baseline-gs` |
| `baseline` | Yes (for Mehregan suite) | No | `none`, `cdbs-130hz`, `periodic-45hz`, `periodic-30hz` |

Variant strings: lowercase, hyphens; must match [benchmarking.md](benchmarking.md) §2 and per-controller specs. Unknown `controller` or `variant` → exit **3** with a list of valid ids from `rl-dbs info`.

---

## 7. Output conventions

### 7.1 Logging (stdout/stderr)

| Stream | Content |
|--------|---------|
| **stdout** | Human-readable progress when not `--quiet`; with `--json` on supported subcommands, **only** JSON lines on stdout. |
| **stderr** | Logs, warnings, errors (default log format: `%(levelname)s %(name)s: %(message)s`). |

Structured training/benchmark events should use JSON lines in log files under `results/` or `artifacts/`, not mixed ad hoc prints on stdout during `benchmark`.

### 7.2 Metrics and artifacts

| Output | Location | Schema |
|--------|----------|--------|
| Benchmark metrics | `results/<suite>/runs/.../metrics.json` | [benchmarking.md](benchmarking.md) §4 |
| Run config | `.../config.json` | `controller`, `variant`, `run_id`, `seed`, hyperparams, checkpoint path |
| Suite manifest | `results/<suite>/manifest.json` | Suite version, env snapshot, git hash |
| Checkpoints | `artifacts/<controller>/` | `{variant}_train{seed}.pt` (PTQ loads FP source checkpoint per variant) |

Do not commit `results/` or large checkpoints ([development/conventions.md](development/conventions.md)).

---

## 8. Cross-platform considerations

| Topic | Requirement |
|-------|-------------|
| **Paths** | Accept `\` and `/`; store paths in JSON as forward slashes or platform-neutral relative paths from repo root. |
| **Working directory** | Commands may be run from any directory if `--suite` / `--results-dir` are absolute or relative to cwd; project discovery uses git root when available. |
| **Shell** | Docs show `uv run rl-dbs ...` without bash-specific syntax; line continuation uses `\` in examples (works in bash, zsh, sh; PowerShell users copy one line or use `` ` ``). |
| **MATLAB bridge** | If plant backend is MATLAB, `train`/`eval`/`benchmark` fail fast with exit **4** and a clear message when MATLAB is unavailable ([development/testing.md](development/testing.md) `@pytest.mark.matlab`). |
| **Console encoding** | UTF-8 preferred; avoid box-drawing in CLI text (reserve for TUI — [tui.md](tui.md)). |
| **Process exit** | Use `sys.exit(code)` so Windows and Unix share semantics. |

---

## 9. Exit codes

| Code | Meaning |
|------|---------|
| **0** | Success. |
| **1** | Runtime failure (training failed, plant error, write error). |
| **2** | Usage error (unknown subcommand, bad flags, mutually exclusive options). |
| **3** | Unknown controller, variant, suite, or config key. |
| **4** | Missing dependency (MATLAB, checkpoint not found, optional backend). |
| **5** | Partial benchmark failure (some runs failed; manifest records failures). |

CI smoke tests may invoke `rl-dbs info` and `rl-dbs benchmark --dry-run` expecting **0**.

---

## 10. Implementation roadmap

| Step | Phase | Status |
|------|-------|--------|
| Spec (this document) | 1 | Done |
| Entry point + `info`, `config show`, `config set --persist`, `summary` | 4 | Done |
| `benchmark` suite runner | 4 | Done |
| `train` / `eval` for `ddpg` (incl. PTQ/QAT variants) | 4 | Done |
| `train` / `eval` for `snn`, `sea_dbs` | 5 | Not started |
| Cross-platform packaging polish | 8+ | Not started |

Prefer thin `rl_adaptive_dbs/*.py` modules (`cli.py`, `train_cmd.py`, `eval_cmd.py`, …) that call into `controllers.*` and `envs` per [benchmarking.md](benchmarking.md) §7.

---

## 11. Consistency checklist

- [ ] Subcommands match [development/roadmap.md](development/roadmap.md) phase plan (`benchmark` before full `train` for all controllers is acceptable).
- [ ] Suite YAML validated against [benchmarking.md](benchmarking.md) §5 before runs start.
- [ ] Every `eval` / `benchmark` run logs `controller`, `variant`, `run_id`, `seed`.
- [ ] Baselines use `controller=baseline` and documented variant ids.
- [ ] `config` keys trace to [environment.md](environment.md) or [plant.md](plant.md).
- [ ] Examples in docs use `uv run rl-dbs`.

---

## 12. Open questions / TBD

### 1. CLI Python package layout

**Fixed:** command name `rl-dbs`, console script `rl_adaptive_dbs.cli:main`.

### 2. Persistent config file

**Fixed:** `.rl-dbs.yaml` (see `.rl-dbs.example.yaml`), parent-directory walk to git root, `config set --persist`. **Open:** TOML variant, JSON Schema validation. **Decide in** later packaging work if needed.

### 3. Parallel benchmark workers

**Fixed:** `--parallel` on `train`, `eval`, and `benchmark` (default **1**). **Process pool** — each worker starts its own `matlab.engine`, runs assigned seed(s), then closes the engine. Not thread-safe for MATLAB. Injected test envs (mock plant) stay sequential. Cap workers at seed/run count automatically.

### 4. Hyperparameter file schema

**Fixed:** `--hyperparams` path overlay. **Open:** JSON Schema per controller. **Decide in** `controllers/<name>/` config modules.

### 5. Checkpoint naming and `artifacts/` layout

**Fixed:** `artifacts/<controller>/{variant}_train{seed}.pt`; suite YAML `checkpoint_dir` (default `artifacts/ddpg`). PTQ variants resolve the full-precision source checkpoint automatically.

### 6. `eval` without `--suite`

**Fixed:** ad hoc rollouts allowed. **Open:** default protocol per controller when `--suite` omitted. **Decide in** controller `evaluate()` API.

### 7. Integration with TUI

**Fixed:** CLI writes `results/` and `train_log.jsonl`; TUI reads them ([tui.md](tui.md)). **Open:** whether `train` emits a TUI-friendly status file. **Decide in** when TUI is implemented.
