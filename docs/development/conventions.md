# Conventions

Contributor rules for **rl-adaptive-dbs** — layout, spec-driven workflow, and cross-controller comparison. Paper-aligned behavior lives in the **specs** under `docs/`. Phases and status: [roadmap.md](roadmap.md).

---

## Spec-driven changes

1. Read the relevant spec before non-trivial implementation.
2. Code under `envs/` or `controllers/<name>/`.
3. Update the spec in the **same change** if behavior or interfaces change.
4. Resolve “intentionally open” items in spec or code—not only in issues or chat.

| Layer | Spec path | Code path |
|-------|-----------|-----------|
| Plant | [plant.md](../plant.md) | `envs/` (dynamics backend) |
| Environment (Mehregan API) | [environment.md](../environment.md) | `envs/` |
| Controller | [controllers/ddpg/replication.md](../controllers/ddpg/replication.md), [controllers/snn/replication.md](../controllers/snn/replication.md), [controllers/sea_dbs/replication.md](../controllers/sea_dbs/replication.md) | `controllers/<name>/` |
| Benchmarks | [benchmarking.md](../benchmarking.md) | `benchmarks/` + `rl-dbs benchmark` |
| CLI | [cli.md](../cli.md) | `rl_adaptive_dbs.cli` (Phase 4 start) |
| User config | [cli.md](../cli.md) §2.3, §5.6 | `rl_adaptive_dbs/user_config.py`, `env_factory.py` |
| TUI | [tui.md](../tui.md) | `rl_adaptive_dbs.tui` (Phase 4 start) |

## Naming and layout

- Python packages match directories: `ddpg`, `snn`, `sea_dbs`.
- Controller docs mirror packages: `docs/controllers/ddpg/replication.md`, etc.
- Default paper-replication benchmark variant: `paper`.
- Distribution name `rl-adaptive-dbs` (hyphen); imports `envs`, `controllers` (underscore, no hyphen).

## Cross-platform code

- Use `pathlib`, `uv run`; avoid hard-coded POSIX-only paths in library code.
- Document OS-specific steps only in [venv.md](venv.md) or [setup.md](../setup.md).

## Documentation

- Cite papers by author/title in prose—not opaque ids like `paper_1`.
- Math in markdown: `$...$` inline, `$$...$$` display ([AGENTS.md](../../AGENTS.md)).

## Controllers vs environment

- **One plant** for all controllers; variants do not fork `envs/`.
- Paper-specific observation/action mismatches → **adapter** in `controllers/<name>/`.
- Do not duplicate plant dynamics inside controller packages.

## Cross-controller benchmarking

All controllers share the **Kumaravelu et al. (2016)** parkinsonian plant, but each paper defines its own **RL interface** (step length, observation, action, reward). The shared `envs/` package implements the **Mehregan et al.** Gym API ([environment.md](../environment.md)); `ddpg` can train on it directly. **Nguyen** and **Ravivarapu** need adapters (see [controllers/snn/replication.md](../controllers/snn/replication.md), [controllers/sea_dbs/replication.md](../controllers/sea_dbs/replication.md)).

| Comparison type | What is held fixed | What differs | When to use |
|-----------------|-------------------|--------------|-------------|
| **Within-paper** (`*_eval` suite) | Plant + that paper’s timing, obs, action, reward | Variants (PTQ, ablations, hyperparameters) | Replication and ablations |
| **Cross-paper** (optional suite) | Plant, seeds, integration step; often **$P_\beta$** logged for all | Per-controller training/eval protocol via adapters | High-level “same plant” tables—not claimable as identical to any single paper’s Table/Figure |

Do **not** put `snn` and `ddpg` in one suite and assume the same **reward_sum** or **episode_length** are comparable without reading the suite manifest: Nguyen uses **100 ms** steps and **α–β** feedback; SEA-DBS uses **2 ms** steps and **Eq. (7)** reward; Mehregan uses **2 s** steps and **Eq. (8)**. Cross-paper runs should log **plant-level** metrics (e.g. raw $P_\beta$, stim duty cycle) plus `adapter: true` and `suite` name. Details: [benchmarking.md](../benchmarking.md) §3, §5.

## Version control

Do **not** commit:

- `.venv/`, `results/`, MATLAB cache files
- `.rl-dbs.yaml`, `.rl-dbs.yml` (local overrides; template: `.rl-dbs.example.yaml`)
- Secrets, machine-local paths, large checkpoints (until an artifact policy exists)
