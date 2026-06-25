# Benchmarking and cross-controller comparison

This document defines how to **compare controllers and variants** (iterations) on the **same** computational environment. **Phase 4** implements the suite runner, `results/` layout, and integration with [`rl-dbs benchmark`](cli.md) and [`rl-dbs-tui`](tui.md); status: [development/roadmap.md](development/roadmap.md) §2.

**Related specs:** Shared plant — [plant.md](plant.md); Mehregan Gym API — [environment.md](environment.md); per-controller training — [controllers/](controllers/); CLI — [cli.md](cli.md); TUI — [tui.md](tui.md).

---

## 1. Goals

| Goal | Notes |
|------|--------|
| **Fair comparison** | Same **parkinsonian plant** (Kumaravelu et al., 2016) and documented seeds; timing, observation, action, and reward follow the **suite** (per-paper replication vs optional cross-paper — §3). |
| **Controller vs controller** | `ddpg`, `snn`, `sea_dbs`, plus **non-RL baselines** on identical eval seeds. |
| **Variant vs variant** | Multiple runs per controller (e.g. full-precision vs PTQ, hyperparameter sweeps, architecture tweaks) distinguished by a stable **variant** id. |
| **Reproducibility** | Every logged run records env config, controller package, variant, seeds, and spec/git metadata. |

---

## 2. Run identity

Each benchmark run should be uniquely described by:

| Field | Example | Meaning |
|-------|---------|---------|
| `controller` | `ddpg`, `snn`, `sea_dbs`, `baseline` | Package or non-learned baseline family. |
| `variant` | `paper`, `ptq-int8`, `init-30hz`, `wider-cnn` | Iteration within that controller (paper replication is just one variant). |
| `run_id` | `20260518-143022-a1b2` | Unique instance (timestamp + short suffix). |
| `seed` | `42` | RNG seed for plant init and eval rollouts. |

**Convention:** `controller` matches `controllers/<name>/`. `variant` is a short slug (lowercase, hyphens); default replication is `paper` unless the paper itself defines named ablations (e.g. Mehregan et al. **30 Hz** init → `init-30hz`).

---

## 3. Evaluation protocol and suites

### 3.1 One plant, multiple RL interfaces

Every controller trains against the **same** Kumaravelu et al. (2016) CBGT model ([plant.md](plant.md)). The **Gymnasium contract in `envs/`** follows **Mehregan et al.** (2 s steps, $P_\beta$-only state, pattern actions, Eq. (8) reward) — [environment.md](environment.md). That is the right default for `ddpg` and for **baselines** shared across papers.

**Nguyen et al.** and **Ravivarapu et al.** use different step durations, observations, actions, and reward shapes (see [controllers/snn/replication.md](controllers/snn/replication.md), [controllers/sea_dbs/replication.md](controllers/sea_dbs/replication.md)). Their packages implement **adapters** around the shared plant wrapper—they do not fork the biophysical network.

### 3.2 Per-paper suites (primary — replication)

Use a **named suite per source paper** so eval matches that paper’s protocol. Controllers and baselines in the same suite are directly comparable (variants, quantization, ablations).

| Suite name | Env / adapter profile | Spec anchors |
|------------|----------------------|--------------|
| `mehregan_eval` | Shared `envs/` API: 2 s steps, $P_\beta$, Eq. (8) | [environment.md](environment.md), [controllers/ddpg/replication.md](controllers/ddpg/replication.md) |
| `nguyen_eval` | Adapter: 100 ms steps, spike obs, α–β (7–35 Hz), **Nguyen Eq. (7)** reward | [controllers/snn/replication.md](controllers/snn/replication.md) |
| `sea_dbs_eval` | Adapter: 2 ms × 30 steps × 150 episodes, binary pulse, **Ravivarapu Eq. (7)** reward | [controllers/sea_dbs/replication.md](controllers/sea_dbs/replication.md) |

**Baselines** for `mehregan_eval` (and optionally reused as plant-only checks in other suites):

| Baseline id | Description |
|-------------|-------------|
| `none` | No stimulation |
| `cdbs-130hz` | Conventional ~130 Hz STN DBS |
| `periodic-45hz` | Periodic 45 Hz (Mehregan et al. comparison) |
| `periodic-30hz` | Periodic 30 Hz ablation where relevant |

**Learned controllers:** After training, evaluate with **fixed seed(s)**. For `mehregan_eval`, follow Mehregan et al. §IV.A.2 (reset / baseline segment, then repeated stimulation steps) once segment timing is resolved in [environment.md](environment.md) §8.

### 3.3 Cross-paper suite (optional — same plant, not same paper table)

An optional suite (e.g. `cross_controller_plant`) may run **multiple controllers on identical plant seeds** and log **plant-level** metrics (raw GPi $P_\beta$, mean stim frequency, episode duration in simulated seconds). It must **not** imply that `reward_sum` or per-step $R$ are interchangeable across controllers unless every run uses an explicitly documented shared reward (not the default).

Record in the manifest: `suite`, `adapter: true|false`, and which metrics are **paper-comparable** vs **plant-only**. Nguyen and SEA-DBS eval in this mode still use their adapters for actions; only metrics defined for all runs belong in the main comparison column.

### 3.4 Adapters

If a controller’s paper used a different observation or action interface, the **adapter** in `controllers/<name>/` implements that paper’s I/O while calling the shared plant. For cross-paper suites, adapters stay on; the manifest documents what was held equal (plant, seeds) vs what was not (reward definition, step count).

---

## 4. Comparable metrics

Log a **common core** on every eval rollout. Which columns are **comparable across controllers** depends on the suite (§3):

| Metric | Description | Cross-paper safe? |
|--------|-------------|-------------------|
| `p_beta_mean` | Mean GPi **beta-band** power $P_\beta$ (**13–35 Hz**, Mehregan / SEA-DBS Eq. (1)); log raw and normalized if used for reward. | Yes as a **shared plant readout**; not the Nguyen training objective |
| `p_beta_final` | $P_\beta$ at end of eval (or last step), same band as above. | Yes (same caveat) |
| `alpha_beta_mean` | Mean GPi **$\alpha$–$\beta$** power (**7–35 Hz**, Nguyen et al. §II.A, §IV). | **Within `nguyen_eval` only** unless all suites compute it for cross-plant tables |
| `alpha_beta_final` | $\alpha$–$\beta$ at end of eval. | Same as `alpha_beta_mean` |
| `reward_sum` | Sum of per-step $R$ over eval episode(s). | **Within-paper suite only** — Mehregan Eq. (8) (linear below $\beta_t$), Nguyen Eq. (7) (energy + $\alpha$–$\beta$), Ravivarapu Eq. (7) (quadratic both branches) are **not** interchangeable |
| `stim_frequency_mean` | Mean stimulation frequency over eval (when defined for that controller/baseline). | Usually yes; document definition in manifest |
| `episode_length` | Steps or simulated seconds completed. | Compare only within same `protocol` / suite |

**Training metrics** (for learning curves, not always comparable across algorithms): episode return, steps to threshold, buffer size, wall time.

**Controller-specific extensions** go in a nested object (e.g. `metrics_extra.quantization = "int8"`) so the core schema stays stable.

---

## 5. Comparison suites

A **suite** is a named, versioned eval config:

```yaml
# Example only — format intentionally open until implemented
name: mehregan_eval
version: 1
protocol: mehregan   # 2 s steps, P_beta, Eq. (8)
seeds: [0, 1, 2, 3, 4]
env_ref: environment.md#5-timing-and-transitions
controllers:
  - { controller: baseline, variant: cdbs-130hz }
  - { controller: baseline, variant: periodic-45hz }
  - { controller: ddpg, variant: paper }
  - { controller: ddpg, variant: ptq-fp16 }   # PTQ after full-precision train (§IV.A.3)
  - { controller: ddpg, variant: ptq-int8 }
  - { controller: ddpg, variant: qat }         # QAT train + eval
```

```yaml
name: nguyen_eval
version: 1
protocol: nguyen
seeds: [0, 1, 2, 3, 4]
controllers:
  - { controller: snn, variant: paper }
```

```yaml
name: sea_dbs_eval
version: 1
protocol: sea_dbs
seeds: [0, 1, 2, 3, 4]
controllers:
  - { controller: sea_dbs, variant: paper }          # full SEA-DBS (PM+GS)
  - { controller: sea_dbs, variant: baseline }       # DDPG w/o PM or GS (Fig. 4–5)
  - { controller: sea_dbs, variant: baseline-pm }    # + predictive model only (Fig. 7)
  - { controller: sea_dbs, variant: baseline-gs }  # + Gumbel-Softmax only (Fig. 7)
```

```yaml
name: cross_controller_plant
version: 1
protocol: cross_paper
seeds: [0, 1, 2]
metrics: [p_beta_mean, alpha_beta_mean, stim_frequency_mean]   # plant readouts; not reward_sum
controllers:
  - { controller: ddpg, variant: paper, adapter: false }
  - { controller: snn, variant: paper, adapter: true }
  - { controller: sea_dbs, variant: paper, adapter: true }
```

Run each suite once; produce one **manifest** plus one result directory per `(controller, variant, seed)` (or aggregate seeds in post-processing — **intentionally open**). Do not mix protocols inside one suite without versioning the suite `name` / `protocol` field.

---

## 6. Results layout (planned)

```
results/
  <suite_name>/
    manifest.json      # suite version, env hash/config, metric definitions, timestamps
    runs/
      <controller>_<variant>_<run_id>/
        config.json      # controller, variant, seeds, hyperparameters, checkpoint path
        metrics.json     # core + extra metrics
        timeseries/      # optional: per-step P_beta, reward, actions
```

- **`results/`** is local output (gitignored); checkpoints may live under `artifacts/` or next to runs — **intentionally open**.
- Do not commit large binaries or MATLAB temp files.

---

## 7. Implementation roadmap

| Step | Status |
|------|--------|
| Spec (this document) | Done |
| `envs/` implements [environment.md](environment.md) | Done |
| Mehregan **PTQ** / **QAT** in `controllers/ddpg/` ([replication.md](controllers/ddpg/replication.md) §6) | Not started |
| Each controller exposes `train()` / `evaluate(seed, checkpoint)` with shared metric dict | Partial (`ddpg` only) |
| Suite runner (`benchmarks/` + `rl-dbs benchmark` per [cli.md](cli.md)) loads suite YAML, runs baselines + controllers (including quantized variants), writes `results/` | Done (Mehregan baselines + `ddpg`; PTQ/QAT pending) |
| Initial `rl-dbs` / `rl-dbs-tui` per Phase 4 ([cli.md](cli.md), [tui.md](tui.md)) | Not started |
| Setup scripts: `scripts/setup.sh` + `scripts/matlab/`; fresh-VM validation ([setup.md](setup.md), [matlab.md](matlab.md)) | In progress |
| Summary script or notebook: table/plot across `controller` × `variant` | Not started (Phase 4+) |

When adding code, prefer a **thin** runner that calls into `controllers.*` and `envs` rather than duplicating plant logic in scripts.

---

## 8. Consistency checklist

- [ ] Suite pins env timing, biomarker band, and reward scaling; all controllers use the same suite for a given table.
- [ ] Every run logs `controller`, `variant`, `run_id`, and `seed`.
- [ ] Baselines (`none`, `cdbs-130hz`, `periodic-45hz`) included in full comparisons.
- [ ] Variants documented in `config.json` (e.g. quantization mode, init frequency).
- [ ] Paper-specific eval quirks called out in `metrics_extra` or suite notes, not silently mixed into core metrics.
