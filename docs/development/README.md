# Development docs

Repo-specific **roadmap, conventions, tooling, and status**. Paper-aligned behavior lives in the top-level specs under `docs/` (`plant.md`, `environment.md`, `controllers/`, etc.).

**Current focus:** **figure replication** for Mehregan et al. — panel checklists, gates, and side-by-side PNGs in [figures/mehregan/replications.md](../figures/mehregan/replications.md). The phase roadmap ([roadmap.md](roadmap.md)) describes long-term architecture; day-to-day work follows **open panels**, not phase numbers.

| Doc | Audience | Contents |
|-----|----------|----------|
| [figures/mehregan/replications.md](../figures/mehregan/replications.md) | Everyone | **Primary goal tracker** — Mehregan panel status, gates, run commands |
| [replication-fidelity.md](replication-fidelity.md) | Everyone | Verified vs divergent vs added (Mehregan DDPG) |
| [roadmap.md](roadmap.md) | Everyone | Figure-first priorities + phase map and implementation status |
| [conventions.md](conventions.md) | Contributors | Spec-driven workflow, figure scripts, naming, layout |
| [venv.md](venv.md) | Everyone | `uv`, lockfile, Python version |
| [testing.md](testing.md) | Contributors | pytest layout, markers, what to test |
| [fresh-validation.md](fresh-validation.md) | Maintainers | Multipass + Sandbox portability gate (not for training) |
| [phase4-results.md](phase4-results.md) | Everyone | Phase 4 `mehregan_eval` outcomes + checklist audit (infrastructure; figures are the live bar) |

**Day-to-day setup:** [setup.md](../setup.md). **Specs:** [plant.md](../plant.md), [environment.md](../environment.md), [benchmarking.md](../benchmarking.md), [cli.md](../cli.md), [tui.md](../tui.md), [figures/mehregan/replications.md](../figures/mehregan/replications.md). **Vendor plant patches:** [reference-material/kumaravelu_vendor_patches.md](../reference-material/kumaravelu_vendor_patches.md).
