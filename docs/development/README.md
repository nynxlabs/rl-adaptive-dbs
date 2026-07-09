# Development docs

Repo-specific **roadmap, conventions, tooling, and status**. Paper-aligned behavior lives in the top-level specs under `docs/` (`plant.md`, `environment.md`, `controllers/`, etc.).

| Doc | Audience | Contents |
|-----|----------|----------|
| [roadmap.md](roadmap.md) | Everyone | Phases and implementation status |
| [conventions.md](conventions.md) | Contributors | Spec-driven workflow, naming, layout |
| [venv.md](venv.md) | Everyone | `uv`, lockfile, Python version |
| [testing.md](testing.md) | Contributors | pytest layout, markers, what to test |
| [fresh-validation.md](fresh-validation.md) | Maintainers | Multipass + Sandbox portability gate (not for training) |
| [phase4-results.md](phase4-results.md) | Everyone | Phase 4 benchmark outcomes + DDPG replication checklist audit |
| [replication-fidelity.md](replication-fidelity.md) | Everyone | **Single source of truth** — what matches Mehregan et al., what doesn't, extensions |

**Day-to-day setup:** [setup.md](../setup.md). **Specs:** [plant.md](../plant.md), [environment.md](../environment.md), [benchmarking.md](../benchmarking.md), [cli.md](../cli.md), [tui.md](../tui.md). **Vendor plant patches:** [reference-material/kumaravelu_vendor_patches.md](../reference-material/kumaravelu_vendor_patches.md).
