# Development docs

Repo-specific **roadmap, conventions, tooling, and status**. Paper-aligned behavior lives in [plant/](../plant/) and [controllers/](../controllers/) ([docs map](../index.md)).

**Current focus:** **figure replication** across Mehregan (Pass), Nguyen, and Ravivarapu — trackers under `docs/controllers/<controller>/README.md`. The phase roadmap ([roadmap.md](roadmap.md)) describes long-term architecture; day-to-day work follows **open panels**, not phase numbers.

| Doc | Audience | Contents |
|-----|----------|----------|
| [docs/controllers/ddpg/README.md](../controllers/ddpg/README.md) | Everyone | Mehregan panel status, gates, run commands (Pass 1b–6b) |
| [docs/controllers/snn/README.md](../controllers/snn/README.md) | Everyone | Nguyen DSQN panel tracker |
| [docs/controllers/sea-dbs/README.md](../controllers/sea-dbs/README.md) | Everyone | Ravivarapu SEA-DBS panel tracker |
| [replication-fidelity.md](replication-fidelity.md) | Everyone | Verified vs divergent vs added (Mehregan DDPG) |
| [roadmap.md](roadmap.md) | Everyone | Figure-first priorities + phase map and implementation status |
| [conventions.md](conventions.md) | Contributors | Spec-driven workflow, figure scripts, naming, layout |
| [venv.md](venv.md) | Everyone | `uv`, lockfile, Python version |
| [testing.md](testing.md) | Contributors | pytest layout, markers, what to test |
| [fresh-validation.md](fresh-validation.md) | Maintainers | Multipass + Sandbox portability gate (not for training) |
| [history/phase4-results.md](history/phase4-results.md) | Everyone | Phase 4 `mehregan_eval` outcomes + checklist audit (infrastructure; figures are the live bar) |
| [history/native-plant-port.md](history/native-plant-port.md) | Maintainers | How the plant was ported from MATLAB to native Python (finished) |

**Day-to-day setup:** [setup.md](../guides/setup.md). **Specs:** [plant.md](../plant/README.md), [environment.md](../plant/environment.md), [benchmarking.md](../guides/benchmarking.md), [cli.md](../guides/cli.md), [tui.md](../guides/tui.md), [docs/controllers/ddpg/README.md](../controllers/ddpg/README.md). **Vendor plant patches:** [plant/vendor-patches.md](../plant/vendor-patches.md).
