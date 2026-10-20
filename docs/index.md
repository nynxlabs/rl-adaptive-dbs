# Docs

Where everything in `docs/` lives. Start with the [README](../README.md) for what this project is and the current status.

| Folder | What's in it |
|--------|--------------|
| [guides/](guides/) | Using the tools: [setup](guides/setup.md), [MATLAB backend](guides/matlab.md), [CLI](guides/cli.md), [TUI](guides/tui.md), [benchmarking](guides/benchmarking.md) |
| [plant/](plant/) | The shared model: the [Kumaravelu et al. (2016) plant](plant/README.md), the [RL environment](plant/environment.md) built on it, and our [patches to the vendor MATLAB code](plant/vendor-patches.md) |
| [controllers/](controllers/) | One folder per replicated paper's controller: its tracker, method spec, extensions, and one page per figure. [Overview and side-by-side showcase](controllers/README.md) |
| [extensions/](extensions/) | Work beyond the papers ([alphabet diversity](extensions/alphabet-diversity/README.md)) |
| [development/](development/) | Contributing: roadmap, conventions, testing, `uv`, fresh-machine validation, replication fidelity; finished snapshots in [development/history/](development/history/) |

## Controllers, papers, and where their files live

Docs are named after the controller; scripts and run outputs are named after the paper.

| Controller | Paper | Docs | Plot scripts | Run outputs (local) |
|------------|-------|------|--------------|---------------------|
| `ddpg` | Mehregan et al. | [controllers/ddpg/](controllers/ddpg/README.md) | `scripts/figures/papers/mehregan/<panel>/plot.py` | `artifacts/figures/papers/mehregan/<panel>/` |
| `snn` | Nguyen et al. | [controllers/snn/](controllers/snn/README.md) | `scripts/figures/papers/nguyen/<panel>/plot.py` | `artifacts/figures/papers/nguyen/<panel>/` |
| `sea_dbs` | Ravivarapu et al. | [controllers/sea-dbs/](controllers/sea-dbs/README.md) | `scripts/figures/papers/ravivarapu/<panel>/plot.py` | `artifacts/figures/papers/ravivarapu/<panel>/` |

Code: `src/controllers/<controller>/`, the plant and environment in `src/envs/`.

## Inside a controller folder

```
docs/controllers/<controller>/
  README.md              tracker: panel status table, links to every figure page
  method.md              the paper's method as we implement it
  extensions.md          work beyond the paper
  figures/<panel>.md     one page per figure
  figures/images/<panel>/  replication PNGs and the paper references they're compared with
```

Every figure page has the same sections: what the paper shows, status, how to reproduce it, gates, deviations and conventions, what was tried, open questions, and notes. The caption and gate tables between `<!-- caption-… -->` / `<!-- gates-… -->` markers are written by the plot and gate scripts; edit around them, not inside.
