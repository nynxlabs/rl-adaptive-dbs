# Probes (diagnostic only)

Do **not** treat this directory as the ship surface. Panel replication lives under `scripts/figures/papers/`.

**Keep here only** reusable / extension diagnostics that still have a docs home. Findings from one-off panel probes belong in `docs/figures/<paper>/<panel>.md` and the panel `plot.py`; delete the probe once that is done.

## Current contents

| Path | Purpose |
|------|---------|
| **`alphabet_diversity/`** | Alphabet-diversity extension sweeps and pipelines — see `docs/extensions/alphabet-diversity/` |

```bash
uv run python -m rl_adaptive_dbs.run scripts/probes/alphabet_diversity/run_plant_continuity_probe.py
```
