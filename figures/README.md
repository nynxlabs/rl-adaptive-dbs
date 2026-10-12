# Figures

Replication PNGs and paper crops: **`figures/<paper>/images/<panel>/`**. Trackers: **`figures/<paper>/replications.md`**.

Panel trackers:

| Paper | Tracker |
|-------|---------|
| Mehregan et al. | [figures/mehregan/replications.md](mehregan/replications.md) |
| Nguyen et al. | [figures/nguyen/replications.md](nguyen/replications.md) |
| Ravivarapu et al. (SEA-DBS) | [figures/ravivarapu/replications.md](ravivarapu/replications.md) |

Plot scripts: `scripts/figures/papers/`. JSON caches: `artifacts/figures/papers/`.

**Source papers.** The papers these figures replicate are kept with the notes, not in git (they are publisher PDFs): `papers/paper_1.{pdf,md}` (Mehregan et al.), `papers/paper_2.{pdf,md}` (Nguyen et al.), and `papers/paper_3.{pdf,md}` (Ravivarapu et al.) under the notes folder that `export.notes_dir` / `RL_DBS_NOTES_DIR` points to ([docs/cli.md](../docs/cli.md) §5.6). The `.md` files are text transcriptions; prefer them over the PDFs when reading methods.
