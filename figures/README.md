# Figures

Replication PNGs and the paper references they're compared with: **`figures/<paper>/images/<panel>/`**. Trackers: **`figures/<paper>/replications.md`**.

The published figures themselves are not in this repository. What each panel folder holds instead:

- **`digitized_v*.png`** (Mehregan et al., Nguyen et al.): our redraw of the values digitized from the paper's figure, labelled "Digitized from Fig. N of …" (`scripts/digitization/redraw_digitized.py`).
- **`arxiv_v*.png`** (Ravivarapu et al.): a crop of the figure in the authors' arXiv version (arXiv:2507.06326v1), © the authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution in [LICENSING.md](../LICENSING.md#figures-from-the-replicated-papers).
- **`paper.png`** (only on your machine, gitignored): a crop of the published figure that `scripts/digitization/paper_crops.py build` makes from your own copy of the paper, for the digitization tools. Never commit it.

Panel trackers:

| Paper | Tracker |
|-------|---------|
| Mehregan et al. | [figures/mehregan/replications.md](mehregan/replications.md) |
| Nguyen et al. | [figures/nguyen/replications.md](nguyen/replications.md) |
| Ravivarapu et al. (SEA-DBS) | [figures/ravivarapu/replications.md](ravivarapu/replications.md) |

Plot scripts: `scripts/figures/papers/`. JSON caches: `artifacts/figures/papers/`.

**Source papers.** The papers these figures replicate are kept with the notes, not in git (they are publisher PDFs): `papers/paper_1.{pdf,md}` (Mehregan et al.), `papers/paper_2.{pdf,md}` (Nguyen et al.), and `papers/paper_3.{pdf,md}` (Ravivarapu et al.) under the notes folder that `export.notes_dir` / `RL_DBS_NOTES_DIR` points to ([docs/cli.md](../docs/cli.md) §5.6). The `.md` files are text transcriptions; prefer them over the PDFs when reading methods.
