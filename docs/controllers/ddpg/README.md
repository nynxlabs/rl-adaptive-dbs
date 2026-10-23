# Mehregan et al. — figure comparisons

**Primary replication tracker** for this repo. Work is scheduled by **panel**, not by roadmap phase: each row below is an exit criterion with automated gates in `plot.py` (manifest `gates` / `gates_pass`), a committed `plot.py`, and side-by-side PNGs.

Side-by-side **paper panel** vs **our replication**. Plot scripts write replication PNGs to `docs/controllers/ddpg/figures/images/`; JSON caches to `artifacts/figures/papers/`. Per-panel gate tables show live **Pass** per check (all required; refreshed by `scripts/digitization/mehregan_gate_status.py` on promote / `--plot-only`).

Method: [method.md](method.md) (paper-accurate DDPG spec) · Extensions beyond the paper: [extensions.md](extensions.md).

<!-- summary:start -->
| Panel | Description | Status |
|-------|-------------|--------|
| Fig 1b | GPi PSD | Fail (`healthy_pd_ratio_near_paper`, rep v6) |
| Fig 2a | GPi $P_\beta$ time series | Pass (rep v4) |
| Fig 2b | Error Index time series | Pass (rep v21) |
| Fig 4a | Training $P_\beta$ vs step | Fail (`early_swing_near_paper`, paper Alg. 1, rep v49) |
| Fig 4b | Training reward vs episode | Fail (`early_reward_near_paper`, paired to 4a, rep v54) |
| Fig 5a | Post-train efficacy @ 45 Hz | Pass (closed loop, Fig 4a model, rep v29) |
| Fig 5b | Post-train efficacy @ 30 Hz | Pass (closed loop, burst alphabet, rep v30) |
| Fig 6a | PTQ / QAT @ 45 Hz | Fail (`qat_elevated_vs_fp32`, closed loop, int8 QAT, rep v74) |
| Fig 6b | PTQ / QAT @ 30 Hz | Fail (`qat_elevated_vs_fp32`, closed loop, int8 QAT, rep v52) |
<!-- summary:end -->

## Figures

| Panel | What it shows | Page |
|-------|---------------|------|
| Fig 1b | GPi PSD | [1b.md](figures/1b.md) |
| Fig 2a | GPi $P_\beta$ time series | [2a.md](figures/2a.md) |
| Fig 2b | Error Index time series | [2b.md](figures/2b.md) |
| Fig 4a | Training $P_\beta$ vs step | [4a.md](figures/4a.md) |
| Fig 4b | Training reward vs episode | [4b.md](figures/4b.md) |
| Fig 5a | Post-train efficacy @ 45 Hz | [5a.md](figures/5a.md) |
| Fig 5b | Post-train efficacy @ 30 Hz | [5b.md](figures/5b.md) |
| Fig 6a | PTQ / QAT @ 45 Hz | [6a.md](figures/6a.md) |
| Fig 6b | PTQ / QAT @ 30 Hz | [6b.md](figures/6b.md) |

## Images and caches

Replication PNGs: `docs/controllers/ddpg/figures/images/`. JSON caches: `artifacts/figures/papers/`. Paper references: `images/<panel>/digitized_v*.png`, our redraw of the values digitized from each published panel (`scripts/digitization/redraw_digitized.py`); the published figures are not in this repository ([LICENSING.md](../../../LICENSING.md#figures-from-the-replicated-papers)). A local, gitignored `paper.png` crop for the digitization tools comes from `scripts/digitization/paper_crops.py build`.
