# Nguyen et al. — figure comparisons

**Primary replication tracker** for the DSQN / closed-loop neuromorphic DBS paper. Spec: [controllers/snn/replication.md](method.md). Work proceeds **in parallel** with Mehregan ([Mehregan replications](../ddpg/README.md)): each row is an exit criterion with automated gates, a committed `plot.py` (when present), and side-by-side PNGs.

Method spec: [method.md](method.md). Work beyond the paper: [extensions.md](extensions.md).

Plot scripts write replication PNGs to `docs/controllers/snn/figures/images/`; JSON caches to `artifacts/figures/papers/nguyen/`. Paper references: `images/<panel>/digitized_v*.png`, our redraw of the values digitized from each published panel (`scripts/digitization/redraw_digitized.py`); the published figures are not in this repository ([LICENSING.md](../../../LICENSING.md#figures-from-the-replicated-papers)). A local, gitignored `paper.png` crop for the digitization tools comes from `scripts/digitization/paper_crops.py build`.

Figs **1–2** are schematics — **not** replication targets.

**Gates:** manifest `gates` in `artifacts/figures/papers/nguyen/<panel>/manifest.json`. Per-panel tables on each figure page refresh from manifest on each `promote` run (`scripts/figures/papers/promote.py` `refresh_nguyen_gate_tables`). Every listed check is required for exit. Digitization mirrors from `scripts/digitization/nguyen_gates.py` (`attach_digitization`).

<!-- summary:start -->
| Panel | Description | Status |
|-------|-------------|--------|
| Fig 3 | GPi α–β distribution (PD Off vs PD On) | Pass (rep v24) |
| Fig 4 | Training reward + episode length | Pass (rep v165) |
| Fig 5 | CBGT spikes + DBS energy over training | Fail (`paper_energy_pearson_0_500`) |
| Fig 6 | α–β + DBS parameters over training | Fail (`paper_alpha_beta_mid_near_paper`) |
| Fig 7 | 50-episode eval (25 steps) | Pass (rep v31) |
<!-- summary:end -->

## Figures

One page per panel: what the paper shows, status, how to reproduce, gates, deviations, everything tried so far, and notes. Scripts: `scripts/figures/papers/nguyen/<panel>/plot.py`.

| Panel | What it shows | Page |
|-------|---------------|------|
| Fig 3 | GPi α–β distribution (PD Off vs PD On) | [figures/3.md](figures/3.md) |
| Fig 4 | Training reward + episode length; training recipe, gates, plant choice (ρ = 1 since Oct 18 2026; ρ = 3 Oct 11–17), the authors' published code | [figures/4.md](figures/4.md) |
| Fig 5 | CBGT spikes + DBS energy over training; spike-count definition, DBS energy and the late energy gap | [figures/5.md](figures/5.md) |
| Fig 6 | α–β + DBS parameters over training; step sizes, late pulse width and frequency vs the paper | [figures/6.md](figures/6.md) |
| Fig 7 | 50-episode eval (25 steps); eval protocol, episode-start (reset window) convention, the step-0 gap | [figures/7.md](figures/7.md) |

## Conventions

Paper-silent settings of the shipped recipe (`fig4_nguyen_config`, Oct 17 2026; detail and evidence in [docs/controllers/snn/figures/4.md](figures/4.md) § Network faithfulness audit and [snn/replication.md](method.md) §12):

| Setting | Choice | Source |
|------|-----------|--------|
| Network | input → one hidden layer of 128 LIF → 9 outputs, β 0.95, Eq. (2) reset, Q = output membrane before reset | paper §III.B |
| Observation | binary spikes of all eight CBGT populations, 80 neurons × 10 bins of 10 ms, one row per SNN step | layout: follow-up code; 10 bins: convention (100 one-ms bins train less stably) |
| Action head | `factored` (3× ternary); spike-count argmax, ties to the higher output membrane | paper (spike counts); tie-break convention |
| Steps per action | 5 Hz / 0.1 ms / 5 nA/cm² | frequency and pulse width read off Fig 6; amplitude from the authors' code |
| Early-stop streak $t_u$ | 5 | authors' code; reproduces Fig 6's late ~81 Hz |
| Reward δ, τ | 0.01 (authors' code), 1,000 (reproduces the late Fig 4a level); raw $d$; rewards × 1e-4 for learning only | code / convention |
| ε | linear 0.9 → 0.05 over 2,000 steps | convention (the code's exponential schedule lands the length drop ~40 episodes late) |
| Target net, loss, optimizer | soft target τ 0.005, SmoothL1, AdamW, value clip 100 | follow-up |
| Updates | every 128 transitions (paper), 64 minibatches, 1,650-transition warm-up | paper / convention |
| Plant | one continuous simulation per episode; the Python port redraws wiring and conductances every 100 ms step; Kumaravelu coupling (ρ = 1, every STN neuron gets the full current) | port behaviour (author question 13); ρ = 1 as in Kumaravelu and the authors' code (the Oct 11 ρ = 3 deviation was dropped Oct 18 2026) |
| Eval episode start | fresh network that keeps its wiring for the whole episode; reset reading at 100 Hz / 300 nA/cm² / 0.3 ms recorded as step 0, then 40 Hz / 0.3 ms / 300 nA/cm² for the agent (eval only; reset window from v29, per-episode wiring from v31) | paper §IV; authors' published code (`MousePDEnv.reset()`); per-episode wiring adopted with the maintainer's OK (Oct 18 2026) |
| Fig 3 α–β window | 100 ms per sample (one RL step); 500 iterations | paper |
