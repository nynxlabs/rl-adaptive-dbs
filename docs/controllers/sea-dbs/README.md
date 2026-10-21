# Ravivarapu et al. — figure comparisons

**Primary replication tracker** for SEA-DBS (*Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*). Work is scheduled by **panel**, not by roadmap phase: each row below is an exit criterion with paper-anchored gates, a committed `plot.py`, and side-by-side PNGs.

Controller spec: [method.md](method.md). Post-replication directions: [extensions.md](extensions.md).

Side-by-side **paper panel** vs **our replication**. Plot scripts write replication PNGs to `docs/controllers/sea-dbs/figures/images/`; JSON caches to `artifacts/figures/papers/`. Per-panel gate tables show live **Pass** per check (all required; refreshed by `scripts/digitization/ravivarapu_gate_status.py` on promote / `--plot-only`).

**Honest replication (oct 12 2026).** Every panel used to be marked Pass, but those passes rested on mechanisms the paper doesn't describe (episode-scheduled actor logit boosts, a no-stim init bias, per-variant plants and learning rates, eval-only windows, hand-picked eval seeds, per-step pulse schedules, PTQ weight noise, loosened gates). All were removed ([method.md](method.md) §15). Every variant now trains with one config (Table I, random init, Eq. (9) as written), and Figs 5–7 evaluate on the training plant setup with only the paper's stimulation frequency changed. Every panel now fails, with the gaps documented in each lab notebook. The two root causes: Eq. (7) has no stimulation cost, so all four variants learn to stimulate every step and their deterministic rollouts coincide; and on Kumaravelu, 30 Hz raises beta. A second pass the same day screened the remaining paper-silent knobs (the Mehregan et al. logits-critic DDPG backbone, Gumbel-max, slow τ annealing, a network the size of the paper's 65 MB, the observation window) and fixed one gate: Fig 4b now compares rewards to the paper in absolute units instead of rescaling the paper by our Baseline's early level. Only the observation window changed the result: $n_{\mathrm{obs}} = 10$ (the paper's 10-step inference horizon) lets the inference traces fall through step 10 ([4a.md](figures/4a.md) § Convention screen).

**Why every panel fails — five root causes (oct 14 2026).** Most failed gates trace to one of these, so the per-panel counts overlap:

1. **Every variant stimulates on every step.** Eq. (7) has no stimulation cost, so always-on is optimal; Baseline, +PM, +GS and SEA-DBS deploy as the same always-on policy and every gate that needs them to differ fails (Figs 5a, 5b, 6, 7, Table II).
2. **SEA-DBS reaches full stimulation by episode ~40** (2 × 64; with the 1 × 32 networks shipped oct 19 2026 it ramps up gradually and reaches it by episode ~60) and then sits on the plant's full-stimulation floor, instead of the paper's gradual 150-episode decline (Fig 4a drop, timing and gap gates; Fig 4b rise gates).
3. **Training starts at ~0.40, not the paper's untreated ~0.45**, because a randomly initialized two-action policy already stimulates about half the time (start-level gates in Figs 4a and 4b).
4. **30 Hz raises beta on Kumaravelu**, so the Fig 5b curves rise where the paper's fall.
5. **The 50 Hz always-on trace is a straight ramp** (the 10-step observation window filling), not the paper's fast-then-slow SEA-DBS curve (Fig 5a SEA-DBS MAE gates).

Separately, our Fig 4b (mean Eq. (7) reward per step) is not Eq. (7) of our Fig 4a PSD the way the paper's is, because P̄ ramps within each episode (`eq7_consistent_with_4a`; [4a.md](figures/4a.md) § Gate audit).

**Gate audit (oct 14 2026).** Every gate is anchored on the paper digitization, and the paper's own curves pass every gate on every panel under both window-mean and linear resampling. Duplicates, coin-flip gates, a gate that tested something the paper doesn't claim (`ptq_traces_distinct`), and gates loose enough to pass clearly wrong curves were removed or tightened; gates that need digitization or paired data fail closed. Details and the noise check: [4a.md](figures/4a.md) § Gate audit.

**Audit pass (oct 17 2026).** The plant now integrates at the paper's 0.02 ms (it ran at the 0.01 ms reference step; same levels, so gate counts move only by realization noise), Table II gives every interval the same 30 plant seeds (the old fixed horizon confounded $n$ with the seed pool), and the Fig 4a/4b plots draw the raw episodes under the rolling mean. Plant probes answered or narrowed several open questions: amplitude and pulse width can't make 30 Hz suppress, and always-on stays optimal under Eq. (7) on a continuous plant too, so root causes 1 and 4 stand under every reading tried ([4a.md](figures/4a.md) § Open-question probes).

**The paper's 2 ms step (oct 17 2026).** A trailing-window mode (`step_mode="trailing"`) runs the paper's literal timing: 30 steps of 2 ms in one continuous 60 ms episode. On Kumaravelu no stimulation lowers beta within 60 ms, both learners stop stimulating, and Fig 4a candidates pass 11–14 of 28 gates (vs 15 for the 150 ms shot), so the 150 ms shot stays the labeled convention ([4a.md](figures/4a.md) § The paper's 2 ms step).

**Candidate readings (oct 17 2026).** Readings beyond the text, behind config flags that default off: the plant carried across episodes (2 ms steps), a stimulation cost in the training reward, and one update per episode. The cost is the only one that splits Baseline and SEA-DBS at inference, and one update per episode gives the best Fig 4a (17 of 28), but none clears a panel, so the shipped default stays as written ([4a.md](figures/4a.md) § Candidate readings, § The paper's 2 ms step).

**Candidate screen (oct 19 2026).** Labelled candidates (each deviates from the paper as we read it, or fills a gap it leaves — our best guess of what the authors may have done; shipped defaults unchanged). Best: **one hidden layer of 32 units** (topology is paper-silent) slows SEA-DBS's saturation without destabilizing it. Single seeds pass 18–21 / 28 on Fig 4a and 16–18 / 20 on Fig 4b (shipped 11–19 and 9–15 over seeds 0–4). Gated on the **mean over training seeds 0–4** (a display/aggregation reading of the paper's smooth curve, same gates), it reaches **Fig 4a 21 / 28, Fig 4b 18 / 20**; seed averaging alone takes the shipped config to 18 / 28 and 12 / 20. Entropy regularization and a lower actor lr slow SEA-DBS too, but on some seeds it settles on never stimulating. PM lr, Polyak τ, more updates per step, a deeper net and richer state inputs don't help. Every candidate actor still stimulates on every inference step, so Figs 5a–7 don't move. Hardened Gumbel-Softmax inference and static int8 PTQ change no action (logit margins 4–620). A partial window carry across Table II reseeds gives the paper's 1/n transient and 3 / 6 gates. Details: [4a.md](figures/4a.md) § Candidate screen (oct 19 2026).

**Network size 1 × 32 (oct 19 2026).** Every panel now trains and evaluates actor, critic and predictive model as one hidden layer of 32 units — a network size the paper doesn't state, our best guess of what the authors may have done (2 × 64 before; [method.md](method.md) §14.9). Seed 0, same gates:

| Panel | 2 × 64 (before) | 1 × 32 (now) |
|-------|-----------------|--------------|
| Fig 4a | 15 / 28 (rep v83) | **18 / 28** (rep v84) |
| Fig 4b | 12 / 20 (rep v21) | **16 / 20** (rep v22) |
| Fig 5a | 12 / 20 (rep v25) | 12 / 20 (rep v26) |
| Fig 5b | 6 / 22 (rep v25) | 6 / 22 (rep v26) |
| Fig 6 | 10 / 15 (rep v25) | 10 / 15 (rep v26) |
| Fig 7 | 7 / 17 (rep v30) | 7 / 17 (rep v31) |
| Table II | 2 / 6 (rep v2) | 2 / 6 (rep v3) |

No panel regressed. SEA-DBS learns gradually now, but every actor still stimulates on every inference step, so Figs 5a–7 and Table II are unchanged. Every variant tried so far: [4a.md](figures/4a.md) § Tried so far.

<!-- summary:start -->
| Panel | Description | Status |
|-------|-------------|--------|
| Fig 4a | Training PSD vs episode | Fail — 18/28 gates pass (rep v85) |
| Fig 4b | Training reward vs episode | Fail — 16/20 gates pass (rep v23) |
| Fig 5a | Inference @ 50 Hz | Fail — 12/20 gates pass (rep v27) |
| Fig 5b | Inference @ 30 Hz | Fail — 6/22 gates pass (rep v27) |
| Fig 6 | FP16 PTQ @ 50 Hz | Fail — 10/15 gates pass (rep v27) |
| Fig 7 | Ablation (Baseline / +PM / +GS / SEA-DBS) | Fail — 7/17 gates pass (rep v32) |
| Table II | Avg PSD with the plant reseeded every n steps | Fail — 2/6 gates pass (rep v4) |
<!-- summary:end -->

Replication PNGs: `docs/controllers/sea-dbs/figures/images/`. JSON caches: `artifacts/figures/papers/`. Paper references: `images/<panel>/arxiv_v*.png`, crops of the figures in the authors' arXiv version — Ravivarapu, Bagwe, Yuan, Yu, Zhang, *Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson's Disease*, arXiv:2507.06326v1, © the authors, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); cropped, otherwise unchanged ([LICENSING.md](../../../LICENSING.md#figures-from-the-replicated-papers)). A local, gitignored `paper.png` crop of the published figure for the digitization tools comes from `scripts/digitization/paper_crops.py build`. Controller spec: [method.md](method.md). Schematics (paper Figs 1, 3) are **out of scope**. Fig 2 (reward curve) is optional polish, not a gate.

---

## Figures

| Panel | What it shows | Page |
|-------|---------------|------|
| Fig 4a | Training PSD vs episode | [figures/4a.md](figures/4a.md) |
| Fig 4b | Training reward vs episode | [figures/4b.md](figures/4b.md) |
| Fig 5a | Inference @ 50 Hz | [figures/5a.md](figures/5a.md) |
| Fig 5b | Inference @ 30 Hz | [figures/5b.md](figures/5b.md) |
| Fig 6 | FP16 PTQ @ 50 Hz | [figures/6.md](figures/6.md) |
| Fig 7 | Ablation (Baseline / +PM / +GS / SEA-DBS) | [figures/7.md](figures/7.md) |
| Table II | Avg PSD with the plant reseeded every n steps | [figures/table2.md](figures/table2.md) |

## Gate design

**Primary gate spec** for SEA-DBS (*Sample-Efficient Reinforcement Learning Controller for Deep Brain Stimulation in Parkinson’s Disease*). Ship status is in the summary above; side-by-side PNGs and run commands are on each [figure page](#figures).

Controller / adapter: [method.md](method.md). Shared plant: [plant.md](../../plant/README.md). Schematics (paper Figs 1–3) are **out of scope**. Fig 2 (reward curve of Eq. (7)) is optional polish, not a gate.

**Defaults (paper §V.A / Table I):** seed `0` (paper seed unspecified — lock one seed for gates), **2 ms** RL step × **30** steps/episode × **150** training episodes, $\gamma=0.99$, buffer **8192**, batch **32**, $\alpha_a=5\times10^{-4}$, $\alpha_c=10^{-3}$. Variants: `baseline` (DDPG, no PM/GS) vs `paper` (full SEA-DBS).

| Panel | Script | Status |
|-------|--------|--------|
| Fig 4a — training PSD vs episode | `scripts/figures/papers/ravivarapu/4a/plot.py` | **Fail, gap documented** — lab notes [4a.md](figures/4a.md) |
| Fig 4b — training reward vs episode | `scripts/figures/papers/ravivarapu/4b/plot.py` | **Fail, gap documented** — paired with 4a |
| Fig 5a — inference @ 50 Hz | `scripts/figures/papers/ravivarapu/5a/plot.py` | **Fail, gap documented** — [5a.md](figures/5a.md) |
| Fig 5b — inference @ 30 Hz | `scripts/figures/papers/ravivarapu/5b/plot.py` | **Fail, gap documented** — [5b.md](figures/5b.md) |
| Fig 6 — FP16 PTQ @ 50 Hz | `scripts/figures/papers/ravivarapu/6/plot.py` | **Fail, gap documented** — [6.md](figures/6.md) |
| Fig 7 — ablation (Baseline / +PM / +GS / SEA-DBS) | `scripts/figures/papers/ravivarapu/7/plot.py` | **Fail, gap documented** — [7.md](figures/7.md) |

Status as of the honest-replication pass (oct 12 2026): every shortcut that produced the earlier passes is removed ([method.md](method.md) §15). Live gates: each [figure page](#figures).

**Digitization:** refined WebPlotDigitizer curves per panel (`curves_fig{4a,4b,5a,5b,6,7}.json`); gates read them through `scripts/digitization/ravivarapu_gates.py`. Absolute y values have ~±0.01–0.02 (Fig 4a) / ~±10–15 (Figs 5–7) calibration uncertainty; **ordering, shared starts, relative drops, and cross-panel ratios** are the hard targets.

Digitization cache (local): `artifacts/figures/papers/ravivarapu/paper_digitization/curves_*.json`.

**Scale note:** Fig 4a y-axis is labeled **(PSD × 10⁻³)** with values ~0.32–0.48. Figs 5–7 use **PSD** on ~300–480. Same biomarker: $0.46$ on Fig 4a ↔ ~$460$ on Figs 5–7.

### Paper digitization — readout summary

Color-masked curve extraction from paper crops (axis box + blue/red/magenta/green/black masks). Values are **paper targets for gate design**, not replication means.

Each figure page keeps its own readout under § Gates → Paper readouts, and its gate design under § Gates → Gate design.

## Conventions (v1)

| Knob | v1 choice |
|------|-----------|
| Seed | `0` (lock for one-seed gates) |
| RL step | **2 ms** × **30** steps/episode (biomarker window convention: see sea_dbs replication.md) |
| Training episodes | **150** |
| Reward | Eq. (7), $\beta_t = 0.35$ |
| Baseline variant | `baseline` — learning DDPG **without** PM/GS (not frozen) |
| Full SEA-DBS | `paper` — PM + GS |
| Inference carrier | Fixed **50 Hz** or **30 Hz** eval knob (not RL action) |
| PTQ | **FP16 PTQ only** (no QAT) |
| Plant integration step | **0.02 ms** (§V.A; 0.01 ms before oct 17 2026, same levels) |
| Gate source | Refined WebPlotDigitizer curves `paper_digitization/curves_fig*.json` |
