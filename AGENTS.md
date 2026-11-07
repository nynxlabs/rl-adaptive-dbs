# Agent notes — rl-adaptive-dbs

For coding agents (any provider) and nynx working in this repo. On nynxbox, machine policy is
in `~/AGENTS.md` (worktrees, ship, heavy jobs, secrets, no PRs). This file covers this repo only.

- **What it is:** a shared testbed for adaptive deep brain stimulation controllers. RL
  controllers and conventional baselines run on simulated brain circuits (plants) through
  adapters, and each published controller is rebuilt and checked with
  [pawtrace](https://github.com/nynxlabs/pawtrace). See `README.md`.
- **Public repo** (`nynxlabs/rl-adaptive-dbs`, Apache-2.0, Copyright (c) 2026 nynxlabs LLC).
  Commit locally in a worktree. Before any push, scan the outgoing diff for secrets, personal
  info and private machine paths. Scaffold and docs pushes are fine; **stop and ask Devin or
  nynx before pushing substantive plant or controller code** (until they lift this).
- **"we" voice:** docs are written as "we", never "I"; Devin by name; nynx / nynxlabs / nezzy /
  nex lowercase. Public text says nothing about the earlier version of this project beyond
  "an earlier version".

## Source rules (hard rules)

These hold for every agent, every provider, local models included. A denial or doubt means
stop and ask Devin through a pawtrace ledger question (`papers/<paper>/ledger.yaml`
`questions:`), not find another way in. Each paper's `spec.yaml` records the same rules in
`rights:` and `policy.off_limits`, so `pawtrace lint` catches a ledger entry that cites an
off-limits source, and `tests/test_layout.py` checks the rights blocks.

1. **Mehregan et al. 2024 and Nguyen et al. 2025 (IEEE ISCAS): `ai_tools: never`.** Our copies
   come from IEEE Xplore, whose terms forbid AI tools. Agents never open these papers' PDFs, or
   any full text, transcription, figure crop, digitization, cheat sheet or note made from them,
   wherever it's stored, and never the earlier version's docs, specs, trackers or digitization
   configs for them. Intake for these two is by Devin, by hand (human-intake forms; see
   § Intake). Agents get these papers' facts **only** from Devin's spec items and his answers
   to ledger questions.
2. **Ravivarapu et al. (SEA-DBS): only arXiv 2507.06326 (CC BY 4.0).** AI intake is allowed from
   that copy. Never the IEEE Xplore version, or anything made from it.
3. **The plant (Kumaravelu, Brocker & Grill 2016) is a clean-room rebuild** from the PMC author
   manuscript (PMC4975943; text mining allowed, quote briefly, don't publish its text).
   Whoever writes plant code must not look at: the earlier version's plant port or plant docs,
   the `kumaravelu-plant` package, the authors' ModelDB code (model 206232) or its GitHub mirror,
   or older plant notes. Every parameter is a spec item in `papers/kumaravelu2016/spec.yaml`
   with its section (provenance), and its value in code cites that item id.
4. **Fleming, Dunn & Lowery 2020 code (GPL-3.0): only through the linked opt-in module.** Devin
   lifted the never-open rule for this one purpose (nov 6 2026). The authors' code is used only
   as published, from their official GPL-3.0 repository pinned to a commit (never ModelDB 262046
   or any unlicensed copy), and never copied into this repo: it's fetched or installed at
   install time. Our glue lives in `extras/fleming2020/` under GPL-3.0 with its own LICENSE, as an
   optional extra; the Apache core never imports it (a test checks). Agents may read the code
   to build that module. The paper (CC BY 4.0) is its own spec, `papers/fleming2020/`, written
   from the paper only: keep notes about the code out of the paper-side spec and ledger, and
   cite code facts only in `extras/fleming2020/`. A later clean-room rebuild from the papers
   must be done by an agent that has never read Fleming's code (`docs/tracker.md`).
   **CL-DBS-RL** (MIT, the Nguyen group's follow-up code) is allowed for gap-fills where Devin's
   Nguyen spec is silent, never over it; its plant files keep rule 3. Other baselines follow
   their paper's license: CC BY papers (e.g. Su 2019, Zhu 2021, Bronte-Stewart 2020) are fine;
   for Velisar 2019 (Elsevier subscription) work from Bronte-Stewart 2020 instead; Gao 2020 only
   from the authors' own posted copy.
5. **The earlier version of this project is off-limits as a source** (code, docs, specs,
   trackers, digitization configs, figures, run artifacts). Lessons from it enter only as open
   questions in `docs/tracker.md`, worded as hypotheses to check against the spec, never as
   paper facts.

6. **Each piece keeps its own license** (`LICENSES.md`, map in `licenses.yaml`). Our own code is
   Apache-2.0.
   - **Vendor** (copy in) only small, stable, permissive pieces: MIT, BSD, Apache-2.0 code, CC BY
     non-code. Keep the license text and notice next to the copy.
   - **Link** (an opt-in extra or install-time fetch, never copied in): GPL/LGPL/AGPL code (opt-in
     modules that import it from its own package, so the core stays Apache-2.0); non-commercial
     or academic-only terms; code whose authors allow use but not redistribution; big maintained
     upstreams (NEURON, Stable-Baselines3, NetPyNE, Gymnasium) as normal dependencies; anything
     needing proprietary tools (MATLAB); large datasets, weights, or data under a use agreement.
   - **Nothing:** code with no license and no written permission (e.g. ModelDB 206232) stays out
     entirely, linked or not, until its rights holders grant use; then follow their grant.
   - Paper rules (1–3) are unchanged. Fleming 2020's code is linked, never vendored (rule 4).
   - Every new plant, controller, baseline, adapter, dataset or dependency gets a
     `licenses.yaml` entry in the same commit; for a paper-based piece, point `spec:` at its
     pawtrace spec instead of repeating its license. `uv run python tools/licenses.py --write`
     renders the table; the tests fail on a stale table or a rule break.

On nynxbox, the exact local paths these rules cover are listed in a private note:
`~/vault/research/rl-adaptive-dbs-testbed/source-rules-local.md`. Read it before starting.

## Intake (pawtrace)

- `pawtrace.yaml` is the project; `papers/<paper>/` holds `spec.yaml` (what the paper says),
  `ledger.yaml` (how each gap got its value, deviations, questions) and `panels/<id>.yaml`
  (gates as config). Run `uv run pawtrace lint` before every commit that touches `papers/`.
  Generated files (`SPEC.md`, `STATUS.md`, `QUESTIONS.md`) are never hand-edited.
- **AI intake** (Ravivarapu, Kumaravelu): `pawtrace playbook intake`. Put the PDF in
  `papers/<paper>/source/` (gitignored; never commit it or its extracted text).
- **Human intake** (Mehregan, Nguyen): Devin fills a plain-language form; an agent turns his
  answers into spec items word for word (`text` = his words, `section` = his citation,
  `status` as he marks it), and asks him about anything unclear through a ledger question.
  The blank form is `papers/human-intake-form.md`. Never pre-fill a form from the paper.
- Gaps are filled in the ledger (`source: convention | cited-work | authors-code | author`),
  never in the spec. Deviations from a stated mechanism go proposed → testing → adopted, and
  need Devin's OK to become the default.

## Layout

| Path | Holds |
| --- | --- |
| `plants/parkinsons/kumaravelu2016/` | the first plant, with tests |
| `controllers/rl/{ddpg,snn,sea_dbs}/` | the three published RL controllers |
| `controllers/baselines/` | conventional controllers |
| `adapters/` | controller ↔ plant glue (actions, observations, timing) |
| `papers/<paper>/` | pawtrace recipe per paper |
| `docs/tracker.md` | what's done, open questions, hypotheses to check |

## Running things

- `uv sync --all-groups`, then `uv run pytest -q` and `uv run pawtrace lint`.
- Anything that can load the CPU for more than a minute runs through `heavy` on nynxbox
  (`heavy --bg …`) with a hard `timeout`. No GPU training without Devin's OK.
