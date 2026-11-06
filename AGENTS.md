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
4. **Fleming, Dunn & Lowery 2020 code (GPL-3.0): never open it.** The paper (CC BY 4.0) is fine.
   **CL-DBS-RL** (MIT, the Nguyen group's follow-up code) is allowed for gap-fills where Devin's
   Nguyen spec is silent, never over it; its plant files keep rule 3. Other baselines follow
   their paper's license: CC BY papers (e.g. Su 2019, Zhu 2021, Bronte-Stewart 2020) are fine;
   for Velisar 2019 (Elsevier subscription) work from Bronte-Stewart 2020 instead; Gao 2020 only
   from the authors' own posted copy.
5. **The earlier version of this project is off-limits as a source** (code, docs, specs,
   trackers, digitization configs, figures, run artifacts). Lessons from it enter only as open
   questions in `docs/tracker.md`, worded as hypotheses to check against the spec, never as
   paper facts.

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
