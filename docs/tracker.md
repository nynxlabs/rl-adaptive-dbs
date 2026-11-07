# Tracker

What's done, what's next, and the open questions we check as we go. Paper facts live in each
paper's pawtrace spec (`papers/<paper>/spec.yaml`), never here.

## Status

| Piece | State |
| --- | --- |
| Repo scaffold, pawtrace project, paper rights | done |
| Ravivarapu intake (arXiv 2507.06326) | spec done: 100 items (32 missing, 12 ambiguous, 1 contradicted), 7 open questions |
| Mehregan, Nguyen intake | waiting on Devin's human-intake forms |
| Kumaravelu 2016 plant (clean-room, from the manuscript) | parked: the printed synapse strengths leave cortex and striatum silent; questions wait for the authors |
| Fleming 2020 circuit (the authors' GPL-3.0 code, linked, `extras/fleming2020/`) | in progress |
| Adapter interfaces, baselines (mock plant), SEA-DBS skeleton | in progress |
| Adapters, baselines, RL controllers | not started |

## Rules carried forward

- **A clean-room Fleming 2020 rebuild, if we ever do one,** must be written by an agent that has
  never read Fleming's code: a fresh agent whose brief lists the code (and `extras/fleming2020/`)
  as off-limits, working only from the papers and `papers/fleming2020/`. That's why notes about
  the code stay out of the paper-side spec and ledger.

## Open questions and hypotheses

Hypotheses to check against the specs, not findings. Each one is settled by a spec item, a
test, or a run, and then moved to the paper's ledger.

- **Plant settings per paper.** Each RL paper may run the 2016 plant with its own settings
  (step length, signal, spectral readout). Check each paper's reported plant behaviour from its
  spec against our plant before training anything on it.
- **Action heads.** For each controller, confirm from the spec exactly what the network outputs
  and how that becomes pulses before building it; whether the stated head trains as written is
  an open question to test early, on small runs.
- **Reward optima.** For each reward, work out on our plant which constant policy (e.g. always
  on, always off) it rewards most. A learner that only finds that policy may still be doing what
  the reward asks.
- **Seeds.** Report pass rates across seeds rather than one seed, and reserve held-out seeds
  for final runs (pawtrace preregistration).
- **One energy measure for every controller.** Sum of I²·PW per second over the pulses actually
  delivered, so baselines and RL controllers compare on the same scale.
