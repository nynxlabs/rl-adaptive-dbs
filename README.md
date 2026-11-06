# rl-adaptive-dbs

A shared testbed for adaptive deep brain stimulation (DBS) controllers. We want reinforcement learning (RL) controllers and the conventional controllers they're measured against to run on the same simulated brain circuits, scored the same way, so their results can be compared directly.

## Status

We're rebuilding this project from scratch, together with [pawtrace](https://github.com/nynxlabs/pawtrace), our pipeline for rebuilding and checking published computational research. An earlier version of this repository has been retired; this one starts over.

Nothing here runs yet. We're starting with one simulated circuit and three published RL controllers:

- **Circuit:** the cortex–basal ganglia–thalamus model of Kumaravelu, Brocker & Grill (2016), J Comput Neurosci, which we rebuild from the paper itself.
- **RL controllers:**
  - Mehregan et al. (2024), IEEE MobileCloud
  - Nguyen et al. (2025), IEEE ISCAS
  - Ravivarapu et al., SEA-DBS ([arXiv:2507.06326](https://arxiv.org/abs/2507.06326))

## What it will hold

- **`plants/`** — simulated circuits, each rebuilt from its paper, with tests and a record of where every parameter comes from.
- **`controllers/rl/`** — the published RL controllers.
- **`controllers/baselines/`** — conventional controllers: no stimulation, continuous DBS, and closed-loop rules such as threshold and proportional–integral control.
- **`adapters/`** — the glue that lets each controller run on every circuit its signals line up with.
- **`papers/`** — one pawtrace recipe per paper: the spec, the figures we rebuild, and the gates each one has to pass.

## License

Apache-2.0. Copyright (c) 2026 nynxlabs LLC. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
