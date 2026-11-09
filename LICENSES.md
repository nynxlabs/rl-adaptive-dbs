# Licenses

rl-adaptive-dbs is a hub: it brings together models, controllers and techniques from other
people's research. We don't need to own all of it. **Each piece keeps its own license**, and we
credit its authors. Our own code is Apache-2.0, Copyright (c) 2026 nynxlabs LLC
([LICENSE](LICENSE), [NOTICE](NOTICE)).

## How a piece gets in

- **Vendored** (copied into this repo): only permissively licensed, small, stable pieces (MIT,
  BSD, Apache-2.0 for code; CC BY for non-code). Its license text and notice stay next to it.
- **Linked** (not in this repo: installed or fetched from upstream at install time):
  - copyleft code (GPL, LGPL, AGPL), as an opt-in module that imports it from its own package,
    so the core stays Apache-2.0;
  - non-commercial or academic-only terms (CC BY-NC, "research use only");
  - code whose authors allow use but not redistribution;
  - big maintained upstreams (e.g. NEURON, Stable-Baselines3, NetPyNE, Gymnasium), as ordinary
    dependencies;
  - anything that needs proprietary tools (e.g. MATLAB);
  - large datasets or model weights, and data under a use agreement.
- **Extra** (our own opt-in package under `extras/<id>/`, with its own license): the glue around
  linked copyleft code is licensed to match it (e.g. GPL-3.0), installed only on request, and
  never imported by the Apache-2.0 core.
- **Reference** (read for facts only): papers and code we take equations, values or conventions
  from, without copying their code or text.
- **Excluded:** code with no license and no written permission stays out entirely, linked or
  not, until its rights holders grant use; then we follow what they grant.

Papers are separate: what we may do with each paper (quote it, publish crops, give it to AI
tools) is set in its pawtrace spec, which we keep separately; its license and AI-tools rule are
repeated in `licenses.yaml` so the table below shows them.

## Map

Generated from [`licenses.yaml`](licenses.yaml) by `uv run python tools/licenses.py --write`;
paper licenses and AI rules are checked against each paper's pawtrace spec where we have it. The
tests fail if this table is stale or a piece breaks the rules above.

<!-- licenses-table:start -->
| Piece | Kind | Mode | License | AI tools | Credit / source |
|---|---|---|---|---|---|
| rl-adaptive-dbs (plants, controllers, adapters, baselines, recipes, tests) | code | own | Apache-2.0 |  | nynxlabs LLC · `.` |
| Kumaravelu 2016 cortex-basal ganglia-thalamus plant (clean-room rebuild) | plant | own | Apache-2.0 (paper: publisher-copyright) | allowed | Model: Kumaravelu, Brocker & Grill (2016), J Comput Neurosci. Our implementation, written from the paper. · `plants/parkinsons/kumaravelu2016/` · Not on main yet. |
| Kumaravelu et al. ModelDB code (model 206232) and its GitHub mirror | code | excluded | none |  | Kumaravelu, Brocker & Grill · `https://modeldb.science/206232` · until: the rights holders grant use in writing |
| So, Kent & Grill 2012 (cited by the plant for synapse conventions) | paper | reference | see spec |  | So, Kent & Grill (2012) · Spec lands with the plant. |
| SEA-DBS controller (Ravivarapu et al.) | controller | own | Apache-2.0 (paper: CC-BY-4.0) | allowed | Method: Ravivarapu et al., arXiv 2507.06326 (CC BY 4.0). Our implementation. · `controllers/parkinsons/rl/sea_dbs/` |
| Mehregan et al. 2024 controller | controller | own | Apache-2.0 (paper: publisher-copyright) | never | Method: Mehregan et al. (2024), IEEE MobileCloud. Our implementation. · `controllers/parkinsons/rl/ddpg/` |
| Nguyen et al. 2025 controller | controller | own | Apache-2.0 (paper: publisher-copyright) | never | Method: Nguyen et al. (2025), IEEE ISCAS. Our implementation. · `controllers/parkinsons/rl/snn/` |
| CL-DBS-RL (Nguyen group follow-up code) | code | reference | MIT |  | Binh Nguyen · `https://github.com/howyoubinh/CL-DBS-RL` · Gap-fill values only, no code copied. If code is ever vendored, keep its MIT license + notice with it. Its plant files port the Kumaravelu model and keep that model's status. |
| Fleming, Dunn & Lowery 2020 model code | code | linked | GPL-3.0 |  | Fleming, Dunn & Lowery · `the authors' official GPL-3.0 repository; URL and pinned commit set in extras/fleming2020 when it lands` · Fetched from the authors' official repository at install time, pinned to a commit, for the opt-in extras/fleming2020 module only; never copied into this repo. |
| pawtrace | tool | linked | Apache-2.0 |  | nynxlabs LLC · `https://github.com/nynxlabs/pawtrace` |
| NumPy | library | linked | BSD-3-Clause |  | NumPy developers · `https://numpy.org` |
<!-- licenses-table:end -->
