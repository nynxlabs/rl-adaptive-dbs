# rl-adaptive-dbs

Exploration of reinforcement learning for adaptive deep brain stimulation, built around published network models and controller experiments.

## Layout

- `controllers/` — controller implementations (placeholders for now: `ddpg`, `snn`, `sea_dbs`).
- `envs/` — RL environments (to be added).
- `docs/` — specs and notes: [environment.md](docs/environment.md) (plant / Gym API), [controller_mehregan.md](docs/controller_mehregan.md) (Mehregan et al. actor–critic training).
- `reference_material/` — third-party models and scripts. Includes the Kumaravelu et al. (2016) MATLAB network model under `reference_material/KumaraveluEtAl2016/` (see that folder’s `readme.txt` for citation and provenance).

## Environment

See [docs/environment.md](docs/environment.md) and [docs/controller_mehregan.md](docs/controller_mehregan.md).
