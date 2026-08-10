# Scripts layout

| Path | Role |
|------|------|
| **`figures/papers/<paper>/<panel>/plot.py`** | Ship surface — train / eval / plot for each paper panel. Prefer these over ad hoc trainers. |
| **`figures/papers/`** helpers | `promote.py`, `paper_overlay.py`, `push_kb_images.py`, `update_report3.py`, `resume_cli.py`, `plot_axes.py`, `overlay_import.py` |
| **`digitization/`** | Curve digitization + panel gate helpers (`paper_gates.py`, `*_gates.py`, WPD/PIL/Engauge tooling) |
| **`lib/`** | Shared helpers used by panel scripts (`paper_protocol_eval`, `pattern_reward_landscape`, runtime guards) |
| **`matlab/`** | MATLAB Engine install / env / verify for the Kumaravelu plant |
| **`plant/`** | Plant fixture exporters (e.g. init draws for Python ↔ MATLAB parity) |
| **`validation/`** | Fresh-host / Multipass / Windows sandbox validation |
| **`reports/`** | Export outreach report Markdown → PDF (`export_pdf.py`) |
| **`probes/`** | Diagnostic-only. Keep only reusable extension probes; delete one-off panel thrash after findings land in `docs/figures/` + panel `plot.py` |
| **`check_mehregan_replication.py`** | Assess a DDPG replication summary JSON against the checklist |
| **`replicate_mehregan_ddpg.py`** | Full Mehregan DDPG train+eval entry (MATLAB plant) |
| **`setup.sh`** | Repo bootstrap |

Launch plant-heavy Python with thread limits applied first:

```bash
uv run python -m rl_adaptive_dbs.run scripts/figures/papers/mehregan/4a/plot.py --push-kb --update-report
```
