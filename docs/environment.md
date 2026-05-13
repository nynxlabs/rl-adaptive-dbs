# Environment

Use a local Python virtual environment so dependencies stay out of version control. This repository ignores `.venv/` (see the root `.gitignore`).

On Linux or WSL, a typical setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install project-specific packages once a `pyproject.toml` or `requirements.txt` is added to this repo.
