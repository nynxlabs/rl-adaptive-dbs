# Python virtual environment (uv)

Use a local virtual environment so dependencies stay out of version control. This repository ignores `.venv/` (see the root `.gitignore`).

[uv](https://docs.astral.sh/uv/) manages the interpreter and venv on **Windows**, **macOS**, and **Linux** (including WSL). The lockfile (`uv.lock`) is shared across platforms.

## Install uv

Install once per machine. See [uv installation](https://docs.astral.sh/uv/getting-started/installation/) for more options (Homebrew, WinGet, PyPI, etc.).

**macOS and Linux** (including WSL):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Create or refresh the venv

From the repository root (same on all platforms):

```bash
uv sync                  # runtime deps + editable install (CI / minimal images)
uv sync --all-groups     # runtime + dev (local development)
```

`uv sync` creates `.venv/` if missing, picks an interpreter compatible with `requires-python` in `pyproject.toml`, installs **runtime** dependencies from `[project]`, and **installs this repository in editable mode** so top-level packages `envs` and `controllers` import correctly. Dev tools (for example `pytest`) live in the `dev` dependency group; use **`uv sync --all-groups`** locally, or **`uv sync --group dev`** for the same result while only `dev` is defined.

Prefer **`uv run`** so you do not need to activate the venv:

```bash
uv run pytest
```

Activate only when you want a traditional shell with `python` on `PATH`:

| Platform | Command |
| --- | --- |
| macOS / Linux / WSL / Git Bash | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

uv does **not** bundle `pip` in `.venv` by default. Use `uv add`, `uv sync`, and `uv run` for normal workflow. For ad-hoc installs from the repo root: **`uv pip install …`** (uv targets `.venv` automatically). Run **`uv pip install pip`** only if you specifically need `python -m pip`.

## Dependencies

Add a library and update the lockfile:

```bash
uv add <package>
uv add --dev <package>   # test or lint tools → dev group
```

Commit `pyproject.toml` and `uv.lock` together so others get the same versions on any OS.

**Optional — MATLAB Engine API** (after MATLAB is installed and licensed; see [matlab.md](matlab.md)):

```bash
uv sync --group matlab
source scripts/matlab-env.sh   # sets LD_LIBRARY_PATH for matlab.engine
```

## Migrating from `python -m venv`

Remove the old `.venv` directory, then sync again.

**macOS / Linux / WSL / Git Bash:**

```bash
rm -rf .venv
uv sync --all-groups
```

**Windows (PowerShell):**

```powershell
Remove-Item -Recurse -Force .venv
uv sync --all-groups
```

## Optional: pin a Python version

To fix the interpreter uv picks on your machine, add a `.python-version` file (for example `3.12`) or adjust `requires-python` in `pyproject.toml`, then run `uv sync` again.
