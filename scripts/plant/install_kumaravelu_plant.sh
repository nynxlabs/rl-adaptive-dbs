#!/usr/bin/env bash
# Install the private kumaravelu_plant package (editable) into this checkout's venv.
#
# The Python port of the Kumaravelu et al. (2016) model, our patches to its
# MATLAB source, and the data it generated live in that package while the
# model's license is pending (LICENSING.md). Without it this repository imports
# and its tests pass with skips, but the plant can't run.
#
# Looks in $KUMARAVELU_PLANT_DIR, else next to the main checkout
# (../kumaravelu-plant). `uv sync` removes it again (it isn't in uv.lock), so
# run this after every `uv sync`; `uv run` leaves it alone.
set -euo pipefail

repo="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
main_checkout="$(cd "$(git -C "$repo" rev-parse --git-common-dir)/.." && pwd)"
dir="${KUMARAVELU_PLANT_DIR:-$main_checkout/../kumaravelu-plant}"

if [[ ! -f "$dir/pyproject.toml" ]]; then
  echo "kumaravelu_plant: no checkout at $dir (set KUMARAVELU_PLANT_DIR); the plant stays unavailable — see LICENSING.md" >&2
  exit 0
fi
cd "$repo"
uv pip install --no-deps -e "$dir"
uv run --no-sync python -c "from envs.plant._kumaravelu import require; print('kumaravelu_plant:', require().__file__)"
