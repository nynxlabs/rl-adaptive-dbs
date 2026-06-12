#!/usr/bin/env bash
# Verify MATLAB install, license, reference model, and Python engine.
set -euo pipefail

_matlab_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_matlab_dir/../.." && pwd)"
cd "$repo_root"

# shellcheck source=scripts/matlab/env.sh
source "$_matlab_dir/env.sh"

fail=0
pass() { echo "  ok  $*"; }
warn() { echo "  warn $*"; fail=1; }
die() { echo "  fail $*"; fail=1; }

echo "=== MATLAB verify (rl-adaptive-dbs) ==="

if [ -x "$MATLAB_ROOT/bin/matlab" ]; then
  pass "MATLAB binary at $MATLAB_ROOT/bin/matlab"
elif [ -x "$MATLAB_ROOT/bin/matlab.exe" ]; then
  pass "MATLAB binary at $MATLAB_ROOT/bin/matlab.exe"
else
  die "MATLAB binary missing — set MATLAB_ROOT (docs/matlab.md §2) or run scripts/matlab/install.sh (Linux/WSL)"
fi

if matlab -batch "license('test','MATLAB'); exit" >/dev/null 2>&1; then
  pass "MATLAB license (batch)"
else
  die "MATLAB not licensed — see docs/matlab.md §2 (existing) or §4 (activate)"
fi

model_dir="$RL_ADAPTIVE_DBS_MATLAB_MODEL"
if [ -f "$model_dir/simulate_network_model.m" ]; then
  pass "reference model at $model_dir"
else
  die "missing simulate_network_model.m under $model_dir"
fi

if matlab -batch "cd('$model_dir'); assert(exist('simulate_network_model','file')==2); disp('model visible'); exit" >/dev/null 2>&1; then
  pass "MATLAB can see simulate_network_model.m"
else
  warn "MATLAB could not load reference model (license or path issue?)"
fi

if matlab -batch "cd('$model_dir'); simulate_network_model(1,1,0,1,true); exit" >/dev/null 2>&1; then
  pass "plant dynamics smoke (simulate_network_model dynamics_only)"
else
  die "plant dynamics smoke failed — see docs/matlab.md §7"
fi

if uv run python -c "import matlab.engine" >/dev/null 2>&1; then
  pass "Python matlab.engine import"
  if uv run python -c "
import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd('$model_dir')
assert eng.which('simulate_network_model')
eng.exit()
" >/dev/null 2>&1; then
    pass "Python engine start + model path"
  else
    warn "matlab.engine import works but start_matlab() failed (license or LD_LIBRARY_PATH?)"
  fi
else
  warn "Python matlab.engine not installed (run: uv sync --group matlab)"
fi

echo "=== done ==="
exit "$fail"
