#!/usr/bin/env bash
# Install MATLAB R2025b to ~/MATLAB via MathWorks Package Manager (Linux/WSL only).
# macOS / Windows: use MathWorks installer — docs/matlab.md §3.2 / §3.3.
set -euo pipefail

release="${MATLAB_RELEASE:-R2025b}"
destination="${MATLAB_ROOT:-$HOME/MATLAB}"
mpm_dir="${MPM_DIR:-$HOME/.cache/rl-adaptive-dbs/mpm}"

if [ -x "$destination/bin/matlab" ]; then
  echo "MATLAB already installed at $destination"
  "$destination/bin/matlab" -batch "fprintf('%s %s\\n', version, release); exit" 2>/dev/null || true
  exit 0
fi

echo "Installing MATLAB $release to $destination"

if ! command -v wget >/dev/null 2>&1; then
  echo "wget is required. Install with: sudo apt-get install -y wget" >&2
  exit 1
fi

mkdir -p "$mpm_dir"
cd "$mpm_dir"

if [ ! -x ./mpm ]; then
  wget -O mpm https://www.mathworks.com/mpm/glnxa64/mpm
  chmod +x mpm
fi

./mpm install --release="$release" --destination="$destination" --products=MATLAB

echo ""
echo "MATLAB installed to $destination"
echo "Next: bash scripts/matlab/setup.sh   # interactive connect + license + verify"
echo "  or: docs/matlab.md §2"
