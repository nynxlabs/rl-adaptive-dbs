#!/usr/bin/env bash
# Open MathWorks Product Authorizer (Linux/WSL; xvfb when headless).
# macOS / Windows: activate via MATLAB GUI — docs/matlab.md §4.
set -euo pipefail

_matlab_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/matlab/env.sh
source "$_matlab_dir/env.sh"

mkdir -p "$MATLAB_ROOT/licenses" "$MATLAB_PREFDIR"

echo "Starting MathWorks Product Authorizer..."
echo "  Sign in with your MathWorks account, or use Advanced Options → license file."
echo "  License file target: $MATLAB_ROOT/licenses/"
echo ""

if [ -z "${DISPLAY:-}" ]; then
  xvfb-run -a "$MATLAB_ROOT/bin/glnxa64/MathWorksProductAuthorizer.sh"
else
  "$MATLAB_ROOT/bin/glnxa64/MathWorksProductAuthorizer.sh"
fi

echo ""
echo "When done, verify:"
echo "  bash scripts/matlab/verify.sh"
