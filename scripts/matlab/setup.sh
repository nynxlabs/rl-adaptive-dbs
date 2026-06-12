#!/usr/bin/env bash
# Interactive MATLAB setup for rl-adaptive-dbs (connect, license, Python engine, verify).
# See docs/matlab.md. Non-interactive: set MATLAB_ROOT + license, then bash scripts/matlab/verify.sh.
set -euo pipefail

_matlab_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$_matlab_dir/../.." && pwd)"
cd "$repo_root"

say() { printf '%s\n' "$*"; }
ask() {
  local prompt="$1" default="${2:-}"
  local reply
  if [[ -n "$default" ]]; then
    read -r -p "$prompt [$default]: " reply
    printf '%s' "${reply:-$default}"
  else
    read -r -p "$prompt: " reply
    printf '%s' "$reply"
  fi
}
ask_yn() {
  local prompt="$1" default="${2:-y}"
  local reply
  read -r -p "$prompt [y/n] (default $default): " reply
  reply="${reply:-$default}"
  [[ "$reply" =~ ^[Yy] ]]
}

detect_os() {
  case "$(uname -s)" in
    Linux)
      if grep -qi microsoft /proc/version 2>/dev/null; then
        printf 'wsl'
      else
        printf 'linux'
      fi
      ;;
    Darwin) printf 'macos' ;;
    MINGW* | MSYS* | CYGWIN*) printf 'windows-bash' ;;
    *) printf 'unknown' ;;
  esac
}

guess_matlab_root() {
  local os="$1" guess=""
  if command -v matlab >/dev/null 2>&1; then
    local bin
    bin="$(readlink -f "$(command -v matlab)" 2>/dev/null || command -v matlab)"
    guess="$(dirname "$(dirname "$bin")")"
  fi
  if [[ -z "$guess" ]]; then
    case "$os" in
      macos)
        [[ -d /Applications/MATLAB_R2025b.app ]] && guess=/Applications/MATLAB_R2025b.app
        ;;
      wsl | linux)
        [[ -d "$HOME/MATLAB" ]] && guess="$HOME/MATLAB"
        [[ -z "$guess" && -d /usr/local/MATLAB/R2025b ]] && guess=/usr/local/MATLAB/R2025b
        ;;
      windows-bash)
        [[ -d "/c/Program Files/MATLAB/R2025b" ]] && guess="/c/Program Files/MATLAB/R2025b"
        ;;
    esac
  fi
  printf '%s' "$guess"
}

say "=== rl-adaptive-dbs — MATLAB setup ==="
say "Repo: $repo_root"
say "Guide: docs/matlab.md"
say ""

if [[ ! -t 0 ]]; then
  say "Not a TTY — run this script in an interactive terminal, or:"
  say "  export MATLAB_ROOT=... MATLAB_RELEASE=R2025b"
  say "  source scripts/matlab/env.sh && bash scripts/matlab/verify.sh"
  exit 1
fi

os="$(detect_os)"
say "Detected OS: $os"
say ""

if ! command -v uv >/dev/null 2>&1; then
  say "uv is required. Install: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if ask_yn "Install Python deps (uv sync --all-groups)?" y; then
  uv sync --all-groups
fi
say ""

default_root="$(guess_matlab_root "$os")"
if [[ -n "$default_root" ]] && { [[ -x "$default_root/bin/matlab" ]] || [[ -x "$default_root/bin/matlab.exe" ]]; }; then
  say "Found MATLAB at: $default_root"
else
  say "MATLAB not detected yet."
  case "$os" in
    wsl | linux)
      if ask_yn "Install MATLAB R2025b to ~/MATLAB via MathWorks Package Manager (Linux/WSL)?" n; then
        export MPM_DIR
        MPM_DIR="$(ask "MPM download cache directory" "${MPM_DIR:-$HOME/.cache/rl-adaptive-dbs/mpm}")"
        export MATLAB_ROOT="${MATLAB_ROOT:-$HOME/MATLAB}"
        bash "$_matlab_dir/install.sh"
        default_root="$HOME/MATLAB"
      else
        say "Install manually: docs/matlab.md §3.1 or https://www.mathworks.com/downloads/"
      fi
      if [[ "$os" == wsl ]] && ask_yn "Install WSL headless packages (xvfb, GTK libs)?" y; then
        say "Running apt-get (may prompt for sudo)..."
        sudo apt-get install -y xvfb libnss3 libxss1 libasound2t64 libgbm1 libxrandr2 \
          libatk-bridge2.0-0 libgtk-3-0 libdrm2 libxdamage1 libxcomposite1 libxfixes3 libxi6 libxtst6
      fi
      ;;
    macos)
      say "Install from MathWorks (macOS installer): docs/matlab.md §3.2"
      say "  https://www.mathworks.com/downloads/"
      ;;
    windows-bash)
      say "Install from MathWorks (Windows): docs/matlab.md §3.3"
      say "  https://www.mathworks.com/downloads/"
      ;;
    *)
      say "See docs/matlab.md for your platform."
      ;;
  esac
fi
say ""

export MATLAB_ROOT="$(ask "MATLAB_ROOT (install folder containing bin/matlab)" "${MATLAB_ROOT:-${default_root:-$HOME/MATLAB}}")"
export MATLAB_RELEASE="$(ask "MATLAB_RELEASE" "${MATLAB_RELEASE:-R2025b}")"
say ""

# shellcheck source=scripts/matlab/env.sh
source "$_matlab_dir/env.sh" || exit 1

if matlab -batch "license('test','MATLAB'); exit" >/dev/null 2>&1; then
  say "License: OK"
else
  say "MATLAB is not licensed yet."
  say "  1) MathWorks account — bash scripts/matlab/activate.sh (Linux/WSL) or open MATLAB GUI"
  say "  2) License file — copy .lic into \$MATLAB_ROOT/licenses/"
  say "  3) Campus license — export MLM_LICENSE_FILE=port@host"
  say ""
  if [[ "$os" == wsl || "$os" == linux ]] && ask_yn "Open activation helper (Linux/WSL only)?" n; then
    bash "$_matlab_dir/activate.sh"
  fi
  if ask_yn "Copy a license file now?" n; then
    lic_path="$(ask "Path to license.lic (or license.dat)")"
    if [[ -f "$lic_path" ]]; then
      mkdir -p "$MATLAB_ROOT/licenses"
      cp "$lic_path" "$MATLAB_ROOT/licenses/"
      say "Copied to $MATLAB_ROOT/licenses/"
      # shellcheck source=scripts/matlab/env.sh
      source "$_matlab_dir/env.sh"
    else
      say "File not found: $lic_path"
    fi
  fi
  if ask_yn "Set MLM_LICENSE_FILE (network license)?" n; then
    export MLM_LICENSE_FILE="$(ask "MLM_LICENSE_FILE (e.g. 27000@license.example.edu)")"
  fi
fi
say ""

if ask_yn "Add MATLAB env to your shell startup (~/.bashrc)?" n; then
  block="# rl-adaptive-dbs MATLAB
export MATLAB_ROOT=\"$MATLAB_ROOT\"
export MATLAB_RELEASE=\"$MATLAB_RELEASE\"
source \"$repo_root/scripts/matlab/env.sh\""
  if [[ -f "$HOME/.bashrc" ]] && grep -q 'rl-adaptive-dbs MATLAB' "$HOME/.bashrc"; then
    say "Already present in ~/.bashrc — edit manually if paths changed."
  else
    printf '\n%s\n' "$block" >>"$HOME/.bashrc"
    say "Appended to ~/.bashrc"
  fi
fi
say ""

say "=== Running verify.sh ==="
bash "$_matlab_dir/verify.sh"
