#!/usr/bin/env bash
# Shell environment for MATLAB (Linux, macOS, WSL, Git Bash on Windows).
# Usage: source scripts/matlab/env.sh

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_repo_root="$(cd "$_script_dir/../.." && pwd)"

export MATLAB_ROOT="${MATLAB_ROOT:-$HOME/MATLAB}"
export MATLAB_RELEASE="${MATLAB_RELEASE:-R2025b}"
export MATLAB_PREFDIR="${MATLAB_PREFDIR:-$HOME/.matlab/$MATLAB_RELEASE}"
export RL_ADAPTIVE_DBS_ROOT="${RL_ADAPTIVE_DBS_ROOT:-$_repo_root}"
export RL_ADAPTIVE_DBS_MATLAB_MODEL="${RL_ADAPTIVE_DBS_MATLAB_MODEL:-$_repo_root/reference-material/KumaraveluEtAl2016}"

_matlab_bin=""
if [ -x "$MATLAB_ROOT/bin/matlab" ]; then
  _matlab_bin="$MATLAB_ROOT/bin/matlab"
elif [ -x "$MATLAB_ROOT/bin/matlab.exe" ]; then
  _matlab_bin="$MATLAB_ROOT/bin/matlab.exe"
fi

if [ -z "$_matlab_bin" ]; then
  echo "matlab/env.sh: MATLAB not found at MATLAB_ROOT=$MATLAB_ROOT" >&2
  echo "  Set MATLAB_ROOT to your install (docs/matlab.md §2)" >&2
  return 1 2>/dev/null || exit 1
fi

# Drop stale MLM_LICENSE_FILE (e.g. path from another machine).
if [ -n "${MLM_LICENSE_FILE:-}" ]; then
  _mlm_ok=0
  IFS=':' read -r -a _mlm_parts <<<"$MLM_LICENSE_FILE"
  for _part in "${_mlm_parts[@]}"; do
    if [ -f "$_part" ] || [ -d "$_part" ]; then
      _mlm_ok=1
      break
    fi
  done
  if [ "$_mlm_ok" -eq 0 ]; then
    unset MLM_LICENSE_FILE
  fi
fi

if [ -z "${MLM_LICENSE_FILE:-}" ]; then
  _license_candidates=()
  [ -d "$MATLAB_ROOT/licenses" ] && _license_candidates+=("$MATLAB_ROOT/licenses")
  [ -d "$HOME/.matlab/${MATLAB_RELEASE}_licenses" ] && _license_candidates+=("$HOME/.matlab/${MATLAB_RELEASE}_licenses")
  if [ "${#_license_candidates[@]}" -gt 0 ]; then
    export MLM_LICENSE_FILE
    MLM_LICENSE_FILE=$(IFS=:; echo "${_license_candidates[*]}")
  fi
fi

export PATH="$MATLAB_ROOT/bin${PATH:+:$PATH}"

_matlab_os="$(uname -s)"
_matlab_arch=""
case "$_matlab_os" in
  Linux) _matlab_arch="glnxa64" ;;
  Darwin)
    case "$(uname -m)" in
      arm64) _matlab_arch="maca64" ;;
      *) _matlab_arch="maci64" ;;
    esac
    ;;
  MINGW* | MSYS* | CYGWIN*) _matlab_arch="win64" ;;
esac

if [ -n "$_matlab_arch" ] && [ -d "$MATLAB_ROOT/bin/$_matlab_arch" ]; then
  case "$_matlab_os" in
    Darwin)
      export DYLD_LIBRARY_PATH="$MATLAB_ROOT/bin/$_matlab_arch${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
      ;;
    MINGW* | MSYS* | CYGWIN*)
      export PATH="$MATLAB_ROOT/bin/$_matlab_arch${PATH:+:$PATH}"
      ;;
    *)
      export LD_LIBRARY_PATH="$MATLAB_ROOT/bin/$_matlab_arch${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      ;;
  esac
fi

if [ "$_matlab_os" = "Linux" ]; then
  _system_libstdc="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
  if [ -f "$_system_libstdc" ] && [ -z "${LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="$_system_libstdc"
  fi
  unset _system_libstdc
fi

unset _matlab_os _matlab_arch _matlab_bin _mlm_ok _mlm_parts _part _license_candidates

matlab() {
  if [ "$(uname -s)" = "Linux" ] && [ -z "${DISPLAY:-}" ]; then
    xvfb-run -a "$MATLAB_ROOT/bin/matlab" "$@"
  elif [ -x "$MATLAB_ROOT/bin/matlab.exe" ]; then
    "$MATLAB_ROOT/bin/matlab.exe" "$@"
  else
    "$MATLAB_ROOT/bin/matlab" "$@"
  fi
}
