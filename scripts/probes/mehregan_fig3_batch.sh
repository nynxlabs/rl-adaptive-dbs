#!/usr/bin/env bash
# Run Fig 3 screen probes in parallel, one single-threaded process each.
# usage: mehregan_fig3_batch.sh <jobfile> <parallel>   jobfile lines: name|timeout_s|args...
set -u
cd "$(dirname "$0")/../.."
JOBS="$1"; P="${2:-8}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RL_DBS_MAX_THREADS=1
run_one() {
  IFS='|' read -r name tmo rest <<< "$1"
  log="logs/f3s-$name.log"
  echo "# start $name $(date -Is) timeout=$tmo args=$rest" > "$log"
  timeout "$tmo" uv run --no-sync python -m rl_adaptive_dbs.run scripts/probes/mehregan_fig3_screen.py \
    $rest --checkpoint "artifacts/probes/mehregan_fig3/screen/$name.pt" \
    --out "artifacts/probes/mehregan_fig3/screen/$name.json" >> "$log" 2>&1 < /dev/null
  echo "# exit $? $(date -Is)" >> "$log"
}
export -f run_one
grep -v '^#' "$JOBS" | grep -v '^$' | xargs -d '\n' -P "$P" -I{} bash -c 'run_one "$@"' _ {}
echo "batch done $(date -Is)"
