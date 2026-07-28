#!/usr/bin/env bash
# Wait for within_step L16 train, then run plant continuity probe.
set -euo pipefail
cd "$(dirname "$0")/../../.."
MAIN_LOG="logs/within-step-l16-train.log"
PROBE_LOG="logs/plant-continuity-probe.log"
export RL_DBS_MAX_THREADS=2

echo "=== waiting for within-step-l16-train ===" | tee -a "$PROBE_LOG"
while true; do
  if grep -q '=== DONE ===' "$MAIN_LOG" 2>/dev/null; then
    echo "train log shows DONE" | tee -a "$PROBE_LOG"
    break
  fi
  if ! tmux has-session -t within-step-l16-train 2>/dev/null; then
    if grep -q 'Traceback\|Error' "$MAIN_LOG" 2>/dev/null; then
      echo "train session ended with error — check $MAIN_LOG" | tee -a "$PROBE_LOG"
      exit 1
    fi
    echo "train tmux gone; waiting for DONE in log..." | tee -a "$PROBE_LOG"
  fi
  if ! pgrep -af 'run_within_step_L16_burst_train' >/dev/null 2>&1; then
    if grep -q '=== DONE ===' "$MAIN_LOG" 2>/dev/null; then
      break
    fi
  fi
  echo "$(date -Is) still training..." | tee -a "$PROBE_LOG"
  sleep 60
done

echo "=== within_step train finished; starting continuity probe ===" | tee -a "$PROBE_LOG"
uv run python -m rl_adaptive_dbs.run --max-threads 2 \
  scripts/probes/alphabet_diversity/run_plant_continuity_probe.py \
  2>&1 | tee -a "$PROBE_LOG"

echo "=== pipeline DONE ===" | tee -a "$PROBE_LOG"
