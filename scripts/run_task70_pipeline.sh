#!/usr/bin/env bash
# TASK-70: after TASK-67 softmax gate passes, retrain DDPG variants and re-run mehregan_eval.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG="artifacts/ddpg/task70_pipeline_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

GATE_JSON="artifacts/ddpg/explore_sl15_seed0_softmax_30ep.json"
GATE_PID="${TASK67_PID:-452384}"
ISSUE_ID="${PAPERCLIP_TASK_ID:-78c29c22-5e7a-40d5-8c37-790d4628fd2b}"

post_comment() {
  local body="$1"
  local status="${2:-}"
  if [[ -z "${PAPERCLIP_API_URL:-}" || -z "${PAPERCLIP_API_KEY:-}" ]]; then
    echo "WARN: PAPERCLIP_* not set; skip API post"
    return 0
  fi
  python3 <<PY
import json, os, urllib.request
api = os.environ["PAPERCLIP_API_URL"]
key = os.environ["PAPERCLIP_API_KEY"]
issue_id = os.environ.get("PAPERCLIP_TASK_ID", "$ISSUE_ID")
body = open("/tmp/task70_comment.md").read()
payload = {"body": body}
req = urllib.request.Request(
    f"{api}/api/issues/{issue_id}/comments",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print("comment", resp.status)
status = """$status"""
if status:
    req2 = urllib.request.Request(
        f"{api}/api/issues/{issue_id}",
        data=json.dumps({"status": status}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="PATCH",
    )
    with urllib.request.urlopen(req2) as resp:
        print("issue", status, resp.status)
PY
}

echo "=== TASK-70 pipeline started $(date -Iseconds) ==="
echo "Waiting for TASK-67 gate: $GATE_JSON (pid $GATE_PID)"

while [[ ! -f "$GATE_JSON" ]]; do
  if ! kill -0 "$GATE_PID" 2>/dev/null; then
    echo "ERROR: TASK-67 process $GATE_PID exited before $GATE_JSON appeared"
  fi
  sleep 120
done

echo "Gate JSON found; validating adaptivity..."
read -r ROLLOUT_UNIQUE OFFLINE_UNIQUE EXPLORE_MODE <<<"$(python3 -c "
import json
d=json.load(open('$GATE_JSON'))
print(d['unique_actions_rollout'], d['unique_actions_offline'], d['exploration_mode'])
")"

echo "rollout_unique=$ROLLOUT_UNIQUE offline_unique=$OFFLINE_UNIQUE mode=$EXPLORE_MODE"

if [[ "$ROLLOUT_UNIQUE" -le 1 || "$OFFLINE_UNIQUE" -le 1 ]]; then
  cat >/tmp/task70_comment.md <<EOF
## TASK-70 blocked — adaptive policy gate failed

TASK-67 softmax 30-episode run completed but did **not** meet acceptance (\`rollout_unique > 1\` and \`offline_unique > 1\`).

| Metric | Value |
|--------|------:|
| rollout_unique | $ROLLOUT_UNIQUE |
| offline_unique | $OFFLINE_UNIQUE |
| exploration_mode | $EXPLORE_MODE |

Gate file: \`$GATE_JSON\`

**Next:** programmer/experimenter to tune exploration (temperature schedule, longer training, or OU noise) before variant retrain + \`mehregan_eval\`.
EOF
  post_comment "" "blocked"
  exit 2
fi

cat >/tmp/task70_comment.md <<EOF
## TASK-70 — adaptive policy confirmed, retrain started

TASK-67 gate **passed** (\`$GATE_JSON\`):

| Metric | Value |
|--------|------:|
| rollout_unique | $ROLLOUT_UNIQUE |
| offline_unique | $OFFLINE_UNIQUE |
| exploration_mode | $EXPLORE_MODE |

Retraining \`paper\`, \`init-30hz\`, \`qat\` (30 episodes, seed 0, state_length=15) then full \`mehregan_eval\`. Log: \`$LOG\`
EOF
post_comment "" "in_progress"

BACKUP_DIR="artifacts/ddpg/pre_explore_backup_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
for v in paper init-30hz qat; do
  if [[ -f "artifacts/ddpg/${v}_train0.pt" ]]; then
    cp "artifacts/ddpg/${v}_train0.pt" "$BACKUP_DIR/"
  fi
done
echo "Backed up prior checkpoints to $BACKUP_DIR"

echo "--- Retrain paper, init-30hz, qat (30 ep, exploration=$EXPLORE_MODE) ---"
uv run python - <<PY
from pathlib import Path
from controllers.ddpg import train
from controllers.ddpg.config import DDPGConfig
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant
from rl_adaptive_dbs.user_config import resolve_config
from scripts.state_length_sweep import _analyze_policy, _rollout_actions

resolved = resolve_config()
env_cfg = MehreganEnvConfig(state_length=15)
plant = PythonPlant(config=resolved.plant)
env = MehreganEnv(plant=plant, config=env_cfg)
mode = "$EXPLORE_MODE"
summary = []
try:
    for variant in ("paper", "init-30hz", "qat"):
        ckpt = Path(f"artifacts/ddpg/{variant}_train0.pt")
        print(f"Training {variant} -> {ckpt}", flush=True)
        cfg = DDPGConfig(
            variant=variant,
            seed=0,
            num_episodes=30,
            exploration_mode=mode,
        )
        result = train(env, cfg, checkpoint_path=ckpt)
        offline = _analyze_policy(result.actor, 15)
        rollout = _rollout_actions(env, result.actor, seed=1000)
        row = {
            "variant": variant,
            "final_reward": float(result.metrics.episode_rewards[-1]),
            "offline_unique": int(offline["unique_actions_offline"]),
            "rollout_unique": len(set(rollout)),
        }
        summary.append(row)
        print(row, flush=True)
finally:
    env.close()

import json
Path("artifacts/ddpg/task70_retrain_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

RESULTS_DIR="results/mehregan_eval_explore"
echo "--- mehregan_eval -> $RESULTS_DIR ---"
uv run rl-dbs benchmark --suite-name mehregan_eval --results-dir "$RESULTS_DIR" 2>&1

SUMMARY_OUT=$(mktemp)
uv run rl-dbs summary --suite-name mehregan_eval --results-dir "$RESULTS_DIR" >"$SUMMARY_OUT" 2>&1 || true

cat >/tmp/task70_comment.md <<EOF
## TASK-70 complete — retrain + mehregan_eval

### TASK-67 gate
- rollout_unique=$ROLLOUT_UNIQUE, offline_unique=$OFFLINE_UNIQUE, mode=$EXPLORE_MODE

### Retrain summary
\`\`\`json
$(cat artifacts/ddpg/task70_retrain_summary.json 2>/dev/null || echo '{}')
\`\`\`

Prior checkpoints backed up to \`$BACKUP_DIR\`.

### mehregan_eval summary ($RESULTS_DIR)
\`\`\`
$(cat "$SUMMARY_OUT")
\`\`\`

Pipeline log: \`$LOG\`

**Remaining:** update \`docs/development/phase4-results.md\` with new metrics and Figure 4/5a qualitative comparison (researcher/docs).
EOF
post_comment "" "done"

echo "=== TASK-70 pipeline finished $(date -Iseconds) ==="
