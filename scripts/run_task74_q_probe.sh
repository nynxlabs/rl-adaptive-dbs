#!/usr/bin/env bash
# TASK-74: paired Q-discrimination probes (logits vs one_hot), 3-ep PythonPlant.
set -euo pipefail
cd "$(dirname "$0")/.."
ART=artifacts/ddpg
mkdir -p "$ART"
LOG="$ART/learning_dynamics_task74.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TASK-74 Q probe start $(date -Iseconds) ==="

echo "--- one_hot (3 ep) ---"
uv run python scripts/ddpg_learning_dynamics.py \
  --episodes 3 \
  --state-length 15 \
  --seed 0 \
  --exploration-mode softmax \
  --log-every 10 \
  --critic-action-input one_hot \
  --out "$ART/learning_dynamics_task74_onehot.json"

echo "--- logits (3 ep) ---"
uv run python scripts/ddpg_learning_dynamics.py \
  --episodes 3 \
  --state-length 15 \
  --seed 0 \
  --exploration-mode softmax \
  --log-every 10 \
  --critic-action-input logits \
  --out "$ART/learning_dynamics_task74_logits.json"

echo "=== TASK-74 Q probe done $(date -Iseconds) ==="

python3 - <<'PY'
import json
from pathlib import Path

art = Path("artifacts/ddpg")
one = json.loads((art / "learning_dynamics_task74_onehot.json").read_text())
log = json.loads((art / "learning_dynamics_task74_logits.json").read_text())
s1 = one.get("summary", {})
s2 = log.get("summary", {})
q1 = s1.get("q_std_onehot_actions_mean", 0.0)
q2 = s2.get("q_std_onehot_actions_mean", 0.0)
ratio = q1 / q2 if q2 > 0 else float("inf")
pass_gate = q1 > 0.15 or ratio > 2.0
print(json.dumps({
    "one_hot_q_std_mean": q1,
    "logits_q_std_mean": q2,
    "ratio_one_hot_over_logits": ratio,
    "acceptance_pass": pass_gate,
    "one_hot_n_logged": s1.get("n_logged_updates"),
    "logits_n_logged": s2.get("n_logged_updates"),
}, indent=2))
Path(art / "learning_dynamics_task74_verdict.json").write_text(
    json.dumps({
        "one_hot": s1,
        "logits": s2,
        "one_hot_q_std_mean": q1,
        "logits_q_std_mean": q2,
        "ratio": ratio,
        "acceptance_pass": pass_gate,
    }, indent=2) + "\n"
)
PY
