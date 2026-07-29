#!/usr/bin/env bash
# Near-hub open-loop diversity @ 45 Hz, then soft-train + logit-margin if ok.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs artifacts/ddpg

echo "=== near-hub diversity @ 45 Hz ==="
uv run python -m rl_adaptive_dbs.run \
  scripts/probes/alphabet_diversity/run_alphabet_diversity_sweep.py \
  --hz 45 \
  --only near_hub_n257 \
  --only near_hub_n513 \
  --out artifacts/ddpg/alphabet_diversity_near_hub.json

python3 - <<'PY'
import json
from pathlib import Path
p = Path("artifacts/ddpg/alphabet_diversity_near_hub.json")
d = json.loads(p.read_text())
ok = [
    r for r in d["runs"]
    if r["mean_hz"] == 45.0 and r["skip_regular"]["diversity_ok"]
]
for r in d["runs"]:
    s = r["skip_regular"]
    print(
        f"{r['key']}: ok={s['diversity_ok']} near={s['n_near_best']} "
        f"margin2={s['margin_best_second']:.3f} best={s['best_action']} "
        f"Pβ={s['best_p_beta']:.1f} uniq={r.get('n_unique_traces')}"
    )
if not ok:
    print("NO diversity_ok — skip soft-train / margin (do not Fig 6a)")
else:
    print(f"{len(ok)} construction(s) diversity_ok — launching margin probe")
Path("artifacts/ddpg/.near_hub_run_margin").write_text("1" if ok else "0")
PY

if [ "$(cat artifacts/ddpg/.near_hub_run_margin)" != "1" ]; then
  echo "=== DONE (diversity gates failed — no Fig 6a) ==="
  exit 0
fi

echo "=== near-hub soft-fp32 + logit margin ==="
uv run python -m rl_adaptive_dbs.run \
  scripts/probes/alphabet_diversity/run_near_hub_margin_probe.py \
  --diversity-json artifacts/ddpg/alphabet_diversity_near_hub.json

echo "=== DONE ==="
