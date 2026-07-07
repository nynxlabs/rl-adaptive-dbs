#!/usr/bin/env python3
"""Check if task67 warmup v2 training completed and report results."""
import json, os, sys

artifact = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "artifacts/ddpg/task67_warmup_v2.json")

if not os.path.exists(artifact):
    print("STILL_RUNNING")
    sys.exit(0)

with open(artifact) as f:
    data = json.load(f)

u_rollout = data.get("unique_actions_rollout", 0)
u_offline = data.get("unique_actions_offline", 0)
passed = u_rollout > 1 and u_offline > 1

print(f"COMPLETE: rollout_unique={u_rollout} offline_unique={u_offline} elapsed={data.get('elapsed_s', 0):.0f}s")
print(f"ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
print(f"dominant_action={data.get('dominant_action')} dominant_fraction={data.get('dominant_fraction'):.3f}")
print(f"episode_rewards: {[round(r, 1) for r in data.get('episode_rewards', [])]}")
print(json.dumps(data, indent=2))
