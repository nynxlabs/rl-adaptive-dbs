
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import torch
import torch.nn.functional as F

from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.trainer import DDPGTrainer
from envs.mehregan.config import MehreganEnvConfig
from envs.mehregan.env import MehreganEnv
from envs.plant.python_backend import PythonPlant

config = DDPGConfig(
    variant="paper",
    num_episodes=15,
    max_episode_steps=30,
    seed=0,
    exploration_mode="softmax",
    exploration_temperature_start=3.0,
    exploration_temperature_end=1.0,
    obs_normalize=True,
    critic_action_input="one_hot",
    reward_normalize=True,
    critic_warmup_steps=50,
    critic_loss_fn="huber",
    logit_noise_std=0.1,
    init_bias_scale=1.0,
    log_episodes=True,
    device="cpu",
)

env_config = MehreganEnvConfig(state_length=15)
plant = PythonPlant()
env = MehreganEnv(plant=plant, config=env_config)
trainer = DDPGTrainer(env, config)

print("Starting training: 15ep, obs_normalize=True, softmax(3->1)", flush=True)
t0 = time.time()
result = trainer.train()
elapsed = time.time() - t0
print(f"Training done: {elapsed:.0f}s", flush=True)

# Greedy rollout
policy = result.policy
policy.eval()
rollout_actions = []
state, info = env.reset(seed=42)
done = False
ep_reward = 0.0
while not done:
    with torch.no_grad():
        state_t = trainer._normalized_state_tensor(state).unsqueeze(0)
        logits = policy(state_t)
        action = int(torch.argmax(logits, dim=-1).item())
    rollout_actions.append(action)
    state, reward, terminated, truncated, info = env.step(action)
    ep_reward += reward
    done = terminated or truncated
unique_rollout = len(set(rollout_actions))

# Offline probe
offline_actions = []
for s in range(20):
    state, _ = env.reset(seed=s)
    with torch.no_grad():
        state_t = trainer._normalized_state_tensor(state).unsqueeze(0)
        logits = policy(state_t)
        action = int(torch.argmax(logits, dim=-1).item())
    offline_actions.append(action)
unique_offline = len(set(offline_actions))

# Diagnostics
with torch.no_grad():
    states_sample = []
    for s in range(10):
        st, _ = env.reset(seed=s)
        states_sample.append(trainer._normalize_obs(st))
    sb = torch.as_tensor(np.stack(states_sample), dtype=torch.float32)
    lb = policy(sb)
    probs = F.softmax(lb, dim=-1)
    logit_margin = float((lb.max(dim=-1).values - lb.median(dim=-1).values).mean())
    entropy = float(-(probs * probs.log()).sum(dim=-1).mean())

env.close()

obs_std_mean = float(np.mean(np.sqrt(np.maximum(trainer._obs_m2 / max(trainer._obs_count, 2), 1e-8))))

report = {
    "state_length": 15, "episodes": 15,
    "obs_normalize": True, "exploration_mode": "softmax",
    "elapsed_s": round(elapsed, 1),
    "episode_rewards": [round(r, 2) for r in result.metrics.episode_rewards],
    "rollout_actions": rollout_actions,
    "unique_actions_rollout": unique_rollout,
    "offline_actions": offline_actions,
    "unique_actions_offline": unique_offline,
    "rollout_reward": round(ep_reward, 2),
    "logit_margin_mean": round(logit_margin, 4),
    "entropy_mean": round(entropy, 4),
    "obs_running_std_mean": round(obs_std_mean, 6),
    "acceptance": unique_rollout > 1 and unique_offline > 1,
}

out = Path("artifacts/ddpg/obsnorm_sl15_softmax_15ep.json")
out.write_text(json.dumps(report, indent=2) + "\n")
print(f"\nResults: {json.dumps(report, indent=2)}", flush=True)
