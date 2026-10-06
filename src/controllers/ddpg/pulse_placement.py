"""Mehregan et al. Figure 3 actor–critic: per-pulse placement over the step.

Fig 3a (actor), read layer by layer:

    Input 1×l → Conv 1×3×32, ReLU, AvgPool 1×2 → Conv 1×3×64, ReLU, AvgPool 1×2
    → Flatten 1×(l·16) → Linear 256, ReLU → Linear 256, ReLU
    → Linear (action logits) 1×(f·l·l/1000) → Reshape (f·l/1000)×l
    → Softmax(dim=1) → Argmax(dim=−1) → action

``l`` is the step length in milliseconds (2 s → 2000) and ``f`` the mean rate,
so the action is **one time slot (ms) for each of the f·l/1000 pulses** of the
step: the mean rate is fixed by construction and the timing is free. Fig 3b
(critic): the same state CNN → 256, the action logits → Linear 256, ReLU,
addition, Linear → 1.

Here ``state_len`` = ``l`` and ``n_slots`` = ``l`` (one slot per ms).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.plant.dbs import DBS_AMPLITUDE_NA_PER_CM2, DBS_PULSE_WIDTH_MS, DbsSpec


class StateCNN(nn.Module):
    """Fig 3a/3b state branch: two conv + avg-pool blocks, flatten, 2× Linear 256."""

    def __init__(self, *, state_len: int, fc_hidden: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (state_len // 4), fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.unsqueeze(1)
        x = F.avg_pool1d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool1d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(x))


class PulseActor(nn.Module):
    """Fig 3a actor: logits over (n_pulses × n_slots)."""

    def __init__(self, *, state_len: int, n_pulses: int, n_slots: int, fc_hidden: int = 256) -> None:
        super().__init__()
        self.n_pulses = n_pulses
        self.n_slots = n_slots
        self.encoder = StateCNN(state_len=state_len, fc_hidden=fc_hidden)
        self.head = nn.Linear(fc_hidden, n_pulses * n_slots)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))

    @staticmethod
    def slots_from_logits(logits: torch.Tensor, n_pulses: int, n_slots: int) -> torch.Tensor:
        """Reshape → softmax(dim=1) → argmax(dim=−1): one slot per pulse."""
        grid = logits.reshape(-1, n_pulses, n_slots)
        return torch.argmax(F.softmax(grid, dim=-1), dim=-1)

    def init_regular(self, *, mean_hz: float, bias_scale: float) -> None:
        """'Action space initialized using regular pulses' (§I): bias pulse k toward
        its regular slot ``round(k·1000/f)`` ms. ``bias_scale = 0`` leaves the head
        bias at zero (exploration then starts from near-uniform slots)."""
        with torch.no_grad():
            self.head.bias.zero_()
            if bias_scale == 0:
                return
            grid = self.head.bias.view(self.n_pulses, self.n_slots)
            for k in range(self.n_pulses):
                grid[k, min(self.n_slots - 1, int(round(k * 1000.0 / mean_hz)))] = bias_scale


class PulseCritic(nn.Module):
    """Fig 3b critic: state CNN branch + action-logit branch, added, Linear → 1."""

    def __init__(self, *, state_len: int, n_logits: int, fc_hidden: int = 256) -> None:
        super().__init__()
        self.encoder = StateCNN(state_len=state_len, fc_hidden=fc_hidden)
        self.action_fc = nn.Linear(n_logits, fc_hidden)
        self.out = nn.Linear(fc_hidden, 1)

    def forward(self, state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        fused = self.encoder(state) + F.relu(self.action_fc(logits))
        return self.out(fused).squeeze(-1)


def pulse_idbs(
    slots_ms: np.ndarray,
    *,
    step_s: float,
    dt_ms: float,
    pulse_width_ms: float = DBS_PULSE_WIDTH_MS,
    amplitude: float = DBS_AMPLITUDE_NA_PER_CM2,
) -> np.ndarray:
    """STN drive for one step: a ``create_dbs_current``-shaped pulse at each slot.

    Two pulses that pick the same slot coincide (one pulse) — the paper's
    argmax does not forbid it.
    """
    n_steps = int(round(step_s * 1000.0 / dt_ms)) + 1
    idbs = np.zeros(n_steps, dtype=np.float64)
    pulse_len = int(round(pulse_width_ms / dt_ms))
    for slot in np.unique(np.asarray(slots_ms, dtype=np.int64)):
        start = int(round(float(slot) / dt_ms))
        idbs[start : min(start + pulse_len, n_steps)] = amplitude
    return idbs


def pulse_dbs_spec(slots_ms: np.ndarray, *, mean_hz: float, step_s: float, dt_ms: float) -> DbsSpec:
    return DbsSpec(
        pick_dbs_freq=DbsSpec.from_frequency_hz(mean_hz).pick_dbs_freq,
        idbs=pulse_idbs(slots_ms, step_s=step_s, dt_ms=dt_ms),
        mean_hz=mean_hz,
    )


def regular_slots(*, mean_hz: float, n_pulses: int, n_slots: int) -> np.ndarray:
    return np.minimum(
        np.round(np.arange(n_pulses) * 1000.0 / mean_hz).astype(np.int64), n_slots - 1
    )
