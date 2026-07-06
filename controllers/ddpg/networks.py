"""CNN actor–critic for discrete STN patterns (Mehregan §III.B)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


class StateEncoder(nn.Module):
    """1-D CNN + adaptive average pool over biomarker window."""

    def __init__(self, *, state_length: int, conv_channels: int, shrink_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(shrink_dim)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self._out_features = conv_channels * 2 * shrink_dim
        self._state_length = state_length

    @property
    def out_features(self) -> int:
        return self._out_features

    def forward(self, state: Tensor) -> Tensor:
        # state: (batch, state_length)
        x = state.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        return x.flatten(1)


class Actor(nn.Module):
    """Maps biomarker state to pattern logits."""

    def __init__(
        self,
        *,
        state_length: int,
        n_actions: int,
        conv_channels: int = 16,
        shrink_dim: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            state_length=state_length,
            conv_channels=conv_channels,
            shrink_dim=shrink_dim,
        )
        self.head = nn.Linear(self.encoder.out_features, n_actions)

    def forward(self, state: Tensor) -> Tensor:
        return self.head(self.encoder(state))

    @staticmethod
    def select_action(logits: Tensor) -> tuple[Tensor, Tensor]:
        """Softmax + argmax; returns ``(action_index, logits)``."""
        action = torch.argmax(logits, dim=-1)
        return action, logits

    def init_toward_action(self, action_index: int, *, bias_scale: float = 2.0) -> None:
        """Bias logits toward ``action_index`` (45 Hz / 30 Hz init experiments)."""
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.weight.zero_()
            self.head.bias[action_index] = bias_scale


class Critic(nn.Module):
    """State–action value on biomarker state and actor logits."""

    def __init__(
        self,
        *,
        state_length: int,
        n_actions: int,
        conv_channels: int = 16,
        shrink_dim: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            state_length=state_length,
            conv_channels=conv_channels,
            shrink_dim=shrink_dim,
        )
        fused = self.encoder.out_features + n_actions
        self.head = nn.Sequential(
            nn.Linear(fused, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.n_actions = n_actions

    def forward(self, state: Tensor, action_logits: Tensor) -> Tensor:
        encoded = self.encoder(state)
        fused = torch.cat([encoded, action_logits], dim=-1)
        return self.head(fused).squeeze(-1)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
            strict=True,
        ):
            target_param.mul_(1.0 - tau)
            target_param.add_(source_param, alpha=tau)


def clone_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)
