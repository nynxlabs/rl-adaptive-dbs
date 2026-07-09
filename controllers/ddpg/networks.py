"""CNN actor–critic for discrete STN patterns (Mehregan §III.B, Figure 3a/3b)."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


def _pooled_length(length: int, pool_kernel: int) -> int:
    """Temporal length after ``AvgPool1d(pool_kernel)`` with stride = kernel.

    When ``length <= 1`` the pool is skipped (paper assumes longer windows; at
    ``state_length=1`` two stride-2 pools would zero the sequence).
    """
    if length <= 1:
        return 1
    return length // pool_kernel


def cnn_flat_dim(*, state_length: int, conv2_out: int, pool_kernel: int) -> int:
    """Flattened feature count after two conv blocks and average pools."""
    length = _pooled_length(_pooled_length(state_length, pool_kernel), pool_kernel)
    return conv2_out * length


class StateEncoder(nn.Module):
    """Shared CNN + two 256-d FC layers (Mehregan Figure 3a state branch)."""

    def __init__(
        self,
        *,
        state_length: int,
        conv1_out: int = 32,
        conv2_out: int = 64,
        pool_kernel: int = 2,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool_kernel = pool_kernel
        self.fc_hidden = fc_hidden
        flat_dim = cnn_flat_dim(
            state_length=state_length,
            conv2_out=conv2_out,
            pool_kernel=pool_kernel,
        )
        self.fc1 = nn.Linear(flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, fc_hidden)

    @property
    def out_features(self) -> int:
        return self.fc_hidden

    def _avg_pool(self, x: Tensor) -> Tensor:
        if x.size(-1) <= 1:
            return x
        return F.avg_pool1d(x, kernel_size=self.pool_kernel, stride=self.pool_kernel)

    def forward(self, state: Tensor) -> Tensor:
        # state: (batch, state_length)
        x = state.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self._avg_pool(x)
        x = F.relu(self.conv2(x))
        x = self._avg_pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class Actor(nn.Module):
    """Maps biomarker state to pattern logits (Figure 3a)."""

    def __init__(
        self,
        *,
        state_length: int,
        n_actions: int,
        conv1_out: int = 32,
        conv2_out: int = 64,
        pool_kernel: int = 2,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            state_length=state_length,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            pool_kernel=pool_kernel,
            fc_hidden=fc_hidden,
        )
        self.head = nn.Linear(fc_hidden, n_actions)

    def forward(self, state: Tensor) -> Tensor:
        return self.head(self.encoder(state))

    @staticmethod
    def select_action(logits: Tensor) -> tuple[Tensor, Tensor]:
        """Greedy argmax on logits; returns ``(action_index, logits)``."""
        action = torch.argmax(logits, dim=-1)
        return action, logits

    def init_toward_action(self, action_index: int, *, bias_scale: float = 2.0) -> None:
        """Bias logits toward ``action_index`` (45 Hz / 30 Hz init experiments).

        Paper §IV.A.1: initialize with regular pulses at mean frequency. We keep
        default PyTorch encoder and head weights and add a modest positive bias on
        the periodic pattern index only — zeroing weights caused instant argmax
        collapse (TASK-169 / TASK-173).
        """
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[action_index] = bias_scale


class Critic(nn.Module):
    """State–action value on biomarker state and actor logits (Figure 3b)."""

    def __init__(
        self,
        *,
        state_length: int,
        n_actions: int,
        conv1_out: int = 32,
        conv2_out: int = 64,
        pool_kernel: int = 2,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(
            state_length=state_length,
            conv1_out=conv1_out,
            conv2_out=conv2_out,
            pool_kernel=pool_kernel,
            fc_hidden=fc_hidden,
        )
        self.action_branch = nn.Sequential(
            nn.Linear(n_actions, fc_hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(fc_hidden, 1)
        self.n_actions = n_actions

    def forward(self, state: Tensor, action_features: Tensor) -> Tensor:
        """``action_features`` is either actor logits or a one-hot action vector (n_actions)."""
        state_emb = self.encoder(state)
        action_emb = self.action_branch(action_features)
        fused = state_emb + action_emb
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
