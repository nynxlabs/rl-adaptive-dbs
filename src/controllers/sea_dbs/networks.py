"""MLP actor, critic, and predictive model for binary SEA-DBS."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Binary logits actor (replication.md §12)."""

    def __init__(
        self,
        *,
        state_dim: int,
        n_actions: int = 2,
        hidden_size: int = 64,
        no_stim_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.net = _mlp(state_dim, hidden_size, n_actions)
        if no_stim_bias != 0.0 and n_actions >= 2:
            # Bias action 0 (no pulse) so early PSD starts high; learning then stims.
            with torch.no_grad():
                self.net[-1].bias[0] += float(no_stim_bias)
                self.net[-1].bias[1] -= float(no_stim_bias)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)

    @staticmethod
    def select_action(logits: Tensor) -> tuple[Tensor, Tensor]:
        action = torch.argmax(logits, dim=-1)
        return action, logits


class Critic(nn.Module):
    """Q(s, a) with one-hot action input."""

    def __init__(self, *, state_dim: int, n_actions: int = 2, hidden_size: int = 64) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.net = _mlp(state_dim + n_actions, hidden_size, 1)

    def forward(self, state: Tensor, action_one_hot: Tensor) -> Tensor:
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.net(x).squeeze(-1)


class PredictiveModel(nn.Module):
    """f_theta(s, a) -> r_hat (Eq. 8)."""

    def __init__(self, *, state_dim: int, n_actions: int = 2, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = _mlp(state_dim + n_actions, hidden_size, 1)

    def forward(self, state: Tensor, action_one_hot: Tensor) -> Tensor:
        x = torch.cat([state, action_one_hot], dim=-1)
        return self.net(x).squeeze(-1)


def gumbel_softmax_sample(
    logits: Tensor,
    *,
    tau: float,
    hard: bool = True,
) -> tuple[Tensor, Tensor]:
    """Gumbel-Softmax for binary actions (Eqs. 11–13).

    Returns ``(relaxed_probs, hard_action_index)``. The hard index is
    Gumbel-max on the raw logits and does **not** depend on ``tau``;
    temperature only scales the straight-through soft probabilities.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    uniform = torch.rand_like(logits).clamp(1e-20, 1.0 - 1e-20)
    gumbels = -torch.log(-torch.log(uniform))
    y = F.softmax((logits + gumbels) / max(tau, 1e-8), dim=-1)
    if hard:
        index = y.argmax(dim=-1)
        y_hard = F.one_hot(index, num_classes=logits.shape[-1]).to(y.dtype)
        y = y_hard - y.detach() + y
        return y, index.squeeze(0) if index.numel() == 1 else index
    index = y.argmax(dim=-1)
    return y, index


def action_one_hot(actions: Tensor, n_actions: int) -> Tensor:
    return F.one_hot(actions.long(), num_classes=n_actions).to(dtype=torch.float32)


def clone_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
            strict=True,
        ):
            target_param.mul_(1.0 - tau)
            target_param.add_(source_param, alpha=tau)
