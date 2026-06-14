"""DDPG network unit tests."""

from __future__ import annotations

import torch

from controllers.ddpg.networks import Actor, Critic


def test_actor_output_shape() -> None:
    actor = Actor(state_length=4, n_actions=41)
    logits = actor(torch.zeros(8, 4))
    assert logits.shape == (8, 41)


def test_actor_select_action() -> None:
    actor = Actor(state_length=1, n_actions=5)
    logits = torch.tensor([[0.1, 0.2, 3.0, 0.0, 0.0]])
    action, out_logits = Actor.select_action(logits)
    assert int(action.item()) == 2
    assert torch.equal(out_logits, logits)


def test_critic_scalar_output() -> None:
    critic = Critic(state_length=2, n_actions=41)
    state = torch.zeros(4, 2)
    logits = torch.zeros(4, 41)
    values = critic(state, logits)
    assert values.shape == (4,)


def test_actor_init_toward_action() -> None:
    actor = Actor(state_length=1, n_actions=41)
    actor.init_toward_action(10)
    logits = actor(torch.zeros(1, 1))
    assert int(torch.argmax(logits).item()) == 10
