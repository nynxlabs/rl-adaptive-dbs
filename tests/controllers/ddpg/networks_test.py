"""DDPG network unit tests."""

from __future__ import annotations

import torch

from controllers.ddpg.networks import Actor, Critic, cnn_flat_dim


def test_actor_output_shape() -> None:
    actor = Actor(state_length=4, n_actions=41)
    logits = actor(torch.zeros(8, 4))
    assert logits.shape == (8, 41)


def test_actor_output_shape_state_length_1() -> None:
    actor = Actor(state_length=1, n_actions=41)
    logits = actor(torch.zeros(4, 1))
    assert logits.shape == (4, 41)


def test_actor_output_shape_state_length_15() -> None:
    actor = Actor(state_length=15, n_actions=41)
    logits = actor(torch.randn(2, 15))
    assert logits.shape == (2, 41)


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


def test_critic_scalar_output_state_length_1() -> None:
    critic = Critic(state_length=1, n_actions=41)
    state = torch.zeros(3, 1)
    one_hot = torch.zeros(3, 41)
    one_hot[:, 5] = 1.0
    values = critic(state, one_hot)
    assert values.shape == (3,)


def test_critic_scalar_output_state_length_15() -> None:
    critic = Critic(state_length=15, n_actions=41)
    state = torch.randn(2, 15)
    logits = torch.randn(2, 41)
    values = critic(state, logits)
    assert values.shape == (2,)


def test_critic_fuses_state_and_action_by_addition() -> None:
    critic = Critic(state_length=4, n_actions=5)
    state = torch.zeros(1, 4)
    logits = torch.zeros(1, 5)
    with torch.no_grad():
        state_emb = critic.encoder(state)
        action_emb = critic.action_branch(logits)
        fused = state_emb + action_emb
        expected = critic.head(fused).squeeze(-1)
    actual = critic(state, logits)
    assert torch.allclose(actual, expected)


def test_cnn_flat_dim_state_length_1_skips_pools() -> None:
    assert cnn_flat_dim(state_length=1, conv2_out=64, pool_kernel=2) == 64


def test_cnn_flat_dim_state_length_15() -> None:
    # 15 -> pool -> 7 -> pool -> 3; flatten 64 * 3
    assert cnn_flat_dim(state_length=15, conv2_out=64, pool_kernel=2) == 192


def test_actor_init_toward_action() -> None:
    """Init nudges logits via head bias only; CNN/head weights stay at PyTorch defaults."""
    torch.manual_seed(0)
    actor = Actor(state_length=1, n_actions=41)
    head_weight_before = actor.head.weight.detach().clone()
    encoder_weight_before = actor.encoder.conv1.weight.detach().clone()

    actor.init_toward_action(10, bias_scale=2.0)

    assert actor.head.bias[10].item() == 2.0
    other_bias = torch.cat([actor.head.bias[:10], actor.head.bias[11:]])
    assert torch.all(other_bias == 0.0)
    assert torch.allclose(actor.head.weight, head_weight_before)
    assert torch.allclose(actor.encoder.conv1.weight, encoder_weight_before)
    assert head_weight_before.abs().sum().item() > 0.0
