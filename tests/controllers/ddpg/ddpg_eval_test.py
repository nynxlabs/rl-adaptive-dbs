"""DDPG eval rollout tests (mock plant)."""

from __future__ import annotations

import pytest

from controllers.ddpg import (
    DDPGConfig,
    EvalConfig,
    evaluate,
    load_actor,
    run_mehregan_eval,
    run_policy_rollout,
    save_checkpoint,
    train,
)
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def eval_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=3, state_length=2),
    )
    yield env
    env.close()


@pytest.fixture
def trained_actor(eval_env: MehreganEnv, tmp_path):
    config = DDPGConfig(
        num_episodes=1,
        batch_size=4,
        min_buffer_size=4,
        buffer_capacity=32,
        seed=3,
    )
    result = train(eval_env, config)
    ckpt = save_checkpoint(
        tmp_path / "actor.pt",
        actor=result.actor,
        config=config,
        state_length=2,
        n_actions=eval_env.action_space.n,
    )
    return result.actor, ckpt


def test_run_policy_rollout(eval_env: MehreganEnv, trained_actor) -> None:
    actor, _ = trained_actor
    rollout = run_policy_rollout(eval_env, actor, seed=1)
    assert rollout.steps == 3
    assert len(rollout.p_beta) == 4  # reset + 3 steps
    assert len(rollout.actions) == 3
    assert rollout.reward_sum == rollout.total_reward


def test_run_mehregan_eval_protocol(eval_env: MehreganEnv, trained_actor) -> None:
    actor, _ = trained_actor
    result = run_mehregan_eval(eval_env, actor, config=EvalConfig(seed=2, eval_steps=2))
    assert result["protocol"] == "mehregan_eval"
    assert result["eval_steps"] == 2
    assert result["steps"] == 2
    assert len(result["p_beta"]) == 3  # reset + 2 policy steps
    assert "p_beta_mean" in result
    assert "stim_frequency_mean" in result


def test_checkpoint_roundtrip(eval_env: MehreganEnv, trained_actor) -> None:
    _, ckpt = trained_actor
    actor, config = load_actor(ckpt, device="cpu")
    assert config.variant == "paper"
    rollout = run_policy_rollout(eval_env, actor, seed=0)
    assert rollout.steps == 3


def test_evaluate_from_checkpoint(eval_env: MehreganEnv, trained_actor) -> None:
    _, ckpt = trained_actor
    result = evaluate(eval_env, ckpt, config=EvalConfig(seed=4, eval_steps=2))
    assert result["protocol"] == "mehregan_eval"
    assert result["steps"] == 2
