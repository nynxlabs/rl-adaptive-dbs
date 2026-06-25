"""DDPG quantization tests."""

from __future__ import annotations

import pytest
import torch

from controllers.ddpg import DDPGConfig, evaluate, save_checkpoint, train
from controllers.ddpg.eval import EvalConfig, run_mehregan_eval
from controllers.ddpg.quantization import (
    QATActor,
    apply_ptq,
    fp_source_variant,
    prepare_actor_for_eval,
    wrap_actor_for_training,
)
from envs.mehregan import MehreganEnv, MehreganEnvConfig
from tests.envs.mock_plant import MockPlant


@pytest.fixture
def q_env() -> MehreganEnv:
    env = MehreganEnv(
        plant=MockPlant(),
        config=MehreganEnvConfig(max_episode_steps=3, state_length=2),
    )
    yield env
    env.close()


def test_fp_source_variant_for_ptq() -> None:
    assert fp_source_variant("ptq-int8") == "paper"
    assert fp_source_variant("paper") == "paper"


def test_qat_actor_forward(q_env: MehreganEnv) -> None:
    config = DDPGConfig(num_episodes=1, batch_size=4, min_buffer_size=4, variant="qat")
    result = train(q_env, config)
    assert result.config.variant == "qat"


def test_ptq_fp16_eval(q_env: MehreganEnv, tmp_path) -> None:
    config = DDPGConfig(num_episodes=1, batch_size=4, min_buffer_size=4, seed=1)
    result = train(q_env, config)
    ckpt = save_checkpoint(
        tmp_path / "paper_train0.pt",
        actor=result.actor,
        config=config,
        state_length=2,
        n_actions=q_env.action_space.n,
    )
    metrics = evaluate(
        q_env,
        ckpt,
        config=EvalConfig(seed=0, eval_steps=2),
        variant="ptq-fp16",
    )
    assert metrics["protocol"] == "mehregan_eval"
    assert metrics["metrics_extra"]["quantization"] == "ptq-fp16"


def test_ptq_int8_prepare(q_env: MehreganEnv) -> None:
    config = DDPGConfig(num_episodes=1, batch_size=4, min_buffer_size=4)
    result = train(q_env, config)
    prepared = prepare_actor_for_eval(result.actor, "ptq-int8")
    out = run_mehregan_eval(q_env, prepared, config=EvalConfig(seed=0, eval_steps=1))
    assert out["steps"] == 1
