"""``rl-dbs train`` command tests."""

from __future__ import annotations

import pytest

from rl_adaptive_dbs.train_cmd import train_controller, validate_train_request


def test_validate_train_rejects_ptq_variants() -> None:
    with pytest.raises(ValueError, match="eval-only"):
        validate_train_request("ddpg", "ptq-int8")
    with pytest.raises(ValueError, match="eval-only"):
        validate_train_request("ddpg", "ptq-fp16")


def test_train_controller_rejects_ptq_before_env() -> None:
    with pytest.raises(ValueError, match="eval-only"):
        train_controller("ddpg", "ptq-int8", seeds=(0,))


def test_validate_train_accepts_qat() -> None:
    validate_train_request("ddpg", "qat")
