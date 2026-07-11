"""SNN / DSQN controller (Nguyen et al.)."""

from __future__ import annotations

from controllers.snn.actions import decode_factored_action, decode_joint_action, select_action
from controllers.snn.adapter import NguyenEnvAdapter
from controllers.snn.buffer import ReplayBuffer, Transition, TransitionBatch
from controllers.snn.config import SNNConfig
from controllers.snn.dbs_params import DBSParameterState
from controllers.snn.encoder import SpikeObservationEncoder
from controllers.snn.energy import dbs_energy_index
from controllers.snn.networks import DSQN, LIFLayer, LIFOutput
from controllers.snn.reward import alpha_beta_power, nguyen_reward
from controllers.snn.trainer import DSQNTrainer, TrainMetrics

__all__ = [
    "DBSParameterState",
    "DSQN",
    "DSQNTrainer",
    "LIFLayer",
    "LIFOutput",
    "NguyenEnvAdapter",
    "ReplayBuffer",
    "SNNConfig",
    "SpikeObservationEncoder",
    "TrainMetrics",
    "Transition",
    "TransitionBatch",
    "alpha_beta_power",
    "dbs_energy_index",
    "decode_factored_action",
    "decode_joint_action",
    "nguyen_reward",
    "select_action",
]
