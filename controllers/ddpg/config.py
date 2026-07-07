"""DDPG hyperparameters and variant slugs (Mehregan et al. §IV.A.1)."""

from __future__ import annotations

from dataclasses import dataclass


def init_baseline_for_variant(variant: str) -> str:
    """Map benchmark variant slug to periodic init frequency."""
    if variant == "init-30hz":
        return "periodic-30hz"
    if variant in ("ptq-fp16", "ptq-int8", "qat", "paper"):
        return "periodic-45hz"
    return "periodic-45hz"


@dataclass(frozen=True)
class DDPGConfig:
    """Training defaults aligned with [replication.md](../../docs/controllers/ddpg/replication.md)."""

    # Paper §IV.A.1 (fixed)
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    buffer_capacity: int = 8192
    batch_size: int = 32
    num_episodes: int = 10
    max_episode_steps: int = 30

    # Not in §IV.A.1 — standard DDPG defaults (documented in replication.md §4.1, §9.3)
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = 1

    # CNN topology (§III.B; channel counts intentionally open)
    conv_channels: int = 16
    shrink_dim: int = 4

    # Benchmark variant (`paper`, `init-30hz`, …)
    variant: str = "paper"

    # Training control
    min_buffer_size: int = 32
    seed: int = 0
    device: str = "cpu"

    # Critic action encoding: ``one_hot`` uses the executed discrete action index (required
    # for correct Q targets under exploration); ``logits`` uses stored actor logits (paper
    # tuple layout; breaks when executed action != argmax(logits)).
    critic_action_input: str = "one_hot"  # one_hot | logits

    # Exploration during training (paper uses greedy argmax at deploy; not specified for
    # online interaction — see replication.md §4.2). Linear decay over total env steps.
    exploration_mode: str = "epsilon"  # epsilon | softmax
    exploration_epsilon_start: float = 0.5
    exploration_epsilon_end: float = 0.1
    exploration_temperature_start: float = 2.0
    exploration_temperature_end: float = 0.5

    # Periodic init bias on actor head (§IV.A.1). Default 2.0 matches benchmark checkpoints;
    # lower values (e.g. 0.5) for learning_v1 experiments — see phase4-results.md §10.4.
    init_bias_scale: float = 2.0

    @property
    def init_baseline(self) -> str:
        return init_baseline_for_variant(self.variant)
