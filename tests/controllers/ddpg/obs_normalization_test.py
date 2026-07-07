
"""Unit tests for observation normalization (TASK-67)."""
import numpy as np
from controllers.ddpg.config import DDPGConfig
from controllers.ddpg.trainer import DDPGTrainer
from tests.envs.mock_plant import MockPlant
from envs.mehregan.env import MehreganEnv


def _make_trainer(obs_normalize: bool = True, state_length: int = 5) -> DDPGTrainer:
    plant = MockPlant()
    env = MehreganEnv(plant=plant)
    config = DDPGConfig(
        obs_normalize=obs_normalize,
        num_episodes=1,
        max_episode_steps=3,
        min_buffer_size=2,
        batch_size=2,
    )
    # Force state_length from config
    env.observation_space = type(env.observation_space)(
        low=env.observation_space.low[:state_length],
        high=env.observation_space.high[:state_length],
        shape=(state_length,),
        dtype=env.observation_space.dtype,
    )
    env._obs_window = type(env._obs_window)(maxlen=state_length)
    return DDPGTrainer(env, config)


def test_obs_normalize_disabled_noop():
    """When obs_normalize=False, _normalize_obs returns input unchanged."""
    trainer = _make_trainer(obs_normalize=False)
    state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    result = trainer._normalize_obs(state)
    np.testing.assert_array_equal(result, state)


def test_obs_normalize_passthrough_before_stats():
    """Before enough samples, normalization passes through."""
    trainer = _make_trainer(obs_normalize=True, state_length=5)
    state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # _obs_count=0, should pass through
    result = trainer._normalize_obs(state)
    np.testing.assert_array_equal(result, state)


def test_obs_normalize_after_stats():
    """After collecting stats, normalization produces z-scores."""
    trainer = _make_trainer(obs_normalize=True, state_length=5)
    # Feed identical observations — std should be ~0 → near-zero output
    for _ in range(10):
        trainer._update_obs_stats(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
    state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    result = trainer._normalize_obs(state)
    # With zero variance, should be 0 (mean-centered)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_obs_normalize_amplifies_differences():
    """Normalization should amplify small differences in the observation."""
    trainer = _make_trainer(obs_normalize=True, state_length=5)
    # Feed observations with very small std (like real P_beta window)
    base = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    for i in range(50):
        noise = np.random.randn(5) * 0.005  # tiny noise, like real obs
        trainer._update_obs_stats(base + noise)
    # Now compare two states that differ by 0.01 (raw)
    s1 = np.array([0.5, 0.5, 0.5, 0.5, 0.505], dtype=np.float32)
    s2 = np.array([0.5, 0.5, 0.5, 0.5, 0.495], dtype=np.float32)
    n1 = trainer._normalize_obs(s1)
    n2 = trainer._normalize_obs(s2)
    # Normalized difference should be MUCH larger than raw difference
    raw_diff = abs(s1[-1] - s2[-1])
    norm_diff = abs(n1[-1] - n2[-1])
    assert norm_diff > raw_diff * 5, (
        f"Normalization should amplify: raw_diff={raw_diff:.4f}, "
        f"norm_diff={norm_diff:.4f}"
    )


def test_obs_normalize_batch():
    """Batch normalization matches per-element normalization."""
    trainer = _make_trainer(obs_normalize=True, state_length=5)
    import torch
    for i in range(20):
        trainer._update_obs_stats(np.random.randn(5).astype(np.float32))
    batch = torch.randn(8, 5)
    normed_batch = trainer._normalize_obs_batch(batch)
    # Compare first element
    single = trainer._normalize_obs(batch[0].numpy())
    np.testing.assert_allclose(normed_batch[0].numpy(), single, atol=1e-5)
