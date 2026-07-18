"""Fixture script for thread_limits tests — run via rl_adaptive_dbs.run."""

from __future__ import annotations

import os

print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '')}")
