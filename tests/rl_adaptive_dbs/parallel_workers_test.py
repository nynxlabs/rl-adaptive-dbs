"""Tests for process-pool scheduling helpers."""

from __future__ import annotations

import os

from rl_adaptive_dbs.parallel_workers import effective_workers, run_in_parallel


def _pid_worker(value: int) -> tuple[int, int]:
    return value, os.getpid()


def test_effective_workers_clamps_to_task_count() -> None:
    assert effective_workers(1, 5) == 1
    assert effective_workers(0, 5) == 1
    assert effective_workers(-2, 5) == 1
    assert effective_workers(4, 2) == 2
    assert effective_workers(8, 3) == 3
    assert effective_workers(3, 0) == 1


def test_run_in_parallel_preserves_order() -> None:
    def _double(value: int) -> int:
        return value * 2

    assert run_in_parallel([1, 2, 3, 4], _double, parallel=1) == [2, 4, 6, 8]
    assert run_in_parallel([], _double, parallel=4) == []


def test_run_in_parallel_uses_process_pool() -> None:
    results = run_in_parallel([0, 1, 2, 3], _pid_worker, parallel=2)
    assert [item[0] for item in results] == [0, 1, 2, 3]
    pids = {item[1] for item in results}
    assert len(pids) >= 2
