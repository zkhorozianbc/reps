"""Focused tests for WorkerPool validation and sampling reproducibility."""

import pytest

from reps.config import REPSWorkersConfig
from reps.worker_pool import WorkerPool
from reps.workers.base import WorkerConfig


def _worker(name: str, weight: float = 1.0, role: str = "exploiter") -> WorkerConfig:
    return WorkerConfig(name=name, impl="single_call", role=role, weight=weight)


def _config(*workers: WorkerConfig):
    return REPSWorkersConfig(types=list(workers))


def test_duplicate_worker_names_raise():
    with pytest.raises(ValueError, match="Duplicate worker name"):
        WorkerPool(_config(_worker("same"), _worker("same")))


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_weights_raise(weight):
    with pytest.raises(ValueError, match="positive finite"):
        WorkerPool(_config(_worker("bad", weight=weight)))


def test_set_allocation_rejects_all_zero_negative_or_non_finite_values():
    pool = WorkerPool(_config(_worker("a"), _worker("b")))

    for allocation in (
        {"a": 0.0, "b": 0.0},
        {"a": -1.0, "b": 0.0},
        {"a": float("nan"), "b": 1.0},
        {"a": float("inf"), "b": 1.0},
    ):
        with pytest.raises(ValueError, match="positive finite"):
            pool.set_allocation(allocation)


def test_seeded_pools_are_reproducible_and_independent():
    cfg = _config(_worker("a", weight=1.0), _worker("b", weight=3.0))
    pool_a = WorkerPool(cfg, random_seed=123)
    pool_b = WorkerPool(cfg, random_seed=123)

    seq_a = []
    seq_b = []
    for _ in range(20):
        seq_a.append(pool_a._sample())
        seq_b.append(pool_b._sample())

    fresh_a = WorkerPool(cfg, random_seed=123)
    fresh_b = WorkerPool(cfg, random_seed=123)
    assert [fresh_a._sample() for _ in range(20)] == seq_a
    assert [fresh_b._sample() for _ in range(20)] == seq_b
