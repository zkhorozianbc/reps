"""Phase 6.1 — pure helper tests for `reps.minibatch.select_instances`.

These cover the contract before any evaluator wiring exists, so a
regression in the helper surfaces as a unit-level failure rather than
bleeding into integration tests.
"""

from __future__ import annotations

import pytest

from reps.config import EvaluatorConfig
from reps.minibatch import VALID_STRATEGIES, select_instances


class TestSelectInstancesEdgeCases:
    def test_empty_all_keys_returns_empty(self):
        assert select_instances([], 3, iteration=0) == []
        assert select_instances([], 3, iteration=42, strategy="random") == []

    def test_size_ge_len_returns_all_keys(self):
        keys = ["a", "b", "c"]
        assert select_instances(keys, 3, iteration=0) == keys
        assert select_instances(keys, 5, iteration=0) == keys

    def test_size_ge_len_returns_copy_not_alias(self):
        keys = ["a", "b", "c"]
        result = select_instances(keys, 5, iteration=0)
        result.append("z")
        assert keys == ["a", "b", "c"]

    def test_zero_or_negative_size_returns_all_keys(self):
        keys = ["a", "b", "c"]
        # `size <= 0` is interpreted as "skip the minibatch path" — same
        # contract as `size >= len(all_keys)`.
        assert select_instances(keys, 0, iteration=0) == keys
        assert select_instances(keys, -1, iteration=0) == keys

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown minibatch strategy"):
            select_instances(["a", "b", "c", "d"], 2, iteration=0, strategy="bogus")


class TestFixedSubsetStrategy:
    def test_same_iteration_window_yields_same_subset(self):
        keys = [f"task_{i}" for i in range(10)]
        # Window 0 covers iterations [0, size). Two candidates evaluated
        # at iteration 0 and iteration 1 with size=4 must see the same
        # subset so per-instance scores are comparable.
        a = select_instances(keys, 4, iteration=0, strategy="fixed_subset")
        b = select_instances(keys, 4, iteration=1, strategy="fixed_subset")
        c = select_instances(keys, 4, iteration=3, strategy="fixed_subset")
        assert a == b == c
        assert len(a) == 4

    def test_window_advances_after_size_iterations(self):
        keys = [f"task_{i}" for i in range(10)]
        first_window = select_instances(keys, 4, iteration=0, strategy="fixed_subset")
        second_window = select_instances(keys, 4, iteration=4, strategy="fixed_subset")
        assert first_window != second_window

    def test_rotation_wraps_around(self):
        keys = [f"task_{i}" for i in range(5)]
        # size=3, n=5. window 0 -> start=0, window 1 -> start=3, window 2 -> start=6 % 5 = 1.
        win0 = select_instances(keys, 3, iteration=0, strategy="fixed_subset")
        win1 = select_instances(keys, 3, iteration=3, strategy="fixed_subset")
        win2 = select_instances(keys, 3, iteration=6, strategy="fixed_subset")
        assert win0 == ["task_0", "task_1", "task_2"]
        # Window 1 wraps: keys 3, 4, then back to 0.
        assert win1 == ["task_3", "task_4", "task_0"]
        # Window 2 starts at index 1.
        assert win2 == ["task_1", "task_2", "task_3"]

    def test_default_strategy_is_fixed_subset(self):
        keys = [f"task_{i}" for i in range(8)]
        default = select_instances(keys, 3, iteration=2)
        explicit = select_instances(keys, 3, iteration=2, strategy="fixed_subset")
        assert default == explicit

    def test_returned_list_is_independent_copy(self):
        keys = [f"task_{i}" for i in range(8)]
        result = select_instances(keys, 3, iteration=0, strategy="fixed_subset")
        result.clear()
        # Subsequent call must not be affected by mutation of a prior return.
        again = select_instances(keys, 3, iteration=0, strategy="fixed_subset")
        assert again == ["task_0", "task_1", "task_2"]


class TestRandomStrategy:
    def test_random_with_same_seed_is_reproducible(self):
        keys = [f"task_{i}" for i in range(20)]
        a = select_instances(keys, 5, iteration=7, strategy="random")
        b = select_instances(keys, 5, iteration=7, strategy="random")
        assert a == b
        assert len(a) == 5
        assert len(set(a)) == 5  # no duplicates

    def test_random_with_different_seeds_varies(self):
        keys = [f"task_{i}" for i in range(20)]
        # Across many distinct seeds, at least two outputs must differ.
        # Sampling 5 of 20 has C(20,5)=15504 distinct outputs ordered, so
        # collisions across small seed sweeps are vanishingly rare.
        seen = {tuple(select_instances(keys, 5, iteration=i, strategy="random")) for i in range(20)}
        assert len(seen) > 1

    def test_random_does_not_mutate_global_rng(self):
        import random as _random

        keys = [f"task_{i}" for i in range(20)]
        _random.seed(12345)
        before = _random.random()
        _random.seed(12345)
        # Calling select_instances must not consume entropy from the global RNG.
        select_instances(keys, 5, iteration=99, strategy="random")
        after = _random.random()
        assert before == after


class TestConfigFields:
    def test_evaluator_config_minibatch_defaults(self):
        cfg = EvaluatorConfig()
        assert cfg.minibatch_size is None
        assert cfg.minibatch_promotion_threshold == 0.5
        assert cfg.minibatch_strategy == "fixed_subset"

    def test_evaluator_config_accepts_minibatch_settings(self):
        cfg = EvaluatorConfig(
            minibatch_size=4,
            minibatch_promotion_threshold=0.7,
            minibatch_strategy="random",
        )
        assert cfg.minibatch_size == 4
        assert cfg.minibatch_promotion_threshold == 0.7
        assert cfg.minibatch_strategy == "random"

    def test_strategy_constants_match(self):
        # Guard against drift between docs/config and helper.
        assert "fixed_subset" in VALID_STRATEGIES
        assert "random" in VALID_STRATEGIES
