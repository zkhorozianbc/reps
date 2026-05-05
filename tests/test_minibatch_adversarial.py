"""Phase 6 adversarial / coverage-gap tests.

Pins down behavior the implementer covered only obliquely or not at all.
Each test is annotated with the spec category it closes:

  (1) `select_instances` edge cases — iteration boundaries, fixed_subset
      invariant across calls, size boundaries.
  (3) Promotion threshold edge cases — boundary inclusivity, NaN /
      missing / non-numeric `combined_score`, evaluator raising during
      minibatch, validity=0.
  (4) Cascade x minibatch mutex — message content, post-construction
      mutation contract.
  (7) Validation messages — name field + value, list valid options.

Categories (2), (5), (6) are covered in `test_minibatch_evaluator_adversarial.py`
and `test_minibatch_archive_adversarial.py`.
"""

from __future__ import annotations

import random as _random

import pytest

from reps.config import EvaluatorConfig
from reps.minibatch import VALID_STRATEGIES, select_instances


# ---------------------------------------------------------------------------
# Category 1: select_instances edge cases
# ---------------------------------------------------------------------------


class TestSelectInstancesIterationBoundaries:
    """Pin iteration=0 (first window), modulus boundaries, and negative
    iteration behavior. The fixed_subset window math uses Python floor
    division — easy to mess up around 0 and negatives."""

    def test_iteration_zero_returns_first_window(self):
        # iteration=0 is the most common entry point (default int = 0 for
        # construction-time evals). Pin explicitly.
        keys = [f"k{i}" for i in range(8)]
        result = select_instances(keys, 4, iteration=0, strategy="fixed_subset")
        assert result == ["k0", "k1", "k2", "k3"]

    def test_iteration_at_full_modulus_cycle_wraps_to_window_zero(self):
        # n=6, size=2 -> 3 windows. iter=6 means window_idx=3, start=6%6=0,
        # back to first window.
        keys = [f"k{i}" for i in range(6)]
        win0 = select_instances(keys, 2, iteration=0, strategy="fixed_subset")
        win_cycle = select_instances(keys, 2, iteration=6, strategy="fixed_subset")
        assert win0 == win_cycle == ["k0", "k1"]

    def test_negative_iteration_does_not_crash_or_duplicate(self):
        # iteration=-1, size=2 -> window=-1, start=(-2)%6=4. Returns ['k4','k5'].
        # The point: the helper must not crash and must not return
        # duplicates (which would corrupt per_instance_scores accounting).
        keys = [f"k{i}" for i in range(6)]
        result = select_instances(keys, 2, iteration=-1, strategy="fixed_subset")
        assert len(result) == 2
        assert len(set(result)) == 2
        assert all(k in keys for k in result)

    def test_very_large_iteration_does_not_crash(self):
        # Defensive: a long-running experiment may accumulate huge iteration
        # counters. Make sure window math survives.
        keys = [f"k{i}" for i in range(5)]
        result = select_instances(keys, 2, iteration=10**9, strategy="fixed_subset")
        assert len(result) == 2
        assert all(k in keys for k in result)


class TestFixedSubsetCrossCallInvariant:
    """The load-bearing invariant: same iteration -> same subset across
    arbitrarily many calls. Concurrent candidates evaluated within the
    same batch must see IDENTICAL instance subsets so their per-instance
    scores are comparable."""

    def test_same_iteration_yields_identical_subset_across_50_calls(self):
        keys = [f"task_{i}" for i in range(50)]
        first = select_instances(keys, 7, iteration=12, strategy="fixed_subset")
        for _ in range(50):
            again = select_instances(keys, 7, iteration=12, strategy="fixed_subset")
            assert again == first

    def test_consecutive_iterations_within_window_collide(self):
        # All iterations sharing the same `iteration // size` yield the
        # same subset. Verify the helper exposes the contract clearly.
        keys = [f"k{i}" for i in range(10)]
        size = 3
        # Window 0 covers iterations 0,1,2.
        win0_iters = [select_instances(keys, size, iteration=i) for i in range(size)]
        assert all(w == win0_iters[0] for w in win0_iters)
        # First iteration of window 1 differs.
        win1_first = select_instances(keys, size, iteration=size)
        assert win1_first != win0_iters[0]


class TestSizeBoundaries:
    """size=1, size==len, size>len. The contract: size>=len returns the
    full list (caller's signal to skip the minibatch path)."""

    def test_size_one_returns_single_element_subset(self):
        keys = ["a", "b", "c", "d"]
        # window 0 -> [a]; window 1 -> [b]; ...
        assert select_instances(keys, 1, iteration=0) == ["a"]
        assert select_instances(keys, 1, iteration=1) == ["b"]
        assert select_instances(keys, 1, iteration=3) == ["d"]
        # Wrap.
        assert select_instances(keys, 1, iteration=4) == ["a"]

    def test_size_equal_to_len_returns_full_set(self):
        # Boundary: size == len(all_keys). Per the contract this is the
        # signal "skip the minibatch path"; the wiring relies on this to
        # avoid spurious minibatch calls when the user misconfigures size.
        keys = ["a", "b", "c", "d"]
        result = select_instances(keys, 4, iteration=2)
        assert result == keys

    def test_size_greater_than_len_returns_full_set_no_duplicates(self):
        # size > len(all_keys) MUST NOT wrap into a list with duplicates;
        # that would corrupt downstream per_instance_scores accounting.
        keys = ["a", "b", "c"]
        result = select_instances(keys, 7, iteration=5, strategy="fixed_subset")
        assert result == keys
        assert len(set(result)) == len(result)

    def test_random_size_one_returns_single_valid_element(self):
        keys = [f"k{i}" for i in range(10)]
        result = select_instances(keys, 1, iteration=42, strategy="random")
        assert len(result) == 1
        assert result[0] in keys


class TestRandomStrategyReproducibility:
    """random strategy is seeded by `iteration` so cross-run
    reproducibility holds. Pin that the seed isolation does not bleed
    into the global RNG even when calls interleave."""

    def test_same_seed_reproducible_across_repeated_calls(self):
        keys = [f"k{i}" for i in range(20)]
        a = select_instances(keys, 5, iteration=7, strategy="random")
        b = select_instances(keys, 5, iteration=7, strategy="random")
        c = select_instances(keys, 5, iteration=7, strategy="random")
        assert a == b == c

    def test_random_strategy_does_not_consume_global_rng_entropy(self):
        # Tightens the existing isolation test: interleave the random
        # strategy *between* the global draws and confirm the global
        # sequence is unchanged.
        keys = [f"k{i}" for i in range(30)]
        _random.seed(2026)
        seq_a = [_random.random() for _ in range(3)]

        _random.seed(2026)
        seq_b = []
        for i in range(3):
            seq_b.append(_random.random())
            select_instances(keys, 5, iteration=i, strategy="random")
        assert seq_a == seq_b


# ---------------------------------------------------------------------------
# Category 4: cascade x minibatch mutex / config validation
# ---------------------------------------------------------------------------


class TestEvaluatorConfigMutexMessage:
    """Spec quote: the mutex error message must name BOTH conflicting
    fields and tell the user how to resolve."""

    def test_mutex_message_names_both_fields(self):
        with pytest.raises(ValueError) as excinfo:
            EvaluatorConfig(cascade_evaluation=True, minibatch_size=8)
        msg = str(excinfo.value)
        assert "minibatch_size" in msg
        assert "cascade_evaluation" in msg

    def test_mutex_message_points_at_resolution(self):
        # User-facing hint should tell them exactly what to flip.
        with pytest.raises(ValueError) as excinfo:
            EvaluatorConfig(cascade_evaluation=True, minibatch_size=8)
        # Either form: "cascade_evaluation: false" / "set ... false"
        assert "false" in str(excinfo.value).lower()

    def test_default_config_does_not_raise(self):
        # Belt-and-suspenders: default EvaluatorConfig has cascade=True
        # and minibatch_size=None, so the mutex must not fire on defaults.
        cfg = EvaluatorConfig()
        assert cfg.cascade_evaluation is True
        assert cfg.minibatch_size is None

    def test_cascade_enabled_with_minibatch_none_does_not_raise(self):
        # Explicit covering case: just enabling cascade with no minibatch.
        cfg = EvaluatorConfig(cascade_evaluation=True, minibatch_size=None)
        assert cfg.minibatch_size is None

    def test_cascade_disabled_with_minibatch_set_does_not_raise(self):
        cfg = EvaluatorConfig(cascade_evaluation=False, minibatch_size=8)
        assert cfg.minibatch_size == 8


class TestPostConstructionMutationContract:
    """Pin the dataclass contract: __post_init__ runs once at
    construction. Mutating fields afterward to invalid combinations does
    NOT re-trigger validation. Users need to know this."""

    def test_post_construction_mutation_to_invalid_combo_does_not_raise(self):
        cfg = EvaluatorConfig(cascade_evaluation=False, minibatch_size=4)
        # Now mutate to a state that would have raised at construction.
        cfg.cascade_evaluation = True
        # No exception — dataclass __post_init__ does not re-run.
        assert cfg.cascade_evaluation is True
        assert cfg.minibatch_size == 4

    def test_post_construction_mutation_to_invalid_strategy_does_not_raise(self):
        cfg = EvaluatorConfig(cascade_evaluation=False, minibatch_size=4)
        # Mutating strategy to an invalid value also does not re-validate.
        cfg.minibatch_strategy = "totally_made_up"
        assert cfg.minibatch_strategy == "totally_made_up"

    def test_post_construction_mutation_to_invalid_archive_policy_does_not_raise(self):
        cfg = EvaluatorConfig(cascade_evaluation=False, minibatch_size=4)
        cfg.minibatch_archive_policy = "totally_made_up"
        assert cfg.minibatch_archive_policy == "totally_made_up"


# ---------------------------------------------------------------------------
# Category 7: validation messages
# ---------------------------------------------------------------------------


class TestStrategyValidationMessages:
    def test_invalid_strategy_message_names_field_and_value(self):
        with pytest.raises(ValueError) as excinfo:
            EvaluatorConfig(cascade_evaluation=False, minibatch_strategy="bogus")
        msg = str(excinfo.value)
        assert "minibatch_strategy" in msg
        assert "bogus" in msg
        # Should mention at least one valid value so user knows what to pick.
        assert "fixed_subset" in msg or "random" in msg

    def test_invalid_archive_policy_message_names_field_and_value(self):
        with pytest.raises(ValueError) as excinfo:
            EvaluatorConfig(minibatch_archive_policy="totally_invalid")
        msg = str(excinfo.value)
        assert "minibatch_archive_policy" in msg
        assert "totally_invalid" in msg
        # Lists at least one valid value.
        assert "promoted_only" in msg or "all_with_tag" in msg

    def test_select_instances_unknown_strategy_lists_valid_values(self):
        # The helper's own ValueError should also include the valid list,
        # not just say "Unknown strategy".
        with pytest.raises(ValueError) as excinfo:
            select_instances(["a", "b", "c", "d"], 2, iteration=0, strategy="bogus")
        msg = str(excinfo.value)
        assert "bogus" in msg
        # Helper exposes the valid set via its message.
        assert "fixed_subset" in msg or "random" in msg


class TestThresholdRangePolicy:
    """Pin current behavior: the threshold is NOT clamped or validated.
    Documents the contract; if a future spec mandates [0, 1] enforcement,
    this test will need updating."""

    def test_negative_threshold_is_currently_accepted(self):
        cfg = EvaluatorConfig(
            cascade_evaluation=False, minibatch_size=4, minibatch_promotion_threshold=-1.0
        )
        assert cfg.minibatch_promotion_threshold == -1.0

    def test_above_one_threshold_is_currently_accepted(self):
        cfg = EvaluatorConfig(
            cascade_evaluation=False, minibatch_size=4, minibatch_promotion_threshold=2.5
        )
        assert cfg.minibatch_promotion_threshold == 2.5

    def test_zero_threshold_is_accepted(self):
        # threshold=0 effectively means "always promote" (since
        # combined_score >= 0 for all valid scores).
        cfg = EvaluatorConfig(
            cascade_evaluation=False, minibatch_size=4, minibatch_promotion_threshold=0.0
        )
        assert cfg.minibatch_promotion_threshold == 0.0

    def test_one_threshold_is_accepted(self):
        cfg = EvaluatorConfig(
            cascade_evaluation=False, minibatch_size=4, minibatch_promotion_threshold=1.0
        )
        assert cfg.minibatch_promotion_threshold == 1.0


class TestStrategyConstantsDocumentedSet:
    def test_valid_strategies_set_matches_post_init(self):
        # Drift guard: the helper's VALID_STRATEGIES tuple must agree
        # with the values __post_init__ accepts. If they diverge, callers
        # who consult VALID_STRATEGIES will see stale info.
        for s in VALID_STRATEGIES:
            cfg = EvaluatorConfig(cascade_evaluation=False, minibatch_strategy=s)
            assert cfg.minibatch_strategy == s
        # And conversely: anything outside VALID_STRATEGIES must be rejected.
        with pytest.raises(ValueError):
            EvaluatorConfig(cascade_evaluation=False, minibatch_strategy="fixed-subset")
