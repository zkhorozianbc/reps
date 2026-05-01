"""Unit tests for reps/pareto.py (Phase 2.1).

Pure functions on Program objects — no I/O, no LLM calls, no databases.
"""
import math
import random

import pytest

from reps.database import Program
from reps.pareto import (
    collect_instance_keys,
    compute_frontier,
    dominates,
    program_score_vector,
    sample_pareto,
)


def _prog(pid, scores=None, combined=None):
    """Tiny Program builder for tests."""
    return Program(
        id=pid,
        code="pass",
        metrics={"combined_score": combined} if combined is not None else {},
        per_instance_scores=scores,
    )


class TestDominates:
    def test_strict_dominance(self):
        assert dominates([1.0, 1.0], [0.5, 0.5]) is True

    def test_equal_vectors_no_dominance(self):
        assert dominates([0.5, 0.5], [0.5, 0.5]) is False

    def test_one_better_one_worse(self):
        assert dominates([1.0, 0.0], [0.0, 1.0]) is False
        assert dominates([0.0, 1.0], [1.0, 0.0]) is False

    def test_one_better_others_equal(self):
        assert dominates([1.0, 0.5], [0.5, 0.5]) is True

    def test_empty_never_dominates(self):
        assert dominates([], []) is False

    def test_length_mismatch_never_dominates(self):
        assert dominates([1.0], [1.0, 0.0]) is False


class TestProgramScoreVector:
    def test_uses_per_instance_scores(self):
        p = _prog("a", {"task_a": 0.7, "task_b": 0.3})
        assert program_score_vector(p, ["task_a", "task_b"]) == [0.7, 0.3]

    def test_missing_key_defaults_to_zero(self):
        p = _prog("a", {"task_a": 0.7})
        assert program_score_vector(p, ["task_a", "task_b"]) == [0.7, 0.0]

    def test_falls_back_to_combined_score(self):
        p = _prog("a", scores=None, combined=0.42)
        # Without per_instance_scores, every requested dim gets the same value.
        assert program_score_vector(p, ["x", "y"]) == [0.42, 0.42]

    def test_falls_back_to_zero_when_no_data(self):
        p = _prog("a")
        assert program_score_vector(p, ["x"]) == [0.0]

    def test_nan_treated_as_neg_inf(self):
        p = _prog("a", {"task_a": float("nan")})
        v = program_score_vector(p, ["task_a"])
        assert v[0] == float("-inf")


class TestCollectInstanceKeys:
    def test_union_across_programs(self):
        programs = [
            _prog("a", {"x": 1.0, "y": 0.5}),
            _prog("b", {"y": 1.0, "z": 0.5}),
        ]
        assert collect_instance_keys(programs) == ["x", "y", "z"]

    def test_falls_back_to_combined_score_when_none(self):
        programs = [_prog("a", combined=0.5), _prog("b", combined=0.7)]
        assert collect_instance_keys(programs) == ["combined_score"]

    def test_deterministic_sorted_order(self):
        programs = [_prog("a", {"z": 1.0, "a": 0.0, "m": 0.5})]
        # Same input twice → same output
        k1 = collect_instance_keys(programs)
        k2 = collect_instance_keys(programs)
        assert k1 == k2 == ["a", "m", "z"]


class TestComputeFrontier:
    def test_empty_input(self):
        assert compute_frontier([]) == []

    def test_single_program_is_frontier(self):
        p = _prog("a", {"x": 0.5})
        assert compute_frontier([p]) == [p]

    def test_strict_dominance_collapses_to_winner(self):
        # A dominates B on every dim — only A on frontier.
        a = _prog("a", {"x": 1.0, "y": 1.0})
        b = _prog("b", {"x": 0.5, "y": 0.5})
        assert compute_frontier([a, b]) == [a]

    def test_complementary_strengths_both_on_frontier(self):
        # Each excels on a different dim — both stay.
        a = _prog("a", {"x": 1.0, "y": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0})
        front = compute_frontier([a, b])
        assert set(p.id for p in front) == {"a", "b"}

    def test_three_dim_pareto(self):
        # Three programs each best on one dim.
        a = _prog("a", {"x": 1.0, "y": 0.0, "z": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0, "z": 0.0})
        c = _prog("c", {"x": 0.0, "y": 0.0, "z": 1.0})
        # Plus a strictly-dominated mediocre program.
        d = _prog("d", {"x": 0.1, "y": 0.1, "z": 0.1})
        # Note: d is NOT dominated by a/b/c alone (each beats it on one dim
        # but loses on others). It IS on the frontier.
        front = compute_frontier([a, b, c, d])
        assert set(p.id for p in front) == {"a", "b", "c", "d"}

    def test_explicit_dominator_removes_inferior(self):
        a = _prog("a", {"x": 1.0, "y": 1.0, "z": 1.0})
        d = _prog("d", {"x": 0.5, "y": 0.5, "z": 0.5})
        front = compute_frontier([a, d])
        assert front == [a]

    def test_ties_all_kept(self):
        a = _prog("a", {"x": 0.5, "y": 0.5})
        b = _prog("b", {"x": 0.5, "y": 0.5})
        front = compute_frontier([a, b])
        assert set(p.id for p in front) == {"a", "b"}

    def test_explicit_instance_keys_subset(self):
        # Restricting to a subset of keys can change who is on the frontier.
        a = _prog("a", {"x": 1.0, "y": 0.0})
        b = _prog("b", {"x": 0.5, "y": 1.0})
        # On {x, y}: complementary, both on frontier.
        assert set(p.id for p in compute_frontier([a, b], ["x", "y"])) == {"a", "b"}
        # On {x} only: a dominates b.
        assert compute_frontier([a, b], ["x"]) == [a]

    def test_falls_back_to_combined_score_when_no_per_instance(self):
        # Without per_instance_scores, the frontier is whoever has the
        # highest combined_score; ties stay together.
        a = _prog("a", combined=0.7)
        b = _prog("b", combined=0.5)
        c = _prog("c", combined=0.7)
        front = compute_frontier([a, b, c])
        assert set(p.id for p in front) == {"a", "c"}

    def test_mixed_per_instance_and_scalar(self):
        # Programs with per_instance_scores compete on the union of keys;
        # programs without it get their combined_score broadcast across
        # all keys.
        a = _prog("a", {"x": 0.6, "y": 0.6})
        b = _prog("b", combined=0.5)  # broadcasts to [0.5, 0.5]
        front = compute_frontier([a, b])
        # a dominates b across both dims.
        assert front == [a]

    def test_nan_program_dominated_by_anything_finite(self):
        # NaN scores should map to -inf via _safe_score, so a NaN program
        # is dominated by anything with finite scores. This matters because
        # failed evaluations (combined_score=NaN) shouldn't pollute the
        # frontier.
        good = _prog("good", {"x": 0.1, "y": 0.1})
        nan_prog = _prog("nan", {"x": float("nan"), "y": float("nan")})
        front = compute_frontier([good, nan_prog])
        assert front == [good]

    def test_nan_program_alone_still_returned(self):
        # If everyone has NaN, no one dominates anyone (-inf vs -inf is a
        # tie); all stay on the frontier so the sampler can't return None.
        a = _prog("a", {"x": float("nan")})
        b = _prog("b", {"x": float("nan")})
        front = compute_frontier([a, b])
        assert set(p.id for p in front) == {"a", "b"}


class TestSamplePareto:
    def test_empty_returns_none(self):
        assert sample_pareto([]) is None

    def test_single_program_returned(self):
        p = _prog("a", {"x": 0.5})
        assert sample_pareto([p]) is p

    def test_only_picks_from_frontier(self):
        # b is strictly dominated and must never be returned.
        a = _prog("a", {"x": 1.0, "y": 1.0})
        b = _prog("b", {"x": 0.0, "y": 0.0})
        rng = random.Random(0)
        for _ in range(50):
            assert sample_pareto([a, b], rng=rng) is a

    def test_uniform_over_complementary_frontier(self):
        # Two complementary frontier members → both should be sampled.
        a = _prog("a", {"x": 1.0, "y": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0})
        rng = random.Random(0)
        seen = set()
        for _ in range(50):
            seen.add(sample_pareto([a, b], rng=rng).id)
        assert seen == {"a", "b"}

    def test_deterministic_with_seeded_rng(self):
        a = _prog("a", {"x": 1.0, "y": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0})
        c = _prog("c", {"x": 0.5, "y": 0.5})  # dominated by neither, on frontier
        s1 = [sample_pareto([a, b, c], rng=random.Random(42)).id for _ in range(10)]
        s2 = [sample_pareto([a, b, c], rng=random.Random(42)).id for _ in range(10)]
        assert s1 == s2
