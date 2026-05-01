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
    select_complementary_pair,
    select_complementary_partner,
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


class TestSelectComplementaryPartner:
    def test_returns_none_for_empty_candidates(self):
        primary = _prog("p", {"x": 0.5})
        assert select_complementary_partner(primary, []) is None

    def test_returns_none_when_only_primary_in_candidates(self):
        primary = _prog("p", {"x": 0.5})
        # Same id excludes the candidate.
        assert select_complementary_partner(primary, [primary]) is None

    def test_picks_candidate_with_most_complementary_strength(self):
        # primary excels on x; among partners, c2 excels on y (perfectly
        # complementary) and c1 excels on x (overlapping).
        primary = _prog("p", {"x": 1.0, "y": 0.0})
        c1 = _prog("c1", {"x": 0.9, "y": 0.0})  # gain = 0 (no dim better)
        c2 = _prog("c2", {"x": 0.0, "y": 1.0})  # gain = 1.0 (y dim)
        c3 = _prog("c3", {"x": 0.0, "y": 0.5})  # gain = 0.5
        assert select_complementary_partner(primary, [c1, c2, c3]).id == "c2"

    def test_breaks_ties_uniformly(self):
        # primary dominates everyone → all gains = 0 → uniform random pick.
        primary = _prog("p", {"x": 1.0, "y": 1.0})
        c1 = _prog("c1", {"x": 0.0, "y": 0.0})
        c2 = _prog("c2", {"x": 0.5, "y": 0.5})
        seen = set()
        rng = random.Random(0)
        for _ in range(100):
            picked = select_complementary_partner(primary, [c1, c2], rng=rng)
            seen.add(picked.id)
        assert seen == {"c1", "c2"}

    def test_excludes_primary_by_id(self):
        primary = _prog("p", {"x": 1.0, "y": 0.0})
        clone = _prog("p", {"x": 0.0, "y": 1.0})  # same id
        c2 = _prog("c2", {"x": 0.5, "y": 0.5})
        rng = random.Random(0)
        # `clone` would otherwise win on complementarity, but same-id excludes it.
        for _ in range(20):
            assert select_complementary_partner(primary, [clone, c2], rng=rng).id == "c2"

    def test_excludes_primary_by_identity_when_no_id(self):
        primary = _prog("", {"x": 1.0, "y": 0.0})
        # Empty id falls back to identity check.
        rng = random.Random(0)
        partner = _prog("", {"x": 0.0, "y": 1.0})
        # primary is excluded; partner is the only candidate left.
        assert select_complementary_partner(primary, [primary, partner], rng=rng) is partner

    def test_explicit_instance_keys(self):
        # Restricting to {x} only — y dimension ignored.
        primary = _prog("p", {"x": 1.0, "y": 0.0})
        c_x = _prog("c_x", {"x": 0.0, "y": 0.0})  # gain on {x} only = 0
        c_y = _prog("c_y", {"x": 0.0, "y": 1.0})  # gain on {x} only = 0
        # On {x}: both candidates tie at 0, so uniform pick.
        rng = random.Random(0)
        seen = set()
        for _ in range(100):
            seen.add(select_complementary_partner(primary, [c_x, c_y], ["x"], rng=rng).id)
        assert seen == {"c_x", "c_y"}

    def test_works_without_per_instance_scores(self):
        # combined_score broadcast: complementary gain is essentially 0 unless
        # candidate has a higher combined_score on the broadcast dim.
        primary = _prog("p", combined=0.7)
        c_lo = _prog("c_lo", combined=0.3)  # gain = 0
        c_hi = _prog("c_hi", combined=0.9)  # gain > 0
        assert select_complementary_partner(primary, [c_lo, c_hi]).id == "c_hi"

    def test_finite_partner_compensates_for_broken_primary(self):
        # If primary's eval failed (NaN → -inf via _safe_score), a finite
        # candidate must still receive gain credit. The naive `isfinite(inf)`
        # check would reject that case and return 0 gain, leaving the
        # uniform-tie-break to pick at random — a regression vs intent.
        primary = _prog("p", {"x": float("nan"), "y": 0.0})
        c_finite = _prog("c_finite", {"x": 0.5, "y": 0.5})
        c_broken = _prog("c_broken", {"x": float("nan"), "y": float("nan")})
        # c_finite contributes 1.0 total gain (0.5 on each dim, primary clamped
        # to 0). c_broken contributes 0.
        rng = random.Random(0)
        for _ in range(20):
            assert select_complementary_partner(
                primary, [c_finite, c_broken], rng=rng
            ).id == "c_finite"


class TestSelectComplementaryPair:
    def test_returns_none_for_empty_or_single(self):
        assert select_complementary_pair([]) is None
        assert select_complementary_pair([_prog("a", {"x": 0.5})]) is None

    def test_picks_most_disjoint_pair(self):
        # a and b are complementary; c is dominated.
        a = _prog("a", {"x": 1.0, "y": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0})
        c = _prog("c", {"x": 0.5, "y": 0.5})
        result = select_complementary_pair([a, b, c])
        assert result is not None
        first, second = result
        # The {a, b} pair has gain=1.0 in either direction (max disjointness);
        # any pair involving c yields gain=0.5 max. So one of them is from {a, b}.
        # Specifically the maximizer is (a→b) or (b→a) since gain=1.0.
        assert {first.id, second.id} == {"a", "b"}

    def test_three_dim_disjoint(self):
        # Three perfectly orthogonal programs — any of the 6 ordered pairs
        # has gain=1.0. All should be reachable.
        a = _prog("a", {"x": 1.0, "y": 0.0, "z": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0, "z": 0.0})
        c = _prog("c", {"x": 0.0, "y": 0.0, "z": 1.0})
        rng = random.Random(0)
        seen_pairs = set()
        for _ in range(200):
            r = select_complementary_pair([a, b, c], rng=rng)
            seen_pairs.add((r[0].id, r[1].id))
        # All 6 ordered pairs should appear over many draws.
        assert len(seen_pairs) == 6

    def test_deterministic_with_seeded_rng(self):
        a = _prog("a", {"x": 1.0, "y": 0.0})
        b = _prog("b", {"x": 0.0, "y": 1.0})
        c = _prog("c", {"x": 0.5, "y": 0.5})
        s1 = [select_complementary_pair([a, b, c], rng=random.Random(7)) for _ in range(10)]
        s2 = [select_complementary_pair([a, b, c], rng=random.Random(7)) for _ in range(10)]
        assert [(p[0].id, p[1].id) for p in s1] == [(p[0].id, p[1].id) for p in s2]
