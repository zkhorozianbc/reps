"""Pareto-frontier utilities for REPS (GEPA-inspired Phase 2).

Pure functions on `Program` lists — no I/O, no global state. Designed to
sit alongside MAP-Elites/island sampling in `database.py`, not replace it.

A program A *dominates* B when, over a set of instance keys, A's score
is ≥ B's on every key and strictly > on at least one. The Pareto frontier
is the set of non-dominated programs.

Per-instance scores come from `Program.per_instance_scores` (populated in
Phase 1.2). Programs with `None` fall back to a one-dimensional vector
{'combined_score': metrics['combined_score']} so the module degrades
gracefully on benchmarks not yet migrated to GEPA-style ASI.

Missing keys for a given program default to 0.0 (most pessimistic) so a
program that excels on tasks A and B isn't artificially dominated by one
that only reports tasks A and B and C with C=0.5.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Tuple

from reps.database import Program


def _safe_score(value: object) -> float:
    """Coerce a metric value to a comparable float. NaN → -inf so failed
    evaluations are dominated by anything finite."""
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return float("-inf")
    return f


def program_score_vector(
    program: Program, instance_keys: List[str]
) -> List[float]:
    """Project a program onto the requested instance keys, in order.

    Uses `per_instance_scores` when present; otherwise falls back to
    `metrics['combined_score']` repeated for every key (so the program is
    a single point that compares only by combined_score across the vector).
    Missing per-instance keys default to 0.0.
    """
    if program.per_instance_scores:
        return [_safe_score(program.per_instance_scores.get(k, 0.0)) for k in instance_keys]
    fallback = _safe_score((program.metrics or {}).get("combined_score", 0.0))
    return [fallback for _ in instance_keys]


def dominates(a: List[float], b: List[float]) -> bool:
    """True iff vector a Pareto-dominates b (≥ on all dims, > on at least one).

    Both vectors must have the same length. Empty vectors → never dominates.
    """
    if len(a) != len(b) or not a:
        return False
    strictly_better = False
    for ai, bi in zip(a, b):
        if ai < bi:
            return False
        if ai > bi:
            strictly_better = True
    return strictly_better


def collect_instance_keys(programs: Iterable[Program]) -> List[str]:
    """Union of per_instance_scores keys across programs, sorted for
    deterministic ordering. Falls back to ['combined_score'] when no
    program exposes per-instance data — keeps the sampler usable on
    pre-Phase-1.2 benchmarks.
    """
    keys: set[str] = set()
    for p in programs:
        if p.per_instance_scores:
            keys.update(p.per_instance_scores.keys())
    if not keys:
        return ["combined_score"]
    return sorted(keys)


def compute_frontier(
    programs: List[Program],
    instance_keys: Optional[List[str]] = None,
) -> List[Program]:
    """Return the Pareto frontier — non-dominated programs.

    O(n^2) — fine for population sizes < a few hundred.
    Tied programs (identical score vectors) all remain on the frontier.
    """
    if not programs:
        return []
    keys = instance_keys if instance_keys is not None else collect_instance_keys(programs)
    vectors = [program_score_vector(p, keys) for p in programs]
    n = len(programs)
    frontier: List[Program] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(vectors[j], vectors[i]):
                dominated = True
                break
        if not dominated:
            frontier.append(programs[i])
    return frontier


def sample_pareto(
    programs: List[Program],
    instance_keys: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Program]:
    """Uniformly pick one program from the Pareto frontier.

    Returns None when the input is empty. When the frontier collapses to
    a single program (every other one is dominated), that program is
    returned deterministically.
    """
    frontier = compute_frontier(programs, instance_keys=instance_keys)
    if not frontier:
        return None
    chooser = rng if rng is not None else random
    return chooser.choice(frontier)


# ---------------------------------------------------------------------------
# System-aware merge (Phase 4): given a primary parent, pick a partner whose
# strengths most complement primary's weaknesses. Definition of "complementary"
# is the total per-dim *gain* over primary:
#
#   gain(primary, candidate) = sum_k max(0, c.scores[k] - p.scores[k])
#
# The candidate that maximizes this contributes the most non-overlapping
# strength. Ties (e.g. primary dominates every candidate so gain=0) are
# broken uniformly at random — the merge worker still gets *some* partner.
# ---------------------------------------------------------------------------


def _complementary_gain(primary_vec: List[float], candidate_vec: List[float]) -> float:
    """Sum of per-dim positive gain candidate has over primary.

    Non-finite scores (e.g. -inf produced by `_safe_score` from NaN evaluations)
    are clamped to 0.0 *for gain accounting only* — `dominates` still treats
    them as -inf so failed evals stay off the frontier. The clamp here means a
    finite candidate gets full credit over a broken-eval primary instead of
    being silently rejected by `isfinite(inf)==False`.
    """
    total = 0.0
    for p, c in zip(primary_vec, candidate_vec):
        p_clean = p if math.isfinite(p) else 0.0
        c_clean = c if math.isfinite(c) else 0.0
        diff = c_clean - p_clean
        if diff > 0.0:
            total += diff
    return total


def select_complementary_partner(
    primary: Program,
    candidates: List[Program],
    instance_keys: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Program]:
    """Pick the candidate that most complements `primary`'s strengths.

    Excludes `primary` itself from `candidates` (matched by id, falling back
    to identity for ad-hoc programs without ids). Returns None when no other
    candidate is available.

    When multiple candidates tie on gain (e.g. primary already dominates them
    all — gain=0 across the board), picks one uniformly at random so the
    merge worker still has a partner.
    """
    if not candidates:
        return None

    others = [
        c for c in candidates
        if (primary.id and c.id != primary.id) or (not primary.id and c is not primary)
    ]
    if not others:
        return None

    keys = instance_keys if instance_keys is not None else collect_instance_keys(
        [primary, *others]
    )
    primary_vec = program_score_vector(primary, keys)

    scored = [(_complementary_gain(primary_vec, program_score_vector(c, keys)), c)
              for c in others]

    best_gain = max(g for g, _ in scored)
    best_candidates = [c for g, c in scored if g == best_gain]

    chooser = rng if rng is not None else random
    return chooser.choice(best_candidates)


def select_complementary_pair(
    frontier: List[Program],
    instance_keys: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[Program, Program]]:
    """Pick two frontier members whose strengths are most disjoint.

    Searches all ordered pairs; the result is `(a, b)` maximizing
    `_complementary_gain(a, b)`. Returns None when the frontier has fewer
    than 2 programs. Ties broken uniformly at random.

    O(n²) over the frontier — fine for typical Pareto-front sizes (<50).
    """
    if not frontier or len(frontier) < 2:
        return None

    keys = instance_keys if instance_keys is not None else collect_instance_keys(frontier)
    vectors = {id(p): program_score_vector(p, keys) for p in frontier}

    best_gain = -1.0
    best_pairs: List[Tuple[Program, Program]] = []
    for a in frontier:
        for b in frontier:
            if a is b:
                continue
            gain = _complementary_gain(vectors[id(a)], vectors[id(b)])
            if gain > best_gain:
                best_gain = gain
                best_pairs = [(a, b)]
            elif gain == best_gain:
                best_pairs.append((a, b))

    if not best_pairs:
        return None

    chooser = rng if rng is not None else random
    return chooser.choice(best_pairs)
