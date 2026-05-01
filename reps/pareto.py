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
from typing import Iterable, List, Optional

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
