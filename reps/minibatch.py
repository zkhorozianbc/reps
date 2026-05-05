"""Minibatch instance selection for evaluator-side promotion (GEPA Phase 6).

When evaluator cost dominates wall time, most candidates are duds. This
module is the pure helper that picks which subset of benchmark instances
to evaluate first, before a candidate is allowed to spend the full-fidelity
budget. Wiring lives in `reps.evaluator.Evaluator.evaluate_with_promotion`.

Two strategies:

  - ``"fixed_subset"`` (default, deterministic): the same iteration window
    picks the same subset, so per-instance scores are comparable across
    candidates evaluated within the same batch. The window size equals
    ``size`` (each consecutive block of ``size`` iterations rotates to the
    next subset). With ``ceil(len(all_keys) / size)`` distinct windows the
    rotation eventually covers the whole instance set.

  - ``"random"``: a fresh random subset per call. Cheaper to reason about,
    but per-instance scores can't be compared across candidates because
    each one sees a different draw. RNG is seeded by ``iteration`` so the
    sequence is still reproducible across runs.

Instance registry source (Phase 6.2 wiring): the harness reads either an
``INSTANCES: list[str]`` module-level constant or a ``list_instances() ->
list[str]`` function from the benchmark evaluator module. Benchmarks that
expose neither cannot opt into minibatch promotion; the wiring raises a
clear ``ValueError`` if `EvaluatorConfig.minibatch_size` is set on such a
benchmark.
"""

from __future__ import annotations

import random
from typing import List


VALID_STRATEGIES = ("fixed_subset", "random")


def select_instances(
    all_keys: List[str],
    size: int,
    *,
    iteration: int,
    strategy: str = "fixed_subset",
) -> List[str]:
    """Pick ``size`` instance keys from ``all_keys``.

    Returns the empty list when ``all_keys`` is empty. Returns
    ``list(all_keys)`` unchanged when ``size <= 0`` or
    ``size >= len(all_keys)`` — callers interpret "subset == full set" as
    "skip the minibatch path".

    ``iteration`` is the deterministic seed for both strategies. For
    ``fixed_subset`` it indexes the rotation window; for ``random`` it
    seeds a per-call ``random.Random`` instance so the draw is reproducible
    without disturbing the global RNG state.
    """
    if not all_keys:
        return []
    if size <= 0 or size >= len(all_keys):
        return list(all_keys)
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown minibatch strategy {strategy!r}. "
            f"Valid: {VALID_STRATEGIES}."
        )

    n = len(all_keys)

    if strategy == "fixed_subset":
        # Rotate a contiguous window of length `size` so the same window
        # (same `iteration // size`) returns the same subset.
        window = iteration // size
        start = (window * size) % n
        if start + size <= n:
            return list(all_keys[start : start + size])
        # Wrap around the end of the list.
        head = all_keys[start:]
        tail = all_keys[: size - len(head)]
        return list(head) + list(tail)

    # strategy == "random": seeded by iteration so the run is reproducible.
    rng = random.Random(iteration)
    return rng.sample(list(all_keys), size)
