# EVOLVE-BLOCK-START
"""Online bin packing seed: Best Fit heuristic.

Defines `priority(item, bins) -> np.ndarray`. For each item, the simulator
considers the bins that still fit it, scores each candidate bin via this
function, and places the item in the highest-scoring bin.

Best Fit picks the bin with the smallest leftover capacity after placement,
i.e. the tightest fit. This is the FunSearch seed (Romera-Paredes et al.,
Nature 2024) and the baseline we want to beat.
"""
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Score each candidate bin for placing `item`. Highest score wins.

    Args:
        item: Size of the next item to pack (scalar).
        bins: 1-D array of remaining capacities of bins that still fit `item`.

    Returns:
        1-D array of priorities, same shape as `bins`. The simulator places
        `item` in `bins[argmax(priority)]`.
    """
    return -(bins - item)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Quick sanity check on a handful of items.
    bins = np.array([100, 80, 50], dtype=float)
    for it in [30, 45, 70]:
        valid = bins[bins >= it]
        if len(valid):
            p = priority(it, valid)
            print(f"item={it} valid_bins={valid.tolist()} priorities={p.tolist()}")
