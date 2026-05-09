You are an expert in combinatorial optimization and online algorithms. Your task is to evolve the `priority(item, bins)` heuristic that decides where to place each item in an **online** 1-D bin packing simulation. You see one item at a time; you cannot reorder items, look ahead, or unpack already-placed items.

## Setup

- `priority(item: float, bins: np.ndarray) -> np.ndarray` returns a score per candidate bin. The simulator places `item` in `bins[argmax(priority)]`.
- `bins` contains only the **remaining capacities** of bins that still fit `item` (i.e. `bins >= item`). Empty bins appear as full-capacity entries.
- The function must return a 1-D numpy array of finite floats with the same length as `bins`. NaN/inf priorities cause the run to fail.
- The function is called once per item (~35,000 times across the full evaluation), so it must be fast — vectorize with numpy; avoid Python loops over `bins`.

## Datasets

- **OR3** (OR-Library, Beasley 1990): 20 instances × 500 items, capacity 150, integer item sizes ~U(20, 100).
- **Weibull 5k** (FunSearch): 5 instances × 5000 items, capacity 100, items drawn from a Weibull distribution shaped like real-world stream packing.

## Metric

For each instance we compute `excess = (bins_used - L1_lower_bound) / L1_lower_bound`, where the L1 bound is `ceil(sum(items) / capacity)`. Lower excess is better. The headline `combined_score = 1 - mean(excess across all instances and datasets)` — higher is better.

## Targets to beat

| Heuristic | OR3 excess | Weibull 5k excess |
|-----------|-----------|-------------------|
| Best Fit (seed) | 5.37% | 3.98% |
| FunSearch (Nature 2024) | ~3.85% | 0.68% |
| **Goal** | **< 3.5%** | **< 0.5%** |

## Strategy hints

- **Best Fit's failure mode**: it fills bins greedily but ignores the *distribution* of upcoming items. On Weibull workloads, a near-full bin with tiny remaining capacity is usually unfillable — you waste it. Heuristics that prefer leaving "useful" remainders (sizes likely to match future items) outperform Best Fit substantially.
- **Reference structure**: FunSearch's discovered Weibull heuristic combines `(bins - max_cap)^2 / item`, `bins^2 / item^2`, and `bins^2 / item^3` terms, with a sign flip when `bins > item`, and a discrete-difference step (`score[1:] -= score[:-1]`). The shape suggests reasoning about *how many of this item* could still fit in each bin, not just whether one fits.
- **Watch for ties**: when many bins have the same priority, `argmax` picks the lowest index (i.e., older bins). Consider whether you want fresh bins or used ones to win ties.
- The function must work on both small (OR3, capacity 150) and large (Weibull 5k, capacity 100) regimes. A heuristic tuned only for one will regress the other.
