# EVOLVE-BLOCK-START
"""Circle packing: 26 equal circles in hexagonal arrangement (rows 5,4,5,4,5,3)."""
import numpy as np


def construct_packing():
    n = 26
    sqrt3 = np.sqrt(3.0)
    # Vertical constraint: 2R + 5*R*sqrt3 = 1
    R = 1.0 / (2.0 + 5.0 * sqrt3)
    # Use slightly smaller radius than R so that distances between centers (== 2R)
    # exceed 2r, creating a tiny non-overlap gap.
    r = R * 0.9999

    row_counts = [5, 4, 5, 4, 5, 3]
    centers = []
    for i, cnt in enumerate(row_counts):
        y = R + i * R * sqrt3
        # Even rows: x = R, 3R, 5R, 7R, 9R
        # Odd rows (shifted): x = 2R, 4R, 6R, 8R
        x0 = R if (i % 2 == 0) else 2.0 * R
        for j in range(cnt):
            x = x0 + 2.0 * R * j
            centers.append([x, y])

    centers = np.array(centers[:n])
    radii = np.full(n, r)
    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
