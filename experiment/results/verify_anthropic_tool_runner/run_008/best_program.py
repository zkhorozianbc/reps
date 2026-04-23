# EVOLVE-BLOCK-START
"""Circle packing seed: 26 equal circles in a uniform grid inside [0,1]x[0,1]."""
import numpy as np


def construct_packing():
    """Place 26 circles in a unit square, maximizing sum of radii.

    Returns:
        (centers, radii, sum_radii) where
            centers: np.array shape (26, 2)
            radii:   np.array shape (26,)
            sum_radii: float
    """
    n = 26
    rows, cols = 6, 5  # 6 rows × 5 cols = 30 slots, use first 26
    r = 0.5 / max(rows, cols)  # radius that fits the grid
    r *= 0.95  # small margin to avoid boundary touching

    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            cx = r + j * (1.0 - 2 * r) / max(cols - 1, 1)
            cy = r + i * (1.0 - 2 * r) / max(rows - 1, 1)
            centers.append([cx, cy])
        if len(centers) >= n:
            break

    centers = np.array(centers[:n])
    radii = np.full(n, r)

    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
