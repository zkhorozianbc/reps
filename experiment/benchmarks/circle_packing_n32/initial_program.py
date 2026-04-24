# EVOLVE-BLOCK-START
"""Circle packing seed: 32 equal circles in a 6x6 grid (first 32 slots) inside [0,1]x[0,1]."""
import numpy as np


def construct_packing():
    """Place 32 circles in a unit square, maximizing sum of radii.

    Uses a 6x6 grid (36 slots) and takes the first 32 — denser than
    the minimal 8x4 rectangular tiling for this n.

    Returns:
        (centers, radii, sum_radii) where
            centers: np.array shape (32, 2)
            radii:   np.array shape (32,)
            sum_radii: float
    """
    n = 32
    rows, cols = 6, 6  # 6 rows x 6 cols = 36 slots, use first 32
    r = 0.5 / max(rows, cols) * 0.95  # safety margin to avoid boundary touching

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
