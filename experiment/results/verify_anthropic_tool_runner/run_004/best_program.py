# EVOLVE-BLOCK-START
"""Circle packing: 25 circles in a 5x5 grid (r=0.1) plus 1 gap circle.

The 5x5 grid with radius 0.1 exactly tiles the unit square (each circle
touches 4 neighbors and 2 boundaries where applicable). The largest
extra circle we can squeeze in has radius sqrt(0.02) - 0.1 in the
central gap between any 4 grid circles. Total sum = 2.5 + ~0.04142.
"""
import numpy as np


def construct_packing():
    """Place 26 circles in the unit square, maximizing sum of radii.

    Returns:
        (centers, radii, sum_radii)
    """
    eps = 1e-9
    r_main = 0.1 - eps
    r_gap = (np.sqrt(0.02) - 0.1) - eps

    centers = np.empty((26, 2))
    radii = np.empty(26)
    k = 0
    for i in range(5):
        for j in range(5):
            centers[k] = (0.1 + 0.2 * j, 0.1 + 0.2 * i)
            radii[k] = r_main
            k += 1
    centers[25] = (0.2, 0.2)
    radii[25] = r_gap

    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
