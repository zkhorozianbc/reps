# EVOLVE-BLOCK-START
"""Circle packing: 5x5 grid of r=0.1 (25 circles) plus 1 interstitial circle.

A regular 5x5 grid with r=0.1 packs exactly into the unit square: the 25
circles touch each other and all four walls. The diagonal gap between any 4
adjacent grid circles admits an additional circle of radius r*(sqrt(2)-1),
yielding a 26th circle at position (0.2, 0.2) tangent to its 4 neighbors.

Sum of radii = 25 * 0.1 + 0.1 * (sqrt(2) - 1) = 2.5 + 0.04142... = 2.54142...
"""
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
    eps = 1e-12  # tiny numerical safety margin

    # 5x5 grid of main circles at r = 0.1, centers on a 0.2 lattice.
    r_main = 0.1 - eps
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + 0.2 * j, 0.1 + 0.2 * i])
            radii.append(r_main)

    # One interstitial circle at the center of 4 adjacent grid circles.
    # Distance from (0.2, 0.2) to (0.1, 0.1) is sqrt(2)*0.1, so max
    # interstitial radius r_int satisfies r_main + r_int <= sqrt(2)*0.1,
    # giving r_int = 0.1*(sqrt(2) - 1) ~ 0.04142.
    r_int = 0.1 * (np.sqrt(2.0) - 1.0) - eps
    centers.append([0.2, 0.2])
    radii.append(r_int)

    centers = np.array(centers, dtype=float)
    radii = np.array(radii, dtype=float)

    assert centers.shape == (n, 2)
    assert radii.shape == (n,)

    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
