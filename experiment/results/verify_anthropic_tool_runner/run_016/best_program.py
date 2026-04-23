# EVOLVE-BLOCK-START
"""Circle packing: 25 circles in 5x5 grid plus 1 small in corner."""
import numpy as np


def construct_packing():
    r = 0.1 * 0.999
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + j * 0.2, 0.1 + i * 0.2])
    r2 = 0.1 * (np.sqrt(2) - 1) / (np.sqrt(2) + 1) * 0.99
    centers.append([1 - r2, 1 - r2])
    radii = np.array([r] * 25 + [r2])
    centers = np.array(centers)
    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
