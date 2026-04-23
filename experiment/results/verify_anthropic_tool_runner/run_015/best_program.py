# EVOLVE-BLOCK-START
"""Circle packing: 5x5 grid plus one gap-filler."""
import numpy as np


def construct_packing():
    eps = 1e-9
    r_big = 0.1 - eps
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + j * 0.2, 0.1 + i * 0.2])
            radii.append(r_big)
    r_gap = (np.sqrt(2) - 1) * 0.1 - eps
    centers.append([0.2, 0.2])
    radii.append(r_gap)
    centers = np.array(centers)
    radii = np.array(radii)
    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
