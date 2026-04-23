# EVOLVE-BLOCK-START
"""Circle packing: 26 circles, 5x5 grid + 1 in a gap."""
import numpy as np


def construct_packing():
    n = 26
    rows, cols = 5, 5
    r = 0.5 / max(rows, cols) - 1e-9
    centers = []
    for i in range(rows):
        for j in range(cols):
            cx = r + j * (1.0 - 2 * r) / (cols - 1)
            cy = r + i * (1.0 - 2 * r) / (rows - 1)
            centers.append([cx, cy])
    r_extra = r * (np.sqrt(2) - 1) - 1e-9
    gx = 0.5 * (centers[0][0] + centers[1][0])
    gy = 0.5 * (centers[0][1] + centers[rows][1])
    centers.append([gx, gy])
    centers = np.array(centers[:n])
    radii = np.full(n, r)
    radii[-1] = r_extra
    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
