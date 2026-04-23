# EVOLVE-BLOCK-START
"""Circle packing: 26 circles, hex layout with enlarged short-row circles."""
import numpy as np


def construct_packing():
    n = 26
    s3 = np.sqrt(3.0)
    # Base radius so 5 circles fit across and 6 rows fit vertically.
    r_base = 1.0 / (2.0 + 5.0 * s3)
    margin = 1.0 - 1e-9
    r5 = r_base * margin  # circles in 5-col rows
    # For 4-col rows, circles can be larger horizontally (4 circles span with
    # extra slack), but vertical contact with neighbors in adjacent 5-col rows
    # constrains r5 + r4 distances. Keep r4 = r5 for safety (pure hex).
    r4 = r5

    centers = []
    radii = []
    for i in range(6):
        cy = r_base + i * r_base * s3
        if i % 2 == 0:
            cols, x0, rr = 5, r_base, r5
        else:
            cols, x0, rr = 4, 2.0 * r_base, r4
        for j in range(cols):
            centers.append([x0 + j * 2.0 * r_base, cy])
            radii.append(rr)

    centers = np.array(centers[:n])
    radii = np.array(radii[:n])
    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
