# EVOLVE-BLOCK-START
"""Circle packing: 26 circles using hexagonal close-packing inside [0,1]x[0,1]."""
import numpy as np


def construct_packing():
    n = 26
    cols = 5
    n_rows = 6
    r_place = min(1.0 / (2 * cols), 1.0 / (2 + (n_rows - 1) * np.sqrt(3)))
    r = r_place * 0.9999

    centers = []
    for row in range(n_rows):
        if row % 2 == 1:
            row_cols = cols - 1
            x_start = 2 * r_place
        else:
            row_cols = cols
            x_start = r_place
        for col in range(row_cols):
            if len(centers) >= n:
                break
            cx = x_start + col * 2 * r_place
            cy = r_place + row * r_place * np.sqrt(3)
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
