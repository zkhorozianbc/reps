# EVOLVE-BLOCK-START
"""Circle packing: 26 equal circles in a hexagonal (staggered) arrangement."""
import numpy as np


def construct_packing():
    """Place 26 circles in a unit square, maximizing sum of radii.

    Uses a hexagonal close-packing layout: 6 rows with 5,4,5,4,5,4 circles
    (27 slots), using the first 26. For 6 rows the binding constraint is
    vertical: 2r + 5*r*sqrt(3) <= 1, giving r ~ 0.0938, significantly larger
    than the square-grid r = 1/12 ~ 0.0833.

    Returns:
        (centers, radii, sum_radii) where
            centers: np.array shape (26, 2)
            radii:   np.array shape (26,)
            sum_radii: float
    """
    n = 26
    row_counts = [5, 4, 5, 4, 5, 4]  # 27 hex-lattice slots

    sqrt3 = np.sqrt(3.0)
    # Reference spacing radius: max equal radius for 6-row hex in unit square.
    # Vertical: 2R + 5*R*sqrt(3) <= 1  => R <= 1/(2 + 5*sqrt(3)) ~ 0.09380
    # Horizontal (row of 5): 10R <= 1  => R <= 0.1
    R = min(1.0 / (2.0 + 5.0 * sqrt3), 0.1)
    # Actual circle radius slightly smaller than spacing radius for safety.
    r = R * 0.995

    centers = []
    # Vertical centering of the hex pattern inside [0,1]
    y_extra = (1.0 - (2.0 * R + 5.0 * R * sqrt3)) * 0.5
    y = R + y_extra
    for nc in row_counts:
        x0 = R if nc == 5 else 2.0 * R  # hex offset for staggered rows
        # Horizontal centering per row
        x_extra = (1.0 - ((2 * nc - 2) * R + 2 * x0)) * 0.5
        for j in range(nc):
            if len(centers) >= n:
                break
            centers.append([x0 + x_extra + j * 2.0 * R, y])
        if len(centers) >= n:
            break
        y += R * sqrt3

    centers = np.array(centers[:n])
    radii = np.full(n, r)

    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
