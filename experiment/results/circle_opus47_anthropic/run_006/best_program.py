# EVOLVE-BLOCK-START
"""Circle packing: 26 circles in [0,1]x[0,1] with variable radii, SLSQP optimization."""
import numpy as np
from scipy.optimize import minimize


def _pack_slsqp(centers0, radii0, maxiter=300, ftol=1e-10):
    n = len(radii0)
    x0 = np.empty(3 * n)
    x0[0::3] = centers0[:, 0]
    x0[1::3] = centers0[:, 1]
    x0[2::3] = radii0

    def neg_sum(x):
        return -np.sum(x[2::3])

    def neg_grad(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    # Boundary constraints as a single vector: returns 4n values >= 0
    def boundary(x):
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        return np.concatenate([xs - rs, 1 - xs - rs, ys - rs, 1 - ys - rs])

    def boundary_jac(x):
        J = np.zeros((4 * n, 3 * n))
        for i in range(n):
            # xs[i] - rs[i] >= 0
            J[i, 3 * i] = 1.0
            J[i, 3 * i + 2] = -1.0
            # 1 - xs[i] - rs[i] >= 0
            J[n + i, 3 * i] = -1.0
            J[n + i, 3 * i + 2] = -1.0
            # ys[i] - rs[i] >= 0
            J[2 * n + i, 3 * i + 1] = 1.0
            J[2 * n + i, 3 * i + 2] = -1.0
            # 1 - ys[i] - rs[i] >= 0
            J[3 * n + i, 3 * i + 1] = -1.0
            J[3 * n + i, 3 * i + 2] = -1.0
        return J

    # Pair non-overlap: d_ij - r_i - r_j >= 0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npairs = len(pairs)

    def nonoverlap(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        out = np.empty(npairs)
        for k, (i, j) in enumerate(pairs):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-15
            out[k] = d - rs[i] - rs[j]
        return out

    def nonoverlap_jac(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        J = np.zeros((npairs, 3 * n))
        for k, (i, j) in enumerate(pairs):
            dx = xs[i] - xs[j]; dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy) + 1e-15
            J[k, 3 * i] = dx / d
            J[k, 3 * i + 1] = dy / d
            J[k, 3 * j] = -dx / d
            J[k, 3 * j + 1] = -dy / d
            J[k, 3 * i + 2] = -1.0
            J[k, 3 * j + 2] = -1.0
        return J

    constraints = [
        {'type': 'ineq', 'fun': boundary, 'jac': boundary_jac},
        {'type': 'ineq', 'fun': nonoverlap, 'jac': nonoverlap_jac},
    ]

    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-4, 0.5))

    res = minimize(neg_sum, x0, jac=neg_grad, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})

    x = res.x
    centers = np.column_stack([x[0::3], x[1::3]])
    radii = x[2::3]
    return centers, radii


def _project_feasible(centers, radii, margin=1e-9):
    """Shrink radii and clamp centers so result is strictly feasible."""
    n = len(radii)
    radii = np.clip(radii, 1e-6, 0.5)
    centers = np.clip(centers, 0.0, 1.0)
    # Boundary: ensure ri <= min(xi, 1-xi, yi, 1-yi)
    for i in range(n):
        lim = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        if radii[i] > lim - margin:
            radii[i] = max(1e-6, lim - margin)
    # Pair: ensure dij >= ri+rj (shrink the larger one)
    for _ in range(3):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] + margin:
                    excess = (radii[i] + radii[j] + margin) - d
                    # shrink both proportionally
                    total = radii[i] + radii[j]
                    if total > 0:
                        radii[i] = max(1e-6, radii[i] - excess * radii[i] / total)
                        radii[j] = max(1e-6, radii[j] - excess * radii[j] / total)
    return centers, radii


def construct_packing():
    n = 26

    # Initialization: 6x5 grid, skip 4 corners of the extra row or use 5x5+1
    rows, cols = 6, 5
    r0 = 0.5 / max(rows, cols) * 0.9
    centers = []
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            cx = r0 + j * (1.0 - 2 * r0) / max(cols - 1, 1)
            cy = r0 + i * (1.0 - 2 * r0) / max(rows - 1, 1)
            centers.append([cx, cy])
    centers = np.array(centers[:n])
    radii = np.full(n, r0)

    centers, radii = _pack_slsqp(centers, radii, maxiter=500, ftol=1e-11)
    centers, radii = _project_feasible(centers, radii, margin=1e-9)

    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
