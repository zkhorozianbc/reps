# EVOLVE-BLOCK-START
"""Circle packing: 5x5+1 + SLSQP with diverse multistarts including edge/corner gap seeds."""
import numpy as np
from scipy.optimize import minimize


def construct_packing():
    eps = 1e-9
    n = 26

    base_grid = [[0.1 + j * 0.2, 0.1 + i * 0.2] for i in range(5) for j in range(5)]
    init_centers = np.array(base_grid + [[0.2, 0.2]])
    init_radii = np.array([0.1] * 25 + [np.sqrt(0.02) - 0.1])
    x0_base = np.column_stack([init_centers, init_radii]).ravel()

    def neg_sum(p):
        return -np.sum(p.reshape(n, 3)[:, 2])

    def neg_sum_grad(p):
        g = np.zeros((n, 3)); g[:, 2] = -1.0
        return g.ravel()

    constraints = []
    for i in range(n):
        constraints.append({"type": "ineq",
            "fun": (lambda p, i=i: p[3*i] - p[3*i+2]),
            "jac": (lambda p, i=i: np.eye(1, 3*n, 3*i).ravel() - np.eye(1, 3*n, 3*i+2).ravel())})
        constraints.append({"type": "ineq",
            "fun": (lambda p, i=i: 1 - p[3*i] - p[3*i+2]),
            "jac": (lambda p, i=i: -np.eye(1, 3*n, 3*i).ravel() - np.eye(1, 3*n, 3*i+2).ravel())})
        constraints.append({"type": "ineq",
            "fun": (lambda p, i=i: p[3*i+1] - p[3*i+2]),
            "jac": (lambda p, i=i: np.eye(1, 3*n, 3*i+1).ravel() - np.eye(1, 3*n, 3*i+2).ravel())})
        constraints.append({"type": "ineq",
            "fun": (lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]),
            "jac": (lambda p, i=i: -np.eye(1, 3*n, 3*i+1).ravel() - np.eye(1, 3*n, 3*i+2).ravel())})
        constraints.append({"type": "ineq",
            "fun": (lambda p, i=i: p[3*i+2]),
            "jac": (lambda p, i=i: np.eye(1, 3*n, 3*i+2).ravel())})

    for i in range(n):
        for j in range(i+1, n):
            def f(p, i=i, j=j):
                dx = p[3*i] - p[3*j]; dy = p[3*i+1] - p[3*j+1]
                return np.sqrt(dx*dx + dy*dy + 1e-30) - p[3*i+2] - p[3*j+2]
            def fj(p, i=i, j=j):
                dx = p[3*i] - p[3*j]; dy = p[3*i+1] - p[3*j+1]
                d = np.sqrt(dx*dx + dy*dy + 1e-30)
                g = np.zeros(3*n)
                g[3*i] = dx/d; g[3*j] = -dx/d
                g[3*i+1] = dy/d; g[3*j+1] = -dy/d
                g[3*i+2] = -1.0; g[3*j+2] = -1.0
                return g
            constraints.append({"type": "ineq", "fun": f, "jac": fj})

    def run_slsqp(x0, maxiter=300):
        try:
            return minimize(neg_sum, x0, jac=neg_sum_grad, method="SLSQP",
                            constraints=constraints,
                            options={"maxiter": maxiter, "ftol": 1e-11})
        except Exception:
            return None

    best_x = None
    best_sum = float(np.sum(init_radii))

    res = run_slsqp(x0_base, 500)
    if res is not None and -res.fun > best_sum:
        best_sum = -res.fun; best_x = res.x

    # Diverse seeds: gap circle in different positions
    alt_gap_positions = [[0.8, 0.8], [0.2, 0.8], [0.8, 0.2], [0.5, 0.5], [0.4, 0.4], [0.6, 0.6]]
    for gp in alt_gap_positions:
        alt = np.array(base_grid + [gp])
        x0 = np.column_stack([alt, init_radii]).ravel()
        res = run_slsqp(x0, 300)
        if res is not None and -res.fun > best_sum:
            best_sum = -res.fun; best_x = res.x

    rng = np.random.default_rng(123)
    for k in range(4):
        pert = x0_base.copy().reshape(n, 3)
        pert[:, 0] += rng.normal(0, 0.015, n)
        pert[:, 1] += rng.normal(0, 0.015, n)
        pert[:, 2] = np.maximum(pert[:, 2] + rng.normal(0, 0.008, n), 0.01)
        res = run_slsqp(pert.ravel(), 300)
        if res is not None and -res.fun > best_sum:
            best_sum = -res.fun; best_x = res.x

    # Chain-restart polish: refine best with tight perturbations
    if best_x is not None:
        res = run_slsqp(best_x.copy(), 500)
        if res is not None and -res.fun > best_sum:
            best_sum = -res.fun; best_x = res.x
        for k in range(5):
            tight = best_x.copy().reshape(n, 3)
            tight[:, 0] += rng.normal(0, 0.003, n)
            tight[:, 1] += rng.normal(0, 0.003, n)
            res = run_slsqp(tight.ravel(), 300)
            if res is not None and -res.fun > best_sum:
                best_sum = -res.fun; best_x = res.x

    if best_x is not None:
        p = best_x.reshape(n, 3)
        centers = p[:, :2].copy()
        radii = np.maximum(p[:, 2], 0.0) * (1 - eps)
    else:
        centers = init_centers.copy()
        radii = init_radii * (1 - eps)

    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
