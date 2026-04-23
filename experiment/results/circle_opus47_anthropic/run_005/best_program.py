# EVOLVE-BLOCK-START
"""Circle packing: 26 circles in unit square maximizing sum of radii.

Structural pivot from the trivial uniform grid seed:
  (1) start from a hexagonal-ish layout,
  (2) run SLSQP over (x_i, y_i, r_i) with analytic-jacobian inequality
      constraints for non-overlap and boundary,
  (3) try several initial layouts and pick the best VALID one.
"""
import numpy as np
from scipy.optimize import minimize

N = 26


def _validate(centers, radii, tol=1e-8):
    for i in range(N):
        if radii[i] <= 0:
            return False
        if (centers[i, 0] - radii[i] < -tol or
                1 - centers[i, 0] - radii[i] < -tol or
                centers[i, 1] - radii[i] < -tol or
                1 - centers[i, 1] - radii[i] < -tol):
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i, 0] - centers[j, 0],
                         centers[i, 1] - centers[j, 1])
            if d - (radii[i] + radii[j]) < -tol:
                return False
    return True


def _shrink_to_valid(centers, radii, safety=1e-9):
    """Binary-search a global radii scale s so that s*radii is feasible
    given the already-feasible-ish centers. Cheap insurance against
    tiny numerical constraint violations from SLSQP."""
    c = centers.copy()
    r = radii.copy()
    # first cap by boundaries
    for i in range(N):
        bmax = min(c[i, 0], 1 - c[i, 0], c[i, 1], 1 - c[i, 1])
        if r[i] > bmax - safety:
            r[i] = max(bmax - safety, 1e-6)
    # then enforce pairwise by uniform shrink factor
    worst = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            s = (d - safety) / max(r[i] + r[j], 1e-12)
            if s < worst:
                worst = s
    if worst < 1.0:
        r = r * worst
    return c, r


def _optimize(centers0, r0_arr, maxiter=250):
    x0 = np.concatenate([centers0.flatten(), r0_arr])

    def neg_sum(v):
        return -float(np.sum(v[2 * N:]))

    def neg_sum_grad(v):
        g = np.zeros_like(v)
        g[2 * N:] = -1.0
        return g

    cons = []
    for i in range(N):
        def fb_xlo(v, i=i): return v[2 * i] - v[2 * N + i]
        def gb_xlo(v, i=i):
            g = np.zeros_like(v); g[2 * i] = 1.0; g[2 * N + i] = -1.0; return g
        def fb_xhi(v, i=i): return 1.0 - v[2 * i] - v[2 * N + i]
        def gb_xhi(v, i=i):
            g = np.zeros_like(v); g[2 * i] = -1.0; g[2 * N + i] = -1.0; return g
        def fb_ylo(v, i=i): return v[2 * i + 1] - v[2 * N + i]
        def gb_ylo(v, i=i):
            g = np.zeros_like(v); g[2 * i + 1] = 1.0; g[2 * N + i] = -1.0; return g
        def fb_yhi(v, i=i): return 1.0 - v[2 * i + 1] - v[2 * N + i]
        def gb_yhi(v, i=i):
            g = np.zeros_like(v); g[2 * i + 1] = -1.0; g[2 * N + i] = -1.0; return g
        cons.append({'type': 'ineq', 'fun': fb_xlo, 'jac': gb_xlo})
        cons.append({'type': 'ineq', 'fun': fb_xhi, 'jac': gb_xhi})
        cons.append({'type': 'ineq', 'fun': fb_ylo, 'jac': gb_ylo})
        cons.append({'type': 'ineq', 'fun': fb_yhi, 'jac': gb_yhi})

    for i in range(N):
        for j in range(i + 1, N):
            def fover(v, i=i, j=j):
                dx = v[2 * i] - v[2 * j]
                dy = v[2 * i + 1] - v[2 * j + 1]
                return float(np.sqrt(dx * dx + dy * dy) - (v[2 * N + i] + v[2 * N + j]))

            def gover(v, i=i, j=j):
                dx = v[2 * i] - v[2 * j]
                dy = v[2 * i + 1] - v[2 * j + 1]
                d = np.sqrt(dx * dx + dy * dy) + 1e-12
                g = np.zeros_like(v)
                g[2 * i] = dx / d
                g[2 * i + 1] = dy / d
                g[2 * j] = -dx / d
                g[2 * j + 1] = -dy / d
                g[2 * N + i] = -1.0
                g[2 * N + j] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': fover, 'jac': gover})

    bnds = [(0.0, 1.0)] * (2 * N) + [(1e-4, 0.5)] * N

    try:
        res = minimize(neg_sum, x0, jac=neg_sum_grad, method='SLSQP',
                       constraints=cons, bounds=bnds,
                       options={'maxiter': maxiter, 'ftol': 1e-11})
        x = res.x
    except Exception:
        x = x0

    centers = x[:2 * N].reshape(N, 2)
    radii = x[2 * N:]
    centers, radii = _shrink_to_valid(centers, radii)
    return centers, radii, float(np.sum(radii))


def _layout_A():
    # rows of 6,5,6,5,4
    rows = [(6, 0.1), (5, 0.3), (6, 0.5), (5, 0.7), (4, 0.9)]
    centers = []
    for count, y in rows:
        xs = np.linspace(0.09, 0.91, count) if count > 1 else [0.5]
        for x in xs:
            centers.append([x, y])
    return np.array(centers[:N])


def _layout_B():
    # rows of 5,6,5,6,4
    rows = [(5, 0.1), (6, 0.3), (5, 0.5), (6, 0.7), (4, 0.9)]
    centers = []
    for count, y in rows:
        xs = np.linspace(0.09, 0.91, count) if count > 1 else [0.5]
        for x in xs:
            centers.append([x, y])
    return np.array(centers[:N])


def _layout_C():
    # 5x5 grid + 1 extra in middle
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + 0.2 * j, 0.1 + 0.2 * i])
    centers.append([0.5, 0.5])  # wrong - duplicates; use edge
    centers[-1] = [0.5, 0.95]
    return np.array(centers[:N])


def _layout_D(seed):
    # random perturbation of layout A
    rng = np.random.default_rng(seed)
    c = _layout_A() + rng.normal(0, 0.02, (N, 2))
    return np.clip(c, 0.05, 0.95)


def construct_packing():
    layouts = [_layout_A(), _layout_B(), _layout_C(),
               _layout_D(1), _layout_D(2)]
    best = None
    best_sum = -1.0
    for c0 in layouts:
        r0 = np.full(N, 0.08)
        centers, radii, s = _optimize(c0, r0, maxiter=250)
        if _validate(centers, radii) and s > best_sum:
            best_sum = s
            best = (centers, radii)
    if best is None:
        # fallback: uniform grid from seed
        r = 0.5 / 6 * 0.95
        centers = []
        for i in range(6):
            for j in range(5):
                if len(centers) >= N:
                    break
                cx = r + j * (1.0 - 2 * r) / 4
                cy = r + i * (1.0 - 2 * r) / 5
                centers.append([cx, cy])
        centers = np.array(centers[:N])
        radii = np.full(N, r)
        return centers, radii, float(np.sum(radii))
    centers, radii = best
    return centers, radii, float(np.sum(radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
