# EVOLVE-BLOCK-START
"""Circle packing: 26 circles in unit square, maximize sum of radii.

Uses SLSQP nonlinear optimization with multi-start from several structured
initial configurations (row-based hex-like patterns with different splits).
Variables per circle: (x, y, r).
"""
import numpy as np
from scipy.optimize import minimize


def _jac_xm(i, n, L):
    g = np.zeros(L); g[2 * i] = 1.0; g[2 * n + i] = -1.0; return g
def _jac_xp(i, n, L):
    g = np.zeros(L); g[2 * i] = -1.0; g[2 * n + i] = -1.0; return g
def _jac_ym(i, n, L):
    g = np.zeros(L); g[2 * i + 1] = 1.0; g[2 * n + i] = -1.0; return g
def _jac_yp(i, n, L):
    g = np.zeros(L); g[2 * i + 1] = -1.0; g[2 * n + i] = -1.0; return g

def _pair(v, i, j, n):
    dx = v[2 * i] - v[2 * j]
    dy = v[2 * i + 1] - v[2 * j + 1]
    rs = v[2 * n + i] + v[2 * n + j]
    return dx * dx + dy * dy - rs * rs

def _pair_jac(v, i, j, n):
    L = len(v)
    g = np.zeros(L)
    dx = v[2 * i] - v[2 * j]
    dy = v[2 * i + 1] - v[2 * j + 1]
    rs = v[2 * n + i] + v[2 * n + j]
    g[2 * i] = 2 * dx
    g[2 * j] = -2 * dx
    g[2 * i + 1] = 2 * dy
    g[2 * j + 1] = -2 * dy
    g[2 * n + i] = -2 * rs
    g[2 * n + j] = -2 * rs
    return g


def _optimize_from(centers0, n, maxiter=400):
    radii = np.full(n, 0.07)
    x0 = np.concatenate([centers0.flatten(), radii])

    def neg_sum(v):
        return -np.sum(v[2 * n:])

    def neg_sum_grad(v):
        g = np.zeros_like(v)
        g[2 * n:] = -1.0
        return g

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq',
                            'fun': (lambda v, i=i: v[2 * i] - v[2 * n + i]),
                            'jac': (lambda v, i=i, n=n: _jac_xm(i, n, len(v)))})
        constraints.append({'type': 'ineq',
                            'fun': (lambda v, i=i: 1.0 - v[2 * i] - v[2 * n + i]),
                            'jac': (lambda v, i=i, n=n: _jac_xp(i, n, len(v)))})
        constraints.append({'type': 'ineq',
                            'fun': (lambda v, i=i: v[2 * i + 1] - v[2 * n + i]),
                            'jac': (lambda v, i=i, n=n: _jac_ym(i, n, len(v)))})
        constraints.append({'type': 'ineq',
                            'fun': (lambda v, i=i: 1.0 - v[2 * i + 1] - v[2 * n + i]),
                            'jac': (lambda v, i=i, n=n: _jac_yp(i, n, len(v)))})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq',
                                'fun': (lambda v, i=i, j=j: _pair(v, i, j, n)),
                                'jac': (lambda v, i=i, j=j: _pair_jac(v, i, j, n))})

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n

    res = minimize(neg_sum, x0, jac=neg_sum_grad, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': maxiter, 'ftol': 1e-10})
    return res.x


def _init_rows(rows_config, n, offset=0.0, rng=None):
    centers = []
    nrows = len(rows_config)
    ys = np.linspace(0.1, 0.9, nrows)
    for ri, count in enumerate(rows_config):
        xs = np.linspace(1.0 / (2 * count), 1 - 1.0 / (2 * count), count)
        if ri % 2 == 1:
            xs = xs + offset
            xs = np.clip(xs, 0.05, 0.95)
        for x in xs:
            centers.append([x, ys[ri]])
    centers = np.array(centers, dtype=float)
    if rng is not None:
        centers += 1e-3 * rng.standard_normal(centers.shape)
    return centers


def _feasibility_repair(centers, radii, n):
    def min_slack(c, r):
        slk = []
        for i in range(n):
            slk += [c[i, 0] - r[i], 1 - c[i, 0] - r[i],
                    c[i, 1] - r[i], 1 - c[i, 1] - r[i]]
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                slk.append(d - (r[i] + r[j]))
        return min(slk)

    ms = min_slack(centers, radii)
    if ms < 0:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if min_slack(centers, radii * mid) >= 0:
                lo = mid
            else:
                hi = mid
        radii = radii * lo
    return centers, radii


def construct_packing():
    n = 26
    rng = np.random.default_rng(0)

    starts = [
        ([5, 6, 5, 5, 5], 0.02),
        ([5, 5, 6, 5, 5], 0.03),
        ([6, 5, 5, 5, 5], 0.02),
        ([5, 5, 5, 5, 6], 0.02),
        ([4, 5, 4, 5, 4, 4], 0.04),
        ([5, 5, 5, 5, 5, 1], 0.02),
    ]

    best_sum = -1.0
    best_c, best_r = None, None
    for rows_config, off in starts:
        if sum(rows_config) != n:
            continue
        c0 = _init_rows(rows_config, n, offset=off, rng=rng)
        v = _optimize_from(c0, n, maxiter=300)
        c = v[:2 * n].reshape(n, 2)
        r = v[2 * n:].copy()
        c, r = _feasibility_repair(c, r, n)
        s = float(np.sum(r))
        if s > best_sum:
            best_sum = s
            best_c, best_r = c, r

    return best_c, best_r, best_sum
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
