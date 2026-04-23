# EVOLVE-BLOCK-START
"""Circle packing: 25 circles in 5x5 grid + 1 interstitial, refined with scipy."""
import numpy as np
from scipy.optimize import minimize


def _initial_config():
    eps = 1e-9
    r_big = 0.1 - eps
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + 0.2 * j, 0.1 + 0.2 * i])
    r_small = r_big * (np.sqrt(2.0) - 1.0) - eps
    centers.append([0.2, 0.2])
    centers = np.array(centers, dtype=float)
    radii = np.array([r_big] * 25 + [r_small], dtype=float)
    return centers, radii


def _refine(centers, radii):
    n = len(radii)
    x0 = np.concatenate([centers.flatten(), radii])

    def unpack(x):
        c = x[: 2 * n].reshape(n, 2)
        r = x[2 * n :]
        return c, r

    def neg_sum(x):
        return -np.sum(x[2 * n :])

    def neg_sum_grad(x):
        g = np.zeros_like(x)
        g[2 * n :] = -1.0
        return g

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def pair_con(x):
        c, r = unpack(x)
        out = np.empty(len(pairs))
        for k, (i, j) in enumerate(pairs):
            d = np.linalg.norm(c[i] - c[j])
            out[k] = d - r[i] - r[j]
        return out

    def bnd_con(x):
        c, r = unpack(x)
        out = np.empty(4 * n)
        out[0::4] = c[:, 0] - r
        out[1::4] = 1.0 - c[:, 0] - r
        out[2::4] = c[:, 1] - r
        out[3::4] = 1.0 - c[:, 1] - r
        return out

    def rad_pos(x):
        return x[2 * n :]

    constraints = [
        {"type": "ineq", "fun": pair_con},
        {"type": "ineq", "fun": bnd_con},
        {"type": "ineq", "fun": rad_pos},
    ]

    try:
        res = minimize(
            neg_sum,
            x0,
            jac=neg_sum_grad,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if res.success or -res.fun > np.sum(radii):
            c, r = unpack(res.x)
            r = r - 1e-9
            r = np.maximum(r, 0.0)
            valid = True
            for i, j in pairs:
                if np.linalg.norm(c[i] - c[j]) + 1e-12 < r[i] + r[j]:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    if c[i, 0] - r[i] < -1e-12 or 1 - c[i, 0] - r[i] < -1e-12:
                        valid = False
                        break
                    if c[i, 1] - r[i] < -1e-12 or 1 - c[i, 1] - r[i] < -1e-12:
                        valid = False
                        break
            if valid and np.sum(r) > np.sum(radii):
                return c, r
    except Exception:
        pass
    return centers, radii


def construct_packing():
    centers, radii = _initial_config()
    base_sum = float(np.sum(radii))
    best_c, best_r = centers, radii
    best_sum = base_sum

    rng = np.random.default_rng(0)
    for trial in range(4):
        c0 = centers.copy()
        r0 = radii.copy()
        if trial > 0:
            c0 = c0 + rng.normal(0, 0.005, c0.shape)
            c0 = np.clip(c0, r0[:, None] + 1e-6, 1 - r0[:, None] - 1e-6)
        c, r = _refine(c0, r0)
        s = float(np.sum(r))
        if s > best_sum:
            best_sum = s
            best_c, best_r = c, r

    return best_c, best_r, float(np.sum(best_r))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
