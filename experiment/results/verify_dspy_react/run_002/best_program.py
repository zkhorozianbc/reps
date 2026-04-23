# EVOLVE-BLOCK-START
"""
Optimized circle packing for 26 circles in [0,1]x[0,1].
Uses LP-sum surrogate optimization + joint center+radii optimization + LP.
"""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist, squareform


def construct_packing():
    """Place 26 circles in a unit square, maximizing sum of radii.

    Returns:
        (centers, radii, sum_radii) where
            centers: np.array shape (26, 2)
            radii:   np.array shape (26,)
            sum_radii: float
    """
    n = 26
    idx_i_global, idx_j_global = np.triu_indices(n, k=1)
    n_pairs = len(idx_i_global)

    def compute_equal_radius(centers):
        bd = np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        )
        pw = pdist(centers)
        if len(pw) == 0:
            return np.min(bd)
        return min(np.min(bd), np.min(pw) / 2.0)

    def compute_variable_radii_lp(centers):
        """Solve LP to maximize sum of radii given fixed centers."""
        D = squareform(pdist(centers))
        bd = np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        )
        c_obj = -np.ones(n)
        A_ub = np.zeros((n_pairs, n))
        b_ub = D[idx_i_global, idx_j_global]
        A_ub[np.arange(n_pairs), idx_i_global] = 1.0
        A_ub[np.arange(n_pairs), idx_j_global] = 1.0
        bounds = [(0.0, float(bd[i])) for i in range(n)]
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return np.maximum(result.x, 0)
        r_eq = compute_equal_radius(centers)
        return np.full(n, r_eq)

    def obj_lp_sum_surrogate(cf, a):
        """Smooth surrogate for LP sum: sum_i soft_min(bd_i, D_ij/2 for all j).

        This is a lower bound on the true LP sum. Maximizing it encourages
        each circle to be as far as possible from boundaries and neighbors.
        """
        centers = cf.reshape(n, 2)
        bd = np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        )
        D = squareform(pdist(centers))
        D_half = D / 2.0
        np.fill_diagonal(D_half, 1e9)
        all_vals = np.hstack([bd.reshape(-1, 1), D_half])
        min_per_row = np.min(all_vals, axis=1)
        shifted = all_vals - min_per_row.reshape(-1, 1)
        shifted_clipped = np.minimum(shifted * a, 500.0)
        soft_min_per_row = min_per_row - np.log(np.sum(np.exp(-shifted_clipped), axis=1)) / a
        return -np.sum(soft_min_per_row)

    def obj_joint(xr, a):
        """Joint objective for centers + radii."""
        c = xr[:2*n].reshape(n, 2)
        r = xr[2*n:]
        bd = np.minimum(
            np.minimum(c[:, 0], 1.0 - c[:, 0]),
            np.minimum(c[:, 1], 1.0 - c[:, 1])
        )
        D = squareform(pdist(c))
        bd_viol = np.sum(np.maximum(r - bd, 0)**2)
        pair_viol = np.sum(np.maximum(
            r[idx_i_global] + r[idx_j_global] - D[idx_i_global, idx_j_global], 0)**2)
        return -np.sum(r) + a * (bd_viol + pair_viol)

    def make_hex_grid(cols=5, offset_start=0):
        centers = []
        dx = 1.0 / cols
        dy = dx * np.sqrt(3) / 2
        row = 0
        while len(centers) < n and row < 20:
            offset = (dx / 2) if ((row + offset_start) % 2 == 1) else 0.0
            y = dy * row + dx / 2
            if y > 1.0:
                break
            for j in range(cols + 2):
                if len(centers) >= n:
                    break
                x = offset + j * dx
                if 0.001 < x < 0.999 and 0.001 < y < 0.999:
                    centers.append([x, y])
            row += 1
        rng = np.random.RandomState(42)
        while len(centers) < n:
            centers.append(rng.uniform(0.05, 0.95, 2).tolist())
        return np.array(centers[:n])

    def make_grid(rows, cols, offset_odd=False):
        centers = []
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                if offset_odd and i % 2 == 1:
                    x = (j + 1.0) / cols
                    if x >= 1.0:
                        continue
                centers.append([x, y])
        rng = np.random.RandomState(1)
        while len(centers) < n:
            centers.append(rng.uniform(0.05, 0.95, 2).tolist())
        return np.array(centers[:n])

    # Diverse initial configs
    init_configs = [
        make_hex_grid(5, 0),
        make_hex_grid(5, 1),
        make_hex_grid(6, 0),
        make_hex_grid(6, 1),
        make_grid(6, 5),
        make_grid(5, 6),
        make_grid(6, 5, offset_odd=True),
        make_grid(5, 6, offset_odd=True),
    ]
    for seed in [0, 1, 2, 3, 42, 7, 13, 99, 17]:
        rng = np.random.RandomState(seed)
        init_configs.append(rng.uniform(0.05, 0.95, (n, 2)))

    best_score = 0.0
    best_centers = None
    best_radii = None
    bounds_c = [(0.0, 1.0)] * (2 * n)
    bounds_xr = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n

    for init in init_configs:
        x0 = np.clip(init.flatten(), 0.02, 0.98)

        # Stage 1: LP-sum surrogate optimization (better proxy than equal-radii)
        for alpha in [5.0, 15.0, 50.0, 150.0, 500.0]:
            res = minimize(obj_lp_sum_surrogate, x0, args=(alpha,), method='L-BFGS-B',
                          bounds=bounds_c,
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-10})
            x0 = res.x

        centers_surr = x0.reshape(n, 2)

        # Stage 2: Joint optimization of centers + radii
        r_lp_init = compute_variable_radii_lp(centers_surr)
        xr0 = np.concatenate([centers_surr.flatten(), r_lp_init])

        for a_pen in [20.0, 100.0, 500.0, 2000.0]:
            res2 = minimize(obj_joint, xr0, args=(a_pen,), method='L-BFGS-B',
                           bounds=bounds_xr,
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10})
            xr0 = res2.x

        c_joint = xr0[:2*n].reshape(n, 2)

        # Stage 3: LP for optimal variable radii
        for centers_cand in [centers_surr, c_joint]:
            r_lp = compute_variable_radii_lp(centers_cand)
            score = float(np.sum(r_lp))
            if score > best_score:
                best_score = score
                best_centers = centers_cand.copy()
                best_radii = r_lp.copy()

    return best_centers, best_radii, float(np.sum(best_radii))
# EVOLVE-BLOCK-END


def run_packing():
    return construct_packing()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")