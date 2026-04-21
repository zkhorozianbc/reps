# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    n = 26
    best_sum, best_centers, best_radii = 0, None, None
    for c0, r0 in _init_configs(n):
        c, r = _opt(c0, r0, n)
        s = np.sum(r)
        if s > best_sum and _valid(c, r, n):
            best_sum, best_centers, best_radii = s, c.copy(), r.copy()
    return best_centers, best_radii, float(np.sum(best_radii))

def _known_packing(n):
    r = 0.10127
    dx, dy = 2*r, r*3**0.5
    pts = []
    for i, (cnt, off) in enumerate([(6,0),(5,1),(6,0),(5,1),(4,0)]):
        y = r + i*dy
        for j in range(cnt):
            pts.append([r + off*r + j*dx, y])
    c = np.array(pts[:n])
    xm, ym = c[:,0].max()+r, c[:,1].max()+r
    sc = min(1/xm, 1/ym)
    return c*sc, np.full(n, r*sc)

def _hex(n, r, rows, offs):
    dx, dy = 2*r, r*3**0.5
    pts = []
    for i, cnt in enumerate(rows):
        y = r + i*dy
        for j in range(cnt):
            pts.append([r + offs[i]*r + j*dx, y])
    return np.array(pts[:n]), np.full(n, r*0.97)

def _init_configs(n):
    cfgs = []
    for sc_fac in [1.0, 0.999, 1.001, 0.998, 1.002]:
        c, r = _known_packing(n)
        cfgs.append((c, r*sc_fac))
    for r, rows, offs in [
        (0.10127,[6,5,6,5,4],[0,1,0,1,0]),
        (0.1013,[6,5,6,5,4],[0,1,0,1,0]),
        (0.1014,[6,5,6,5,4],[0,1,0,1,0]),
        (0.1012,[6,5,6,5,4],[0,1,0,1,0]),
        (0.1013,[5,6,5,6,4],[1,0,1,0,0]),
    ]:
        cfgs.append(_hex(n, r, rows, offs))
    for rows, offs in [([6,5,6,5,4],[0,1,0,1,0]),([5,6,5,6,4],[1,0,1,0,0])]:
        r = 0.1013
        c, rv = _hex(n, r, rows, offs)
        xm = c[:,0].max()+r; ym = c[:,1].max()+r
        sc = min(1/xm, 1/ym)
        cfgs.append((c*sc, np.full(n, r*sc*0.98)))
    return cfgs

def _opt(centers, radii, n):
    ii, jj = np.triu_indices(n, 1)
    def fv(x): return -np.sum(np.exp(x[2*n:]))
    def cwv(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.r_[c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r]
    def ccv(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.sqrt(((c[ii]-c[jj])**2).sum(1)) - r[ii] - r[jj]
    cons = [{'type':'ineq','fun':cwv},{'type':'ineq','fun':ccv}]
    opts6 = {'maxiter':6000,'ftol':1e-13}
    opts8 = {'maxiter':8000,'ftol':1e-14}

    # Stage 1: uniform radius
    def fu(x): return -n*np.exp(x[-1])
    def cwu(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[-1])
        return np.r_[c.ravel()-r, 1-r-c.ravel()]
    def ccu(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[-1])
        return np.sqrt(((c[ii]-c[jj])**2).sum(1)) - 2*r
    x0 = np.r_[centers.ravel(), np.log(radii[0])]
    try:
        res = minimize(fu, x0, method='SLSQP',
            constraints=[{'type':'ineq','fun':cwu},{'type':'ineq','fun':ccu}],
            options={'maxiter':4000,'ftol':1e-13})
        c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[-1])
        rv = np.full(n, r)
        if _valid(c, rv, n): centers, radii = c, rv
    except: pass

    x0 = np.r_[centers.ravel(), np.log(radii)]
    best_c, best_r = centers.copy(), radii.copy()

    # Stage 2: varied radii, progressive tightening
    for ftol in [1e-9, 1e-11, 1e-13, 1e-14]:
        try:
            res = minimize(fv, x0, method='SLSQP', constraints=cons,
                options={'maxiter':6000,'ftol':ftol})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r,n) and np.sum(r)>np.sum(best_r):
                best_c, best_r = c.copy(), r.copy(); x0 = res.x
        except: pass

    # Stage 3: perturbed restarts
    for seed, scale in enumerate([0.001,0.002,0.003,0.005,0.008,0.001,0.002,0.004,0.003,0.006,0.001,0.002,0.0005,0.0015]):
        np.random.seed(seed+100)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        xp[:2*n] += np.random.randn(2*n)*scale
        try:
            res = minimize(fv, xp, method='SLSQP', constraints=cons, options=opts6)
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r,n) and np.sum(r)>np.sum(best_r):
                best_c, best_r = c.copy(), r.copy()
                try:
                    res2 = minimize(fv, res.x, method='SLSQP', constraints=cons, options=opts8)
                    c2, r2 = res2.x[:2*n].reshape(n,2), np.exp(res2.x[2*n:])
                    if _valid(c2,r2,n) and np.sum(r2)>np.sum(best_r):
                        best_c, best_r = c2.copy(), r2.copy()
                except: pass
        except: pass

    # Stage 4: combined perturbations
    for seed in range(8):
        np.random.seed(seed+50)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        xp[:2*n] += np.random.randn(2*n)*0.002
        xp[2*n:] += np.random.randn(n)*0.005
        try:
            res = minimize(fv, xp, method='SLSQP', constraints=cons, options=opts6)
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r,n) and np.sum(r)>np.sum(best_r):
                best_c, best_r = c.copy(), r.copy()
        except: pass

    # Stage 5: large perturbations
    for seed in range(6):
        np.random.seed(seed+300)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        xp[:2*n] += np.random.randn(2*n)*0.015
        try:
            res = minimize(fv, xp, method='SLSQP', constraints=cons,
                options={'maxiter':6000,'ftol':1e-12})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r,n) and np.sum(r)>np.sum(best_r):
                best_c, best_r = c.copy(), r.copy()
                try:
                    res2 = minimize(fv, res.x, method='SLSQP', constraints=cons, options=opts6)
                    c2, r2 = res2.x[:2*n].reshape(n,2), np.exp(res2.x[2*n:])
                    if _valid(c2,r2,n) and np.sum(r2)>np.sum(best_r):
                        best_c, best_r = c2.copy(), r2.copy()
                except: pass
        except: pass

    # Stage 6: final polish
    try:
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        res = minimize(fv, xp, method='SLSQP', constraints=cons, options=opts8)
        c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
        if _valid(c,r,n) and np.sum(r)>np.sum(best_r):
            best_c, best_r = c.copy(), r.copy()
    except: pass

    return best_c, best_r

def _valid(c, r, n, tol=1e-6):
    if np.any(c[:,0]-r < -tol) or np.any(c[:,0]+r > 1+tol): return False
    if np.any(c[:,1]-r < -tol) or np.any(c[:,1]+r > 1+tol): return False
    ii, jj = np.triu_indices(n, 1)
    d = np.sqrt(((c[ii]-c[jj])**2).sum(1))
    return np.all(d >= r[ii]+r[jj]-tol)
# EVOLVE-BLOCK-END

def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")