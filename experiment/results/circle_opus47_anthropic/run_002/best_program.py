# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 26

def construct_packing():
    best_s, best_c, best_r = 0, None, None
    for c0, r0 in _configs():
        c, r = _opt(c0, r0)
        if _valid(c, r):
            s = r.sum()
            if s > best_s:
                best_s, best_c, best_r = s, c.copy(), r.copy()
    for seed in range(6):
        np.random.seed(seed + 7000)
        c0 = best_c + np.random.randn(N, 2) * 0.002
        r0 = best_r.copy()
        c, r = _opt(c0, r0)
        if _valid(c, r) and r.sum() > best_s:
            best_s, best_c, best_r = r.sum(), c.copy(), r.copy()
    return best_c, best_r, float(best_s)

def _hex(rows, offs, r):
    dx, dy = 2*r, r*np.sqrt(3)
    pts = []
    for i, cnt in enumerate(rows):
        y = r + i*dy
        for j in range(cnt):
            pts.append([r + offs[i]*r + j*dx, y])
    pts = np.array(pts[:N])
    xm, ym = pts[:,0].max()+r, pts[:,1].max()+r
    sc = min(1/xm, 1/ym)
    return pts*sc, np.full(N, r*sc*0.98)

def _grid(nx, ny):
    pts = []
    dx, dy = 1/nx, 1/ny
    for i in range(ny):
        for j in range(nx):
            pts.append([dx/2+j*dx, dy/2+i*dy])
            if len(pts) >= N: break
        if len(pts) >= N: break
    r = min(dx,dy)/2*0.95
    return np.array(pts[:N]), np.full(N, r)

def _asym(big_r, small_r):
    pts = [[big_r, big_r]]
    rs = [big_r]
    r = small_r
    dx, dy = 2*r, r*np.sqrt(3)
    nrows = int((1-r)/dy)+1
    for i in range(nrows):
        y = r + i*dy
        x_start = r + (r if i%2 else 0)
        for j in range(8):
            x = x_start + j*dx
            if x+r > 1-1e-9: break
            d = np.sqrt((x-big_r)**2+(y-big_r)**2)
            if d < big_r+r+1e-9: continue
            pts.append([x,y]); rs.append(r)
            if len(pts)>=N: break
        if len(pts)>=N: break
    while len(pts)<N:
        pts.append([0.5+0.01*len(pts), 0.5]); rs.append(0.02)
    return np.array(pts[:N]), np.array(rs[:N])

def _configs():
    cfgs = []
    hex_patterns = [
        ([6,5,6,5,4],[0,1,0,1,0]),
        ([5,6,5,6,4],[1,0,1,0,0]),
        ([6,5,6,5,4],[1,0,1,0,1]),
        ([5,5,5,5,6],[0,1,0,1,0]),
        ([5,5,6,5,5],[1,0,0,0,1]),
        ([6,6,5,5,4],[0,0,1,1,0]),
        ([4,5,6,5,6],[0,1,0,1,0]),
        ([6,5,5,5,5],[0,1,1,1,1]),
        ([5,6,5,5,5],[1,0,1,1,1]),
        ([5,5,5,6,5],[1,1,1,0,1]),
        ([4,6,5,6,5],[1,0,1,0,1]),
        ([5,5,5,5,5,1],[0,1,0,1,0,0]),
        ([6,5,6,5,4],[0,1,0,1,1]),
        ([6,5,6,5,4],[0,0,0,0,0]),
    ]
    for rows, offs in hex_patterns:
        cfgs.append(_hex(rows, offs, 0.101))
    cfgs.append(_grid(6,5))
    cfgs.append(_grid(5,6))
    for br, sr in [(0.15,0.095),(0.18,0.09),(0.12,0.098),(0.16,0.092)]:
        cfgs.append(_asym(br, sr))
    base_c, base_r = _hex([6,5,6,5,4],[0,1,0,1,0],0.101)
    for seed in range(6):
        np.random.seed(seed)
        c = base_c + np.random.randn(N,2)*0.008
        cfgs.append((c, base_r.copy()))
    for seed in range(4):
        np.random.seed(seed+2000)
        c = np.random.rand(N,2)*0.8+0.1
        cfgs.append((c, np.full(N, 0.09)))
    return cfgs

def _opt(centers, radii):
    n = N
    ii, jj = np.triu_indices(n, 1)

    def fv(x): return -np.sum(np.exp(x[2*n:]))
    def dfv(x):
        g = np.zeros_like(x)
        g[2*n:] = -np.exp(x[2*n:])
        return g
    def cwv(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.r_[c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r]
    def ccv(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.sqrt(((c[ii]-c[jj])**2).sum(1)+1e-30) - r[ii] - r[jj]
    cons = [{'type':'ineq','fun':cwv},{'type':'ineq','fun':ccv}]

    def fu(x): return -np.exp(x[-1])
    def cwu(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[-1])
        return np.r_[c.ravel()-r, 1-r-c.ravel()]
    def ccu(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[-1])
        return np.sqrt(((c[ii]-c[jj])**2).sum(1)+1e-30) - 2*r

    x0 = np.r_[centers.ravel(), np.log(max(radii[0], 0.01))]
    try:
        res = minimize(fu, x0, method='SLSQP',
            constraints=[{'type':'ineq','fun':cwu},{'type':'ineq','fun':ccu}],
            options={'maxiter':2000,'ftol':1e-12})
        c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[-1])
        rv = np.full(n, r)
        if _valid(c, rv): centers, radii = c, rv
    except: pass

    x0 = np.r_[centers.ravel(), np.log(np.clip(radii, 0.005, 0.25))]
    best_c, best_r = centers.copy(), radii.copy()

    for ftol in [1e-9, 1e-11, 1e-13]:
        try:
            res = minimize(fv, x0, jac=dfv, method='SLSQP', constraints=cons,
                options={'maxiter':5000,'ftol':ftol})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r) and r.sum()>best_r.sum():
                best_c, best_r = c.copy(), r.copy(); x0 = res.x
        except: pass

    scales = [0.001,0.002,0.003,0.005,0.008,0.012,0.002,0.004,0.006,0.003,0.007,0.015,0.005,0.01,0.02]
    for seed, scale in enumerate(scales):
        np.random.seed(seed+100)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        xp[:2*n] += np.random.randn(2*n)*scale
        xp[2*n:] += np.random.randn(n)*scale*0.3
        try:
            res = minimize(fv, xp, jac=dfv, method='SLSQP', constraints=cons,
                options={'maxiter':5000,'ftol':1e-13})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r) and r.sum()>best_r.sum():
                best_c, best_r = c.copy(), r.copy()
                try:
                    res2 = minimize(fv, res.x, jac=dfv, method='SLSQP', constraints=cons,
                        options={'maxiter':5000,'ftol':1e-14})
                    c2, r2 = res2.x[:2*n].reshape(n,2), np.exp(res2.x[2*n:])
                    if _valid(c2,r2) and r2.sum()>best_r.sum():
                        best_c, best_r = c2.copy(), r2.copy()
                except: pass
        except: pass

    for seed in range(10):
        np.random.seed(seed+500)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        idx_small = np.argsort(best_r)[:8]
        boost = np.zeros(n)
        boost[idx_small] = 0.05 + np.random.rand(len(idx_small))*0.08
        xp[2*n:] += boost
        xp[:2*n] += np.random.randn(2*n)*0.004
        try:
            res = minimize(fv, xp, jac=dfv, method='SLSQP', constraints=cons,
                options={'maxiter':5000,'ftol':1e-13})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r) and r.sum()>best_r.sum():
                best_c, best_r = c.copy(), r.copy()
        except: pass

    for seed in range(8):
        np.random.seed(seed+800)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        idx = np.argsort(best_r)[:3]
        for k in idx:
            xp[2*k:2*k+2] = np.random.rand(2)*0.9 + 0.05
        try:
            res = minimize(fv, xp, jac=dfv, method='SLSQP', constraints=cons,
                options={'maxiter':5000,'ftol':1e-13})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r) and r.sum()>best_r.sum():
                best_c, best_r = c.copy(), r.copy()
        except: pass

    for target_idx in range(n):
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        xp[2*n + target_idx] += 0.15
        for k in range(n):
            if k != target_idx:
                xp[2*n + k] -= 0.01
        np.random.seed(target_idx+3000)
        xp[:2*n] += np.random.randn(2*n)*0.003
        try:
            res = minimize(fv, xp, jac=dfv, method='SLSQP', constraints=cons,
                options={'maxiter':3000,'ftol':1e-12})
            c, r = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(c,r) and r.sum()>best_r.sum():
                best_c, best_r = c.copy(), r.copy()
        except: pass

    return best_c, best_r

def _valid(c, r, tol=1e-6):
    n = len(r)
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