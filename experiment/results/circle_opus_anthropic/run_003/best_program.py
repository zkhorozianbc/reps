# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    n = 26
    ii, jj = np.triu_indices(n, 1)
    
    def fv(x): return -np.sum(np.exp(x[2*n:]))
    def wv(x):
        cc, rr = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.r_[cc[:,0]-rr, 1-cc[:,0]-rr, cc[:,1]-rr, 1-cc[:,1]-rr]
    def sv(x):
        cc, rr = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.sqrt(((cc[ii]-cc[jj])**2).sum(1)) - rr[ii] - rr[jj]
    cons = [{'type':'ineq','fun':wv},{'type':'ineq','fun':sv}]
    
    def fu(x): return -n*np.exp(x[-1])
    def wu(x):
        cc, rr = x[:2*n].reshape(n,2), np.exp(x[-1])
        return np.r_[cc[:,0]-rr, 1-cc[:,0]-rr, cc[:,1]-rr, 1-cc[:,1]-rr]
    def su(x):
        cc, rr = x[:2*n].reshape(n,2), np.exp(x[-1])
        return np.sqrt(((cc[ii]-cc[jj])**2).sum(1)) - 2*rr
    ucons = [{'type':'ineq','fun':wu},{'type':'ineq','fun':su}]
    
    def _hex(r0, rows, offs):
        pts = []
        dx, dy = 2*r0, r0*3**0.5
        for i, cnt in enumerate(rows):
            y = r0 + i*dy
            for j in range(cnt):
                pts.append([r0 + offs[i]*r0 + j*dx, y])
        c = np.array(pts[:n])
        sc = min(1/(c[:,0].max()+r0), 1/(c[:,1].max()+r0))
        return c*sc, np.full(n, r0*sc)
    
    def _optU(c, radii):
        x0 = np.r_[c.ravel(), np.log(radii[0])]
        try:
            res = minimize(fu, x0, method='SLSQP', constraints=ucons, options={'maxiter':3000,'ftol':1e-13})
            cu, ru = res.x[:2*n].reshape(n,2), np.exp(res.x[-1])
            rv = np.full(n, ru)
            if _valid(cu, rv, n): return cu, rv
        except: pass
        return c, radii
    
    def _optV(c, radii):
        xp = np.r_[c.ravel(), np.log(radii)]
        bc, br = c.copy(), radii.copy()
        for ftol in [1e-9, 1e-11, 1e-13]:
            try:
                res = minimize(fv, xp, method='SLSQP', constraints=cons, options={'maxiter':6000,'ftol':ftol})
                cr, rr = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
                if _valid(cr,rr,n) and np.sum(rr)>np.sum(br):
                    bc, br = cr.copy(), rr.copy(); xp = res.x
            except: pass
        return bc, br
    
    best_c, best_r, best_s = None, None, 0
    cfgs = [
        (0.10127,[6,5,6,5,4],[0,1,0,1,0]),
        (0.1013,[5,6,5,6,4],[1,0,1,0,0]),
        (0.1013,[6,5,6,5,4],[1,0,1,0,0]),
        (0.1010,[6,5,6,5,4],[0,1,0,1,0]),
        (0.1015,[5,5,6,5,5],[0,1,0,1,0]),
        (0.1020,[5,5,5,6,5],[0,1,0,1,0]),
        (0.1008,[6,6,5,5,4],[0,0,1,1,0]),
    ]
    for r0, rows, offs in cfgs:
        c, radii = _hex(r0, rows, offs)
        c, radii = _optU(c, radii)
        c, radii = _optV(c, radii)
        s = np.sum(radii)
        if s > best_s and _valid(c, radii, n):
            best_c, best_r, best_s = c.copy(), radii.copy(), s
    
    for seed in range(30):
        np.random.seed(seed+42)
        xp = np.r_[best_c.ravel(), np.log(best_r)]
        sc = 0.001*(1+(seed%10))
        xp[:2*n] += np.random.randn(2*n)*sc
        if seed > 12: xp[2*n:] += np.random.randn(n)*0.008
        if seed > 20:
            idx = np.random.choice(n, 2, replace=False)
            xp[2*idx[0]:2*idx[0]+2], xp[2*idx[1]:2*idx[1]+2] = xp[2*idx[1]:2*idx[1]+2].copy(), xp[2*idx[0]:2*idx[0]+2].copy()
        try:
            res = minimize(fv, xp, method='SLSQP', constraints=cons, options={'maxiter':6000,'ftol':1e-13})
            cr, rr = res.x[:2*n].reshape(n,2), np.exp(res.x[2*n:])
            if _valid(cr,rr,n) and np.sum(rr)>best_s:
                best_c, best_r, best_s = cr.copy(), rr.copy(), np.sum(rr)
        except: pass
    
    return best_c, best_r, float(best_s)

def _valid(c, r, n, tol=1e-6):
    if np.any(c[:,0]-r<-tol) or np.any(c[:,0]+r>1+tol): return False
    if np.any(c[:,1]-r<-tol) or np.any(c[:,1]+r>1+tol): return False
    ii, jj = np.triu_indices(n, 1)
    return np.all(np.sqrt(((c[ii]-c[jj])**2).sum(1)) >= r[ii]+r[jj]-tol)
# EVOLVE-BLOCK-END

def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")