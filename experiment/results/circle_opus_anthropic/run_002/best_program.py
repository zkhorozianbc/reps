# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct_packing():
    n = 26
    ii, jj = np.triu_indices(n, 1)

    def fv(x):
        return -np.sum(np.exp(x[2*n:]))
    def gv(x):
        g = np.zeros_like(x); g[2*n:] = -np.exp(x[2*n:]); return g
    def cwv(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.r_[c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r]
    def ccv(x):
        c, r = x[:2*n].reshape(n,2), np.exp(x[2*n:])
        return np.sqrt(((c[ii]-c[jj])**2).sum(1)) - r[ii] - r[jj]
    cons = [{'type':'ineq','fun':cwv},{'type':'ineq','fun':ccv}]

    best_c, best_r, best_sum = None, None, 0.0

    def try_opt(x0, mi=20000):
        nonlocal best_c, best_r, best_sum
        for ft in [1e-13, 1e-16]:
            try:
                res = minimize(fv, x0, method='SLSQP', jac=gv, constraints=cons,
                    options={'maxiter':mi,'ftol':ft})
                c = res.x[:2*n].reshape(n,2); r = np.exp(res.x[2*n:])
                tol=1e-6; d=np.sqrt(((c[ii]-c[jj])**2).sum(1))-r[ii]-r[jj]
                if np.all(c[:,0]-r>=-tol) and np.all(c[:,0]+r<=1+tol) and np.all(c[:,1]-r>=-tol) and np.all(c[:,1]+r<=1+tol) and np.all(d>=-tol) and np.sum(r)>best_sum:
                    best_c,best_r,best_sum = c.copy(),r.copy(),np.sum(r)
                x0 = res.x
            except: pass

    def make_hex(rows, r0, sm=0, cx_off=0.0, cy_off=0.0):
        pts = []; dx, dy = 2*r0, r0*3**0.5
        for i, cnt in enumerate(rows):
            off = ((i+sm)%2)*r0; y = r0 + i*dy + cy_off
            for j in range(cnt):
                pts.append([r0+off+j*dx+cx_off, y])
        c0 = np.array(pts[:n])
        mx = max(c0[:,0].max()+r0, c0[:,1].max()+r0)
        mn = min(c0[:,0].min()-r0, c0[:,1].min()-r0)
        sc = 1.0/(mx - min(mn, 0))
        if mn < 0: c0 -= mn
        c0 *= sc
        return np.r_[c0.ravel()*1.0, np.full(n, np.log(r0*sc))]

    rcs = [(6,5,6,5,4),(5,6,5,6,4),(5,5,6,5,5),(4,5,6,5,6),
           (6,5,5,6,4),(5,6,6,5,4),(4,6,5,6,5),(6,4,6,5,5),
           (5,5,5,6,5),(6,6,5,5,4),(5,5,5,5,6),(6,5,6,4,5)]
    for rows in rcs:
        if sum(rows)!=n: continue
        for r0 in [0.092,0.094,0.097,0.0985,0.1005,0.1013,0.1025,0.104,0.107,0.11]:
            for sm in [0,1]:
                try_opt(make_hex(rows, r0, sm))

    if best_c is not None:
        for seed in range(350):
            np.random.seed(seed*17+5)
            xp = np.r_[best_c.ravel(), np.log(best_r)]
            s = 0.0001 + 0.035*(seed%30)/29.0
            xp[:2*n] += np.random.randn(2*n)*s
            xp[2*n:] += np.random.randn(n)*s*0.7
            try_opt(xp, mi=18000)

    return best_c, best_r, float(best_sum)
# EVOLVE-BLOCK-END

def run_packing():
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")