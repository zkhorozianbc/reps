# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    e=1e-12; r=.1-e; g=np.linspace(r,1-r,5); h=(g[0]+g[1])/2
    centers=np.array([(x,y)for y in g for x in g]+[(h,h)])
    radii=np.r_[np.full(25,r),np.hypot(h-g[0],h-g[0])-r-e]
    return centers,radii,float(radii.sum())
# EVOLVE-BLOCK-END

def run_packing():
    return construct_packing()

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
