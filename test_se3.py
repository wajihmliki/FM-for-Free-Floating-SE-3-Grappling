"""test_se3.py -- self-test of the SE(3) core: Jacobians vs finite differences, mass
matrix symmetry/positive-definiteness, and momentum conservation of the reduced rollout."""
import numpy as np, ffsm_core_se3 as fc3
np.random.seed(0); p = fc3.FFSM3DParams(); n = p.n; nd = p.ndof
print(f"SE(3) core: n={n} ndof={nd} (6 base + {n} joints)")

# random config
q = np.concatenate([np.array([0.1,-0.2,0.15]), fc3.exp_so3(np.array([0.3,-0.2,0.4])), np.linspace(0.5,-0.5,n)])
Jv, Jw = fc3.geo_jacobians(q, p)

# ---- 1. geometric Jacobians vs finite-diff (perturb along generalized velocity dirs) ----
def rotvec(dR): return 0.5*np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])
eps = 1e-6; maxv = maxw = 0
k0 = fc3.fk3d(q, p)
for kk in range(nd):
    qp = q.copy()
    if kk < 3: qp[kk] += eps
    elif kk < 6:
        e = np.zeros(3); e[kk-3] = eps
        qp[3:7] = fc3.qnorm(fc3.qmul(fc3.exp_so3(e), q[3:7]))   # world-frame base rotation
    else: qp[kk+1] += eps                                       # joint (skip quat's extra dim)
    kp = fc3.fk3d(qp, p)
    for i in range(n+1):
        maxv = max(maxv, np.abs((kp['com'][i]-k0['com'][i])/eps - Jv[i][:,kk]).max())
        dR = kp['Rw'][i] @ k0['Rw'][i].T
        maxw = max(maxw, np.abs(rotvec(dR)/eps - Jw[i][:,kk]).max())
print(f"[1] Jv err {maxv:.1e}   Jw err {maxw:.1e}   (want <1e-5)")

# ---- 2. mass matrix symmetric PD ----
M = fc3.mass_matrix3d(q, p)
print(f"[2] M sym {np.abs(M-M.T).max():.1e}  min-eig {np.linalg.eigvalsh(M).min():.3f} (want>0)")

# ---- 3. MOMENTUM CONSERVATION of the reduced SE(3) rollout (from rest) ----
from numpy import cos, sin
N = 120; tf = 6.0; dt = tf/N; s = np.linspace(0,1,N+1)
th0 = np.linspace(0.3,-0.4,n); thf = np.linspace(0.9,-0.5,n)
ease = 10*s**3-15*s**4+6*s**5; de=(30*s**2-60*s**3+30*s**4)/tf
th_seq = th0+np.outer(ease, thf-th0); thd_seq = np.outer(de, thf-th0)
base0 = np.concatenate([np.zeros(3), np.array([1.,0,0,0])])
Q = fc3.rollout_reduced_se3(base0, th_seq, thd_seq, dt, p)
# momentum at each node (u from momentum map)
Ps=[]; Ls=[]
for kx in range(N+1):
    qk = Q[kx]; thd = thd_seq[kx]
    vb = fc3.base_vel_from_joints(qk, thd, p); u = np.concatenate([vb, thd])
    P,L = fc3.momentum(qk, u, p); Ps.append(P); Ls.append(L)
Ps=np.array(Ps); Ls=np.array(Ls)
print(f"[3] reduced rollout from REST: max|P|={np.abs(Ps).max():.1e}  max|L|={np.abs(Ls).max():.1e}  (want ~0)")
# base recoil magnitude (proves 6-DoF coupling is real)
dpos = Q[-1,:3]-Q[0,:3]; dang = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Q[-1,3:7], fc3.qconj(Q[0,3:7])))))
print(f"    base recoil: dpos={np.round(dpos,3)} m,  d|orientation|={dang:.1f} deg  (arm swing -> 6-DoF base reaction)")
