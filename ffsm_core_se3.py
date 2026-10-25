"""ffsm_core_se3.py -- free-floating space manipulator in SE(3): dynamics, momentum map,
true-dynamics certificate.

A free-floating spacecraft base (position R^3 + orientation SO(3)) carrying a 6R arm
grapples a tumbling target. Linear and angular momentum are conserved; from rest the
6-DoF base motion is the momentum-determined reaction to arm motion.

  config q = [p0(3), quat0(4), theta(n)]   (len 7+n)
  vel    u = [v0(3), w0(3), thetadot(n)]   (len 6+n)   spatial (world-frame) base twist

Contents: forward kinematics, geometric Jacobians, spatial mass matrix M(q), momentum
h(q,u), the momentum map [v0; w0] = -M_bb^{-1} M_bj thetadot and the momentum-consistent
base reconstruction Phi (rollout_reduced_se3*), Coriolis terms, forward/inverse dynamics,
the tumbling target, capsule collision clearances, and verify3d(), the feasibility
certificate (true-dynamics RK4 rollout of the commanded torque from rest; 5 cm / 3 deg
grasp gate, collision, torque limits).
"""
import numpy as np

# ----------------------------- SO(3) helpers (scalar-first quat, body->world) -----------------------------
def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qnorm(q): return q/(np.linalg.norm(q)+1e-12)
def exp_so3(rv):
    a = np.linalg.norm(rv)
    if a < 1e-9: return qnorm(np.array([1.0, rv[0]/2, rv[1]/2, rv[2]/2]))
    ax = rv/a; return np.array([np.cos(a/2), *(np.sin(a/2)*ax)])
def log_so3(q):
    q = qnorm(q);  w = np.clip(q[0], -1, 1) if q[0] >= 0 else np.clip(-q[0], -1, 1)
    qq = q if q[0] >= 0 else -q; s = np.sqrt(max(1-qq[0]*qq[0], 0.0))
    return np.zeros(3) if s < 1e-9 else 2*np.arccos(np.clip(qq[0], -1, 1))*qq[1:]/s
def rotmat(q):
    w, x, y, z = qnorm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

from dataclasses import dataclass, field
@dataclass
class FFSM3DParams:
    m0: float = 50.0
    I0: np.ndarray = field(default_factory=lambda: np.diag([8.0, 8.0, 10.0]))
    base_attach: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.0, 0.0]))
    # 6R spatial arm (anthropomorphic + spherical wrist) -> can match a full SE(3) grasp pose
    axis: np.ndarray = field(default_factory=lambda: np.array(
        [[0, 0, 1.], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0.]]))
    l: np.ndarray = field(default_factory=lambda: np.array(
        [[0, 0, 0.15], [0.6, 0, 0], [0.5, 0, 0], [0.12, 0, 0], [0.08, 0, 0], [0.12, 0, 0.]]))
    r: np.ndarray = field(default_factory=lambda: np.array(
        [[0, 0, 0.07], [0.3, 0, 0], [0.25, 0, 0], [0.07, 0, 0], [0, 0, 0], [0.07, 0, 0.]]))
    m: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 2.0, 1.0, 0.6, 0.5]))
    Ilink: np.ndarray = field(default_factory=lambda: np.array(
        [np.diag([.02, .02, .02]), np.diag([.02, .2, .2]), np.diag([.02, .12, .12]),
         np.diag([.01, .02, .02]), np.diag([.01, .02, .02]), np.diag([.01, .02, .02])]))
    tau_max: np.ndarray = field(default_factory=lambda: np.array([12.0, 12.0, 8.0, 4.0, 3.0, 3.0]))
    @property
    def n(self): return len(self.m)
    @property
    def ndof(self): return 6 + self.n

# ----------------------------- forward kinematics -----------------------------
def fk3d(q, p):
    p0 = q[:3]; R0 = rotmat(q[3:7]); th = q[7:]; n = p.n
    Rw = [R0]                                  # Rw[i] = world orientation of link i (i=0 base)
    axis_w = [];  jp = [p0 + R0 @ p.base_attach]   # jp[0] = arm root
    com = [p0]                                  # com[0] = base CoM (at p0)
    for i in range(n):
        aw = Rw[i] @ p.axis[i]; axis_w.append(aw)          # joint i axis in world (in link i-1 frame -> world via Rw[i]... use Rw[i]=parent)
        Ri = Rw[i] @ rotmat(exp_so3(p.axis[i] * th[i]))    # link i orientation
        Rw.append(Ri)
        com.append(jp[i] + Ri @ p.r[i])
        jp.append(jp[i] + Ri @ p.l[i])
    ee_pos = jp[n]; ee_R = Rw[n]
    return dict(p0=p0, R0=R0, Rw=Rw, axis_w=axis_w, jp=jp, com=com, ee_pos=ee_pos, ee_R=ee_R)

def geo_jacobians(q, p):
    """Jv[i], Jw[i] (3 x ndof) for each link i=0..n (0=base). World frame."""
    n = p.n; nd = p.ndof; k = fk3d(q, p); com = k['com']; jp = k['jp']; aw = k['axis_w']; p0 = k['p0']
    Jv = [np.zeros((3, nd)) for _ in range(n+1)]; Jw = [np.zeros((3, nd)) for _ in range(n+1)]
    for i in range(n+1):
        Jv[i][:, 0:3] = np.eye(3)
        Jv[i][:, 3:6] = -skew(com[i] - p0)                 # d/dw0: w0 x (com-p0)
        Jw[i][:, 3:6] = np.eye(3)
        for j in range(1, i+1):                            # joints upstream of link i
            Jv[i][:, 6+(j-1)] = np.cross(aw[j-1], com[i] - jp[j-1])
            Jw[i][:, 6+(j-1)] = aw[j-1]
    return Jv, Jw

def mass_matrix3d(q, p):
    n = p.n; nd = p.ndof; Jv, Jw = geo_jacobians(q, p); k = fk3d(q, p); Rw = k['Rw']
    masses = np.concatenate([[p.m0], p.m])
    Iw = [k['R0'] @ p.I0 @ k['R0'].T] + [Rw[i+1] @ p.Ilink[i] @ Rw[i+1].T for i in range(n)]
    M = np.zeros((nd, nd))
    for i in range(n+1):
        M += masses[i] * (Jv[i].T @ Jv[i]) + Jw[i].T @ Iw[i] @ Jw[i]
    return 0.5*(M + M.T)

# ----------------------------- momentum -----------------------------
def momentum(q, u, p):
    """Physical linear P (3) and angular L about origin (3)."""
    n = p.n; Jv, Jw = geo_jacobians(q, p); k = fk3d(q, p); com = k['com']; Rw = k['Rw']
    masses = np.concatenate([[p.m0], p.m])
    Iw = [k['R0'] @ p.I0 @ k['R0'].T] + [Rw[i+1] @ p.Ilink[i] @ Rw[i+1].T for i in range(n)]
    P = np.zeros(3); L = np.zeros(3)
    for i in range(n+1):
        vi = Jv[i] @ u; wi = Jw[i] @ u
        P += masses[i]*vi
        L += masses[i]*np.cross(com[i], vi) + Iw[i] @ wi
    return P, L

def base_vel_from_joints(q, thd, p):
    """From-rest momentum map: M_bb [v0;w0] + M_bj thd = 0 -> [v0;w0] = -M_bb^{-1} M_bj thd."""
    M = mass_matrix3d(q, p)
    return np.linalg.solve(M[:6, :6], -M[:6, 6:] @ thd)

def rollout_reduced_se3(base0, th_seq, thd_seq, dt, p, substeps=4):
    """Integrate base pose (p0 in R^3, quat0 on SO(3)) from the momentum map along a joint path.
       base0 = [p0(3), quat0(4)]. Returns Q (N+1, 7+n)."""
    N = len(th_seq) - 1; n = p.n
    Q = np.zeros((N+1, 7+n)); Q[0, :7] = base0; Q[0, 7:] = th_seq[0]
    p0 = base0[:3].copy(); qb = qnorm(base0[3:7].copy())
    h = dt/substeps
    for kk in range(N):
        for s in range(substeps):
            f = s/substeps; th = th_seq[kk]*(1-f)+th_seq[kk+1]*f; thd = thd_seq[kk]*(1-f)+thd_seq[kk+1]*f
            q = np.concatenate([p0, qb, th]); vb = base_vel_from_joints(q, thd, p)
            v0, w0 = vb[:3], vb[3:6]
            p0 = p0 + h*v0; qb = qnorm(qmul(exp_so3(w0*h), qb))     # world-frame twist
        Q[kk+1, :3] = p0; Q[kk+1, 3:7] = qb; Q[kk+1, 7:] = th_seq[kk+1]
    return Q

# ----------------------------- Coriolis + true dynamics (the gate) -----------------------------
def _perturb(q, k, eps, p):
    """Config perturbed along generalized-velocity direction k (manifold-aware)."""
    qp = q.copy()
    if k < 3: qp[k] += eps
    elif k < 6:
        e = np.zeros(3); e[k-3] = eps
        qp[3:7] = qnorm(qmul(exp_so3(e), q[3:7]))          # world-frame base rotation
    else: qp[7+(k-6)] += eps
    return qp

def _dM(q, p, eps=1e-6):
    nd = p.ndof; dM = np.zeros((nd, nd, nd))
    for k in range(nd):
        Mp = mass_matrix3d(_perturb(q, k, eps, p), p)
        Mm = mass_matrix3d(_perturb(q, k, -eps, p), p)
        dM[:, :, k] = (Mp - Mm)/(2*eps)
    return dM

def coriolis3d(q, u, p):
    """C(q,u)u via Christoffel symbols from manifold finite-diff dM/dq."""
    nd = p.ndof; dM = _dM(q, p); c = np.zeros(nd)
    for kk in range(nd):
        G = 0.5*(dM[kk] + dM[kk].T - dM[:, :, kk])
        c[kk] = u @ G @ u
    return c

def forward_dynamics3d(q, u, tau, p):
    """u_dot from M u_dot + C(q,u)u = S tau. Free-floating: zero generalized force on base."""
    M = mass_matrix3d(q, p); c = coriolis3d(q, u, p)
    Q = np.zeros(p.ndof); Q[6:] = tau
    return np.linalg.solve(M, Q - c)

def _integrate_state(q, u, udot, h):
    """Advance (q,u) by h: base pos/joints Euler-ish in the RK stage; base quat via exp map."""
    p0 = q[:3] + h*u[:3]
    qb = qnorm(qmul(exp_so3(u[3:6]*h), q[3:7]))
    th = q[7:] + h*u[6:]
    un = u + h*udot
    return np.concatenate([p0, qb, th]), un

def integrate_true3d(q0, u0, tau_seq, dt, p, substeps=10):
    """Forward-integrate TRUE spatial dynamics (RK4-in-velocity, exp-map on SO(3)), ZOH torque.
       tau_seq: (N, n). Returns Q (N+1, 7+n), U (N+1, 6+n)."""
    N = len(tau_seq); h = dt/substeps; nq = 7+p.n
    Q = np.zeros((N+1, nq)); U = np.zeros((N+1, p.ndof)); Q[0] = q0; U[0] = u0
    q = q0.copy(); u = u0.copy()
    for kx in range(N):
        tau = tau_seq[kx]
        for _ in range(substeps):
            a1 = forward_dynamics3d(q, u, tau, p)
            q2, u2 = _integrate_state(q, u, a1, 0.5*h); a2 = forward_dynamics3d(q2, u2, tau, p)
            q3, u3 = _integrate_state(q, u, a2, 0.5*h); a3 = forward_dynamics3d(q3, u3, tau, p)
            q4, u4 = _integrate_state(q, u, a3, h);     a4 = forward_dynamics3d(q4, u4, tau, p)
            udot = (a1 + 2*a2 + 2*a3 + a4)/6
            umid = (u + u2 + u3 + u4) / 4.0            # avg twist for the config step
            # config update with RK4-averaged twist
            p0 = q[:3] + h*(u[:3] + 2*u2[:3] + 2*u3[:3] + u4[:3])/6
            wbar = (u[3:6] + 2*u2[3:6] + 2*u3[3:6] + u4[3:6])/6
            qb = qnorm(qmul(exp_so3(wbar*h), q[3:7]))
            th = q[7:] + h*(u[6:] + 2*u2[6:] + 2*u3[6:] + u4[6:])/6
            q = np.concatenate([p0, qb, th]); u = u + h*udot
        Q[kx+1] = q; U[kx+1] = u
    return Q, U

# ----------------------------- inverse dynamics (path -> torque) -----------------------------
def inverse_dynamics3d(th, thd, thdd, base0, dt, p):
    """Reduced (momentum-consistent) path -> commanded joint torque. base0=[p0(3),quat0(4)].
       Returns Q (N+1,7+n), U (N+1,ndof), tau (N+1,n), base_resid."""
    N = len(th)-1
    Q = rollout_reduced_se3(base0, th, thd, dt, p)
    U = np.array([np.concatenate([base_vel_from_joints(Q[i], thd[i], p), thd[i]]) for i in range(N+1)])
    tau = np.zeros((N+1, p.n)); base_resid = 0.0
    for i in range(N+1):
        M = mass_matrix3d(Q[i], p); c = coriolis3d(Q[i], U[i], p)
        Mbb, Mbj, Mjb, Mjj = M[:6, :6], M[:6, 6:], M[6:, :6], M[6:, 6:]
        vdot_b = np.linalg.solve(Mbb, -(Mbj @ thdd[i] + c[:6]))
        tau[i] = Mjb @ vdot_b + Mjj @ thdd[i] + c[6:]
        base_resid = max(base_resid, np.abs(Mbb @ vdot_b + Mbj @ thdd[i] + c[:6]).max())
    return Q, U, tau, base_resid

# ----------------------------- SE(3) tumbling target + collision + gate -----------------------------
@dataclass
class Target3D:
    p0: np.ndarray = field(default_factory=lambda: np.array([2.3, 0.3, 0.2]))
    v:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat0: np.ndarray = field(default_factory=lambda: np.array([1., 0, 0, 0]))
    omega: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.05, 0.08]))   # tumble axis*rate
    grasp_offset: np.ndarray = field(default_factory=lambda: np.array([-0.45, 0, 0]))  # in target frame
    grasp_quat: np.ndarray = field(default_factory=lambda: exp_so3(np.array([0, np.pi, 0])))  # EE rel target
    body_radius: float = 0.30
    def frame(self, t):
        c = self.p0 + self.v*t
        Rq = qnorm(qmul(exp_so3(self.omega*t), self.quat0))
        gp = c + rotmat(Rq) @ self.grasp_offset
        gq = qnorm(qmul(Rq, self.grasp_quat))
        return c, Rq, gp, gq

LINK_RADIUS = 0.06; BASE_RADIUS = 0.6
def collision_clearances3d(q, p, tgt, t):
    k = fk3d(q, p); jp = k['jp']; p0 = k['p0']; n = p.n
    def seg_pt(a, b, pt):
        ab = b-a; u = np.clip((pt-a)@ab/((ab@ab)+1e-12), 0, 1); return np.linalg.norm(pt-(a+u*ab))
    def seg_seg(a, b, c, d):
        ts = np.linspace(0, 1, 6); pa = a+np.outer(ts, b-a); pc = c+np.outer(ts, d-c)
        return min(np.linalg.norm(pa[i]-pc[j]) for i in range(6) for j in range(6))
    segs = [(jp[i], jp[i+1]) for i in range(n)]; clr = []
    for i in range(n): clr.append(seg_pt(*segs[i], p0) - (BASE_RADIUS+LINK_RADIUS))
    for i in range(n):
        for j in range(i+3, n): clr.append(seg_seg(*segs[i], *segs[j]) - 2*LINK_RADIUS)
    c, _, _, _ = tgt.frame(t)
    for i in range(n-1): clr.append(seg_pt(*segs[i], c) - (tgt.body_radius+LINK_RADIUS))
    return np.array(clr) if clr else np.array([1.0])

def build_inst3d(p, tgt, th0, base0, tf, N):
    return dict(p=p, tgt=tgt, th0=np.asarray(th0, float), base0=np.asarray(base0, float),
                tf=tf, N=N, dt=tf/N, tau_max=p.tau_max)

def verify3d(inst, tau_seq, substeps=12, pos_tol=0.05, ang_tol=np.deg2rad(3.0)):
    """GATE: integrate TRUE spatial dynamics from REST with commanded torque; decide feasibility
       by gripper SE(3) pose error to the moving grasp frame + collision + torque limits."""
    p = inst['p']; tgt = inst['tgt']; dt = inst['dt']; N = inst['N']
    if tau_seq is None: return dict(feasible=False, pos_err=9.9, ang_err_deg=180.0, max_vio=9.9)
    tau_seq = np.asarray(tau_seq)[:N]
    q0 = np.concatenate([inst['base0'], inst['th0']]); u0 = np.zeros(p.ndof)
    Q, U = integrate_true3d(q0, u0, tau_seq, dt, p, substeps=substeps)
    k = fk3d(Q[-1], p); _, _, gp, gq = tgt.frame(N*dt)
    pos_err = float(np.linalg.norm(k['ee_pos'] - gp))
    from numpy.linalg import norm
    ee_q = _rot_to_quat(k['ee_R'])
    ang_err = float(norm(log_so3(qmul(qconj(ee_q), gq))))
    max_tau = float(np.abs(tau_seq).max())
    min_clr = min(collision_clearances3d(Q[i], p, tgt, i*dt).min() for i in range(N+1))
    max_vio = float(max(0.0, -min_clr))
    tau_ok = bool(np.all(np.abs(tau_seq) <= p.tau_max + 1e-6))
    P0, L0 = momentum(Q[0], U[0], p); Pf, Lf = momentum(Q[-1], U[-1], p)
    feasible = bool(pos_err <= pos_tol and ang_err <= ang_tol and max_vio <= 1e-3 and tau_ok)
    return dict(feasible=feasible, pos_err=pos_err, ang_err_deg=float(np.rad2deg(ang_err)),
                max_vio=max_vio, max_tau=max_tau, tau_ok=tau_ok,
                P_drift=float(np.linalg.norm(Pf-P0)), L_drift=float(np.linalg.norm(Lf-L0)), Q=Q)

def _rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr+1.0)*2; w = 0.25*S; x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/S; x=0.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/S; x=(R[0,1]+R[1,0])/S; y=0.25*S; z=(R[1,2]+R[2,1])/S
    else:
        S = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/S; x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=0.25*S
    return qnorm(np.array([w, x, y, z]))

def ee_quat(q, p): return _rot_to_quat(fk3d(q, p)['ee_R'])

# ----------------------------- vectorized fast dynamics (cross-checked vs the reference above) -----------------------------
def _jacs_fast(q, p):
    k = fk3d(q, p); com = np.array(k['com']); jp = np.array(k['jp']); aw = np.array(k['axis_w']); p0 = k['p0']
    n = p.n; nd = p.ndof; nl = n + 1
    Jv = np.zeros((nl, 3, nd)); Jw = np.zeros((nl, 3, nd))
    Jv[:, :, 0:3] = np.eye(3); Jw[:, :, 3:6] = np.eye(3)
    d = com - p0
    Jv[:, :, 3:6] = -np.stack([skew(d[i]) for i in range(nl)])
    for j in range(n):
        cr = np.cross(aw[j][None, :], com - jp[j])          # (nl,3)
        mask = np.arange(nl) >= j + 1
        Jv[mask, :, 6+j] = cr[mask]; Jw[mask, :, 6+j] = aw[j]
    return Jv, Jw, k

def mass_matrix3d_fast(q, p):
    n = p.n; Jv, Jw, k = _jacs_fast(q, p); Rw = k['Rw']
    masses = np.concatenate([[p.m0], p.m])
    Iw = np.stack([k['R0'] @ p.I0 @ k['R0'].T] + [Rw[i+1] @ p.Ilink[i] @ Rw[i+1].T for i in range(n)])
    M = np.einsum('l,lki,lkj->ij', masses, Jv, Jv) + np.einsum('lki,lkm,lmj->ij', Jw, Iw, Jw)
    return 0.5*(M + M.T)

def base_vel_from_joints_fast(q, thd, p):
    M = mass_matrix3d_fast(q, p); return np.linalg.solve(M[:6, :6], -M[:6, 6:] @ thd)

def rollout_reduced_se3_fast(base0, th_seq, thd_seq, dt, p, substeps=4):
    N = len(th_seq)-1; n = p.n; Q = np.zeros((N+1, 7+n)); Q[0, :7] = base0; Q[0, 7:] = th_seq[0]
    p0 = base0[:3].copy(); qb = qnorm(base0[3:7].copy()); h = dt/substeps
    for kk in range(N):
        for s in range(substeps):
            f = s/substeps; th = th_seq[kk]*(1-f)+th_seq[kk+1]*f; thd = thd_seq[kk]*(1-f)+thd_seq[kk+1]*f
            vb = base_vel_from_joints_fast(np.concatenate([p0, qb, th]), thd, p)
            p0 = p0 + h*vb[:3]; qb = qnorm(qmul(exp_so3(vb[3:6]*h), qb))
        Q[kk+1, :3] = p0; Q[kk+1, 3:7] = qb; Q[kk+1, 7:] = th_seq[kk+1]
    return Q

def rollout_reduced_se3_eucl(base0, th_seq, thd_seq, dt, p, substeps=4):
    """FM-Eucl base reconstruction: integrate the momentum angular velocity with a EUCLIDEAN
       (R^4 quaternion Euler) update + renormalize, instead of the SO(3) exp-map. This is the
       genuine 'treat SO(3) as flat R^4' error -> accumulates off-manifold -> momentum-INconsistent.
       Returns Q (N+1,7+n) and U (N+1,6+n) (the twists used)."""
    N = len(th_seq)-1; n = p.n; Q = np.zeros((N+1, 7+n)); U = np.zeros((N+1, 6+n))
    Q[0, :7] = base0; Q[0, 7:] = th_seq[0]; p0 = base0[:3].copy(); qb = qnorm(base0[3:7].copy()); h = dt/substeps
    for kk in range(N):
        for s in range(substeps):
            f = s/substeps; th = th_seq[kk]*(1-f)+th_seq[kk+1]*f; thd = thd_seq[kk]*(1-f)+thd_seq[kk+1]*f
            vb = base_vel_from_joints_fast(np.concatenate([p0, qb, th]), thd, p); v0, w0 = vb[:3], vb[3:6]
            p0 = p0 + h*v0
            qdot = 0.5*qmul(np.array([0.0, *w0]), qb)          # R^4 quaternion rate (world)
            qb = qnorm(qb + h*qdot)                            # EUCLIDEAN Euler step + renormalize
        Q[kk+1, :3] = p0; Q[kk+1, 3:7] = qb; Q[kk+1, 7:] = th_seq[kk+1]
        vb = base_vel_from_joints_fast(Q[kk+1], thd_seq[kk+1], p); U[kk+1] = np.concatenate([vb, thd_seq[kk+1]])
    U[0, :6] = base_vel_from_joints_fast(Q[0], thd_seq[0], p); U[0, 6:] = thd_seq[0]
    return Q, U

def torque_from_path(Q, U, dt, p):
    """Joint torque for an IMPOSED (base, joints) trajectory: tau_joint = [M Udot + C U]_joint.
       For a momentum-inconsistent base this torque, applied to the free-floating base, will NOT
       reproduce the imposed base (base follows true momentum instead) -> the FM-Eucl failure."""
    N = Q.shape[0]-1; Udot = np.gradient(U, dt, axis=0); tau = np.zeros((N+1, p.n))
    for i in range(N+1):
        M = mass_matrix3d_fast(Q[i], p); c = coriolis3d(Q[i], U[i], p)
        tau[i] = (M @ Udot[i] + c)[6:]
    return tau
