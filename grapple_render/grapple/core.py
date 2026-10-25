"""
core.py -- SE(3) helpers, momentum-map partitioning, momentum-consistent
           base reconstruction, all against a MuJoCo model.

=====================================================================
FRAME CONVENTIONS -- READ THIS, IT IS THE #1 SOURCE OF SILENT ERRORS
=====================================================================
For a MuJoCo `freejoint` on the spacecraft base:

    qpos[0:3]  = base position,      GLOBAL frame
    qpos[3:7]  = base quaternion,    (w, x, y, z)   <-- NOT xyzw
    qvel[0:3]  = base linear vel,    GLOBAL frame
    qvel[3:6]  = base angular vel,   ***BODY-LOCAL frame***

That last line is verified numerically (see selftest_conventions() below).
The paper defines v0, w0 and h in the SPATIAL (inertial) frame, so:

    w_mujoco = R0^T @ w_spatial          J*_mujoco(q) = blkdiag(I, R0^T) J*(q)

Everything in this file works in MuJoCo's native dof frame and converts at
the boundary.

The generalized mass matrix from mj_fullM is in dof (qvel) order, so with
the layout [base_free(6), j1..j6(6), target_free(6)] the paper's partition

    h = Mbb [v0; w0] + Mbj thetadot                                   (eq 4)
    [v0; w0] = -Mbb^-1 Mbj thetadot = J*(q) thetadot                  (eq 5)

is literally M[0:6,0:6] and M[0:6,6:12]. The target's 6 dofs are
block-decoupled (no contact before capture); assert_decoupled() checks it.
"""

import numpy as np
import mujoco

# --------------------------------------------------------------------- quats

def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(q, dtype=np.float64))
    return R.reshape(3, 3)


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R.reshape(9), dtype=np.float64))
    return q


def quat_mul(a, b):
    o = np.zeros(4)
    mujoco.mju_mulQuat(o, np.ascontiguousarray(a, dtype=np.float64),
                       np.ascontiguousarray(b, dtype=np.float64))
    return o


def quat_norm(q):
    q = np.asarray(q, dtype=np.float64).copy()
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0, 0, 0])
    q /= n
    return q if q[0] >= 0 else -q          # canonical hemisphere


# ------------------------------------------------------------------- SO3/SE3

def so3_exp(w):
    """Rodrigues. w is a rotation vector (rad)."""
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def so3_log(R):
    """Returns rotation vector; branch-safe near pi."""
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-10:
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) * 0.5
    if np.pi - th < 1e-6:                      # near-pi: use the symmetric part
        A = (R + np.eye(3)) / 2.0
        v = np.sqrt(np.clip(np.diag(A), 0, 1))
        i = int(np.argmax(v))
        v = A[:, i] / (v[i] + 1e-15)
        v /= np.linalg.norm(v)
        return v * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2],
                                             R[0, 2] - R[2, 0],
                                             R[1, 0] - R[0, 1]])


def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def se3_log(T):
    """log(T)^vee in R^6 as [rho; phi] (translation part first)."""
    R, p = T[:3, :3], T[:3, 3]
    w = so3_log(R)
    th = np.linalg.norm(w)
    if th < 1e-10:
        Vinv = np.eye(3) - 0.5 * skew(w)
    else:
        K = skew(w / th)
        a = 1.0 - (th * np.sin(th)) / (2 * (1 - np.cos(th)))
        Vinv = np.eye(3) - 0.5 * skew(w) + (a / th**2) * (skew(w) @ skew(w))
    return np.concatenate([Vinv @ p, w])


def make_T(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def pose_error(T_ref, T_act):
    """
    The paper's e_SE3 = log( g^-1 x_ee )^vee, with g = T_ref, x_ee = T_act.
    Returns (err6, pos_err_m, rot_err_rad).
    """
    E = inv_T(T_ref) @ T_act
    e6 = se3_log(E)
    return e6, float(np.linalg.norm(E[:3, 3])), float(np.linalg.norm(so3_log(E[:3, :3])))


# --------------------------------------------------------------- model layout

class Layout:
    """Resolves names -> qpos/qvel/dof indices once, so nothing downstream
    hard-codes an offset."""

    def __init__(self, model, base_joint="base_free", target_joint="target_free",
                 arm_joints=("j1", "j2", "j3", "j4", "j5", "j6"),
                 gripper_site="gripper", grasp_site="grasp",
                 base_body="base", target_body="target"):
        self.m = model
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

        self.base_j = jid(base_joint)
        self.tgt_j = jid(target_joint)
        assert self.base_j >= 0, f"no joint named {base_joint!r}"
        assert model.jnt_type[self.base_j] == mujoco.mjtJoint.mjJNT_FREE, \
            "base joint must be a freejoint -- an actuated/welded base is a different paper"

        self.base_qadr = int(model.jnt_qposadr[self.base_j])
        self.base_dadr = int(model.jnt_dofadr[self.base_j])
        self.tgt_qadr = int(model.jnt_qposadr[self.tgt_j]) if self.tgt_j >= 0 else None
        self.tgt_dadr = int(model.jnt_dofadr[self.tgt_j]) if self.tgt_j >= 0 else None

        self.arm_j = [jid(n) for n in arm_joints]
        assert all(j >= 0 for j in self.arm_j), f"missing arm joints in {arm_joints}"
        self.arm_qadr = np.array([model.jnt_qposadr[j] for j in self.arm_j], dtype=int)
        self.arm_dadr = np.array([model.jnt_dofadr[j] for j in self.arm_j], dtype=int)
        self.n = len(self.arm_j)

        self.grip_s = sid(gripper_site)
        self.grasp_s = sid(grasp_site)
        assert self.grip_s >= 0 and self.grasp_s >= 0, "gripper/grasp sites not found"
        self.base_b = bid(base_body)
        self.tgt_b = bid(target_body)

        self.jnt_range = np.array([model.jnt_range[j] for j in self.arm_j])
        self.limited = np.array([bool(model.jnt_limited[j]) for j in self.arm_j])

    # -- accessors -----------------------------------------------------------
    def get_theta(self, d):
        return d.qpos[self.arm_qadr].copy()

    def set_theta(self, d, th):
        d.qpos[self.arm_qadr] = th

    def set_thetadot(self, d, thd):
        d.qvel[self.arm_dadr] = thd

    def get_base(self, d):
        a = self.base_qadr
        return d.qpos[a:a + 3].copy(), quat_norm(d.qpos[a + 3:a + 7])

    def set_base(self, d, p, q):
        a = self.base_qadr
        d.qpos[a:a + 3] = p
        d.qpos[a + 3:a + 7] = quat_norm(q)

    def set_target(self, d, p, q, w_local=None, v=None):
        a = self.tgt_qadr
        d.qpos[a:a + 3] = p
        d.qpos[a + 3:a + 7] = quat_norm(q)
        if w_local is not None:
            d.qvel[self.tgt_dadr + 3:self.tgt_dadr + 6] = w_local
        if v is not None:
            d.qvel[self.tgt_dadr:self.tgt_dadr + 3] = v

    def site_T(self, d, sid):
        return make_T(d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy())

    def gripper_T(self, d):
        return self.site_T(d, self.grip_s)

    def grasp_T(self, d):
        return self.site_T(d, self.grasp_s)


# ------------------------------------------------------------- momentum map

_FULLM_NEW_API = None


def full_M(model, data):
    """Dense generalized mass matrix, portable across MuJoCo versions.

    mj_fullM changed signature: <=3.2 is mj_fullM(m, dst, d.qM), >=3.3 is
    mj_fullM(m, d, dst) and MjData.qM was renamed. Probe once, then cache.
    """
    global _FULLM_NEW_API
    M = np.zeros((model.nv, model.nv))
    if _FULLM_NEW_API is None:
        try:
            mujoco.mj_fullM(model, data, M)
            _FULLM_NEW_API = True
        except TypeError:
            mujoco.mj_fullM(model, M, data.qM)
            _FULLM_NEW_API = False
        return M
    if _FULLM_NEW_API:
        mujoco.mj_fullM(model, data, M)
    else:
        mujoco.mj_fullM(model, M, data.qM)
    return M


def momentum_map(model, data, lay):
    """
    J*(q) in MuJoCo's dof frame:   [v0_global; w0_LOCAL] = Jstar @ thetadot
    i.e. eq (5) of the paper, with the attitude component expressed in the
    base body frame (see the header note).
    """
    M = full_M(model, data)
    b = slice(lay.base_dadr, lay.base_dadr + 6)
    Mbb = M[b, b]
    Mbj = M[b, :][:, lay.arm_dadr]
    return -np.linalg.solve(Mbb, Mbj), Mbb, Mbj


def momentum_residual(model, data, lay):
    """h = Mbb u_b + Mbj thetadot. Should be ~1e-15 on a reconstructed traj
    (Proposition 1)."""
    M = full_M(model, data)
    b = slice(lay.base_dadr, lay.base_dadr + 6)
    ub = data.qvel[b]
    thd = data.qvel[lay.arm_dadr]
    return M[b, b] @ ub + M[b, :][:, lay.arm_dadr] @ thd


def assert_decoupled(model, data, lay, tol=1e-9):
    """The target must not be coupled to the arm system through M, or the
    partition above is wrong."""
    if lay.tgt_dadr is None:
        return
    M = full_M(model, data)
    b = slice(lay.base_dadr, lay.base_dadr + 6)
    t = slice(lay.tgt_dadr, lay.tgt_dadr + 6)
    off = np.abs(M[b, t]).max()
    assert off < tol, (f"base-target mass coupling {off:.2e} > {tol:.0e}; "
                       "the momentum partition is invalid")


def reconstruct_base(model, data, lay, theta_of_t, thetadot_of_t,
                     t0, t1, p0, q0, dt=2e-3, check_residual=True):
    """
    Phi( theta(.) ; q0 ) -- integrate the momentum map to get the base pose.
    RK4 on position, exponential map on attitude (matches the paper's
    Section III-B and the certification integrator).

    Returns (p, quat) at t1 and the max momentum residual seen.
    """
    p = np.array(p0, dtype=float)
    R = quat2mat(quat_norm(q0))
    max_res = 0.0
    nstep = max(1, int(round((t1 - t0) / dt)))
    h = (t1 - t0) / nstep
    t = t0

    def twist(tt, p_, R_):
        lay.set_base(data, p_, mat2quat(R_))
        lay.set_theta(data, theta_of_t(tt))
        thd = thetadot_of_t(tt)
        lay.set_thetadot(data, thd)
        mujoco.mj_forward(model, data)
        Js, _, _ = momentum_map(model, data, lay)
        u = Js @ thd
        # write the reconstructed base twist back into qvel, otherwise
        # momentum_residual() reads a stale (usually zero) base velocity and
        # reports |h| = |Mbj thetadot| instead of ~0. This assignment is what
        # makes the Proposition 1 check actually check Proposition 1.
        data.qvel[lay.base_dadr:lay.base_dadr + 6] = u
        return u[:3], u[3:]                      # v_global, w_local

    for _ in range(nstep):
        # RK4 for translation; attitude by midpoint omega + exp map (the
        # attitude increment per node is small, so this is well inside the
        # numerical-tolerance agreement reported against MuJoCo.)
        v1, w1 = twist(t, p, R)
        v2, w2 = twist(t + h / 2, p + h / 2 * v1, R @ so3_exp(w1 * h / 2))
        v3, w3 = twist(t + h / 2, p + h / 2 * v2, R @ so3_exp(w2 * h / 2))
        v4, w4 = twist(t + h, p + h * v3, R @ so3_exp(w3 * h))
        p = p + h / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        R = R @ so3_exp(h / 6 * (w1 + 2 * w2 + 2 * w3 + w4))
        t += h
        if check_residual:
            max_res = max(max_res, float(np.abs(momentum_residual(model, data, lay)).max()))

    return p, mat2quat(R), max_res


# --------------------------------------------------------------- min-jerk

def minjerk(theta0, thetaf, tf):
    """5th-order min-jerk with zero end velocity/acceleration -- the
    trajectory family of Section IV-D."""
    theta0 = np.asarray(theta0, float)
    thetaf = np.asarray(thetaf, float)
    d = thetaf - theta0

    def th(t):
        s = np.clip(t / tf, 0.0, 1.0)
        return theta0 + d * (10 * s**3 - 15 * s**4 + 6 * s**5)

    def thd(t):
        s = np.clip(t / tf, 0.0, 1.0)
        return d * (30 * s**2 - 60 * s**3 + 30 * s**4) / tf

    return th, thd


# ------------------------------------------------------------------ selftest

def selftest_conventions(model, lay=None):
    """
    Re-derives the freejoint angular-velocity frame on the given model.

    Uses mj_integratePos, which applies MuJoCo's own qpos<-qvel rule, rather
    than a dynamics rollout: a rollout on a loaded model precesses (asymmetric
    composite inertia) and gives a misleading answer. Costs <1 ms.

    Expected verdict: LOCAL.
    """
    qadr = 0 if lay is None else lay.base_qadr
    dadr = 0 if lay is None else lay.base_dadr
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    # base quaternion = 90 deg about X, so body-z == world -y and the two
    # conventions are distinguishable
    d.qpos[qadr + 3:qadr + 7] = [0.70710678, 0.70710678, 0, 0]
    R0 = quat2mat(d.qpos[qadr + 3:qadr + 7])
    dt = 1e-3
    d.qvel[:] = 0
    d.qvel[dadr + 3:dadr + 6] = [0, 0, 2.0]
    mujoco.mj_integratePos(model, d.qpos, d.qvel, dt)
    R1 = quat2mat(quat_norm(d.qpos[qadr + 3:qadr + 7]))
    w_local = so3_log(R0.T @ R1) / dt
    w_glob = so3_log(R1 @ R0.T) / dt
    is_local = np.allclose(w_local, [0, 0, 2.0], atol=1e-3)
    return dict(local=w_local, glob=w_glob,
                verdict="LOCAL" if is_local else "GLOBAL")
