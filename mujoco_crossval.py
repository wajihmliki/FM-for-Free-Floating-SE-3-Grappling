"""mujoco_crossval.py -- independent-simulator cross-validation (Sec. VII-F).

Applies the same joint-torque sequence from rest in MuJoCo and in integrate_true3d and
compares base recoil (position + orientation), joint angles and the gripper path.

    python3 mujoco_crossval.py --seed 400002 --N 40
"""
import argparse, numpy as np
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from build_mjcf_se3 import build_model


def mujoco_rollout(tau_seq, base0, th0, N, dt, p):
    import mujoco
    model, meta = build_model(p); data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
    # initial condition: base pose + joints, zero velocity
    data.qpos[:3] = base0[:3]; data.qpos[3:7] = base0[3:7]; data.qpos[7:] = th0; data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    n_sub = max(1, int(round(dt/model.opt.timestep)))
    Q = np.zeros((N+1, 7+p.n)); EE = np.zeros((N+1, 3))
    Q[0, :3] = data.qpos[:3]; Q[0, 3:7] = data.qpos[3:7]; Q[0, 7:] = data.qpos[7:]; EE[0] = data.site_xpos[site]
    for k in range(N):
        data.ctrl[:] = tau_seq[k]
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        Q[k+1, :3] = data.qpos[:3]; Q[k+1, 3:7] = data.qpos[3:7]; Q[k+1, 7:] = data.qpos[7:]
        EE[k+1] = data.site_xpos[site]
    return Q, EE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=400002); ap.add_argument('--N', type=int, default=40)
    ap.add_argument('--tf', type=float, default=6.0); ap.add_argument('--substeps', type=int, default=12)
    a = ap.parse_args()
    p = fc3.FFSM3DParams(); dt = a.tf/a.N
    inst = d3.rand_inst3d(a.seed, a.N, a.tf, p); base0 = inst['base0']; th0 = inst['th0']
    # a realistic torque: from a demonstrator reaching trajectory
    from ffsm_reach_eval_se3 import _mj
    ease, de = _mj(a.N, a.tf); s = np.linspace(0,1,a.N+1)
    dde = (60*s-180*s**2+120*s**3)/a.tf**2
    b, be = d3.grapple3d(inst, branches=5, certify=False); dth = be['thf']-th0
    th = th0+np.outer(ease, dth); thd = np.outer(de, dth); thdd = np.outer(dde, dth)
    _, _, tau, _ = fc3.inverse_dynamics3d(th, thd, thdd, base0, dt, p)

    # reference: the certificate's true-dynamics integrator
    q0 = np.concatenate([base0, th0])
    Q_ref, U_ref = fc3.integrate_true3d(q0, np.zeros(p.ndof), tau[:a.N], dt, p, substeps=a.substeps)
    EE_ref = np.array([fc3.fk3d(Q_ref[k], p)['ee_pos'] for k in range(a.N+1)])

    # mujoco
    Q_mj, EE_mj = mujoco_rollout(tau, base0, th0, a.N, dt, p)

    # compare
    base_pos_err = np.linalg.norm(Q_ref[:, :3]-Q_mj[:, :3], axis=1)
    base_ang_err = np.array([np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(fc3.qconj(Q_ref[k,3:7]), Q_mj[k,3:7]))))
                             for k in range(a.N+1)])
    ee_err = np.linalg.norm(EE_ref-EE_mj, axis=1)
    joint_err = np.abs(Q_ref[:, 7:]-Q_mj[:, 7:]).max(axis=1)
    recoil_ref = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Q_ref[-1,3:7], fc3.qconj(Q_ref[0,3:7])))))
    recoil_mj  = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Q_mj[-1,3:7],  fc3.qconj(Q_mj[0,3:7])))))

    print(f"=== MuJoCo vs integrate_true3d (seed {a.seed}, N={a.N}) ===")
    print(f"  base recoil:  ref {recoil_ref:.2f} deg   mujoco {recoil_mj:.2f} deg")
    print(f"  max base-position error : {base_pos_err.max()*100:.2f} cm")
    print(f"  max base-orientation err: {base_ang_err.max():.2f} deg")
    print(f"  max joint-angle error   : {np.rad2deg(joint_err.max()):.2f} deg")
    print(f"  max gripper-path error  : {ee_err.max()*100:.2f} cm   (final {ee_err[-1]*100:.2f} cm)")
    ok = base_pos_err.max() < 0.03 and base_ang_err.max() < 2.0 and ee_err.max() < 0.05
    print("  ->", "MATCH" if ok else "MISMATCH (check joint axis frames, base_attach / link nesting, inertias)")


if __name__ == '__main__':
    main()
