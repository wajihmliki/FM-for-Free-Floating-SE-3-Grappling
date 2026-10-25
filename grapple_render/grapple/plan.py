"""plan.py -- target tumble rollout, collision screening, and the trajectory schema.

=====================  .npz SCHEMA (what render.py reads)  =====================
  t            (K,)      time stamps, seconds
  theta_fm     (K, n)    arm joint trajectory, momentum-projected warm start
  theta_cold   (K, n)    arm joint trajectory, recoil-naive warm start
  base_fm      (K, 7)    base pose [x y z qw qx qy qz]  (momentum map)
  base_cold    (K, 7)    base pose, recoil-naive theta executed free-floating
  target       (K, 7)    target body pose
  grasp        (K, 7)    grasp frame g(t) in world
  grip_fm      (K, 7)    gripper pose, momentum-projected
  grip_cold    (K, 7)    gripper pose, recoil-naive
  meta         json str  seed, tf, recoil_deg, reach_fm_cm, reach_cold_cm,
                         reach_fm_deg, reach_cold_deg, momentum_residual
==============================================================================
"""

import json
import numpy as np
import mujoco

from .core import mat2quat, quat_norm, assert_decoupled


# --------------------------------------------------------------- target roll

def roll_target(model, lay, p_t0, q_t0, w_local, tf, K):
    """Torque-free rigid-body tumble of the target, integrated by MuJoCo.

    Contacts are disabled for this rollout, so it is a true torque-free
    integration regardless of the arm configuration."""
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    lay.set_target(d, p_t0, q_t0, w_local=w_local)
    saved_flags = int(model.opt.disableflags)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    mujoco.mj_forward(model, d)
    assert_decoupled(model, d, lay)

    ts = np.linspace(0, tf, K)
    tgt = np.zeros((K, 7))
    grasp = np.zeros((K, 7))
    k, dt = 0, model.opt.timestep
    for step in range(int(round(tf / dt)) + 1):
        t = step * dt
        while k < K and ts[k] <= t + 1e-9:
            a = lay.tgt_qadr
            tgt[k] = np.concatenate([d.qpos[a:a + 3], quat_norm(d.qpos[a + 3:a + 7])])
            T = lay.grasp_T(d)
            grasp[k] = np.concatenate([T[:3, 3], mat2quat(T[:3, :3])])
            k += 1
        mujoco.mj_step(model, d)
        mujoco.mj_forward(model, d)
    model.opt.disableflags = saved_flags
    return ts, tgt, grasp


def min_clearance(model, lay, theta, base, target):
    """Worst signed penetration along a trajectory, in metres (negative = the
    arm is inside the target)."""
    d = mujoco.MjData(model)
    worst = 0.0
    for k in range(len(theta)):
        lay.set_base(d, base[k, :3], base[k, 3:])
        lay.set_theta(d, theta[k])
        a = lay.tgt_qadr
        d.qpos[a:a + 3] = target[k, :3]
        d.qpos[a + 3:a + 7] = quat_norm(target[k, 3:])
        mujoco.mj_forward(model, d)
        for i in range(d.ncon):
            worst = min(worst, float(d.contact[i].dist))
    return worst
