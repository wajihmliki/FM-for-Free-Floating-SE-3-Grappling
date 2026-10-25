#!/usr/bin/env python3
"""run_selftest.py -- five checks on the render model before trusting any rendered pose:
freejoint angular-velocity frame, unactuated base, zero gravity, base-target mass
decoupling, and Proposition 1 (momentum residual of reconstructed trajectories).

    MUJOCO_GL=egl python run_selftest.py --model ffsm_render.xml
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ffsm_render.xml")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from grapple.core import (Layout, selftest_conventions, full_M, momentum_map,
                              momentum_residual, assert_decoupled, minjerk,
                              reconstruct_base, quat2mat, so3_log)

    m = mujoco.MjModel.from_xml_path(a.model)
    lay = Layout(m)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok: fails += 1

    print("\n1. freejoint angular-velocity frame")
    st = selftest_conventions(m, lay)
    check("qvel[3:6] is body-local", st["verdict"] == "LOCAL",
          f"measured local={np.round(st['local'],4)}")

    print("\n2. base is unactuated")
    base_dofs = set(range(lay.base_dadr, lay.base_dadr + 6))
    driven = set()
    for i in range(m.nu):
        t = m.actuator_trntype[i]
        if t == mujoco.mjtTrn.mjTRN_JOINT:
            j = m.actuator_trnid[i, 0]
            driven.update(range(m.jnt_dofadr[j], m.jnt_dofadr[j] + 6
                                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else
                                m.jnt_dofadr[j] + 1))
    check("no actuator drives a base dof", not (base_dofs & driven))

    print("\n3. zero gravity, no external wrench")
    check("gravity is zero", float(np.linalg.norm(m.opt.gravity)) < 1e-12,
          f"g={np.array(m.opt.gravity)}")

    print("\n4. base-target mass decoupling (momentum partition validity)")
    try:
        assert_decoupled(m, d, lay); check("M[base, target] == 0", True)
    except AssertionError as e:
        check("M[base, target] == 0", False, str(e))

    print("\n5. Proposition 1: momentum conserved by construction")
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(3):
        th0 = rng.uniform(-0.8, 0.8, lay.n)
        thf = th0 + rng.uniform(-1.2, 1.2, lay.n)
        th, thd = minjerk(th0, thf, 6.0)
        _, _, res = reconstruct_base(m, d, lay, th, thd, 0.0, 6.0,
                                     np.zeros(3), np.array([1.,0,0,0]),
                                     dt=5e-3, check_residual=True)
        worst = max(worst, res)
    check("max |h(q,u)| < 1e-10 over random joint trajectories",
          worst < 1e-10, f"max |h| = {worst:.2e}")

    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else str(fails) + ' CHECK(S) FAILED'}\n")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
