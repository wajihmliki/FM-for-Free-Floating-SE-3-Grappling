"""verify_prop2.py -- numerical check of Propositions 1 and 2 on held-out scenarios.

Prop. 1: momentum residual of the reconstructed trajectory (~machine precision).
Prop. 2: the recoil-naive (fixed-base IK) terminal error matches the displacement
caused by the neglected base recoil, and grows with recoil.

    python3 verify_prop2.py
"""
import numpy as np, ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from ffsm_reach_eval_se3 import _mj, cold_thf
p = fc3.FFSM3DParams(); N = 50; tf = 6.0; ease, de = _mj(N, tf)
print(f"{'recoil':>7} {'mom_resid':>11} {'cold_err':>9} {'recoil_disp':>12} {'match':>6}")
rows = []
for i in range(12):
    inst = d3.rand_inst3d(400000+i, N, tf, p); base0 = inst['base0']; _,_,gp,gq = inst['tgt'].frame(tf)
    b, be = d3.grapple3d(inst, branches=5, certify=False); dth = be['thf']-inst['th0']
    thd = np.outer(de, dth)
    # Prop 1: momentum residual of the reconstructed trajectory
    Q = fc3.rollout_reduced_se3_fast(base0, inst['th0']+np.outer(ease,dth), thd, tf/N, p, 4)
    resid = max(np.linalg.norm(fc3.momentum(Q[k],
              np.concatenate([fc3.base_vel_from_joints_fast(Q[k],thd[k],p), thd[k]]), p))
              for k in range(0, N+1, 5))
    recoil = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Q[-1,3:7], fc3.qconj(Q[0,3:7])))))
    # Prop 2: cold aims fixed-base; measure actual error vs recoil-induced displacement
    cth = cold_thf(inst); cdth = cth-inst['th0']
    ee_fixed = fc3.fk3d(np.concatenate([base0, cth]), p)['ee_pos']     # where cold THINKS it lands
    Qc = fc3.rollout_reduced_se3_fast(base0, inst['th0']+np.outer(ease,cdth), np.outer(de,cdth), tf/N, p, 4)
    ee_actual = fc3.fk3d(Qc[-1], p)['ee_pos']                          # where it ACTUALLY lands
    cold_err = np.linalg.norm(ee_actual-gp)*100
    recoil_disp = np.linalg.norm(ee_actual-ee_fixed)*100
    match = abs(cold_err-recoil_disp)/max(cold_err,1e-6) < 0.35
    rows.append((recoil, cold_err, recoil_disp))
    print(f"{recoil:>6.1f} {resid:>11.1e} {cold_err:>8.1f} {recoil_disp:>11.1f} {str(match):>6}")
r = np.array(rows)
print(f"\ncorrelation(recoil, cold_err) = {np.corrcoef(r[:,0], r[:,1])[0,1]:.2f}  (Prop 2: should be strongly positive)")
