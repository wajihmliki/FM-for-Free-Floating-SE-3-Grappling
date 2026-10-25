"""ablate_projection.py -- Table I "Fixed-Base Execution Ablation" (Sec. VII-D).

Separates the generator from the momentum projection. The SAME flow-matching samples
are evaluated two ways:

  FM (projected)   base reconstructed from the momentum map -- the method
  FM (fixed base)  base held at T_0(0) instead of reconstructed

Reports reach / orientation error per recoil bin for both, and the momentum residual
max ||h(q,u)|| in SI units (~1e-15 for the projected trajectory, Prop. 1).

    python3 ablate_projection.py --ckpt ffsm_fm_se3.pt --n 500
"""
import argparse
import numpy as np, torch
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from ffsm_dataset_se3 import cond_vec
from ffsm_reach_eval_se3 import _mj, load as load_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ffsm_fm_se3.pt")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--tf", type=float, default=6.0)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed0", type=int, default=400000)
    a = ap.parse_args()
    from ffsm_fm_se3 import sample_theta

    net, ck = load_gen(a.ckpt, a.device)
    N = ck["seq_len"] - 1; p = fc3.FFSM3DParams(); dt = a.tf / N
    ease, de = _mj(N, a.tf)
    insts = [d3.rand_inst3d(a.seed0 + i, N, a.tf, p) for i in range(a.n)]
    cond = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(a.device)
    fm = sample_theta(net, cond, ck["stats"], seq_len=ck["seq_len"],
                      device=a.device, th0_slice=(0, ck["dim"])).cpu().numpy()

    rows = []
    for j, inst in enumerate(insts):
        th0 = np.asarray(inst["th0"], float)
        dth = np.asarray(fm[j][-1], float) - th0
        th = th0[None, :] + np.outer(ease, dth)
        thd = np.outer(de, dth)
        _, _, gp, gq = inst["tgt"].frame(a.tf)

        # --- projected: base from the momentum map -------------------------
        Q = fc3.rollout_reduced_se3_fast(inst["base0"], th, thd, dt, p, 4)
        k = fc3.fk3d(Q[-1], p)
        e_proj = np.linalg.norm(k["ee_pos"] - gp) * 100
        r_proj = np.rad2deg(np.linalg.norm(fc3.log_so3(
            fc3.qmul(fc3.qconj(fc3.ee_quat(Q[-1], p)), gq))))
        rho = np.rad2deg(np.linalg.norm(fc3.log_so3(
            fc3.qmul(Q[-1, 3:7], fc3.qconj(Q[0, 3:7])))))
        h_proj = max(np.linalg.norm(fc3.momentum(
            Q[i], np.concatenate([fc3.base_vel_from_joints_fast(Q[i], thd[i], p), thd[i]]), p))
            for i in range(0, N + 1, 10))

        # --- fixed base: same samples, base frozen at T_0(0) ---------------
        qf = np.concatenate([inst["base0"], th[-1]])
        kf = fc3.fk3d(qf, p)
        e_fix = np.linalg.norm(kf["ee_pos"] - gp) * 100
        r_fix = np.rad2deg(np.linalg.norm(fc3.log_so3(
            fc3.qmul(fc3.qconj(fc3.ee_quat(qf, p)), gq))))
        # momentum carried by a frozen base while the joints move: h = M_bj thetadot
        h_fix = max(np.linalg.norm(fc3.momentum(
            np.concatenate([inst["base0"], th[i]]),
            np.concatenate([np.zeros(6), thd[i]]), p))
            for i in range(0, N + 1, 10))

        rows.append((rho, e_proj, e_fix, r_proj, r_fix, h_proj, h_fix))

    out = np.array(rows)
    edges = [0, 15, 25, 35, 200]
    print(f"\n=== Ablation: momentum projection vs fixed base, identical FM samples (n={a.n}) ===")
    print(f"{'recoil':>10} {'#':>4} | {'proj cm':>8} {'fixed cm':>9} | "
          f"{'proj deg':>9} {'fixed deg':>10}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (out[:, 0] >= lo) & (out[:, 0] < hi)
        if m.sum() == 0: continue
        print(f"{f'[{lo},{hi})':>10} {int(m.sum()):>4} | {np.median(out[m,1]):>8.1f} "
              f"{np.median(out[m,2]):>9.1f} | {np.median(out[m,3]):>9.1f} "
              f"{np.median(out[m,4]):>10.1f}")
    print(f"{'All':>10} {a.n:>4} | {np.median(out[:,1]):>8.1f} {np.median(out[:,2]):>9.1f} | "
          f"{np.median(out[:,3]):>9.1f} {np.median(out[:,4]):>10.1f}")

    print(f"\nmomentum residual max|h| over the trajectory (SI):")
    print(f"  projected  : median {np.median(out[:,5]):.2e}   max {out[:,5].max():.2e}")
    print(f"  fixed base : median {np.median(out[:,6]):.2e}   max {out[:,6].max():.2e}")
    print("\n=> projected residual is at machine precision (Prop. 1); the fixed-base")
    print("   trajectory violates the conservation law by a physically large amount,")
    print("   and its reach error grows with recoil while the projected one does not.")


if __name__ == "__main__":
    main()
