"""ffsm_reach_eval_se3.py -- shared reach evaluation utilities (single sample, no refiner).

For a terminal joint posture: min-jerk ease from theta_0 -> momentum-consistent rollout
-> terminal gripper error to the moving SE(3) grasp frame, plus the base recoil
rho = ||log(R0(0)^T R0(tf))||. Also defines the Cold (fixed-base IK) and Oracle
(momentum-aware IK) terminal postures and the checkpoint loader used by every script.

    python3 ffsm_reach_eval_se3.py --ckpt ffsm_fm_se3.pt --ddpm_ckpt ffsm_ddpm_se3.pt --n 500
"""
import argparse, numpy as np, torch
from scipy.optimize import least_squares
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from ffsm_dataset_se3 import cond_vec


def _mj(N, tf):
    s = np.linspace(0, 1, N+1); return 10*s**3-15*s**4+6*s**5, (30*s**2-60*s**3+30*s**4)/tf


def reach_of(inst, thf, ease, de):
    p, N, tf, dt = inst['p'], inst['N'], inst['tf'], inst['dt']
    _, _, gp, gq = inst['tgt'].frame(tf); dth = thf - inst['th0']
    Q = fc3.rollout_reduced_se3_fast(inst['base0'], inst['th0'][None,:]+np.outer(ease, dth), np.outer(de, dth), dt, p, 2)
    k = fc3.fk3d(Q[-1], p)
    pe = np.linalg.norm(k['ee_pos']-gp)*100
    ae = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(fc3.qconj(fc3.ee_quat(Q[-1], p)), gq))))
    recoil = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Q[-1,3:7], fc3.qconj(Q[0,3:7])))))
    return pe, ae, recoil


def cold_thf(inst):
    p, tf, base0, th0 = inst['p'], inst['tf'], inst['base0'], inst['th0']; _, _, gp, gq = inst['tgt'].frame(tf)
    def ee(t): q = np.concatenate([base0, t]); k = fc3.fk3d(q, p); return k['ee_pos'], fc3.ee_quat(q, p)
    return least_squares(lambda t: np.concatenate([ee(t)[0]-gp, 0.5*fc3.log_so3(fc3.qmul(fc3.qconj(ee(t)[1]), gq))]),
                         th0, bounds=(-2.7, 2.7), max_nfev=120).x


def oracle_thf(inst, ease, de):
    p, N, tf, base0, th0 = inst['p'], inst['N'], inst['tf'], inst['base0'], inst['th0']; _, _, gp, gq = inst['tgt'].frame(tf)
    def feat(t):
        th = th0+np.outer(ease, t-th0); thd = np.outer(de, t-th0)
        Q = fc3.rollout_reduced_se3_fast(base0, th, thd, tf/N, p, 2); k = fc3.fk3d(Q[-1], p); return k['ee_pos'], fc3.ee_quat(Q[-1], p)
    return least_squares(lambda t: np.concatenate([feat(t)[0]-gp, 0.5*fc3.log_so3(fc3.qmul(fc3.qconj(feat(t)[1]), gq))]),
                         th0, bounds=(-2.7, 2.7), max_nfev=120).x


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    from ffsm_fm_se3 import FFSM3DVelocityNet
    net = FFSM3DVelocityNet(seq_len=ck['seq_len'], cond_dim=ck['cond_dim'], dim=ck['dim'],
                            hidden=ck['hidden'], n_layers=ck['n_layers'], n_heads=ck['n_heads'])
    net.load_state_dict(ck['state_dict']); net.to(device).eval(); return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ffsm_fm_se3.pt'); ap.add_argument('--ddpm_ckpt', default=None)
    ap.add_argument('--n', type=int, default=500); ap.add_argument('--tf', type=float, default=6.0)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed0', type=int, default=400000)
    a = ap.parse_args()
    from ffsm_fm_se3 import sample_theta
    net, ck = load(a.ckpt, a.device); N = ck['seq_len']-1; p = fc3.FFSM3DParams(); ease, de = _mj(N, a.tf)
    insts = [d3.rand_inst3d(a.seed0+i, N, a.tf, p) for i in range(a.n)]
    conds = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(a.device)
    fm = sample_theta(net, conds, ck['stats'], seq_len=ck['seq_len'], device=a.device, th0_slice=(0, ck['dim'])).cpu().numpy()
    dd = None
    if a.ddpm_ckpt:
        from ffsm_ddpm_se3 import sample_theta_ddpm
        dnet, dck = load(a.ddpm_ckpt, a.device)
        dd = sample_theta_ddpm(dnet, conds, dck['stats'], seq_len=dck['seq_len'], device=a.device, th0_slice=(0, dck['dim'])).cpu().numpy()
    rows = []
    for j, inst in enumerate(insts):
        pe_fm, _, rec = reach_of(inst, fm[j][-1], ease, de)
        pe_cold, _, _ = reach_of(inst, cold_thf(inst), ease, de)
        pe_orc, _, _ = reach_of(inst, oracle_thf(inst, ease, de), ease, de)
        pe_dd = reach_of(inst, dd[j][-1], ease, de)[0] if dd is not None else np.nan
        rows.append((rec, pe_cold, pe_dd, pe_fm, pe_orc))
    out = np.array(rows)
    edges = [0, 15, 25, 35, 200]; has_dd = dd is not None
    print(f"\n=== SE(3) warm-start REACH quality (median gripper->target, cm), single sample, n={a.n} ===")
    hdr = f"{'recoil bin':>12} {'#':>5} {'cold':>8}" + (f" {'DDPM':>8}" if has_dd else "") + f" {'FM':>8} {'oracle':>8}"
    print(hdr)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (out[:,0]>=lo)&(out[:,0]<hi)
        if m.sum()==0: continue
        row = f"{f'[{lo},{hi})':>12} {int(m.sum()):>5} {np.median(out[m,1]):>7.1f}"
        if has_dd: row += f" {np.median(out[m,2]):>7.1f}"
        row += f" {np.median(out[m,3]):>7.1f} {np.median(out[m,4]):>7.1f}"; print(row)
    allrow = f"{'ALL':>12} {a.n:>5} {np.median(out[:,1]):>7.1f}"
    if has_dd: allrow += f" {np.median(out[:,2]):>7.1f}"
    allrow += f" {np.median(out[:,3]):>7.1f} {np.median(out[:,4]):>7.1f}"; print(allrow)
    print("\n=> FM & DDPM should be small and flat (learned recoil-aware); cold grows with recoil; oracle ~0.")


if __name__ == '__main__':
    main()
