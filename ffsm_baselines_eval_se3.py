"""ffsm_baselines_eval_se3.py -- Table I (warm-start reach quality) and Table III (tail).

Single unrefined sample per method on 500 held-out scenarios (seed0 400000):
Cold, NN, Regress, DDPM, FM (ours), Oracle. Each terminal posture -> min-jerk ease ->
momentum-consistent rollout -> terminal gripper error (cm) and orientation error (deg)
to the moving grasp frame, binned by base recoil. Also prints the median / p90 / max
tail (Table III), the NN near-vs-novel split, and the correlation between recoil and
Cold reach error. Raw per-scenario results are saved to reach_raw.npz (bootstrap_ci.py).

    python3 ffsm_baselines_eval_se3.py --ckpt ffsm_fm_se3.pt --ddpm_ckpt ffsm_ddpm_se3.pt \
        --regress_ckpt ffsm_regress_se3.pt --data ffsm_grapple_se3.pkl --n 500
"""

import argparse
import numpy as np
import torch
import ffsm_core_se3 as fc3
import ffsm_demo_se3 as d3
from ffsm_dataset_se3 import cond_vec
from ffsm_reach_eval_se3 import _mj, reach_of, cold_thf, oracle_thf, load as load_gen


def rot_of(inst, thf, ease, de, p, tf):
    dth = thf - inst['th0']
    Q = fc3.rollout_reduced_se3_fast(
        inst['base0'],
        inst['th0'] + np.outer(ease, dth),
        np.outer(de, dth),
        tf / (len(ease) - 1),
        p,
        4,
    )
    _, _, _, gq = inst['tgt'].frame(tf)
    return np.degrees(
        np.linalg.norm(fc3.log_so3(fc3.qmul(fc3.qconj(fc3.ee_quat(Q[-1], p)), gq)))
    )


def nn_setup(data_pkl):
    import pickle
    d = pickle.load(open(data_pkl, 'rb'))
    C = d['cond'].astype(np.float64)
    TH = d['theta'].astype(np.float64)  # (M,24),(M,T,n)
    mu = C.mean(0)
    sd = C.std(0) + 1e-9
    return (C - mu) / sd, TH[:, -1, :], mu, sd  # normalized cond, terminal joints


def nn_query(cn, thf_bank, mu, sd, cond_vec_query):
    q = (cond_vec_query - mu) / sd
    dists = np.linalg.norm(cn - q[None], axis=1)
    j = int(np.argmin(dists))
    return thf_bank[j], float(dists[j])  # nearest terminal joints + distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ffsm_fm_se3.pt')
    ap.add_argument('--ddpm_ckpt', default='ffsm_ddpm_se3.pt')
    ap.add_argument('--regress_ckpt', default='ffsm_regress_se3.pt')
    ap.add_argument('--data', default='ffsm_grapple_se3.pkl')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--tf', type=float, default=6.0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed0', type=int, default=400000)
    a = ap.parse_args()

    from ffsm_fm_se3 import sample_theta
    from ffsm_ddpm_se3 import sample_theta_ddpm
    from ffsm_regress_se3 import sample_regress

    net, ck = load_gen(a.ckpt, a.device)
    N = ck['seq_len'] - 1
    p = fc3.FFSM3DParams()
    ease, de = _mj(N, a.tf)
    dnet, dck = load_gen(a.ddpm_ckpt, a.device)
    rnet, rck = load_gen(a.regress_ckpt, a.device)
    cn_bank, thf_bank, mu, sd = nn_setup(a.data)

    insts = [d3.rand_inst3d(a.seed0 + i, N, a.tf, p) for i in range(a.n)]
    conds_np = np.stack([cond_vec(i) for i in insts])
    conds = torch.from_numpy(conds_np).float().to(a.device)

    fm = sample_theta(
        net,
        conds,
        ck['stats'],
        seq_len=ck['seq_len'],
        device=a.device,
        th0_slice=(0, ck['dim']),
    ).cpu().numpy()
    dd = sample_theta_ddpm(
        dnet,
        conds,
        dck['stats'],
        seq_len=dck['seq_len'],
        device=a.device,
        th0_slice=(0, dck['dim']),
    ).cpu().numpy()
    rg = sample_regress(
        rnet,
        conds,
        rck['stats'],
        seq_len=rck['seq_len'],
        device=a.device,
        th0_slice=(0, rck['dim']),
    ).cpu().numpy()

    # Columns recorded:
    # 0: recoil
    # 1..6: position errors (cold, NN, regress, DDPM, FM, oracle)
    # 7..12: orientation errors (cold, NN, regress, DDPM, FM, oracle)
    # 13: nn_dist
    rows = []
    for j, inst in enumerate(insts):
        # Joint configurations
        thf_cold = cold_thf(inst)
        thf_nn, nn_d = nn_query(cn_bank, thf_bank, mu, sd, conds_np[j])
        thf_rg = rg[j][-1]
        thf_dd = dd[j][-1]
        thf_fm = fm[j][-1]
        thf_orc = oracle_thf(inst, ease, de)

        # Position errors
        pe_fm, _, rec = reach_of(inst, thf_fm, ease, de)
        pe_cold, _, _ = reach_of(inst, thf_cold, ease, de)
        pe_orc, _, _ = reach_of(inst, thf_orc, ease, de)
        pe_dd, _, _ = reach_of(inst, thf_dd, ease, de)
        pe_rg, _, _ = reach_of(inst, thf_rg, ease, de)
        pe_nn, _, _ = reach_of(inst, thf_nn, ease, de)

        # Orientation errors (deg)
        re_cold = rot_of(inst, thf_cold, ease, de, p, a.tf)
        re_nn = rot_of(inst, thf_nn, ease, de, p, a.tf)
        re_rg = rot_of(inst, thf_rg, ease, de, p, a.tf)
        re_dd = rot_of(inst, thf_dd, ease, de, p, a.tf)
        re_fm = rot_of(inst, thf_fm, ease, de, p, a.tf)
        re_orc = rot_of(inst, thf_orc, ease, de, p, a.tf)

        rows.append((
            rec,
            pe_cold,
            pe_nn,
            pe_rg,
            pe_dd,
            pe_fm,
            pe_orc,
            re_cold,
            re_nn,
            re_rg,
            re_dd,
            re_fm,
            re_orc,
            nn_d,
        ))

    out = np.array(rows)
    np.savez('reach_raw.npz', out=out)

    edges = [0, 15, 25, 35, 200]
    print(
        f"\n=== SE(3) six-way warm-start reach (median gripper->target cm), single sample, n={a.n} ==="
    )
    print(
        f"{'recoil':>10} {'#':>4} {'cold':>7} {'NN':>7} {'regress':>8} {'DDPM':>7} {'FM':>7} {'oracle':>7}"
    )
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (out[:, 0] >= lo) & (out[:, 0] < hi)
        if m.sum() == 0:
            continue
        med = [np.median(out[m, k]) for k in (1, 2, 3, 4, 5, 6)]
        print(
            f"{f'[{lo},{hi})':>10} {int(m.sum()):>4} {med[0]:>7.1f} {med[1]:>7.1f} {med[2]:>8.1f} {med[3]:>7.1f} {med[4]:>7.1f} {med[5]:>7.1f}"
        )
    med = [np.median(out[:, k]) for k in (1, 2, 3, 4, 5, 6)]
    print(
        f"{'ALL':>10} {a.n:>4} {med[0]:>7.1f} {med[1]:>7.1f} {med[2]:>8.1f} {med[3]:>7.1f} {med[4]:>7.1f} {med[5]:>7.1f}"
    )

    print(
        f"\n=== SE(3) six-way warm-start orientation error (median deg), single sample, n={a.n} ==="
    )
    print(
        f"{'recoil':>10} {'#':>4} {'cold':>7} {'NN':>7} {'regress':>8} {'DDPM':>7} {'FM':>7} {'oracle':>7}"
    )
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (out[:, 0] >= lo) & (out[:, 0] < hi)
        if m.sum() == 0:
            continue
        med = [np.median(out[m, k]) for k in (7, 8, 9, 10, 11, 12)]
        print(
            f"{f'[{lo},{hi})':>10} {int(m.sum()):>4} {med[0]:>7.1f} {med[1]:>7.1f} {med[2]:>8.1f} {med[3]:>7.1f} {med[4]:>7.1f} {med[5]:>7.1f}"
        )
    med = [np.median(out[:, k]) for k in (7, 8, 9, 10, 11, 12)]
    print(
        f"{'ALL':>10} {a.n:>4} {med[0]:>7.1f} {med[1]:>7.1f} {med[2]:>8.1f} {med[3]:>7.1f} {med[4]:>7.1f} {med[5]:>7.1f}"
    )

    # NN novel-vs-near split (median NN distance as the boundary)
    thr = np.median(out[:, 13])
    near = out[:, 13] <= thr
    novel = ~near
    print(
        f"\n--- generalization check: FM vs NN, split at NN-distance median ({thr:.2f}) ---"
    )
    for name, msk in [('NEAR  training', near), ('NOVEL (far)   ', novel)]:
        if msk.sum() == 0:
            continue
        print(
            f"  {name}: n={int(msk.sum()):3d}   NN {np.median(out[msk,2]):5.1f}cm   FM {np.median(out[msk,5]):5.1f}cm   regress {np.median(out[msk,3]):5.1f}cm"
        )
    print(f"\ncorrelation(recoil, cold reach error) r = {np.corrcoef(out[:, 0], out[:, 1])[0, 1]:.2f}")

    # tail comparison (p90)
    print("\n--- tail (90th percentile reach, cm) ---")
    for k, nm in [(3, 'regress'), (4, 'DDPM'), (5, 'FM')]:
        print(
            f"  {nm:>8}: median {np.median(out[:,k]):5.1f}   p90 {np.percentile(out[:,k],90):5.1f}   max {out[:,k].max():5.1f}"
        )


if __name__ == '__main__':
    main()
