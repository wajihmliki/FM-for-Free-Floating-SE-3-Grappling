"""ffsm_ood_eval_se3.py -- Table VI: out-of-distribution reach (200 scenarios, no retraining).

Pushes held-out scenarios beyond the training ranges along the physically meaningful
axes (tumble rate, target distance) and reports the median single-sample reach of FM
and Cold, plus the fraction of grasp points that lie inside the arm's kinematic
workspace (|base_attach| + sum |l_i| from the base origin).

    python3 ffsm_ood_eval_se3.py --ckpt ffsm_fm_se3.pt --data ffsm_grapple_se3.pkl --n 200
"""
import argparse, pickle, numpy as np, torch
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from ffsm_dataset_se3 import cond_vec
from ffsm_reach_eval_se3 import _mj, reach_of, cold_thf, load as load_gen


def bounds_from_data(data_pkl, n_layout):
    """Infer training ranges for tumble |omega| and target distance from the dataset conditioning."""
    d = pickle.load(open(data_pkl, 'rb')); C = d['cond']
    # layout: th0(n) center(3) quat0(4) omega(3) grasp_offset(3) grasp_quat(4) tf(1)
    n = n_layout
    center = C[:, n:n+3]; omega = C[:, n+7:n+10]
    dist = np.linalg.norm(center, axis=1); wmag = np.linalg.norm(omega, axis=1)
    return dict(dist=(dist.min(), dist.max()), wmag=(wmag.min(), wmag.max()))


def make_ood_inst(seed, N, tf, p, dist_scale=1.0, omega_scale=1.0, base_bounds=None):
    """Draw a scenario then push it OOD: scale the target distance and tumble beyond training."""
    inst = d3.rand_inst3d(seed, N, tf, p); tgt = inst['tgt']
    # push tumble rate
    tgt.omega = tgt.omega * omega_scale
    # push target distance (scale center outward from base)
    c = tgt.p0.copy(); r = np.linalg.norm(c)
    if r > 1e-6:
        tgt.p0 = c * dist_scale
    inst['tgt'] = tgt
    return inst


def eval_arm(insts, thfs, ease, de):
    errs = []
    for inst, thf in zip(insts, thfs):
        errs.append(reach_of(inst, thf, ease, de)[0])
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ffsm_fm_se3.pt'); ap.add_argument('--data', default='ffsm_grapple_se3.pkl')
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--tf', type=float, default=6.0)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed0', type=int, default=600000)
    a = ap.parse_args()
    from ffsm_fm_se3 import sample_theta
    net, ck = load_gen(a.ckpt, a.device); N = ck['seq_len']-1; p = fc3.FFSM3DParams(); ease, de = _mj(N, a.tf)
    n = p.n
    b = bounds_from_data(a.data, n)
    print(f"training bounds: target dist {b['dist'][0]:.2f}-{b['dist'][1]:.2f} m, "
          f"tumble |omega| {b['wmag'][0]:.3f}-{b['wmag'][1]:.3f} rad/s")

    # conditions: in-distribution control, then OOD along each axis (1.5x, 2x beyond)
    configs = [
        ("In-distribution", 1.0, 1.0),
        ("Tumble x2",       1.0, 2.0),
        ("Tumble x3",       1.0, 3.0),
        ("Distance x1.15",  1.15, 1.0),
        ("Distance x1.40",  1.40, 1.0),
    ]
    max_reach = float(np.linalg.norm(p.base_attach) + np.sum(np.linalg.norm(p.l, axis=1)))
    print(f"max reach from base origin = |base_attach| + sum|l_i| = {max_reach:.3f} m")
    print(f"\n{'condition':>18} {'FM_cm':>7} {'cold_cm':>8} {'FM_p90':>7} {'within_reach':>13}")
    for name, ds, os_ in configs:
        insts = [make_ood_inst(a.seed0+i, N, a.tf, p, ds, os_) for i in range(a.n)]
        conds = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(a.device)
        fm = sample_theta(net, conds, ck['stats'], seq_len=ck['seq_len'], device=a.device, th0_slice=(0, ck['dim'])).cpu().numpy()
        fm_err = eval_arm(insts, [fm[j][-1] for j in range(a.n)], ease, de)
        cold_err = eval_arm(insts, [cold_thf(i) for i in insts], ease, de)
        within = np.mean([np.linalg.norm(i['tgt'].frame(a.tf)[2] - i['base0'][:3]) <= max_reach
                          for i in insts])
        print(f"{name:>18} {np.median(fm_err):>7.1f} {np.median(cold_err):>8.1f} "
              f"{np.percentile(fm_err,90):>7.1f} {100*within:>12.0f}%")


if __name__ == '__main__':
    main()
