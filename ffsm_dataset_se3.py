"""ffsm_dataset_se3.py -- certified SE(3) grapple demonstrations (training set).

Uses the demonstrator (ffsm_demo_se3.grapple3d: momentum-aware multi-start IK) and keeps
only trajectories that pass the full-integration certificate (verify3d). The flow model
generates the JOINT trajectory; the 6-DoF base is reconstructed from the momentum map, so
demos store joints + base pose + the conditioning needed to rebuild the scenario.

Saved (pickle):
  theta  (n_demos, N+1, n)     joint trajectory (the generation target)
  base   (n_demos, N+1, 7)     base pose [p0(3), quat0(4)]
  tau    (n_demos, N+1, n)     commanded joint torque
  cond   (n_demos, 24)         [th0(n), tgt_center(3), tgt_quat0(4), tgt_omega(3),
                                grasp_offset(3), grasp_quat(4), tf(1)]
  meta   dict (N, tf, n, cond_layout)

    python3 ffsm_dataset_se3.py --n_unique 1200 --N 60 --tf 6.0 --workers 16 --out ffsm_grapple_se3.pkl
"""
import argparse, pickle
from functools import partial
import numpy as np
import ffsm_core_se3 as fc3
import ffsm_demo_se3 as d3


def cond_vec(inst):
    t = inst['tgt']
    return np.concatenate([inst['th0'], t.p0, t.quat0, t.omega, t.grasp_offset, t.grasp_quat,
                           [inst['tf']]]).astype(np.float32)


def worker(seed, N, tf, branches, substeps):
    p = fc3.FFSM3DParams()
    inst = d3.rand_inst3d(seed, N, tf, p)
    best, _ = d3.grapple3d(inst, branches=branches, substeps=substeps, certify=True)
    if best is None or not best.get('feasible', False):
        return None
    # re-integrate the certified torque to store exactly what the gate saw
    q0 = np.concatenate([inst['base0'], inst['th0']])
    Q, U = fc3.integrate_true3d(q0, np.zeros(p.ndof), best['tau'][:N], tf/N, p, substeps=substeps)
    theta = Q[:, 7:].astype(np.float32)
    base = Q[:, :7].astype(np.float32)
    tau = np.vstack([best['tau'][:N], best['tau'][N-1:N]]).astype(np.float32)
    return dict(theta=theta, base=base, tau=tau, cond=cond_vec(inst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_unique', type=int, default=1200)
    ap.add_argument('--N', type=int, default=60); ap.add_argument('--tf', type=float, default=6.0)
    ap.add_argument('--branches', type=int, default=8); ap.add_argument('--substeps', type=int, default=10)
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--seed0', type=int, default=11)
    ap.add_argument('--max_attempts', type=int, default=0); ap.add_argument('--out', default='ffsm_grapple_se3.pkl')
    a = ap.parse_args()
    max_attempts = a.max_attempts or a.n_unique * 3
    fn = partial(worker, N=a.N, tf=a.tf, branches=a.branches, substeps=a.substeps)
    seeds = list(range(a.seed0, a.seed0 + max_attempts)); demos = []; attempted = 0
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for r in pool.imap_unordered(fn, seeds, chunksize=1):
                attempted += 1
                if r is not None: demos.append(r)
                if len(demos) >= a.n_unique: break
                if attempted % 20 == 0:
                    print(f"  {len(demos)}/{a.n_unique} certified  ({attempted} attempts, "
                          f"{len(demos)/max(attempted,1):.0%})")
    else:
        for s in seeds:
            r = fn(s); attempted += 1
            if r is not None: demos.append(r)
            if len(demos) >= a.n_unique: break
    if not demos:
        print("NO SE(3) demos certified."); return
    data = dict(theta=np.stack([d['theta'] for d in demos]), base=np.stack([d['base'] for d in demos]),
                tau=np.stack([d['tau'] for d in demos]), cond=np.stack([d['cond'] for d in demos]),
                meta=dict(N=a.N, tf=a.tf, n=fc3.FFSM3DParams().n, so3=True,
                          cond_layout='th0(n) tgt_center(3) tgt_quat0(4) tgt_omega(3) grasp_offset(3) grasp_quat(4) tf(1)',
                          generator='ffsm_grapple_se3_verified'))
    pickle.dump(data, open(a.out, 'wb'))
    print(f"\nSaved {len(demos)} verified SE(3) demos -> {a.out}  "
          f"(feasibility {len(demos)/max(attempted,1):.1%}, attempted {attempted})")
    print(f"  theta {data['theta'].shape}  base {data['base'].shape}  tau {data['tau'].shape}  cond {data['cond'].shape}")


if __name__ == '__main__':
    main()
