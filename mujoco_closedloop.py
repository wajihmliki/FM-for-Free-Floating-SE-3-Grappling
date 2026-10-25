"""mujoco_closedloop.py -- closed-loop execution in MuJoCo (Table V, PD and MPC rows).

The warm start serves as a reference for feedback rather than an open-loop command:

  --mode pd   : PD joint tracking of the warm-start joint trajectory.
  --mode mpc  : receding-horizon re-solve: every --mpc_every nodes, re-refine the
                terminal grasp posture from the current (drifted) state with the
                reduced-coordinate refiner, re-time the min-jerk reference, and keep
                PD tracking.

Same perturbations as mujoco_execute.py (--perturb).

    python3 mujoco_closedloop.py --mode pd  --ckpt ffsm_fm_se3.pt --n 200 --perturb
    python3 mujoco_closedloop.py --mode mpc --ckpt ffsm_fm_se3.pt --n 200 --perturb
"""
import argparse, pickle, numpy as np, torch
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from build_mjcf_se3 import build_mjcf_xml
from ffsm_reach_eval_se3 import _mj, cold_thf, load as load_gen
from ffsm_dataset_se3 import cond_vec


def build_model(p, damping=0.0, inertia_scale=1.0):
    import mujoco
    xml = build_mjcf_xml(p)
    if damping > 0: xml = xml.replace('damping="0"', f'damping="{damping:.4g}"')
    model = mujoco.MjModel.from_xml_string(xml)
    if inertia_scale != 1.0:
        for b in range(2, model.nbody):
            model.body_inertia[b] *= inertia_scale; model.body_mass[b] *= inertia_scale
    return model


def ref_from_thf(inst, thf, ease, de, dde, p):
    """min-jerk joint reference (pos/vel/acc) from th0 to thf."""
    dth = thf - inst['th0']
    return (inst['th0']+np.outer(ease, dth), np.outer(de, dth), np.outer(dde, dth))


def pd_track(model, data, mujoco, site, th_ref, thd_ref, base0, th0, N, dt, p,
             kp, kd, torque_noise, rng, refiner=None, inst=None, ease=None, de=None, dde=None, mpc_every=0):
    """Execute in MuJoCo, PD-tracking th_ref (optionally re-planning every mpc_every nodes)."""
    data.qpos[:3]=base0[:3]; data.qpos[3:7]=base0[3:7]; data.qpos[7:]=th0; data.qvel[:]=0
    mujoco.mj_forward(model, data)
    n_sub = max(1, int(round(dt/model.opt.timestep)))
    thr, thdr = th_ref.copy(), thd_ref.copy()
    tau_max = p.tau_max
    for k in range(N):
        # optional MPC re-plan from current (drifted) joint state toward the (moving) grasp
        if mpc_every and refiner is not None and k > 0 and k % mpc_every == 0 and k < N-2:
            cur_th = data.qpos[7:].copy()
            base_now = np.concatenate([data.qpos[:3], data.qpos[3:7]])
            thf_new = refiner(inst, cur_th, base_now, k, N)      # re-refine terminal posture
            m = N-k
            seg = cur_th[None,:] + np.outer(np.linspace(0,1,m+1)**3*(10-15*np.linspace(0,1,m+1)+6*np.linspace(0,1,m+1)**2), thf_new-cur_th)
            thr[k:] = seg; thdr[k:] = np.gradient(seg, dt, axis=0)
        for _ in range(n_sub):
            q = data.qpos[7:]; qd = data.qvel[6:]
            tau = kp*(thr[k]-q) + kd*(thdr[k]-qd)
            tau = np.clip(tau, -tau_max, tau_max)
            if torque_noise>0: tau = tau*(1.0+torque_noise*rng.standard_normal(tau.shape))
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
    return data.site_xpos[site].copy()


def make_refiner(p, tf):
    from scipy.optimize import least_squares
    def refiner(inst, cur_th, base_now, k, N):
        dt = tf/N; tk = k*dt; _,_,gp,gq = inst['tgt'].frame(tf)   # aim at capture-time grasp
        def feat(thf):
            m = N-k; s = np.linspace(0,1,m+1); ez = 10*s**3-15*s**4+6*s**5
            th = cur_th+np.outer(ez, thf-cur_th); thd = np.gradient(th, dt, axis=0)
            Q = fc3.rollout_reduced_se3_fast(base_now, th, thd, dt, p, 2); kk = fc3.fk3d(Q[-1], p)
            return kk['ee_pos'], fc3.ee_quat(Q[-1], p)
        sol = least_squares(lambda t: np.concatenate([feat(t)[0]-gp, 0.5*fc3.log_so3(fc3.qmul(fc3.qconj(feat(t)[1]), gq))]),
                            np.clip(cur_th,-2.69,2.69), bounds=(-2.7,2.7), max_nfev=40)
        return sol.x
    return refiner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['pd','mpc'], default='pd')
    ap.add_argument('--ckpt', default='ffsm_fm_se3.pt'); ap.add_argument('--data', default='ffsm_grapple_se3.pkl')
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--tf', type=float, default=6.0)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed0', type=int, default=700000)
    ap.add_argument('--perturb', action='store_true'); ap.add_argument('--damping', type=float, default=0.05)
    ap.add_argument('--inertia_scale', type=float, default=1.1); ap.add_argument('--torque_noise', type=float, default=0.02)
    ap.add_argument('--kp', type=float, default=40.0); ap.add_argument('--kd', type=float, default=8.0)
    ap.add_argument('--mpc_every', type=int, default=10)
    a = ap.parse_args()
    import mujoco
    from ffsm_fm_se3 import sample_theta
    net, ck = load_gen(a.ckpt, a.device); N = ck['seq_len']-1; p = fc3.FFSM3DParams(); dt = a.tf/N
    ease, de = _mj(N, a.tf); s = np.linspace(0,1,N+1); dde = (60*s-180*s**2+120*s**3)/a.tf**2
    rng = np.random.default_rng(0)
    damp = a.damping if a.perturb else 0.0; iscale = a.inertia_scale if a.perturb else 1.0; tnoise = a.torque_noise if a.perturb else 0.0
    model = build_model(p, damp, iscale); data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
    refiner = make_refiner(p, a.tf) if a.mode=='mpc' else None
    mpc_every = a.mpc_every if a.mode=='mpc' else 0

    insts = [d3.rand_inst3d(a.seed0+i, N, a.tf, p) for i in range(a.n)]
    conds = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(a.device)
    fm = sample_theta(net, conds, ck['stats'], seq_len=ck['seq_len'], device=a.device, th0_slice=(0, ck['dim'])).cpu().numpy()

    rows = []
    for j, inst in enumerate(insts):
        _,_,gp,_ = inst['tgt'].frame(a.tf)
        thr, thdr, _ = ref_from_thf(inst, fm[j][-1], ease, de, dde, p)
        ee_fm = pd_track(model, data, mujoco, site, thr, thdr, inst['base0'], inst['th0'], N, dt, p,
                         a.kp, a.kd, tnoise, rng, refiner, inst, ease, de, dde, mpc_every)
        thr_c, thdr_c, _ = ref_from_thf(inst, cold_thf(inst), ease, de, dde, p)
        ee_cold = pd_track(model, data, mujoco, site, thr_c, thdr_c, inst['base0'], inst['th0'], N, dt, p,
                           a.kp, a.kd, tnoise, rng, refiner, inst, ease, de, dde, mpc_every)
        Qr,_ = fc3.integrate_true3d(np.concatenate([inst['base0'],inst['th0']]), np.zeros(p.ndof),
               torque_from_thf(inst, fm[j][-1], ease, de, dde, p, dt)[:N], dt, p, substeps=8)
        recoil = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Qr[-1,3:7], fc3.qconj(Qr[0,3:7])))))
        rows.append((recoil, np.linalg.norm(ee_fm-gp)*100, np.linalg.norm(ee_cold-gp)*100))
    out = np.array(rows); edges=[0,15,25,35,200]
    tag = f"{a.mode.upper()} closed-loop" + (f", PERTURBED (damp {damp}, inertia x{iscale}, noise {tnoise})" if a.perturb else ", nominal")
    print(f"\n=== MuJoCo {tag}: reach at capture, median cm, n={a.n} ===")
    print(f"{'recoil':>10} {'#':>4} {'FM':>7} {'cold':>7}")
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(out[:,0]>=lo)&(out[:,0]<hi)
        if m.sum()==0: continue
        print(f"{f'[{lo},{hi})':>10} {int(m.sum()):>4} {np.median(out[m,1]):>7.1f} {np.median(out[m,2]):>7.1f}")
    print(f"{'ALL':>10} {a.n:>4} {np.median(out[:,1]):>7.1f} {np.median(out[:,2]):>7.1f}")


def torque_from_thf(inst, thf, ease, de, dde, p, dt):
    dth = thf-inst['th0']; th=inst['th0']+np.outer(ease,dth); thd=np.outer(de,dth); thdd=np.outer(dde,dth)
    _,_,tau,_=fc3.inverse_dynamics3d(th,thd,thdd,inst['base0'],dt,p); return tau


if __name__ == '__main__':
    main()
