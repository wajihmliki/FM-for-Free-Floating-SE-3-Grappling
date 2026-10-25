"""mujoco_execute.py -- open-loop execution in MuJoCo (Table V, "Open-loop playback").

Re-executes every warm-start source (cold / NN / regress / DDPM / FM) open loop in
MuJoCo and measures the gripper-to-grasp error at capture, binned by recoil. With
--perturb, MuJoCo adds effects the warm start does not model: viscous joint damping,
+10% arm inertia and 2% multiplicative actuator noise.

    python3 mujoco_crossval.py                                   # check agreement first
    python3 mujoco_execute.py --ckpt ffsm_fm_se3.pt --ddpm_ckpt ffsm_ddpm_se3.pt \
        --regress_ckpt ffsm_regress_se3.pt --data ffsm_grapple_se3.pkl --n 200 --perturb
"""
import argparse, pickle, numpy as np, torch
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from build_mjcf_se3 import build_mjcf_xml
from ffsm_reach_eval_se3 import _mj, cold_thf, load as load_gen
from ffsm_dataset_se3 import cond_vec


def build_perturbed_model(p, damping=0.0, inertia_scale=1.0):
    """MuJoCo model, optionally with joint damping and scaled arm inertias (model mismatch)."""
    import mujoco
    xml = build_mjcf_xml(p)
    if damping > 0:
        xml = xml.replace('damping="0"', f'damping="{damping:.4g}"')
    model = mujoco.MjModel.from_xml_string(xml)
    if inertia_scale != 1.0:
        for b in range(2, model.nbody):                 # arm links only; base left as-is
            model.body_inertia[b] *= inertia_scale
            model.body_mass[b] *= inertia_scale
    return model


def torque_from_thf(inst, thf, ease, de, dde, p, dt):
    dth = thf - inst['th0']
    th = inst['th0']+np.outer(ease, dth); thd = np.outer(de, dth); thdd = np.outer(dde, dth)
    _, _, tau, _ = fc3.inverse_dynamics3d(th, thd, thdd, inst['base0'], dt, p)
    return tau


def mujoco_exec(model, data, mujoco, site, tau_seq, base0, th0, N, dt, torque_noise=0.0, rng=None):
    data.qpos[:3] = base0[:3]; data.qpos[3:7] = base0[3:7]; data.qpos[7:] = th0; data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    n_sub = max(1, int(round(dt/model.opt.timestep)))
    for k in range(N):
        tau = tau_seq[k].copy()
        if torque_noise > 0 and rng is not None:
            tau = tau * (1.0 + torque_noise*rng.standard_normal(tau.shape))
        data.ctrl[:] = tau
        for _ in range(n_sub): mujoco.mj_step(model, data)
    return data.site_xpos[site].copy()


def nn_setup(data_pkl, n):
    d = pickle.load(open(data_pkl, 'rb')); C = d['cond'].astype(np.float64); TH = d['theta'].astype(np.float64)
    mu = C.mean(0); sd = C.std(0)+1e-9
    return (C-mu)/sd, TH[:, -1, :], mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ffsm_fm_se3.pt'); ap.add_argument('--ddpm_ckpt', default=None)
    ap.add_argument('--regress_ckpt', default=None); ap.add_argument('--data', default='ffsm_grapple_se3.pkl')
    ap.add_argument('--n', type=int, default=200); ap.add_argument('--N', type=int, default=60)
    ap.add_argument('--tf', type=float, default=6.0); ap.add_argument('--device', default='cuda')
    ap.add_argument('--seed0', type=int, default=700000)
    ap.add_argument('--perturb', action='store_true'); ap.add_argument('--damping', type=float, default=0.05)
    ap.add_argument('--inertia_scale', type=float, default=1.1); ap.add_argument('--torque_noise', type=float, default=0.02)
    a = ap.parse_args()
    import mujoco
    from ffsm_fm_se3 import sample_theta
    net, ck = load_gen(a.ckpt, a.device); N = ck['seq_len']-1; p = fc3.FFSM3DParams(); dt = a.tf/N
    ease, de = _mj(N, a.tf); s = np.linspace(0,1,N+1); dde = (60*s-180*s**2+120*s**3)/a.tf**2
    rng = np.random.default_rng(0)

    damp = a.damping if a.perturb else 0.0
    iscale = a.inertia_scale if a.perturb else 1.0
    tnoise = a.torque_noise if a.perturb else 0.0
    model = build_perturbed_model(p, damping=damp, inertia_scale=iscale); data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
    tag = f"PERTURBED (damping={damp}, inertia x{iscale}, torque noise {tnoise})" if a.perturb else "nominal (clone-check)"

    insts = [d3.rand_inst3d(a.seed0+i, N, a.tf, p) for i in range(a.n)]
    conds_np = np.stack([cond_vec(i) for i in insts]); conds = torch.from_numpy(conds_np).float().to(a.device)
    fm = sample_theta(net, conds, ck['stats'], seq_len=ck['seq_len'], device=a.device, th0_slice=(0, ck['dim'])).cpu().numpy()
    dd = rg = None
    if a.ddpm_ckpt:
        from ffsm_ddpm_se3 import sample_theta_ddpm
        dnet, dck = load_gen(a.ddpm_ckpt, a.device)
        dd = sample_theta_ddpm(dnet, conds, dck['stats'], seq_len=dck['seq_len'], device=a.device, th0_slice=(0, dck['dim'])).cpu().numpy()
    if a.regress_ckpt:
        from ffsm_regress_se3 import sample_regress
        rnet, rck = load_gen(a.regress_ckpt, a.device)
        rg = sample_regress(rnet, conds, rck['stats'], seq_len=rck['seq_len'], device=a.device, th0_slice=(0, rck['dim'])).cpu().numpy()
    cn_bank, thf_bank, mu, sd = nn_setup(a.data, p.n)

    def reach_mj(thf, inst):
        tau = torque_from_thf(inst, thf, ease, de, dde, p, dt)
        ee = mujoco_exec(model, data, mujoco, site, tau, inst['base0'], inst['th0'], N, dt, tnoise, rng)
        _, _, gp, _ = inst['tgt'].frame(a.tf); return np.linalg.norm(ee-gp)*100

    rows = []
    for j, inst in enumerate(insts):
        q0 = np.concatenate([inst['base0'], inst['th0']])
        tau_fm = torque_from_thf(inst, fm[j][-1], ease, de, dde, p, dt)
        Qr, _ = fc3.integrate_true3d(q0, np.zeros(p.ndof), tau_fm[:N], dt, p, substeps=8)
        recoil = np.rad2deg(np.linalg.norm(fc3.log_so3(fc3.qmul(Qr[-1,3:7], fc3.qconj(Qr[0,3:7])))))
        r = [recoil, reach_mj(cold_thf(inst), inst)]
        nn_thf = thf_bank[int(np.argmin(np.linalg.norm(cn_bank-((conds_np[j]-mu)/sd)[None], axis=1)))]
        r.append(reach_mj(nn_thf, inst))
        r.append(reach_mj(rg[j][-1], inst) if rg is not None else np.nan)
        r.append(reach_mj(dd[j][-1], inst) if dd is not None else np.nan)
        r.append(reach_mj(fm[j][-1], inst))
        rows.append(r)
    out = np.array(rows)
    edges = [0,15,25,35,200]
    print(f"\n=== MuJoCo reach at capture [{tag}], median cm, n={a.n} ===")
    print(f"{'recoil':>10} {'#':>4} {'cold':>7} {'NN':>7} {'regress':>8} {'DDPM':>7} {'FM':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (out[:,0]>=lo)&(out[:,0]<hi)
        if m.sum()==0: continue
        med=[np.nanmedian(out[m,k]) for k in (1,2,3,4,5)]
        print(f"{f'[{lo},{hi})':>10} {int(m.sum()):>4} {med[0]:>7.1f} {med[1]:>7.1f} {med[2]:>8.1f} {med[3]:>7.1f} {med[4]:>7.1f}")
    med=[np.nanmedian(out[:,k]) for k in (1,2,3,4,5)]
    print(f"{'ALL':>10} {a.n:>4} {med[0]:>7.1f} {med[1]:>7.1f} {med[2]:>8.1f} {med[3]:>7.1f} {med[4]:>7.1f}")


if __name__ == '__main__':
    main()
