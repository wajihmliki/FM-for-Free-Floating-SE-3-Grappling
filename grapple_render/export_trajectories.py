#!/usr/bin/env python3
"""export_trajectories.py -- export held-out FM and recoil-naive trajectories for
rendering (Figs. 3-4).

Draws a pool of held-out scenarios (seed0 900000), samples the flow model once per
scenario, and picks the median-recoil scenario inside each recoil bin
((5,15), (20,30), (35,45) deg). Selection never looks at reach error. For each pick
it writes one .npz (schema in grapple/plan.py) with:
  * FM:   min-jerk path to the generated terminal posture (as in Table I),
  * Cold: min-jerk path to the fixed-base IK terminal posture,
  * both bases reconstructed from MuJoCo's own mass matrix via the momentum map,
  * the target and grasp frame g(t) prescribed from the scenario's frame(t).

    python export_trajectories.py --model ffsm_render.xml --out traj/
    MUJOCO_GL=egl python run_render.py --npz 'traj/*.npz' --model ffsm_render.xml --out fig/
    python plot_fig4.py --npz 'traj/*.npz' --out fig/fig_error_vs_time.pdf

Frame note: the paper defines the base twist in the SPATIAL frame; MuJoCo's freejoint
qvel angular block is BODY-LOCAL. Base poses enter as qpos (frame agnostic), and the
reconstruction here uses MuJoCo's own convention, so the two are never mixed.
"""

import argparse
import json
import os
import sys

import numpy as np


def select_by_recoil_bin(all_scenarios, bins=((5, 15), (20, 30), (35, 45))):
    """Deterministic, FM-blind scenario selection.

    Picks the MEDIAN-recoil scenario inside each bin and never inspects reach
    error. Each scenario carries a 'recoil_deg' field computed from the
    momentum-projected rollout (Sec. VI).
    """
    picks, rule = [], []
    for lo, hi in bins:
        inb = sorted([s for s in all_scenarios if lo <= s["recoil_deg"] <= hi],
                     key=lambda s: s["recoil_deg"])
        if not inb:
            raise SystemExit(f"no held-out scenario with recoil in [{lo}, {hi}] deg")
        pick = inb[len(inb) // 2]
        picks.append(pick)
        rule.append(f"[{lo},{hi}]: median of {len(inb)} -> seed {pick['seed']} "
                    f"({pick['recoil_deg']:.1f} deg)")
    print("[select] FM-blind, median-of-bin:\n    " + "\n    ".join(rule))
    return picks


_CACHE = {}          # code_root -> (net, ck, params, N, device)
_FM_SAMPLES = {}     # seed -> (K, n) FM trajectory, so recoil and export use the SAME draw


def _load_pipeline(code_root, device="cuda", ckpt=None):
    """Load the model once. Returns (net, ck, p, N, device)."""
    if code_root in _CACHE:
        return _CACHE[code_root]
    import os, sys, torch
    sys.path.insert(0, code_root)
    import ffsm_core_se3 as fc3
    from ffsm_reach_eval_se3 import load as load_gen
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if ckpt is None:
        ckpt = os.path.join(code_root, "ffsm_fm_se3.pt")
    net, ck = load_gen(ckpt, device)
    p = fc3.FFSM3DParams()
    N = ck["seq_len"] - 1
    _CACHE[code_root] = (net, ck, p, N, device)
    return _CACHE[code_root]


def _thd_of(th, tf):
    """Joint rates on the node grid, consistent with the exporter's spline."""
    from scipy.interpolate import CubicSpline
    K = len(th)
    ts = np.linspace(0.0, tf, K)
    return CubicSpline(ts, th, axis=0)(ts, 1)


def _omega_body_from_target(tgt, tf):
    """Body-local target angular velocity, derived from the target's OWN frame().

    Differencing frame() avoids any assumption about the frame of tgt.omega.
    """
    import ffsm_core_se3 as fc3
    dt = min(1e-3, tf * 1e-3)
    q0 = np.asarray(tgt.frame(0.0)[1], float)
    q1 = np.asarray(tgt.frame(dt)[1], float)
    R0, R1 = fc3.rotmat(q0), fc3.rotmat(q1)
    return fc3.log_so3_mat(R0.T @ R1) / dt if hasattr(fc3, "log_so3_mat") else \
        fc3.log_so3(fc3.qmul(fc3.qconj(q0), q1)) / dt


def _grasp_body(tgt, tf):
    """Grasp pose expressed in the target BODY frame, plus a consistency check.

    The MJCF site is fixed in the body frame, but tgt.frame() reports the grasp in
    WORLD. Invert the target pose at t=0 to get the body-frame offset, then verify
    it reproduces tgt.frame() at later times.
    """
    import ffsm_core_se3 as fc3
    c0, q0, gp0, gq0 = tgt.frame(0.0)
    R0 = fc3.rotmat(np.asarray(q0, float))
    off = R0.T @ (np.asarray(gp0, float) - np.asarray(c0, float))
    qb = fc3.qmul(fc3.qconj(np.asarray(q0, float)), np.asarray(gq0, float))
    worst = 0.0
    for t in (0.25 * tf, 0.5 * tf, tf):
        c, q, gp, _ = tgt.frame(t)
        pred = np.asarray(c, float) + fc3.rotmat(np.asarray(q, float)) @ off
        worst = max(worst, float(np.linalg.norm(pred - np.asarray(gp, float))))
    if worst > 2e-3:
        print(f"    !! grasp is not body-fixed: body-frame offset mispredicts "
              f"tgt.frame() by {worst*1000:.1f} mm. The MJCF site model is wrong.")
    return dict(grasp_offset=off, grasp_quat=qb)


def _t_world(tgt, tf, N):
    """Target BODY pose over the node grid, from the scenario's frame(t)."""
    ts = np.linspace(0.0, tf, N + 1)
    return np.array([np.concatenate([np.asarray(tgt.frame(t)[0], float),
                                     np.asarray(tgt.frame(t)[1], float)]) for t in ts])


def _g_world(tgt, tf, N):
    """Grasp trajectory g(t) from the scenario's frame(t)."""
    ts = np.linspace(0.0, tf, N + 1)
    return np.array([np.concatenate([np.asarray(tgt.frame(t)[2], float),
                                     np.asarray(tgt.frame(t)[3], float)]) for t in ts])


def _recoil_deg(inst, th, tf, p):
    """Geodesic base-attitude angle over the maneuver (Sec. VI)."""
    import ffsm_core_se3 as fc3
    K = len(th); dt = tf / (K - 1)
    Q = fc3.rollout_reduced_se3_fast(inst["base0"], th, _thd_of(th, tf), dt, p, 4)
    dq = fc3.qmul(Q[-1, 3:7], fc3.qconj(Q[0, 3:7]))
    return float(np.degrees(np.linalg.norm(fc3.log_so3(dq))))


def load_scenarios(n, code_root, seed0=900000, pool=400, tf=6.0, device="cuda", ckpt=None):
    """Held-out scenarios with recoil_deg, ready for select_by_recoil_bin()."""
    import sys, torch
    sys.path.insert(0, code_root)
    import ffsm_demo_se3 as d3
    from ffsm_dataset_se3 import cond_vec
    from ffsm_fm_se3 import sample_theta
    net, ck, p, N, dev = _load_pipeline(code_root, device, ckpt)

    insts = [d3.rand_inst3d(seed0 + i, N, tf, p) for i in range(pool)]
    conds = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(dev)
    fm = sample_theta(net, conds, ck["stats"], seq_len=ck["seq_len"],
                      device=dev, th0_slice=(0, ck["dim"])).cpu().numpy()

    out = []
    for i, inst in enumerate(insts):
        seed = seed0 + i
        th_raw = np.asarray(fm[i], float)
        _FM_SAMPLES[seed] = th_raw                   # reuse this exact draw at export
        # recoil is defined on the trajectory that is actually executed, i.e. the
        # min-jerk path to the generated terminal posture (Table I's construction)
        from ffsm_reach_eval_se3 import _mj as _mjp
        _ease, _ = _mjp(N, tf)
        th = inst["th0"][None, :] + np.outer(_ease, th_raw[-1] - inst["th0"])
        c0, q0, _, _ = inst["tgt"].frame(0.0)
        out.append(dict(
            seed=seed, tf=float(tf),
            theta0=np.asarray(inst["th0"], float),
            base0=np.asarray(inst["base0"], float),
            target0=np.concatenate([np.asarray(c0, float), np.asarray(q0, float)]),
            omega_g=_omega_body_from_target(inst["tgt"], tf),
            recoil_deg=_recoil_deg(inst, th, tf, p),
            # grasp pose in the TARGET BODY frame, derived from the target's own
            # frame() rather than guessed from a conditioning-vector layout
            **_grasp_body(inst["tgt"], tf),
            g_ref=_g_world(inst["tgt"], tf, N),
            tgt_ref=_t_world(inst["tgt"], tf, N),
            _inst=inst,
        ))
    print(f"[load] {len(out)} held-out scenarios, recoil "
          f"{min(s['recoil_deg'] for s in out):.1f}-{max(s['recoil_deg'] for s in out):.1f} deg")
    return out


def sample_fm(scenario, code_root):
    """Min-jerk to the FM terminal posture -- the same construction Table I measures
    (ffsm_reach_eval_se3.reach_of uses only the generated terminal posture)."""
    import sys
    sys.path.insert(0, code_root)
    from ffsm_reach_eval_se3 import _mj
    _, _, p, N, _ = _load_pipeline(code_root)
    seed = int(scenario["seed"])
    if seed not in _FM_SAMPLES:
        raise RuntimeError(f"no cached FM sample for seed {seed}; call load_scenarios first")
    tf = float(scenario["tf"])
    ease, _ = _mj(N, tf)
    th0 = np.asarray(scenario["theta0"], float)
    thf = np.asarray(_FM_SAMPLES[seed][-1], float)      # terminal posture only
    return th0[None, :] + np.outer(ease, thf - th0)


def sample_cold(scenario, code_root):
    """Recoil-naive: fixed-base IK terminal posture + min-jerk, as in Table I."""
    import sys
    sys.path.insert(0, code_root)
    from ffsm_reach_eval_se3 import _mj, cold_thf
    net, ck, p, N, dev = _load_pipeline(code_root)
    tf = float(scenario["tf"])
    ease, _ = _mj(N, tf)
    th0 = np.asarray(scenario["theta0"], float)
    thf = np.asarray(cold_thf(scenario["_inst"]), float)
    return th0[None, :] + np.outer(ease, thf - th0)


def export(model_path, scenarios, out_dir, code_root, reconstruct=True,
           arm_joints=("j1", "j2", "j3", "j4", "j5", "j6"),
           base_joint="base_free", target_joint="target_free",
           gripper_site="gripper", grasp_site="grasp"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from grapple.core import (Layout, quat2mat, mat2quat, quat_norm, make_T,
                              so3_log, pose_error, reconstruct_base)
    from grapple.plan import roll_target, min_clearance

    model = mujoco.MjModel.from_xml_path(model_path)
    lay = Layout(model, base_joint=base_joint, target_joint=target_joint,
                 arm_joints=arm_joints, gripper_site=gripper_site,
                 grasp_site=grasp_site)
    d = mujoco.MjData(model)
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for i, sc in enumerate(scenarios):
        tf = float(sc["tf"])
        # the grasp site varies per scenario; the MJCF only holds a placeholder
        if "grasp_offset" in sc:
            model.site_pos[lay.grasp_s] = np.asarray(sc["grasp_offset"], float)
        if "grasp_quat" in sc:
            model.site_quat[lay.grasp_s] = quat_norm(np.asarray(sc["grasp_quat"], float))
        th_fm = np.asarray(sample_fm(sc, code_root), float)
        th_cd = np.asarray(sample_cold(sc, code_root), float)
        K = len(th_fm)
        assert len(th_cd) == K, "FM and cold trajectories must share the node grid"
        ts = np.linspace(0.0, tf, K)

        # The target motion is given (the model is conditioned on it and the
        # certificate uses it); prescribing it from frame(t) measures against
        # exactly the same g(t) as Table I.
        if "tgt_ref" in sc and "g_ref" in sc and len(sc["g_ref"]) == K:
            tgt = np.asarray(sc["tgt_ref"], float)
            grasp = np.asarray(sc["g_ref"], float)
        else:
            _, tgt, grasp = roll_target(model, lay, sc["target0"][:3],
                                        sc["target0"][3:], sc["omega_g"], tf, K)

        out = {}
        for tag, th in (("fm", th_fm), ("cold", th_cd)):
            base = np.zeros((K, 7))
            grip = np.zeros((K, 7))
            p = np.asarray(sc["base0"][:3], float)
            q = quat_norm(sc["base0"][3:])
            max_res = 0.0
            if reconstruct:
                from scipy.interpolate import CubicSpline
                cs = CubicSpline(ts, th, axis=0)
                th_of = lambda t: cs(t)
                thd_of = lambda t: cs(t, 1)
            for k in range(K):
                if k > 0 and reconstruct:
                    p, q, r = reconstruct_base(model, d, lay, th_of, thd_of,
                                               ts[k - 1], ts[k], p, q, dt=2e-3,
                                               check_residual=True)
                    max_res = max(max_res, r)
                elif k > 0:
                    bp = sc[f"base_{tag}"][k]      # externally supplied base trajectory
                    p, q = np.asarray(bp[:3], float), quat_norm(bp[3:])
                lay.set_base(d, p, q)
                lay.set_theta(d, th[k])
                mujoco.mj_forward(model, d)
                base[k] = np.concatenate([p, q])
                T = lay.gripper_T(d)
                grip[k] = np.concatenate([T[:3, 3], mat2quat(T[:3, :3])])
            out[f"base_{tag}"] = base
            out[f"grip_{tag}"] = grip
            out[f"_res_{tag}"] = max_res

        Tg = make_T(grasp[-1, :3], quat2mat(grasp[-1, 3:]))
        _, pf, rf = pose_error(Tg, make_T(out["grip_fm"][-1, :3],
                                          quat2mat(out["grip_fm"][-1, 3:])))
        _, pc, rc = pose_error(Tg, make_T(out["grip_cold"][-1, :3],
                                          quat2mat(out["grip_cold"][-1, 3:])))
        R0 = quat2mat(out["base_fm"][0, 3:])
        Rf = quat2mat(out["base_fm"][-1, 3:])
        meta = dict(seed=int(sc["seed"]), tf=tf, n_nodes=K, converged=True,
                    recoil_deg=float(np.degrees(np.linalg.norm(so3_log(R0.T @ Rf)))),
                    reach_fm_cm=float(pf * 100), reach_cold_cm=float(pc * 100),
                    reach_fm_deg=float(np.degrees(rf)),
                    reach_cold_deg=float(np.degrees(rc)),
                    source="flow_matching")
        # momentum residual is only reported when it was measured here
        if reconstruct:
            meta["momentum_residual"] = float(max(out.pop("_res_fm"),
                                                  out.pop("_res_cold")))
        else:
            out.pop("_res_fm", None); out.pop("_res_cold", None)

        clr = min_clearance(model, lay, th_fm, out["base_fm"], tgt)
        meta["min_clearance_fm_mm"] = float(clr * 1000)
        meta["collision_free"] = bool(clr >= -1e-4)
        if not meta["collision_free"]:
            print(f"    !! arm penetrates target by {-clr*1000:.1f} mm")

        p_out = os.path.join(out_dir, f"scenario_{i+1}_seed{sc['seed']}.npz")
        np.savez_compressed(p_out, t=ts, theta_fm=th_fm, theta_cold=th_cd,
                            target=tgt, grasp=grasp, meta=json.dumps(meta), **out)
        written.append(p_out)
        print(f"[export] {p_out}  rho={meta['recoil_deg']:.1f} deg  "
              f"FM {meta['reach_fm_cm']:.2f} cm  COLD {meta['reach_cold_cm']:.2f} cm")

        for name, val, tol in (("FM reach", pf * 100, 6.0),
                               ("orientation", np.degrees(rf), 3.0)):
            if val > tol:
                print(f"    note: unrefined {name} {val:.2f} is above the {tol} gate")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ffsm_render.xml")
    ap.add_argument("--code-root", dest="code_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                    help="directory containing ffsm_core_se3.py etc. (the repo root)")
    ap.add_argument("--ckpt", default=None, help="FM checkpoint (default <code-root>/ffsm_fm_se3.pt)")
    ap.add_argument("--scenarios", type=int, default=3)
    ap.add_argument("--out", default="traj")
    ap.add_argument("--pool", type=int, default=400,
                    help="held-out scenarios drawn before recoil binning")
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arm-joints", dest="arm_joints", default="j1,j2,j3,j4,j5,j6")
    ap.add_argument("--base-joint", dest="base_joint", default="base_free")
    ap.add_argument("--target-joint", dest="target_joint", default="target_free")
    ap.add_argument("--gripper-site", dest="gripper_site", default="gripper")
    ap.add_argument("--grasp-site", dest="grasp_site", default="grasp")
    ap.add_argument("--no-reconstruct", action="store_true",
                    help="use externally supplied base poses instead of "
                         "reconstructing them here (needs base_fm/base_cold "
                         "in each scenario dict)")
    a = ap.parse_args()
    pool = load_scenarios(a.scenarios, a.code_root, pool=a.pool, seed0=a.seed0,
                          device=a.device, ckpt=a.ckpt)
    scs = select_by_recoil_bin(pool)
    export(a.model, scs, a.out, a.code_root, reconstruct=not a.no_reconstruct,
           arm_joints=tuple(a.arm_joints.split(",")), base_joint=a.base_joint,
           target_joint=a.target_joint, gripper_site=a.gripper_site,
           grasp_site=a.grasp_site)


if __name__ == "__main__":
    main()
