"""scp_warmstart_study_se3.py -- Table IV: solver-side warm-start study (Sec. IV-D, VII-E).

Runs the reduced-coordinate sequential convex refinement from each initialization
(Cold / Regress / DDPM / FM / Oracle) on the same 500 held-out scenarios as Table I and
reports initial error, iterations, certified rate, wall-clock and final error.

The refinement:
  decision vector z = (theta_f, b_1..b_K), K=4 -> (K+1)n = 30 variables,
      theta(s) = theta_0 + ease(s)(theta_f - theta_0) + sum_k b_k psi_k(s),
      ease(s) = 10s^3 - 15s^4 + 6s^5,  psi_k(s) = s^(k+2) (1-s)^3;
  the spacecraft base is reconstructed from the momentum map at every iterate, so all
  iterates are momentum-consistent (Prop. 1);
  residual r(z) = [e_p ; w e_R] with w = 0.5 m/rad (SE(3) terminal tracking error);
  each iteration: forward-difference Jacobian J = dr/dz, then the convex QP
      min_d ||r + J d||^2 + lam ||d||^2   s.t. ||d||_inf <= rho_TR, |z + d| <= z_max
  solved with OSQP, with trust-region accept/reject on actual vs predicted reduction;
  converged when the terminal error is inside 0.4x the certificate gate (2 cm, 1.2 deg)
  with no torque violation, or when ||d||_inf < 1e-4.
Torque limits are checked post hoc (inverse dynamics + full nonlinear integration
against the 5 cm / 3 deg certificate), not linearized inside the loop.

    python3 scp_warmstart_study_se3.py --smoke
    python3 scp_warmstart_study_se3.py --n 500 --ddpm_ckpt ffsm_ddpm_se3.pt --regress_ckpt ffsm_regress_se3.pt
"""
import argparse, time, numpy as np, torch
from collections import Counter
from scipy.interpolate import CubicSpline
import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
from ffsm_dataset_se3 import cond_vec
from ffsm_reach_eval_se3 import _mj, cold_thf, oracle_thf, load as load_gen

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:
    HAVE_CVXPY = False

JOINT_LIM = 2.7          # rad, matches the refiner bounds used elsewhere
GATE_POS = 0.05          # m
GATE_ROT = np.deg2rad(3.0)

# ---------------------------------------------------------------- trajectory parameterization
from numpy.polynomial import Polynomial as _P

def _basis(K):
    ease = _P([0, 0, 0, 10, -15, 6])
    onem3 = _P([1, -3, 3, -1])                       # (1-s)^3
    psis = [_P([0] * (k + 2) + [1]) * onem3 for k in range(1, K + 1)]
    return ease, psis


def traj_from_knots(th0, z, tf, N, n):
    """z[0] = terminal posture; z[1:] = shape coefficients. Returns (th, thd, thdd)."""
    K = z.shape[0] - 1
    ease, psis = _basis(K)
    s = np.linspace(0.0, 1.0, N + 1)
    thf = z[0]; b = z[1:]
    e0, e1, e2 = ease(s), ease.deriv(1)(s), ease.deriv(2)(s)
    th = th0[None, :] + np.outer(e0, thf - th0)
    thd = np.outer(e1, thf - th0) / tf
    thdd = np.outer(e2, thf - th0) / tf ** 2
    for k in range(K):
        p0, p1, p2 = psis[k](s), psis[k].deriv(1)(s), psis[k].deriv(2)(s)
        th = th + np.outer(p0, b[k])
        thd = thd + np.outer(p1, b[k]) / tf
        thdd = thdd + np.outer(p2, b[k]) / tf ** 2
    return th, thd, thdd


def knots_from_traj(th_traj, tf, N, K):
    """Least-squares fit of (theta_f, b) to a generated joint trajectory."""
    th0 = th_traj[0]; n = th_traj.shape[1]
    ease, psis = _basis(K)
    s = np.linspace(0.0, 1.0, th_traj.shape[0] - 1 + 1)
    A = np.column_stack([ease(s)] + [psis[k](s) for k in range(K)])   # (N+1, K+1)
    Y = th_traj - th0[None, :]
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)                      # (K+1, n)
    z = np.zeros((K + 1, n))
    z[0] = th0 + coef[0]                                             # ease coefficient -> theta_f
    z[1:] = coef[1:]
    return z


def knots_from_terminal(th0, thf, tf, N, K, ease_unused=None):
    """Pure min-jerk to a terminal posture: shape coefficients zero."""
    n = th0.shape[0]
    z = np.zeros((K + 1, n)); z[0] = thf
    return z

# ---------------------------------------------------------------- residuals
def residuals(z, inst, p, tf, N, dt, substeps=4, w_eff=0.10, reach_only=False):
    """r = [pos(3); rot(3); torque violation(n); effort(n)]."""
    n = p.n; th0 = inst['th0']; base0 = inst['base0']
    th, thd, thdd = traj_from_knots(th0, z, tf, N, n)
    Q = fc3.rollout_reduced_se3_fast(base0, th, thd, dt, p, substeps)   # momentum-consistent base
    k = fc3.fk3d(Q[-1], p)
    _, _, gp, gq = inst['tgt'].frame(tf)
    pos_err = k['ee_pos'] - gp
    q_ee = fc3.ee_quat(Q[-1], p)
    rot_err = fc3.log_so3(fc3.qmul(fc3.qconj(q_ee), gq))
    if reach_only:
        r = np.concatenate([pos_err, 0.5 * rot_err])
        diag = dict(pos=np.linalg.norm(pos_err), rot=np.linalg.norm(rot_err),
                    tau_viol=0.0, tau=None, th=th, thd=thd, thdd=thdd)
        return r, diag
    _, _, tau, _ = fc3.inverse_dynamics3d(th, thd, thdd, base0, dt, p)
    viol = np.maximum(np.abs(tau).max(axis=0) - p.tau_max, 0.0)         # per-joint max violation
    effort = np.sqrt((tau ** 2).mean(axis=0)) / p.tau_max               # per-joint RMS / limit
    r = np.concatenate([pos_err, 0.5 * rot_err, 0.5 * viol, w_eff * effort])
    diag = dict(pos=np.linalg.norm(pos_err), rot=np.linalg.norm(rot_err),
                tau_viol=float(viol.max()), tau=tau, th=th, thd=thd, thdd=thdd)
    return r, diag


def jacobian_fd(z, inst, p, tf, N, dt, r0, eps=1e-4, reach_only=False):
    """Finite-difference Jacobian of r wrt the flattened decision vector."""
    zf = z.reshape(-1); m = r0.size; J = np.zeros((m, zf.size))
    for i in range(zf.size):
        zp = zf.copy(); zp[i] += eps
        rp, _ = residuals(zp.reshape(z.shape), inst, p, tf, N, dt, reach_only=reach_only)
        J[:, i] = (rp - r0) / eps
    return J

# ---------------------------------------------------------------- convex subproblem
def solve_subproblem(J, r, rho, z_flat, lam=1e-3, LIM_VEC=None):
    """min ||r + J d||^2 + lam||d||^2  s.t. ||d||_inf <= rho, joint limits. Convex QP."""
    if LIM_VEC is None:
        LIM_VEC = np.full(z_flat.size, JOINT_LIM)
    if HAVE_CVXPY:
        d = cp.Variable(J.shape[1])
        obj = cp.Minimize(cp.sum_squares(J @ d + r) + lam * cp.sum_squares(d))
        cons = [cp.norm_inf(d) <= rho,
                z_flat + d <= LIM_VEC, z_flat + d >= -LIM_VEC]
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if d.value is not None:
                return np.asarray(d.value).ravel()
        except Exception:
            pass
    H = J.T @ J + lam * np.eye(J.shape[1])
    d = -np.linalg.solve(H, J.T @ r)
    d = np.clip(d, -rho, rho)
    return np.clip(z_flat + d, -LIM_VEC, LIM_VEC) - z_flat

# ---------------------------------------------------------------- the SCP solver
def scp_refine(z0, inst, p, tf, N, dt, max_iter=30, rho0=0.35, tol_step=1e-4, reach_only=False):
    """Returns (z, iters, converged, history). Converged = residual inside a tight inner tolerance."""
    z = z0.copy(); rho = rho0
    r, diag = residuals(z, inst, p, tf, N, dt, reach_only=reach_only)
    cost = float(r @ r)
    iters = 0; converged = False

    def inner_ok(d):
        return d['pos'] < 0.4 * GATE_POS and d['rot'] < 0.4 * GATE_ROT and d['tau_viol'] <= 0

    if inner_ok(diag):
        return z, 0, True, diag

    for it in range(max_iter):
        iters = it + 1
        J = jacobian_fd(z, inst, p, tf, N, dt, r, reach_only=reach_only)
        lim = np.full(z.shape, 1.5); lim[0] = JOINT_LIM
        d = solve_subproblem(J, r, rho, z.reshape(-1), LIM_VEC=lim.reshape(-1))
        if np.max(np.abs(d)) < tol_step:
            converged = inner_ok(diag)
            break
        z_try = (z.reshape(-1) + d).reshape(z.shape)
        r_try, diag_try = residuals(z_try, inst, p, tf, N, dt, reach_only=reach_only)
        cost_try = float(r_try @ r_try)
        pred = cost - float((r + J @ d) @ (r + J @ d))
        ratio = (cost - cost_try) / pred if pred > 1e-12 else -1.0
        if ratio > 0.1:
            z, r, diag, cost = z_try, r_try, diag_try, cost_try
            rho = min(rho * 1.6, 1.0) if ratio > 0.7 else rho
        else:
            rho *= 0.4
            if rho < 1e-4:
                break
        if inner_ok(diag):
            converged = True
            break
    return z, iters, converged, diag

# ---------------------------------------------------------------- full-nonlinear gate check
def gate_check(z, inst, p, tf, N, dt, substeps=8):
    """Roll the commanded torque through the TRUE dynamics from rest;
    returns failure condition ('pass', 'pos', 'rot', 'torque') alongside pos/rot errors."""
    n = p.n
    th, thd, thdd = traj_from_knots(inst['th0'], z, tf, N, n)
    _, _, tau, _ = fc3.inverse_dynamics3d(th, thd, thdd, inst['base0'], dt, p)

    # 1. Torque-limit check
    if np.any(np.abs(tau) > p.tau_max[None, :] * 1.0 + 1e-9):
        return "torque", np.inf, np.inf

    q0 = np.concatenate([inst['base0'], inst['th0']])
    Q, _ = fc3.integrate_true3d(q0, np.zeros(p.ndof), tau[:N], dt, p, substeps=substeps)
    k = fc3.fk3d(Q[-1], p)
    _, _, gp, gq = inst['tgt'].frame(tf)

    pos_m = float(np.linalg.norm(k['ee_pos'] - gp))
    rot_rad = float(np.linalg.norm(fc3.log_so3(fc3.qmul(fc3.qconj(fc3.ee_quat(Q[-1], p)), gq))))

    pos_cm = pos_m * 100.0
    rot_deg = np.rad2deg(rot_rad)

    # 2. Position / orientation error checks
    pos_bad, rot_bad = pos_m >= GATE_POS, rot_rad >= GATE_ROT
    if pos_bad and rot_bad:
        return "both", pos_cm, rot_deg
    if pos_bad:
        return "pos", pos_cm, rot_deg
    if rot_bad:
        return "rot", pos_cm, rot_deg

    return "pass", pos_cm, rot_deg

# ---------------------------------------------------------------- study driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ffsm_fm_se3.pt')
    ap.add_argument('--ddpm_ckpt', default=None); ap.add_argument('--regress_ckpt', default=None)
    ap.add_argument('--n', type=int, default=500); ap.add_argument('--tf', type=float, default=6.0)
    ap.add_argument('--K', type=int, default=4, help='shape basis terms (decision vars = (K+1)*n)')
    ap.add_argument('--max_iter', type=int, default=30)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed0', type=int, default=400000)
    ap.add_argument('--torque_in_residual', action='store_true',
                    help='also linearize torque violation/effort inside the loop '
                         '(default: reach-only residual, torque checked post hoc)')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    reach_only = not a.torque_in_residual

    p = fc3.FFSM3DParams(); n = p.n
    if a.smoke:
        N = 20; tf = 4.0; dt = tf / N; ease, de = _mj(N, tf)
        inst = d3.rand_inst3d(1234, N, tf, p)
        z0 = knots_from_terminal(inst['th0'], cold_thf(inst), tf, N, a.K, ease)
        t0 = time.time(); z, it, conv, diag = scp_refine(z0, inst, p, tf, N, dt, max_iter=6,
                                                         reach_only=reach_only)
        reason, pos, rot = gate_check(z, inst, p, tf, N, dt, substeps=4)
        print(f"SMOKE: cvxpy={HAVE_CVXPY} iters={it} converged={conv} "
              f"pos={diag['pos']*100:.1f}cm rot={np.rad2deg(diag['rot']):.1f}deg "
              f"gate_reason={reason} ({pos:.1f}cm/{rot:.1f}deg) {time.time()-t0:.1f}s -> OK")
        return

    net, ck = load_gen(a.ckpt, a.device); N = ck['seq_len'] - 1; tf = a.tf; dt = tf / N
    ease, de = _mj(N, tf)
    from ffsm_fm_se3 import sample_theta
    insts = [d3.rand_inst3d(a.seed0 + i, N, tf, p) for i in range(a.n)]
    conds = torch.from_numpy(np.stack([cond_vec(i) for i in insts])).float().to(a.device)
    fm = sample_theta(net, conds, ck['stats'], seq_len=ck['seq_len'], device=a.device,
                      th0_slice=(0, ck['dim'])).cpu().numpy()
    dd = rg = None
    if a.ddpm_ckpt:
        from ffsm_ddpm_se3 import sample_theta_ddpm
        dnet, dck = load_gen(a.ddpm_ckpt, a.device)
        dd = sample_theta_ddpm(dnet, conds, dck['stats'], seq_len=dck['seq_len'],
                               device=a.device, th0_slice=(0, dck['dim'])).cpu().numpy()
    if a.regress_ckpt:
        from ffsm_regress_se3 import sample_regress
        rnet, rck = load_gen(a.regress_ckpt, a.device)
        rg = sample_regress(rnet, conds, rck['stats'], seq_len=rck['seq_len'],
                            device=a.device, th0_slice=(0, rck['dim'])).cpu().numpy()

    arms = ['cold', 'FM']
    if rg is not None: arms.insert(1, 'regress')
    if dd is not None: arms.insert(-1, 'DDPM')
    arms.append('oracle')
    res = {k: dict(iters=[], conv=[], gate=[], wall=[], pos=[], init=[], tqrej=[]) for k in arms}
    gate_breakdown = {k: Counter() for k in arms}
    recoils = []; oracle_gen = []

    print(f"cvxpy={HAVE_CVXPY}  n={a.n}  K={a.K} (decision vars={(a.K+1)*n})  max_iter={a.max_iter}\n")
    for j, inst in enumerate(insts):
        inits = {}
        inits['cold']   = knots_from_terminal(inst['th0'], cold_thf(inst), tf, N, a.K, ease)
        inits['FM']     = knots_from_traj(fm[j], tf, N, a.K)
        if dd is not None: inits['DDPM']    = knots_from_traj(dd[j], tf, N, a.K)
        if rg is not None: inits['regress'] = knots_from_traj(rg[j], tf, N, a.K)
        t_or = time.time(); thf_or = oracle_thf(inst, ease, de); oracle_gen.append(time.time() - t_or)
        inits['oracle'] = knots_from_terminal(inst['th0'], thf_or, tf, N, a.K, ease)

        # Recoil label from the FM rollout
        thF, thdF, _ = traj_from_knots(inst['th0'], inits['FM'], tf, N, n)
        QF = fc3.rollout_reduced_se3_fast(inst['base0'], thF, thdF, dt, p, 2)
        recoils.append(np.rad2deg(np.linalg.norm(fc3.log_so3(
            fc3.qmul(QF[-1, 3:7], fc3.qconj(QF[0, 3:7]))))))
            
        for name in arms:
            z0 = inits[name]
            r0, d0 = residuals(z0, inst, p, tf, N, dt, reach_only=reach_only)
            t0 = time.time()
            z, it, conv, diag = scp_refine(z0, inst, p, tf, N, dt, max_iter=a.max_iter,
                                           reach_only=reach_only)
            wall = time.time() - t0
            reason, pos_cm, rot_deg = gate_check(z, inst, p, tf, N, dt)
            
            # Record detailed gate status breakdown
            gate_breakdown[name][reason] += 1
            ok = (reason == "pass")

            R = res[name]
            R['iters'].append(it); R['conv'].append(conv); R['gate'].append(ok)
            R['wall'].append(wall); R['init'].append(d0['pos'] * 100)
            R['tqrej'].append(reason == 'torque')
            R['pos'].append(pos_cm if np.isfinite(pos_cm) else np.nan)
            
        if (j + 1) % 5 == 0:
            print(f"  ...{j+1}/{a.n} scenarios")

    rec = np.array(recoils)
    print(f"\n=== SCP warm-start study (reduced-coordinate sequential convex refinement), n={a.n} ===")
    print(f"{'init':>9} {'init_err':>9} {'iters(med)':>11} {'converged':>10} {'gate pass':>10} "
          f"{'tq-rej':>7} {'wall s(med)':>12} {'final cm':>9}")
    for name in arms:
        R = res[name]
        pos = np.array(R['pos'], dtype=float)
        print(f"{name:>9} {np.median(R['init']):>8.1f}c {np.median(R['iters']):>11.1f} "
              f"{100*np.mean(R['conv']):>9.0f}% {100*np.mean(R['gate']):>9.0f}% "
              f"{100*np.mean(R['tqrej']):>6.0f}% {np.median(R['wall']):>12.2f} "
              f"{(np.nanmedian(pos) if np.isfinite(pos).any() else float('nan')):>9.2f}")
    print("  ('converged' = inner reduced-model tolerance; 'gate pass' = full nonlinear certificate;")
    print("   'tq-rej' = rejected for exceeding torque limits; 'final cm' excludes tq-rejected cases.)")

    # Printed Failure Breakdown per Method Table
    print(f"\n=== Gate Failure Breakdown per Method (n={a.n}) ===")
    print(f"{'Method':>10} | {'PASS':>6} | {'POS Fail':>8} | {'ROT Fail':>8} | {'BOTH':>6} | {'TORQUE Fail':>11}")
    print("-" * 64)
    for name in arms:
        c = gate_breakdown[name]
        print(f"{name:>10} | {c['pass']:>6} | {c['pos']:>8} | {c['rot']:>8} | {c['both']:>6} | {c['torque']:>11}")
    print(f"\nwall clock excludes generation; oracle multi-start IK generation: "
          f"median {np.median(oracle_gen):.2f} s per scenario")

    # Paired comparison
    if 'cold' in res and 'FM' in res:
        it_fm = np.array(res['FM']['iters']); it_cold = np.array(res['cold']['iters'])
        w_fm = np.array(res['FM']['wall']);   w_cold = np.array(res['cold']['wall'])
        g_fm = np.array(res['FM']['gate']);   g_cold = np.array(res['cold']['gate'])
        print(f"\n--- paired FM vs cold (identical scenarios) ---")
        print(f"  fewer iterations : {100*np.mean(it_fm < it_cold):.0f}%   "
              f"equal {100*np.mean(it_fm == it_cold):.0f}%   more {100*np.mean(it_fm > it_cold):.0f}%")
        print(f"  median iteration ratio (cold/FM): {np.median(it_cold/np.maximum(it_fm,1)):.2f}")
        print(f"  median wall-clock ratio (cold/FM): {np.median(w_cold/np.maximum(w_fm,1e-6)):.2f}")
        print(f"  gate pass: FM {100*g_fm.mean():.0f}%   cold {100*g_cold.mean():.0f}%")
        hi = rec >= 35
        if hi.sum() > 0:
            print(f"  high-recoil (>=35 deg, n={int(hi.sum())}): "
                  f"iters FM {np.median(it_fm[hi]):.1f} vs cold {np.median(it_cold[hi]):.1f}; "
                  f"gate FM {100*g_fm[hi].mean():.0f}% vs cold {100*g_cold[hi].mean():.0f}%")


if __name__ == '__main__':
    main()
