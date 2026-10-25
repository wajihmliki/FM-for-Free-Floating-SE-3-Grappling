"""ffsm_demo_se3.py -- SE(3) grapple scenarios and the demonstrator (6R arm).

rand_inst3d(seed, ...) draws one held-out or training scenario in the on-orbit
servicing regime: target center at 1.8-2.2 m, near-face grasp handle, tumble rate
||omega_g|| <= 0.17 rad/s, tf = 6 s.

grapple3d(): momentum-aware multi-start terminal IK (the gripper meets the moving SE(3)
grasp frame after the base has recoiled), min-jerk ease from the start posture, a fast
reduced-rollout screen for reach and collision across IK branches, then certification
of the winner with the true-dynamics gate (verify3d).
"""
import numpy as np
from scipy.optimize import least_squares
import ffsm_core_se3 as fc3


def _minjerk(N, tf):
    s = np.linspace(0, 1, N + 1)
    return (10*s**3 - 15*s**4 + 6*s**5, (30*s**2 - 60*s**3 + 30*s**4)/tf,
            (60*s - 180*s**2 + 120*s**3)/tf**2)


def _seeds(th0, n, extra):
    S = [th0]
    flips = [[0,1,-1,0,-1,0],[0,-1,1,0,1,0],[1,1,-1,0,0,1],[-1,1,1,0,0,-1],[0,1,1,0,-1,0],[0,-1,-1,0,1,0]]
    for fl in flips[:extra]:
        S.append(np.clip(th0 + 0.8*np.array(fl[:n]), -2.6, 2.6))
    return S


def grapple3d(inst, branches=8, substeps=10, certify=True, verbose=False):
    """Reach-only multi-branch IK -> fast reduced feasibility (reach+collision) -> gate-certify.
       Returns (best_certified|None, best_effort) with fast fields; best has 'tau' if certify."""
    p, tgt = inst['p'], inst['tgt']; th0, base0, N, tf = inst['th0'], inst['base0'], inst['N'], inst['tf']
    ease, de, dde = _minjerk(N, tf); _, _, gp, gq = tgt.frame(tf); dt = tf/N
    def feat(thf, rb=2):
        th = th0 + np.outer(ease, thf-th0); thd = np.outer(de, thf-th0)
        Q = fc3.rollout_reduced_se3_fast(base0, th, thd, dt, p, substeps=rb); k = fc3.fk3d(Q[-1], p)
        return k['ee_pos'], fc3.ee_quat(Q[-1], p), Q, th, thd
    def ik_resid(thf):
        ee, eq, _, _, _ = feat(thf)
        return np.concatenate([ee-gp, 0.5*fc3.log_so3(fc3.qmul(fc3.qconj(eq), gq))])
    rng = np.random.default_rng(0)
    seeds = _seeds(th0, p.n, min(branches-1, 6)) + [rng.uniform(-2.2, 2.2, p.n) for _ in range(max(0, branches-7))]
    best = None; best_eff = None
    for st in seeds:
        sol = least_squares(ik_resid, st, bounds=(-2.7, 2.7), max_nfev=200, xtol=1e-12)
        ee, eq, Q, th, thd = feat(sol.x)
        pe = np.linalg.norm(ee-gp); ae = np.linalg.norm(fc3.log_so3(fc3.qmul(fc3.qconj(eq), gq)))
        mc = min(fc3.collision_clearances3d(Q[i], p, tgt, i*dt).min() for i in range(0, N+1, 2))
        fast_ok = (pe < 0.05 and ae < np.deg2rad(3) and mc > 0)
        info = dict(pos_err=pe, ang_err_deg=np.rad2deg(ae), min_clr=mc, fast_ok=fast_ok, thf=sol.x)
        if verbose:
            print(f"  seed: pos={pe*100:5.2f}cm ang={np.rad2deg(ae):5.2f} min_clr={mc:+.3f} fast_ok={fast_ok}")
        if best_eff is None or pe < best_eff['pos_err']: best_eff = info
        if fast_ok and (best is None or pe < best['pos_err']): best = info
        if fast_ok: break
    if best is not None and certify:                    # certify the winner via the true-dynamics gate
        thf = best['thf']; th = th0 + np.outer(ease, thf-th0); thd = np.outer(de, thf-th0); thdd = np.outer(dde, thf-th0)
        _, _, tau, _ = fc3.inverse_dynamics3d(th, thd, thdd, base0, dt, p)
        g = fc3.verify3d(inst, tau, substeps=substeps); g.update(thf=thf, tau=tau); best = g
    return best, best_eff


OMEGA_MAX = 0.17          # rad/s, maximum target tumble rate


def _quat_from_xaxis(xhat):
    xhat = xhat/np.linalg.norm(xhat); up = np.array([0,0,1.]) if abs(xhat[2])<0.9 else np.array([0,1.,0])
    yh = np.cross(up, xhat); yh /= np.linalg.norm(yh); zh = np.cross(xhat, yh)
    return fc3._rot_to_quat(np.column_stack([xhat, yh, zh]))

def rand_inst3d(seed, N, tf, p=None):
    """Grasp handle on the target's NEAR face (toward the base), gripper approaching along
       base->target, so the 6R arm extends to grasp. Target center at 1.8-2.2 m,
       ||omega_g|| <= 0.17 rad/s."""
    rng = np.random.default_rng(seed); p = p or fc3.FFSM3DParams()
    d = rng.uniform(1.8, 2.2); az = rng.uniform(-0.4, 0.4); el = rng.uniform(-0.25, 0.25)
    center = d*np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
    chat = center/np.linalg.norm(center)
    gp0 = center - 0.45*chat                                   # grasp point on near face
    gq0 = fc3.qnorm(fc3.qmul(_quat_from_xaxis(chat), fc3.exp_so3(rng.standard_normal(3)*0.15)))
    omega = rng.standard_normal(3)*0.045
    wn = np.linalg.norm(omega)
    if wn > OMEGA_MAX: omega = omega * (OMEGA_MAX / wn)           # ||omega_g|| <= 0.17 rad/s
    tgt = fc3.Target3D(p0=center, v=np.zeros(3), quat0=np.array([1.,0,0,0]),
                       omega=omega,
                       grasp_offset=gp0-center, grasp_quat=gq0, body_radius=0.30)
    th0 = np.array([0.2, 0.5, 0.4, 0.0, 0.3, 0.0])[:p.n] + rng.standard_normal(p.n)*0.05
    base0 = np.concatenate([np.zeros(3), np.array([1., 0, 0, 0])])
    return fc3.build_inst3d(p, tgt, th0, base0, tf, N)

if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20); ap.add_argument('--N', type=int, default=40)
    ap.add_argument('--tf', type=float, default=6.0); ap.add_argument('--branches', type=int, default=8)
    ap.add_argument('--seed0', type=int, default=2025)
    ap.add_argument('--fast', action='store_true', help='reduced feasibility only (no gate) -- fast preflight')
    a = ap.parse_args()
    p = fc3.FFSM3DParams(); feas = 0; t0 = time.time()
    for i in range(a.n):
        inst = rand_inst3d(a.seed0 + i, a.N, a.tf, p)
        b, be = grapple3d(inst, branches=a.branches, certify=not a.fast)
        ok = (b is not None) if a.fast else (b is not None and b.get('feasible', False))
        feas += ok
    tag = "reduced-feasible" if a.fast else "gate-certified"
    print(f"=== SE(3) preflight ({tag}): N={a.N} tf={a.tf}, n={a.n} ===")
    print(f"  demonstrator feasible: {feas}/{a.n}   ({time.time()-t0:.0f}s)")
