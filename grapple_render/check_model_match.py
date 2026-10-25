#!/usr/bin/env python3
"""check_model_match.py -- MuJoCo reconstruction vs ffsm_core_se3 for the same joints.

Reconstructs the base pose with the MJCF's mass matrix (reconstruct_base) and with
ffsm_core_se3.rollout_reduced_se3_fast for the same joint trajectory. Sub-millimetre /
sub-0.1 deg agreement means the render model matches the dynamics.

    python check_model_match.py --model ffsm_render.xml
"""
import argparse, os, sys
import numpy as np

# Locate the grapple package. Works whether you run from the package dir, copy
# this script elsewhere (as long as the package came with it), or point --pkg
# at it explicitly.
def _find_pkg(explicit=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ([explicit] if explicit else []) + [here, os.getcwd()]:
        if cand and os.path.isdir(os.path.join(cand, "grapple")):
            sys.path.insert(0, cand)
            return cand
    sys.exit("ERROR: cannot find the 'grapple' package; run from grapple_render/ "
             "or pass --pkg /path/to/grapple_render.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ffsm_render.xml")
    ap.add_argument("--code-root", dest="code_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pkg", default=None,
                    help="path to the grapple_render package, if not the cwd")
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--tf", type=float, default=6.0)
    a = ap.parse_args()
    _find_pkg(a.pkg)
    if not os.path.exists(a.model):
        sys.exit(f"ERROR: no such model file: {a.model} (build it with build_render_mjcf.py)")
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from scipy.interpolate import CubicSpline
    from grapple.core import Layout, reconstruct_base, quat2mat, so3_log

    sys.path.insert(0, a.code_root)
    import ffsm_core_se3 as fc3
    import ffsm_demo_se3 as d3
    from export_trajectories import _load_pipeline, _thd_of

    net, ck, p, N, dev = _load_pipeline(a.code_root, ckpt=a.ckpt)
    inst = d3.rand_inst3d(a.seed, N, a.tf, p)
    K = N + 1
    ts = np.linspace(0.0, a.tf, K)

    rng = np.random.default_rng(0)
    th0 = np.asarray(inst["th0"], float)
    th = th0[None, :] + np.outer(10 * (ts / a.tf) ** 3 - 15 * (ts / a.tf) ** 4
                                 + 6 * (ts / a.tf) ** 5,
                                 rng.uniform(-0.8, 0.8, len(th0)))

    Q = fc3.rollout_reduced_se3_fast(inst["base0"], th, _thd_of(th, a.tf),
                                     a.tf / (K - 1), p, 4)

    model = mujoco.MjModel.from_xml_path(a.model)
    lay = Layout(model)
    d = mujoco.MjData(model)
    cs = CubicSpline(ts, th, axis=0)
    pp = np.asarray(inst["base0"][:3], float)
    qq = np.asarray(inst["base0"][3:7], float)
    ep, eq = [], []
    for k in range(1, K):
        pp, qq, _ = reconstruct_base(model, d, lay, lambda t: cs(t),
                                     lambda t: cs(t, 1), ts[k - 1], ts[k],
                                     pp, qq, dt=2e-3, check_residual=False)
        ep.append(np.linalg.norm(pp - np.asarray(Q[k, :3], float)))
        R1, R2 = quat2mat(qq), quat2mat(np.asarray(Q[k, 3:7], float))
        eq.append(np.degrees(np.linalg.norm(so3_log(R1.T @ R2))))

    print(f"\n  max base position disagreement : {max(ep)*1000:9.4f} mm")
    print(f"  max base attitude disagreement : {max(eq):9.4f} deg")
    # the two integrators use different step sizes; agreement well below the
    # 5 cm / 3 deg certificate gate is what matters
    ok = max(ep) < 1e-3 and max(eq) < 0.1
    print(f"\n  {'MATCH' if ok else 'MISMATCH (check masses/inertias, joint axes, freejoint angular-velocity frame)'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
