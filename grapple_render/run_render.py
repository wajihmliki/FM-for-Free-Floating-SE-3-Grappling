#!/usr/bin/env python3
"""
run_render.py -- render exported trajectories: stills, close-up insets, the Fig. 3
filmstrip, an error-vs-time companion plot and (optionally) HUD video clips.

    MUJOCO_GL=egl python run_render.py --npz 'traj/*.npz' --model ffsm_render.xml --out fig/ --no-video

The .npz files come from export_trajectories.py (schema in grapple/plan.py). Playback
is kinematic: qpos is set directly from the exported trajectory, whose base pose is the
momentum-consistent reconstruction, so the picture is exactly the evaluated trajectory.
"""

import argparse
import glob
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ffsm_render.xml")
    ap.add_argument("--out", default="fig")
    ap.add_argument("--npz", nargs="+", required=True,
                    help="trajectory files written by export_trajectories.py")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--azimuth", type=float, default=None,
                    help="camera azimuth in deg; default derives it from base->grasp")
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from grapple.core import Layout, selftest_conventions
    from grapple.dualize import dualize
    from grapple import figure
    from grapple.render import GrappleRenderer, Traj

    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    model = mujoco.MjModel.from_xml_path(a.model)
    lay = Layout(model)
    st = selftest_conventions(model, lay)
    print(f"[check] freejoint angular velocity frame: {st['verdict']} "
          f"(measured {np.round(st['local'], 4)})")
    print(f"[check] nq={model.nq} nv={model.nv} nu={model.nu}  "
          f"offscreen={model.vis.global_.offwidth}x{model.vis.global_.offheight}")

    npz_files = sorted(sum([glob.glob(p) for p in a.npz], []))
    if not npz_files:
        raise SystemExit(f"no files matched {a.npz}")
    print(f"[load]  {len(npz_files)} trajectories")

    # solid robot = momentum-projected run, translucent ghost = recoil-naive run
    dual = os.path.join(a.out, "dual_model.xml")
    dualize(a.model, dual, root_body="base", prefix="gh_")
    print(f"[model] dual scene -> {dual}")

    R = GrappleRenderer(dual, width=a.width, height=a.height, ss=a.ss)
    print(f"[render] {a.width}x{a.height} at {a.ss}x supersampling")

    panels, metas, curves, insets = [], [], [], []
    for i, p in enumerate(npz_files):
        tr = Traj(p)
        tag = f"sc{i+1}"
        print(f"[render] scenario {i+1}: stills")
        panels.append(_stills(R, tr, os.path.join(a.out, "panels"), tag,
                              a.azimuth, a.elevation))
        cu = R.render_closeup(tr, tr.K - 1)
        cup = os.path.join(a.out, "panels", f"{tag}_closeup.png")
        cu.save(cup, optimize=True)
        insets.append(cup)
        metas.append(tr.meta)
        curves.append(_error_curve(R, tr))
        if not a.no_video:
            mp4 = os.path.join(a.out, f"grapple_{tag}.mp4")
            print(f"[render] scenario {i+1}: clip -> {mp4}")
            R.render_clip(tr, mp4, fps=a.fps)

    cols = ["$t = 0$  (start)", "approach", "pre-capture", "$t = t_f$  (capture)"]
    pdf = os.path.join(a.out, "fig_grapple_filmstrip.pdf")
    png = os.path.join(a.out, "fig_grapple_filmstrip.png")
    figure.filmstrip(panels, metas, pdf, png, col_labels=cols,
                     crop=(0.04, 0.055, 0.965, 0.955), insets=insets)
    print(f"[figure] {pdf}")

    epdf = os.path.join(a.out, "fig_error_vs_time_render.pdf")
    figure.error_curves(curves, epdf, epdf.replace(".pdf", ".png"))
    print(f"[figure] {epdf}  (Fig. 4 itself is produced by plot_fig4.py)")

    summary = [dict(file=os.path.basename(f), **m) for f, m in zip(npz_files, metas)]
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] {time.time()-t_start:.1f}s  ->  {a.out}/")
    for s in summary:
        print(f"   rho={s['recoil_deg']:5.1f} deg   FM {s['reach_fm_cm']:6.2f} cm   "
              f"COLD {s['reach_cold_cm']:7.2f} cm")


def _stills(R, traj, out_dir, tag, azimuth, elevation, fracs=(0.0, 0.42, 0.75, 1.0)):
    os.makedirs(out_dir, exist_ok=True)
    trails = R._trails(traj)
    R.auto_camera(traj, azimuth=azimuth, elevation=elevation)
    out = []
    for i, f in enumerate(fracs):
        k = int(round(f * (traj.K - 1)))
        img = R.frame(traj, k, trails=trails, show_miss=(f > 0.5), hud=False)
        p = os.path.join(out_dir, f"{tag}_{i}.png")
        img.save(p, optimize=True)
        out.append(p)
    return out


def _error_curve(R, traj):
    from grapple.core import pose_error
    e_fm, e_cd = [], []
    for k in range(traj.K):
        R._set_kinematic(traj.at(k))
        Tg = R.fm.grasp_T(R.data)
        _, a, _ = pose_error(Tg, R.fm.gripper_T(R.data))
        _, b, _ = pose_error(Tg, R.cold.gripper_T(R.data))
        e_fm.append(a); e_cd.append(b)
    return (traj.t, e_fm, e_cd, traj.meta)


if __name__ == "__main__":
    main()
