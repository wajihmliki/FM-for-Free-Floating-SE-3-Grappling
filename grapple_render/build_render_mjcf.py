"""build_render_mjcf.py -- render-ready MJCF matching FFSM3DParams exactly.

Adds to the dynamics model of build_mjcf_se3.py the names grapple/Layout expects
(base_free, j1..j6, gripper, target_free, grasp), a free-floating target body,
collision geometry on the arm and target, visual panels, and a large offscreen buffer.

Inertias are read from FFSM3DParams, so MuJoCo's mass matrix -- and therefore
reconstruct_base -- matches the dynamics used everywhere else.

    python3 build_render_mjcf.py --out ffsm_render.xml
"""
import argparse, os, sys, numpy as np


def _inertia(I):
    return f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"


def build(p, target_half=0.25, target_mass=80.0):
    n = p.n
    L = ['<mujoco model="ffsm_se3_render">',
         '  <option gravity="0 0 0" integrator="RK4" timestep="0.002"/>',
         '  <compiler angle="radian" inertiafromgeom="false"/>',
         '  <visual><global offwidth="3840" offheight="2160"/></visual>',
         '  <asset>',
         '    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.1 0.12"',
         '             rgb2="0.12 0.12 0.15" width="300" height="300"/>',
         '  </asset>',
         '  <worldbody>',
         '    <light pos="2 -2 3" dir="-0.5 0.5 -1" diffuse="0.9 0.9 0.9"/>',
         '    <light pos="-2 2 2" dir="0.5 -0.5 -1" diffuse="0.35 0.35 0.4"/>',
         '    <body name="base" pos="0 0 0">',
         '      <freejoint name="base_free"/>',
         f'      <inertial pos="0 0 0" mass="{p.m0:.6g}" fullinertia="{_inertia(p.I0)}"/>',
         '      <geom name="base_geom" type="box" size="0.3 0.3 0.22"',
         '            rgba="0.72 0.74 0.78 1" contype="1" conaffinity="1"/>',
         '      <geom name="panel_l" type="box" size="0.45 0.02 0.22" pos="-0.78 0 0"',
         '            rgba="0.10 0.16 0.38 1" contype="0" conaffinity="0"/>',
         '      <geom name="panel_r" type="box" size="0.45 0.02 0.22" pos="0.78 0 0"',
         '            rgba="0.10 0.16 0.38 1" contype="0" conaffinity="0"/>']
    ba, ind, opened = p.base_attach, '      ', 1
    for i in range(n):
        pos = ba if i == 0 else p.l[i - 1]
        ax = p.axis[i]
        L += [f'{ind}<body name="link{i+1}" pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}">',
              f'{ind}  <joint name="j{i+1}" type="hinge" axis="{ax[0]:.6g} {ax[1]:.6g} {ax[2]:.6g}"'
              ' pos="0 0 0" limited="false" damping="0"/>',
              f'{ind}  <inertial pos="{p.r[i][0]:.6g} {p.r[i][1]:.6g} {p.r[i][2]:.6g}"'
              f' mass="{p.m[i]:.6g}" fullinertia="{_inertia(p.Ilink[i])}"/>',
              f'{ind}  <geom name="g{i+1}" type="capsule" fromto="0 0 0'
              f' {p.l[i][0]:.6g} {p.l[i][1]:.6g} {p.l[i][2]:.6g}" size="0.035"'
              ' rgba="0.95 0.55 0.12 1" contype="1" conaffinity="1"/>']
        ind += '  '; opened += 1
    tip = p.l[n - 1]
    L.append(f'{ind}<site name="gripper" pos="{tip[0]:.6g} {tip[1]:.6g} {tip[2]:.6g}"'
             ' size="0.025" rgba="0.15 0.9 0.45 1"/>')
    for _ in range(opened):
        ind = ind[:-2] if len(ind) > 6 else '    '
        L.append(f'{ind}</body>' if ind != '    ' else '    </body>')
    h = target_half
    L += ['    <body name="target" pos="3 0 0">',
          '      <freejoint name="target_free"/>',
          f'      <inertial pos="0 0 0" mass="{target_mass:.6g}"'
          f' diaginertia="{0.4*target_mass*h*h:.6g} {0.4*target_mass*h*h:.6g}'
          f' {0.4*target_mass*h*h:.6g}"/>',
          f'      <geom name="tgt_geom" type="box" size="{h:.6g} {h:.6g} {h:.6g}"'
          ' rgba="0.70 0.16 0.13 1" contype="1" conaffinity="1"/>',
          f'      <geom name="tgt_band" type="box" size="{h*0.12:.6g} {h*1.02:.6g} {h*1.02:.6g}"'
          ' pos="0 0 0" rgba="0.95 0.78 0.12 1" contype="0" conaffinity="0"/>',
          '      <site name="grasp" pos="0 0 0" size="0.03" rgba="0.95 0.78 0.12 1"/>',
          '    </body>',
          '  </worldbody>',
          '  <actuator>']
    L += [f'    <motor name="m{i+1}" joint="j{i+1}" gear="1" ctrllimited="false"/>' for i in range(n)]
    L += ['  </actuator>', '</mujoco>']
    return '\n'.join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-root", dest="code_root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--out", default="ffsm_render.xml")
    ap.add_argument("--target-half", type=float, default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.code_root)
    import ffsm_core_se3 as fc3, ffsm_demo_se3 as d3
    p = fc3.FFSM3DParams()
    half = a.target_half
    if half is None:
        try:
            tgt = d3.rand_inst3d(900000, 60, 6.0, p)["tgt"]
            half = float(getattr(tgt, "body_radius", 0.25))
            print(f"[build] target half-extent {half:.3f} m (from Target3D.body_radius)")
        except Exception as e:
            half = 0.25
            print(f"[build] could not read Target3D.body_radius ({e}); using {half} m. "
                  f"Pass --target-half to match your geometry.")
    xml = build(p, target_half=half)
    open(a.out, "w").write(xml)
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    print(f"[build] wrote {a.out}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}")
    print(f"[build] expect nq={7+p.n+7} nv={6+p.n+6} nu={p.n} nbody={2+p.n+1}")
    print(f"[build] arm+base mass {p.m0 + p.m.sum():.2f} kg (unchanged from FFSM3DParams)")
