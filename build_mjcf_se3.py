"""build_mjcf_se3.py -- MuJoCo model of the free-floating base + 6R arm that matches
ffsm_core_se3.FFSM3DParams exactly (link vectors l, CoM offsets r, masses m, inertias
Ilink, joint axes, base m0/I0/base_attach), read programmatically.

Conventions matched to fk3d:
  - base body: free joint, mass m0, inertia I0, CoM at the body origin (p0);
  - arm root attached at base_attach in the base frame;
  - joint i rotates about axis[i] (parent link frame); link i is offset by l[i] to the
    next joint, with CoM r[i], mass m[i], inertia Ilink[i];
  - zero gravity; base unactuated (free joint, no motor); joints torque-actuated;
  - RK4 integrator, 2 ms timestep.

    python3 build_mjcf_se3.py          # writes ffsm_se3.xml
"""
import numpy as np
import ffsm_core_se3 as fc3


def _inertia_str(I):
    # MuJoCo fullinertia: ixx iyy izz ixy ixz iyz
    return f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"


def build_mjcf_xml(p=None):
    if p is None: p = fc3.FFSM3DParams()
    n = p.n
    # base: free body at origin, CoM at origin, inertia I0
    xml = ['<mujoco model="ffsm_se3">',
           '  <option gravity="0 0 0" integrator="RK4" timestep="0.002"/>',
           '  <compiler angle="radian" inertiafromgeom="false"/>',
           '  <worldbody>',
           '    <body name="base" pos="0 0 0">',
           '      <freejoint name="base_free"/>',
           f'      <inertial pos="0 0 0" mass="{p.m0:.6g}" fullinertia="{_inertia_str(p.I0)}"/>',
           '      <geom name="base_geom" type="box" size="0.3 0.3 0.22" rgba="0.5 0.55 0.6 0.3" contype="0" conaffinity="0"/>']
    # arm root offset: base_attach in base frame
    ba = p.base_attach
    indent = '      '
    # We nest bodies. Each joint i is placed at the *current* joint position; the link geometry
    # (CoM r[i], and the offset l[i] to the next joint) lives in that body's frame.
    # Body i origin = joint i location. Joint i rotates about axis[i]. Link i CoM at r[i]; child
    # (joint i+1) at l[i].
    open_bodies = 1  # base is open
    # first arm body placed at base_attach relative to base
    for i in range(n):
        pos = ba if i == 0 else p.l[i-1]
        ax = p.axis[i]
        xml.append(f'{indent}<body name="link{i}" pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}">')
        xml.append(f'{indent}  <joint name="j{i}" type="hinge" axis="{ax[0]:.6g} {ax[1]:.6g} {ax[2]:.6g}" pos="0 0 0" limited="false" damping="0"/>')
        xml.append(f'{indent}  <inertial pos="{p.r[i][0]:.6g} {p.r[i][1]:.6g} {p.r[i][2]:.6g}" mass="{p.m[i]:.6g}" fullinertia="{_inertia_str(p.Ilink[i])}"/>')
        xml.append(f'{indent}  <geom name="g{i}" type="capsule" fromto="0 0 0 {p.l[i][0]:.6g} {p.l[i][1]:.6g} {p.l[i][2]:.6g}" size="0.03" rgba="0.9 0.5 0.1 0.5" contype="0" conaffinity="0"/>')
        indent += '  '; open_bodies += 1
    # end-effector site at the tip of the last link (offset l[n-1] from last joint)
    xml.append(f'{indent}<site name="ee" pos="{p.l[n-1][0]:.6g} {p.l[n-1][1]:.6g} {p.l[n-1][2]:.6g}" size="0.02"/>')
    # close all bodies
    for _ in range(open_bodies):
        indent = indent[:-2] if len(indent) > 6 else '    '
        xml.append(f'{indent}</body>' if indent != '    ' else '    </body>')
    xml.append('  </worldbody>')
    # actuators: torque motors on the 6 joints only (base free joint unactuated)
    xml.append('  <actuator>')
    for i in range(n):
        xml.append(f'    <motor name="m{i}" joint="j{i}" gear="1" ctrllimited="false"/>')
    xml.append('  </actuator>')
    xml.append('</mujoco>')
    return '\n'.join(xml)


def build_model(p=None):
    import mujoco
    xml = build_mjcf_xml(p)
    model = mujoco.MjModel.from_xml_string(xml)
    meta = dict(xml=xml, n=(p or fc3.FFSM3DParams()).n)
    return model, meta


if __name__ == "__main__":
    import mujoco
    p = fc3.FFSM3DParams()
    xml = build_mjcf_xml(p); open('ffsm_se3.xml', 'w').write(xml)
    print(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    print(f"\nMuJoCo model built: nq={model.nq} nv={model.nv} nu={model.nu} (expect nq=7+{p.n}, nv=6+{p.n}, nu={p.n})")
    print(f"total mass = {sum(model.body_mass):.2f} (expect {p.m0 + p.m.sum():.2f})")
