"""render.py -- offscreen rendering of a free-floating grapple (MuJoCo).

Kinematic playback: qpos is set directly from the trajectory. The base pose comes from
the momentum map, so the picture is exactly the evaluated trajectory and the numbers
in the HUD are the numbers in the tables.

Rendering notes:
  * the offscreen framebuffer defaults to 640x480 -> set <global offwidth/offheight>
  * offscreen MSAA is <quality offsamples>, not the onscreen setting
  * MUJOCO_GL must be egl (headless GPU) or osmesa (CPU)
  * freejoint qvel[3:6] is BODY-LOCAL angular velocity (see core.py header)
  * mjv decor geoms are added after update_scene and before render
"""

import os
import json
import math
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from .core import Layout, quat_norm, pose_error

# --------------------------------------------------------------------- style

C_FM = (0.13, 0.85, 0.47, 1.0)          # momentum-projected: green
C_COLD = (0.95, 0.35, 0.28, 1.0)        # recoil-naive: red
C_GRASP = (1.00, 0.84, 0.20, 1.0)       # grasp frame: amber
C_ERR = (1.00, 0.25, 0.25, 1.0)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _font(size):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ----------------------------------------------------------------- decor api

def _add(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g,
                        int(gtype),
                        np.asarray(size, dtype=np.float64),
                        np.asarray(pos, dtype=np.float64),
                        np.asarray(mat, dtype=np.float64).reshape(9),
                        np.asarray(rgba, dtype=np.float32))
    g.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1
    return g


def _connector(scene, a, b, width, rgba, gtype=mujoco.mjtGeom.mjGEOM_CAPSULE):
    g = _add(scene, gtype, [width, width, width], np.zeros(3), np.eye(3), rgba)
    if g is None:
        return
    mujoco.mjv_connector(g, int(gtype), float(width),
                         np.asarray(a, dtype=np.float64),
                         np.asarray(b, dtype=np.float64))


def draw_frame(scene, T, length=0.20, width=0.010, alpha=1.0):
    """SE(3) triad. The paper is about SE(3) poses, so show orientation, not
    just a dot, so the grasp frame can be seen rotating."""
    p, R = T[:3, 3], T[:3, :3]
    for i, c in enumerate(((1, .25, .25), (.35, 1, .35), (.35, .55, 1))):
        _connector(scene, p, p + R[:, i] * length, width, (*c, alpha),
                   mujoco.mjtGeom.mjGEOM_ARROW1)


def draw_trail(scene, pts, rgba, width=0.013, max_seg=140):
    if len(pts) < 2:
        return
    idx = np.linspace(0, len(pts) - 1, min(len(pts), max_seg)).astype(int)
    p = np.asarray(pts)[idx]
    n = len(p)
    for i in range(n - 1):
        a = 0.25 + 0.75 * (i / max(1, n - 2))       # fade the tail
        _connector(scene, p[i], p[i + 1], width, (*rgba[:3], rgba[3] * a))


def draw_miss(scene, p_grip, p_grasp, rgba=C_ERR, width=0.006):
    """The error vector, drawn. This is the figure's punchline."""
    _connector(scene, p_grip, p_grasp, width, rgba)
    _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [width * 3.2] * 3, p_grip,
         np.eye(3), rgba)


# ------------------------------------------------------------------- traject

class Traj:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.t = z["t"]
        self.theta_fm = z["theta_fm"]
        self.theta_cold = z["theta_cold"]
        self.base_fm = z["base_fm"]
        self.base_cold = z["base_cold"]
        self.target = z["target"]
        self.grasp = z["grasp"]
        self.meta = json.loads(str(z["meta"]))
        self.K = len(self.t)

    def at(self, k):
        return dict(theta_fm=self.theta_fm[k], theta_cold=self.theta_cold[k],
                    base_fm=self.base_fm[k], base_cold=self.base_cold[k],
                    target=self.target[k], grasp=self.grasp[k])


# ------------------------------------------------------------------ renderer

class GrappleRenderer:
    def __init__(self, model_path, width=1920, height=1080, ss=2,
                 ghost_prefix="gh_", max_geom=20000):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.W, self.H, self.ss = int(width), int(height), int(ss)
        rw, rh = self.W * self.ss, self.H * self.ss
        ow = int(self.model.vis.global_.offwidth)
        oh = int(self.model.vis.global_.offheight)
        if rw > ow or rh > oh:
            raise SystemExit(
                f"offscreen buffer is {ow}x{oh} but you asked to render "
                f"{rw}x{rh}.\nAdd to your MJCF:\n"
                f'  <visual><global offwidth="{rw}" offheight="{rh}"/></visual>\n'
                "(this is the single most common cause of blurry MuJoCo figures)")

        self.renderer = mujoco.Renderer(self.model, rh, rw, max_geom=max_geom)
        self.scene = self.renderer.scene
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1

        self.opt = mujoco.MjvOption()
        for f in (mujoco.mjtVisFlag.mjVIS_JOINT, mujoco.mjtVisFlag.mjVIS_ACTUATOR,
                  mujoco.mjtVisFlag.mjVIS_COM, mujoco.mjtVisFlag.mjVIS_CONSTRAINT,
                  mujoco.mjtVisFlag.mjVIS_PERTFORCE):
            self.opt.flags[f] = 0
        self.opt.sitegroup[3] = 0        # hide the raw site markers; we draw triads

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        self.fm = Layout(self.model)
        gp = ghost_prefix
        self.cold = Layout(self.model,
                           base_joint=gp + "base_free",
                           target_joint="target_free",
                           arm_joints=tuple(gp + f"j{i}" for i in range(1, 7)),
                           gripper_site=gp + "gripper",
                           grasp_site="grasp",
                           base_body=gp + "base",
                           target_body="target")

    # -- camera --------------------------------------------------------------
    def set_camera(self, lookat, distance, azimuth, elevation):
        self.cam.lookat[:] = lookat
        self.cam.distance = float(distance)
        self.cam.azimuth = float(azimuth)
        self.cam.elevation = float(elevation)

    def auto_camera(self, traj, pad=1.35, azimuth=138.0, elevation=-15.0):
        """Establishing shot framing base, target and grasp path (Fig. 3 uses
        azimuth 138 deg, elevation -15 deg)."""
        pts = np.vstack([traj.base_fm[:, :3], traj.target[:, :3], traj.grasp[:, :3]])
        lo, hi = pts.min(0), pts.max(0)
        c = (lo + hi) / 2
        ext = float(np.linalg.norm(hi - lo))
        if azimuth is None:
            azimuth = 138.0
        self.set_camera(c, max(2.4, ext * pad), float(azimuth), elevation)

    # -- state ---------------------------------------------------------------
    def _set_kinematic(self, s):
        d = self.data
        self.fm.set_base(d, s["base_fm"][:3], s["base_fm"][3:])
        self.fm.set_theta(d, s["theta_fm"])
        self.cold.set_base(d, s["base_cold"][:3], s["base_cold"][3:])
        self.cold.set_theta(d, s["theta_cold"])
        a = self.fm.tgt_qadr
        d.qpos[a:a + 3] = s["target"][:3]
        d.qpos[a + 3:a + 7] = quat_norm(s["target"][3:])
        mujoco.mj_forward(self.model, d)

    # -- one frame -----------------------------------------------------------
    def frame(self, traj, k, trails=None, show_frames=True, show_miss=True,
              hud=True, orbit_deg=0.0, decor_scale=None):
        """decor_scale: triad/trail/marker sizes are in METRES, so a decoration
        tuned for a 4 m establishing shot swallows a 0.6 m close-up. Left at
        None it is derived from the camera distance."""
        s = traj.at(k)
        self._set_kinematic(s)
        ds = float(np.clip(self.cam.distance / 4.0, 0.16, 1.25)) \
            if decor_scale is None else float(decor_scale)

        if orbit_deg:
            self.cam.azimuth += orbit_deg

        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)

        T_grip = self.fm.gripper_T(self.data)
        T_cold = self.cold.gripper_T(self.data)
        T_grasp = self.fm.grasp_T(self.data)

        if trails is not None:
            draw_trail(self.scene, trails["fm"][:k + 1], C_FM, width=0.013 * ds)
            draw_trail(self.scene, trails["cold"][:k + 1], C_COLD, width=0.013 * ds)
            draw_trail(self.scene, trails["grasp"][:k + 1], C_GRASP, width=0.011 * ds)

        if show_frames:
            draw_frame(self.scene, T_grasp, length=0.26 * ds, width=0.013 * ds)
            draw_frame(self.scene, T_grip, length=0.18 * ds, width=0.010 * ds)
            draw_frame(self.scene, T_cold, length=0.18 * ds, width=0.010 * ds, alpha=0.55)

        if show_miss:
            draw_miss(self.scene, T_cold[:3, 3], T_grasp[:3, 3], C_COLD, width=0.011 * ds)
            _add(self.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.034 * ds] * 3,
                 T_grasp[:3, 3], np.eye(3), C_GRASP)

        px = self.renderer.render()
        img = Image.fromarray(px)
        if self.ss > 1:
            img = img.resize((self.W, self.H), Image.LANCZOS)

        if hud:
            _, e_fm, r_fm = pose_error(T_grasp, T_grip)
            _, e_cd, r_cd = pose_error(T_grasp, T_cold)
            self._hud(img, traj, k, e_fm, r_fm, e_cd, r_cd)
        return img

    # -- overlay -------------------------------------------------------------
    def _hud(self, img, traj, k, e_fm, r_fm, e_cd, r_cd):
        d = ImageDraw.Draw(img, "RGBA")
        S = self.H / 1080.0
        f_ttl = _font(int(30 * S))
        f_big = _font(int(40 * S))
        f_med = _font(int(25 * S))
        f_sml = _font(int(20 * S))
        m = traj.meta
        pad = int(26 * S)

        # ---- title block
        d.rectangle([0, 0, self.W, int(74 * S)], fill=(6, 8, 14, 205))
        d.text((pad, int(20 * S)),
               "Free-floating SE(3) grapple  |  momentum-projected vs recoil-naive warm-start",
               font=f_ttl, fill=(232, 236, 244, 255))
        hdr = f"seed {m['seed']}   base recoil {m['recoil_deg']:.1f}°"
        try:
            hw = d.textlength(hdr, font=f_med)
        except Exception:
            hw = len(hdr) * 14 * S
        d.text((self.W - pad - hw, int(24 * S)), hdr,
               font=f_med, fill=(168, 178, 196, 255))

        # ---- readout panel
        w, h = int(560 * S), int(232 * S)
        x0, y0 = pad, self.H - h - pad
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=(6, 8, 14, 200))
        d.rectangle([x0, y0, x0 + w, y0 + h], outline=(70, 80, 100, 220), width=max(1, int(2 * S)))

        d.text((x0 + int(20 * S), y0 + int(14 * S)),
               f"t = {traj.t[k]:5.2f} / {m['tf']:.1f} s", font=f_med,
               fill=(190, 198, 214, 255))

        yy = y0 + int(56 * S)
        sw = int(16 * S)
        d.rectangle([x0 + int(20 * S), yy + int(12 * S), x0 + int(20 * S) + sw, yy + int(12 * S) + sw],
                    fill=tuple(int(255 * c) for c in C_FM[:3]) + (255,))
        d.text((x0 + int(46 * S), yy), "momentum-projected", font=f_sml, fill=(206, 214, 228, 255))
        d.text((x0 + int(46 * S), yy + int(24 * S)),
               f"{e_fm*100:6.2f} cm   {math.degrees(r_fm):5.2f}°", font=f_big,
               fill=tuple(int(255 * c) for c in C_FM[:3]) + (255,))

        yy += int(88 * S)
        d.rectangle([x0 + int(20 * S), yy + int(12 * S), x0 + int(20 * S) + sw, yy + int(12 * S) + sw],
                    fill=tuple(int(255 * c) for c in C_COLD[:3]) + (255,))
        d.text((x0 + int(46 * S), yy), "recoil-naive (ghost)", font=f_sml, fill=(206, 214, 228, 255))
        d.text((x0 + int(46 * S), yy + int(24 * S)),
               f"{e_cd*100:6.2f} cm   {math.degrees(r_cd):5.2f}°", font=f_big,
               fill=tuple(int(255 * c) for c in C_COLD[:3]) + (255,))

        # ---- momentum residual, bottom right: the Prop-1 receipt, on screen
        res = m.get("momentum_residual", None)
        if res is not None and res > 0:
            txt = f"max |h(q,u)| = {res:.1e}   (momentum conserved by construction)"
            try:
                tw = d.textlength(txt, font=f_sml)
            except Exception:
                tw = len(txt) * 11 * S
            d.text((self.W - pad - tw, self.H - pad - int(30 * S)), txt,
                   font=f_sml, fill=(150, 160, 180, 235))

    # -- clip ----------------------------------------------------------------
    def render_clip(self, traj, out_mp4, fps=30, orbit_total=18.0,
                    hold_end=1.2, crf=18, preset="medium", verbose=True):
        """Node trajectories are coarse (K=61 over 6 s). Interpolate to fps for
        a smooth clip; SLERP the quaternions, don't lerp them."""
        import imageio.v2 as imageio
        from scipy.spatial.transform import Rotation as Rot, Slerp
        from scipy.interpolate import CubicSpline

        t = traj.t
        tt = np.linspace(t[0], t[-1], int(round((t[-1] - t[0]) * fps)))

        def spl(a):
            return CubicSpline(t, a, axis=0)(tt)

        def slerp(q):
            r = Rot.from_quat(np.column_stack([q[:, 1:], q[:, 0]]))  # wxyz->xyzw
            out = Slerp(t, r)(tt).as_quat()
            return np.column_stack([out[:, 3], out[:, :3]])          # back to wxyz

        dense = Traj.__new__(Traj)
        dense.t = tt
        dense.K = len(tt)
        dense.meta = traj.meta
        dense.theta_fm = spl(traj.theta_fm)
        dense.theta_cold = spl(traj.theta_cold)
        dense.base_fm = np.column_stack([spl(traj.base_fm[:, :3]), slerp(traj.base_fm[:, 3:])])
        dense.base_cold = np.column_stack([spl(traj.base_cold[:, :3]), slerp(traj.base_cold[:, 3:])])
        dense.target = np.column_stack([spl(traj.target[:, :3]), slerp(traj.target[:, 3:])])
        dense.grasp = np.column_stack([spl(traj.grasp[:, :3]), slerp(traj.grasp[:, 3:])])

        trails = self._trails(dense)
        self.auto_camera(dense)
        n = dense.K
        step = orbit_total / max(1, n - 1)

        w = imageio.get_writer(out_mp4, fps=fps, codec="libx264",
                               quality=None, ffmpeg_params=[
                                   "-crf", str(crf), "-preset", preset,
                                   "-pix_fmt", "yuv420p"],
                               macro_block_size=1)
        try:
            for k in range(n):
                img = self.frame(dense, k, trails=trails, orbit_deg=step)
                w.append_data(np.asarray(img))
                if verbose and k % 30 == 0:
                    print(f"    frame {k+1}/{n}")
            for _ in range(int(hold_end * fps)):        # freeze on capture
                w.append_data(np.asarray(img))
        finally:
            w.close()
        return out_mp4

    def _trails(self, traj):
        fm, cd, gr = [], [], []
        for k in range(traj.K):
            self._set_kinematic(traj.at(k))
            fm.append(self.fm.gripper_T(self.data)[:3, 3].copy())
            cd.append(self.cold.gripper_T(self.data)[:3, 3].copy())
            gr.append(self.fm.grasp_T(self.data)[:3, 3].copy())
        return dict(fm=np.array(fm), cold=np.array(cd), grasp=np.array(gr))

    def render_closeup(self, traj, k, hud=False, trails=None, pad=3.0,
                       min_dist=0.95, square=True, azimuth=None, elevation=None):
        """Tight view framing the grasp frame AND both grippers. A 3.8 cm error
        is invisible in a 4 m establishing shot; this is the panel that actually
        shows the claim, so it goes in as an inset on the capture frame.

        Framing is derived from the three points, not a fixed distance --
        otherwise a large miss falls outside the crop, which is exactly the
        case you most need to see."""
        self._set_kinematic(traj.at(k))
        pts = np.vstack([self.fm.grasp_T(self.data)[:3, 3],
                         self.fm.gripper_T(self.data)[:3, 3],
                         self.cold.gripper_T(self.data)[:3, 3]])
        c = pts.mean(0)
        spread = float(np.linalg.norm(pts.max(0) - pts.min(0)))

        # Look at the grasp FROM THE SPACECRAFT SIDE, offset for a 3/4 view.
        # Inheriting the establishing shot's azimuth puts the camera behind the
        # target half the time, and the inset then shows a red wall.
        p_base = self.fm.get_base(self.data)[0]
        u = c - p_base
        auto_az = np.degrees(np.arctan2(u[1], u[0])) + 34.0
        az = auto_az if azimuth is None else azimuth
        el = -14.0 if elevation is None else elevation
        saved = (self.cam.lookat.copy(), self.cam.distance,
                 self.cam.azimuth, self.cam.elevation)
        self.set_camera(c, float(np.clip(spread * pad, min_dist, 2.20)), az, el)
        try:
            img = self.frame(traj, k, trails=trails, show_frames=True,
                             show_miss=True, hud=hud)
        finally:
            self.cam.lookat[:] = saved[0]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = saved[1:]
        if square:
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w - side) // 2, (h - side) // 2,
                            (w + side) // 2, (h + side) // 2))
        return img

    def render_stills(self, traj, out_dir, fracs=(0.0, 0.4, 0.72, 1.0), tag="s"):
        os.makedirs(out_dir, exist_ok=True)
        trails = self._trails(traj)
        self.auto_camera(traj)
        paths = []
        for i, f in enumerate(fracs):
            k = int(round(f * (traj.K - 1)))
            img = self.frame(traj, k, trails=trails,
                             show_miss=(f > 0.5), hud=True)
            p = os.path.join(out_dir, f"{tag}_{i}_t{traj.t[k]:04.1f}.png")
            img.save(p, optimize=True)
            paths.append(p)
        return paths
