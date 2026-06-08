#!/usr/bin/env python3
"""
view_standing.py  --  load GeckoBody-R, settle into the standing pose, render
the head camera + external views to PNG, and (where a display exists) open the
interactive viewer.

Headless-safe: always writes PNGs via EGL; the interactive viewer is attempted
and skipped gracefully if no display is available.

Usage:
    python view_standing.py [path/to/gecko_body_r.xml] [--no-view]
"""
import os, sys, gc, contextlib
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

DEFAULT_XML = Path(__file__).resolve().parent.parent / "morphology" / "gecko_body_r.xml"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
XML = Path(args[0]) if args else DEFAULT_XML
DO_VIEW = "--no-view" not in sys.argv
OUT = Path.cwd() / "gecko_out"; OUT.mkdir(exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand"))

# settle while holding the neutral standing targets
d.ctrl[:] = 0.0
for _ in range(int(1.5 / m.opt.timestep)):
    mujoco.mj_step(m, d)

# ---- stance summary --------------------------------------------------------
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
G = mujoco.mjtGeom
def geom_lowest_z(i):
    p = d.geom_xpos[i]; R = d.geom_xmat[i].reshape(3, 3); s = m.geom_size[i]; t = m.geom_type[i]
    if t == G.mjGEOM_SPHERE:    return p[2] - s[0]
    if t == G.mjGEOM_CAPSULE:   return p[2] - abs(R[2, 2]) * s[1] - s[0]
    if t == G.mjGEOM_ELLIPSOID: return p[2] - np.linalg.norm(R[2, :3] * s[:3])
    if t == G.mjGEOM_BOX:       return p[2] - float(np.abs(R[2, :3]) @ s[:3])
    return p[2] - m.geom_rbound[i]
def body_minz(n):
    b = bid(n)
    zs = [geom_lowest_z(i) for i in range(m.ngeom)
          if m.geom_bodyid[i] == b and m.geom_group[i] == 3]
    return (min(zs) if zs else 0.0) * 1000
def sens(n):
    s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)
    return float(d.sensordata[m.sensor_adr[s]])
print(f"model            : {XML}")
print(f"root height      : {d.qpos[2]*1000:.2f} mm")
print(f"belly clearance  : mid {body_minz('trunk_middle'):.2f} mm | post {body_minz('trunk_posterior'):.2f} mm")
print("foot contact (N) : " + "  ".join(
    f"{f.split('_',1)[1]}={sens(f):.3f}" for f in
    ("touch_fore_L", "touch_fore_R", "touch_hind_L", "touch_hind_R")))
print(f"settled max|qvel|: {np.abs(d.qvel).max():.4f} rad/s")

# ---- render views ----------------------------------------------------------
def render(camera=None, w=640, h=480, free=None):
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        r = mujoco.Renderer(m, h, w)
        if free is not None:
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = free.get("lookat", [0, 0, 0.02])
            cam.distance = free.get("distance", 0.32)
            cam.azimuth = free.get("azimuth", 135)
            cam.elevation = free.get("elevation", -20)
            r.update_scene(d, camera=cam)
        else:
            r.update_scene(d, camera=camera)
        img = r.render().copy()
        del r; gc.collect()
    return img

try:
    import imageio.v2 as imageio
    saved = []
    imageio.imwrite(OUT / "stand_head_cam_64.png", render(camera="head_cam", w=64, h=64));   saved.append("stand_head_cam_64.png")
    imageio.imwrite(OUT / "stand_head_cam.png",    render(camera="head_cam", w=256, h=256));  saved.append("stand_head_cam.png")
    imageio.imwrite(OUT / "stand_side.png",        render(free=dict(azimuth=90,  elevation=-12))); saved.append("stand_side.png")
    imageio.imwrite(OUT / "stand_persp.png",       render(free=dict(azimuth=135, elevation=-25))); saved.append("stand_persp.png")
    imageio.imwrite(OUT / "stand_top.png",         render(free=dict(azimuth=90,  elevation=-85))); saved.append("stand_top.png")
    print("\nrendered:", ", ".join(saved), f"\n  -> {OUT}")
except Exception as e:
    print("render failed:", str(e)[:160])

# ---- interactive viewer (guarded) -----------------------------------------
def has_display():
    import platform
    if platform.system() in ("Darwin", "Windows"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

if DO_VIEW and not has_display():
    print("\n(no display detected -> skipping interactive viewer; PNGs above show the pose.")
    print(" run on a machine with a display, or pass nothing on a desktop, for the live viewer.)")
elif DO_VIEW:
    try:
        import mujoco.viewer
        print("\nopening interactive viewer (Esc / close window to quit)...")
        with mujoco.viewer.launch_passive(m, d) as v:
            import time
            while v.is_running():
                d.ctrl[:] = 0.0
                mujoco.mj_step(m, d)
                v.sync()
                time.sleep(m.opt.timestep)
    except Exception as e:
        print("interactive viewer unavailable:", str(e)[:120])
        print("PNGs above contain the standing pose.")
