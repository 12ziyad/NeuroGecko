#!/usr/bin/env python3
"""
watch_random.py  --  drive GeckoBody-R with smoothed random actions, render a
video, and report stability statistics (NaN, velocity bounds, tail-tip speed).

Smoothed (low-pass) random control approximates what a trained policy emits,
so the motion is representative rather than spasmodic. Use --harsh for the
full-range step-command stress test instead.

Headless-safe: always writes a video via EGL; the interactive viewer is
attempted and skipped gracefully if no display exists.

Usage:
    python watch_random.py [path/to/gecko_body_r.xml] [--harsh] [--secs N] [--no-view]
"""
import os, sys, gc, contextlib
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

DEFAULT_XML = Path(__file__).resolve().parent.parent / "morphology" / "gecko_body_r.xml"
argv = sys.argv[1:]
HARSH = "--harsh" in argv
DO_VIEW = "--no-view" not in argv
SECS = 8.0
skip = set()
if "--secs" in argv:
    i = argv.index("--secs"); SECS = float(argv[i + 1]); skip |= {i, i + 1}
pos_args = [a for k, a in enumerate(argv) if not a.startswith("--") and k not in skip]
XML = Path(pos_args[0]) if pos_args else DEFAULT_XML
SEED = 0
OUT = Path.cwd() / "gecko_out"; OUT.mkdir(exist_ok=True)

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
kstand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
tail5 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tail5")
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

def tip_speed():
    v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, tail5, v, 0)
    return float(np.linalg.norm(v[3:6]))

rng = np.random.default_rng(SEED)
mujoco.mj_resetDataKeyframe(m, d, kstand); mujoco.mj_forward(m, d)

FPS = 50
frame_every = max(1, int((1.0 / FPS) / m.opt.timestep))
frames = []
stats = dict(nan=False, qvel=0.0, pos=0.0, tail=0.0, tail_mean=0.0)
n_tail = 0
a = np.zeros(m.nu); tg = np.zeros(m.nu); hold = 0
nsteps = int(SECS / m.opt.timestep)

renderer = None
try:
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        renderer = mujoco.Renderer(m, 480, 640)
except Exception as e:
    print("offscreen renderer unavailable:", str(e)[:120])

for k in range(nsteps):
    if HARSH:
        if hold <= 0:
            tg = lo + (hi - lo) * rng.random(m.nu); hold = int(0.12 / m.opt.timestep)
        a = tg; hold -= 1
    else:
        a = 0.985 * a + 0.015 * (lo + (hi - lo) * rng.random(m.nu))  # smoothed
    d.ctrl[:] = a
    mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qpos)):
        stats["nan"] = True; print(f"NaN at t={k*m.opt.timestep:.3f}s"); break
    stats["qvel"] = max(stats["qvel"], float(np.abs(d.qvel).max()))
    stats["pos"]  = max(stats["pos"], float(np.abs(d.xpos[1:]).max()))
    ts = tip_speed(); stats["tail"] = max(stats["tail"], ts)
    stats["tail_mean"] += ts; n_tail += 1
    if renderer is not None and k % frame_every == 0:
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = d.subtree_com[bid("trunk_middle")]
            cam.distance, cam.azimuth, cam.elevation = 0.34, 130, -18
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

if renderer is not None:
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        del renderer; gc.collect()

stats["tail_mean"] /= max(1, n_tail)
mode = "HARSH full-range steps" if HARSH else "smoothed (policy-like)"
print(f"model        : {XML}")
print(f"control mode : {mode}, {SECS:.1f}s, seed {SEED}")
print(f"finite       : {not stats['nan']}")
print(f"max |qvel|   : {stats['qvel']:.2f} rad/s")
print(f"max |pos|    : {stats['pos']:.3f} m   (bounded => no detachment/explosion)")
print(f"tail-tip     : max {stats['tail']:.2f} m/s | mean {stats['tail_mean']:.2f} m/s "
      f"({'no violent whip' if stats['tail'] < (3.0 if HARSH else 1.5) else 'WHIP?'})")

# ---- write video -----------------------------------------------------------
if frames:
    try:
        import imageio.v2 as imageio
        mp4 = OUT / ("random_harsh.mp4" if HARSH else "random_smooth.mp4")
        try:
            imageio.mimwrite(mp4, frames, fps=FPS, quality=8)
            print(f"\nvideo: {mp4} ({len(frames)} frames @ {FPS} fps)")
        except Exception:
            gif = mp4.with_suffix(".gif")
            imageio.mimwrite(gif, frames[::2], duration=2.0 / FPS)
            print(f"\nvideo (gif fallback): {gif}")
    except Exception as e:
        print("video write failed:", str(e)[:160])

# ---- interactive viewer (guarded) -----------------------------------------
def has_display():
    import platform
    if platform.system() in ("Darwin", "Windows"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

if DO_VIEW and not has_display():
    print("\n(no display detected -> skipping interactive viewer; the video above shows the rollout.)")
elif DO_VIEW:
    try:
        import mujoco.viewer, time
        print("\nopening interactive viewer with live random actions (close to quit)...")
        mujoco.mj_resetDataKeyframe(m, d, kstand); mujoco.mj_forward(m, d)
        a = np.zeros(m.nu); rng = np.random.default_rng(SEED)
        with mujoco.viewer.launch_passive(m, d) as v:
            while v.is_running():
                a = 0.985 * a + 0.015 * (lo + (hi - lo) * rng.random(m.nu))
                d.ctrl[:] = a; mujoco.mj_step(m, d); v.sync()
                time.sleep(m.opt.timestep)
    except Exception as e:
        print("interactive viewer unavailable:", str(e)[:120])
        print("The rendered video above shows the random-action rollout.")
