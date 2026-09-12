"""Show what is under the skin: the physics body, and the joints that move it.

WHAT THIS IS AND IS NOT. It is NOT an anatomical skeleton. There are no bones in
this model, no skull, no vertebrae, no femur. What it shows is the body the
PHYSICS uses: 48 capsules, ellipsoids, spheres and boxes that carry every gram
of the 38 g budget, collide with the ground, and are what all 14 morphology
gates measure. The skin is a separate surface bound over the top and weighs
nothing (#318 measured that: with it and without it the animal's trajectory is
bit-identical). So this is the real animal and the skin is the costume, not the
other way round.

The x-ray section blends two render passes rather than using a transparency
flag, because that flag applies to geoms and would not make a SKIN see-through.
Two passes also mean the mix is a number here instead of a renderer setting.

Usage:  python tools/skeleton_view.py [--seconds 18] [--no-video]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools._tinyfont import draw_text                        # noqa: E402

FPS = 50
#: How much of the skin survives in the x-ray section. INVENTED: a look.
XRAY_SKIN = 0.42

#: Geom groups: 1 is the visual surface, 3 is the collision body. The skin is
#: its own list and sits in group 1.
def _option(mujoco, *, skin, visual, collision, joints=False):
    o = mujoco.MjvOption()
    o.geomgroup[:] = 0
    if visual:
        o.geomgroup[1] = 1
    if collision:
        o.geomgroup[3] = 1
    o.skingroup[:] = 0
    if skin:
        o.skingroup[1] = 1
    if joints:
        o.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--bones-only", action="store_true",
                    help="skip the skin and x-ray sections; film the physics body alone")
    ap.add_argument("--joints", action="store_true", default=None,
                    help="draw the joint arrows (default: on for --bones-only)")
    ap.add_argument("--out", default="artifacts/video/skeleton.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from envs.gecko_walk_env import GeckoWalkEnv

    env = GeckoWalkEnv(xml_path="morphology/gecko_world_v1.xml",
                       gait_profile="lab", control_mode="cpg_residual",
                       residual_scale=0.25, privileged_target=True)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    skin_only = _option(mujoco, skin=True, visual=False, collision=False)
    bones_only = _option(mujoco, skin=False, visual=False, collision=True)
    bones_joints = _option(mujoco, skin=False, visual=False, collision=True, joints=True)

    def draw(option):
        renderer.update_scene(env.data, camera=camera, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        return renderer.render().astype(np.float32)

    # ---- stills -----------------------------------------------------------
    env.reset(seed=args.seed)
    for _ in range(400):
        forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0][:2]
        env.target = env.data.xpos[env._trunk][:2] + forward * 5.0
        try:
            _, env._prev_dist, _ = env._target_egocentric()
        except Exception:
            pass
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    camera.lookat[:] = env.data.xpos[env._trunk]
    camera.distance, camera.azimuth, camera.elevation = 0.23, 115, -14
    skin, bones, joints = draw(skin_only), draw(bones_only), draw(bones_joints)
    xray = skin * XRAY_SKIN + bones * (1.0 - XRAY_SKIN)
    grid = np.concatenate([
        np.concatenate([skin, xray], axis=1),
        np.concatenate([bones, joints], axis=1)], axis=0).astype(np.uint8)
    out_png = REPO / "artifacts" / "video" / "skeleton.png"
    imageio.imwrite(out_png, grid)
    print(f"stills: {out_png}  (skin | x-ray / physics body | with joints)")

    if args.no_video:
        env.close()
        renderer.close()
        return

    # ---- clip: skin, then x-ray, then the body alone ----------------------
    frames = []
    smooth_look = smooth_heading = None
    total = int(args.seconds / env.dt)
    env.reset(seed=args.seed)
    for i in range(total):
        forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0][:2]
        env.target = env.data.xpos[env._trunk][:2] + forward * 5.0
        try:
            _, env._prev_dist, _ = env._target_egocentric()
        except Exception:
            pass
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        here = env.data.xpos[env._trunk][:2].copy()
        f3 = env.data.xmat[env._trunk].reshape(3, 3)[:, 0]
        heading = math.degrees(math.atan2(f3[1], f3[0]))
        if smooth_heading is None:
            smooth_heading = heading
        else:
            delta = (heading - smooth_heading + 180.0) % 360.0 - 180.0
            smooth_heading += delta * 0.02
        look = np.array([here[0], here[1], 0.022])
        smooth_look = (look.copy() if smooth_look is None
                       else smooth_look + (look - smooth_look) * 0.08)
        camera.lookat[:] = smooth_look
        camera.distance = 0.215
        camera.elevation = -12.0
        camera.azimuth = smooth_heading + 100.0 + 28.0 * math.sin(
            2.0 * math.pi * 0.055 * (i / FPS))

        t = i / max(1, total - 1)
        if args.bones_only:
            show_joints = True if args.joints is None else args.joints
            frame = draw(bones_joints if show_joints else bones_only)
            label = "THE PHYSICS BODY   48 shapes, 36 joints, 38 g"
        elif t < 0.22:
            frame, label = draw(skin_only), "THE SKIN"
        elif t < 0.66:
            # fade the skin out over the first second of this section
            u = min(1.0, (t - 0.22) / 0.06)
            mix = 1.0 - (1.0 - XRAY_SKIN) * (u * u * (3.0 - 2.0 * u))
            frame = draw(skin_only) * mix + draw(bones_only) * (1.0 - mix)
            label = "X-RAY: the physics body underneath"
        else:
            frame, label = draw(bones_joints), "48 SHAPES AND 36 JOINTS, no skin"
        frame = frame.astype(np.uint8).copy()
        draw_text(frame, label, 12, 12, scale=3)
        draw_text(frame, ("every gram and every gate lives here; the skin is a costume"
                          if args.bones_only
                          else "the skin weighs nothing and changes no physics"), 12, 42, scale=2)
        frames.append(frame)

    renderer.close()
    env.close()
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=FPS, quality=6, macro_block_size=1)
    print(f"written: {out}  ({len(frames)} frames, {len(frames) / FPS:.1f} s)")


if __name__ == "__main__":
    main()
