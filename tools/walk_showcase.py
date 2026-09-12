"""Film the accepted walker with the skin on, from three angles in one take.

THE CONTROLLER IS THE ACCEPTED ONE AND NOTHING ELSE. Zero residual, no trained
policy, the hand-written base with its contact reflex -- the controller ledger
row #71 accepted at 19 strides, 1.1892 Hz measured against 1.1888 commanded and
duty 0.733. `tools/accepted_walker_clip.py` is the canonical evidence clip and
this reuses its exact setup; the only thing that differs is the camera, which
is closer and moves, because the point here is to SEE the animal.

  Do not be tempted to switch the policy on to make the walk look livelier.
  #11 refuted the trained residual (2 x 3.01 M steps; best trained 3/6 against
  base 4/6) and it visibly limps: front-foot duty 0.203 / 0.591 against the
  base's balanced pair. That measurement is in the canonical clip's docstring.

WHAT THE SKIN CHANGES HERE, AND WHAT IT DOES NOT. It changes nothing the
physics sees: #318 measured the meshed and mesh-free bodies as bit-identical on
mass, inertia, joint ranges, damping and actuator gains, identical in `qpos`
after 300 steps, and identical in all 4096 pixels of the image the animal's own
eye receives. The gait in this clip is the same gait that was there before the
animal had a skin. What it changes is that you can now watch the body bend
instead of watching 23 rigid shells slide past each other.

Usage:  python tools/walk_showcase.py [--seconds 24] [--out PATH]
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

#: (fraction of the clip this shot ends at, azimuth offset from the animal's
#: heading, elevation, distance, caption). INVENTED: these are camera moves.
SHOTS = (
    (0.40,  90.0,  -8.0, 0.200, "side"),
    (0.75, 143.0, -16.0, 0.185, "three-quarter"),
    (1.00,  90.0, -60.0, 0.265, "overhead"),
)


def ease(t):
    """Smoothstep, so the camera never starts or stops abruptly."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def shot_at(fraction):
    """Interpolate the camera between shots, blending over the last fifth."""
    previous = (0.0, SHOTS[0][1], SHOTS[0][2], SHOTS[0][3], SHOTS[0][4])
    for end, az, el, dist, name in SHOTS:
        if fraction <= end:
            span = end - previous[0]
            local = (fraction - previous[0]) / span if span > 0 else 1.0
            blend = ease((local - 0.80) / 0.20) if local > 0.80 else 0.0
            index = SHOTS.index((end, az, el, dist, name))
            nxt = SHOTS[min(index + 1, len(SHOTS) - 1)]
            return (az + (nxt[1] - az) * blend,
                    el + (nxt[2] - el) * blend,
                    dist + (nxt[3] - dist) * blend,
                    name if blend < 0.5 else nxt[4])
        previous = (end, az, el, dist, name)
    return SHOTS[-1][1], SHOTS[-1][2], SHOTS[-1][3], SHOTS[-1][4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--quality", type=int, default=6)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="declared lab-parameter change, e.g. spine_amp=0.61; "
                         "same protocol as eval/session2_controller.py")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="multiply every shot's camera distance; 2.5 gives a wide view")
    ap.add_argument("--out", default="artifacts/video/walk_showcase.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco

    from envs.gecko_walk_env import GeckoWalkEnv

    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key.strip()] = float(value)
    if overrides:
        print("declared lab-parameter changes:", overrides)
    env = GeckoWalkEnv(xml_path="morphology/gecko_world_v1.xml",
                       gait_profile="lab", control_mode="cpg_residual",
                       residual_scale=0.25, privileged_target=True,
                       lab_parameters=overrides or None)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    options = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    frames = []
    duty = {"fl_duty": [], "fr_duty": [], "hl_duty": [], "hr_duty": []}
    smooth_look = None
    smooth_heading = None
    path = 0.0

    env.reset(seed=args.seed)
    previous = env.data.xpos[env._trunk][:2].copy()
    total = int(args.seconds / env.dt)
    try:
        for i in range(total):
            forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0][:2]
            env.target = env.data.xpos[env._trunk][:2] + forward * 5.0
            try:
                _, env._prev_dist, _ = env._target_egocentric()
            except Exception:
                pass
            # ZERO residual: the accepted controller, with the policy off.
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))

            here = env.data.xpos[env._trunk][:2].copy()
            path += float(np.linalg.norm(here - previous))
            previous = here
            for key in duty:
                if key in info:
                    duty[key].append(float(info[key]))

            # The trunk bobs once per stride; a camera locked to it shakes the
            # whole world (#181). Follow a smoothed point, and a smoothed
            # heading, so a single misstep does not swing the shot.
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

            az_off, elevation, distance, name = shot_at(i / max(1, total - 1))
            camera.azimuth = smooth_heading + az_off
            camera.elevation = elevation
            camera.distance = distance * args.zoom
            camera.lookat[:] = smooth_look

            renderer.update_scene(env.data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            frame = renderer.render()
            fl = float(np.nanmean(duty["fl_duty"])) if duty["fl_duty"] else float("nan")
            fr = float(np.nanmean(duty["fr_duty"])) if duty["fr_duty"] else float("nan")
            draw_text(frame, "ACCEPTED WALKER   no trained policy", 12, 12, scale=2)
            # THE TARGET GOES ON SCREEN NEXT TO THE VALUE. Front-foot duty is
            # one of the two gates this walker FAILS, and a number shown alone
            # under the word ACCEPTED reads as a pass. Same figure the canonical
            # clip prints: FL 0.447 / FR 0.444 against a published 0.70.
            draw_text(frame, f"front duty {fl:.2f}/{fr:.2f} vs 0.70 target"
                             f"   {path * 100:.0f} cm", 12, 36, scale=2)
            draw_text(frame, name, 12, args.height - 28, scale=2)
            frames.append(frame)

            if terminated or truncated:
                env.reset(seed=args.seed + i)
                previous = env.data.xpos[env._trunk][:2].copy()
    finally:
        renderer.close()
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=FPS, quality=args.quality,
                     macro_block_size=1)

    seconds = len(frames) * env.dt
    print(f"written: {out}  ({len(frames)} frames, {seconds:.1f} s)")
    for key in ("fl_duty", "fr_duty", "hl_duty", "hr_duty"):
        if duty[key]:
            print(f"  {key:8} {float(np.nanmean(duty[key])):.3f}")
    print(f"  travelled {path * 100:.1f} cm in {seconds:.1f} s "
          f"= {path / seconds * 100:.1f} cm/s")


if __name__ == "__main__":
    main()
