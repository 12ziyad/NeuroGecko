"""The head, close enough to see it move: neck rotation, blink, breathing, jaw.

WHY THIS EXISTS. Every previous clip was filmed from the hunt camera, far
enough away that a 22 deg head saccade is a few pixels and a 5 deg throat pulse
is invisible. The user could not see the neck rotating at all. This films the
head from close range with the camera locked to the animal, and prints the
joint angles so a movement that is too small to see is at least readable.

WHAT IS ON SCREEN, and where each number comes from:

  neck + head yaw   the SEARCH: saccade and fixate (brain/search.py). Jump
                    amplitude, dwell and settle are INVENTED or derived from
                    this body's own servo, never from an animal (#278, #279).
  eyelids           PUBLISHED ORGAN, INVENTED behaviour. Eublepharids are the
                    only geckos with movable eyelids; no blink rate exists for
                    this species or any lizard.
  throat            BREATHING, not gills. The pump is published in direction
                    for geckos; no frequency exists for any of them (#302).
  jaw               PUBLISHED: ~37 deg peak of an ~80 ms cycle, target species
                    (Delheusy, Brillet & Bels 1995).

Usage:  python tools/closeup_video.py [--mode search|strike] [--seed 2]
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
sys.path.insert(0, str(REPO / "tools"))

from _tinyfont import draw_text  # noqa: E402


def trunk_yaw_deg(env):
    q = env.walk_env.data.qpos[3:7]
    return math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                   1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["search", "strike"], default="search")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"artifacts/video/closeup_{args.mode}.mp4"

    import imageio.v2 as imageio
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    autonomous = args.mode == "search"
    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False,
        eye=True, strike=True, homeostasis=autonomous,
        action_selection=autonomous, behaviour_control=autonomous,
        smell=autonomous, meals_owed_at_start=1.0 if autonomous else 0.0,
        gaze_pitch_gain=2.0,
        render_mode="rgb_array", max_steps=100000, seed=args.seed,
        view_mode="hunt", camera_smoothing=0.92)
    if autonomous:
        env.eye.tectum.motion_floor = 0.08
        env.eye.tectum.flow_gain_yaw = 0.006
    env.reset(seed=args.seed)

    m = env.walk_env.model
    d = env.walk_env.data
    Q = {n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
         for n in ("neck_yaw", "head_yaw", "jaw", "eyelids", "gular",
                   "spine_lat_2", "tail_yaw_1")}

    # A camera that rides the head, close in. Added to the RENDER model only --
    # it is never the animal's own `head_cam` and sees nothing the brain uses.
    head_bid = m.body("head").id
    frames = []
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    try:
        for i in range(args.steps):
            if autonomous:
                env.autonomous_step()
            else:
                dxy = np.asarray(env.food_xy) - env._nose_xy()
                hd = math.radians(((math.degrees(math.atan2(dxy[1], dxy[0]))
                                    - trunk_yaw_deg(env)) + 180.0) % 360.0 - 180.0)
                a[0], a[1], a[2], a[3] = math.cos(hd), math.sin(hd), -0.2, 1.0
                env.step(a)

            ang = {n: math.degrees(d.qpos[q]) for n, q in Q.items()}
            frame = env.render()
            # crop to the head and scale back up: the cheapest close-up that
            # needs no new camera in a hash-checked model.
            h, w = frame.shape[:2]
            cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
            hx, hy = w // 2, h // 2
            half = max(80, int(min(h, w) * 0.22))
            y0 = max(0, min(h - 2 * half, hy - half))
            x0 = max(0, min(w - 2 * half, hx - half))
            crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
            k = max(1, min(h // crop.shape[0], w // crop.shape[1]))
            big = np.repeat(np.repeat(crop, k, axis=0), k, axis=1)
            canvas = np.zeros_like(frame)
            canvas[:big.shape[0], :big.shape[1]] = big[:h, :w]
            frame = canvas

            y = 10
            for line in [
                f"CLOSE UP - {args.mode.upper()}",
                f"neck yaw {ang['neck_yaw']:+6.1f}   head yaw {ang['head_yaw']:+6.1f}"
                f"   = LOOKING {ang['neck_yaw'] + ang['head_yaw']:+6.1f} deg",
                f"eyelids {ang['eyelids']:5.1f} deg {'SHUT' if ang['eyelids'] > 35 else 'open'}"
                f"   throat {ang['gular']:+5.1f} deg (breathing)"
                f"   jaw {ang['jaw']:5.1f} deg",
                f"body curve {ang['spine_lat_2']:+6.1f}   tail base {ang['tail_yaw_1']:+6.1f}"
                f"   {'SAME side' if ang['spine_lat_2'] * ang['tail_yaw_1'] > 0 else 'OPPOSITE'}",
            ]:
                draw_text(frame, line, 8, y, scale=2)
                y += 20
            frames.append(frame)
    finally:
        env.close()

    path = REPO / out
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), frames, fps=50, quality=8, macro_block_size=1)
    print(f"written: {path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
