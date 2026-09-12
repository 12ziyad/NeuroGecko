"""The same hunt, seen by the old eye and the new one, frame for frame.

WHY THIS IS A FAIR COMPARISON AND NOT TWO RUNS SPLICED TOGETHER. The approach
is the scripted, oracle-driven creep from `tools/hunt_video.py`, and the eye
controls nothing -- it is read and logged, never steered by. So the physics of
this hunt does not depend on which eye is watching it. That means BOTH eyes can
be run on the SAME rendered frames, in one simulation, and every difference on
screen is the eye and nothing else. Rendering twice and cutting them together
would have left the seeds, the contacts and the prey's wander free to diverge.

WHAT THE TWO EYES ARE. Identical optics, identical cell grid, identical
thresholds, identical efference-copy gains. One difference: how many control
frames the prey-motion map is differenced across.

    OLD   motion_window = 1     what shipped before session 12
    NEW   motion_window = 8     the measured optimum (ledger #230, #235)

WHAT THE COUNTERS MEAN. "on screen" is the oracle: the prey's true position
projected through the same camera matrix the eye reads, so it is the number of
frames on which any detector could possibly be right. "spotted" is the eye
reporting a target. The pair of them is the recall this whole session was
about, and the reason #220 could not compute it is that the first column did
not exist.

Usage:  python tools/eye_before_after_video.py [--steps 700] [--seed 3]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from _tinyfont import draw_text                              # noqa: E402
from hunt_decomposition import approach_action               # noqa: E402
from tectum_diagnosis import camera_frame_bearing, project   # noqa: E402

PIX = 64
OLD_WINDOW = 1
NEW_WINDOW = 8


def _text(frame, x, y, line, scale=2):
    draw_text(frame, line, x, y, scale=scale)


def _bar(frame, x, y, w, h, on, rgb):
    """A filled block when the eye is speaking, an outline when it is not."""
    if on:
        frame[y:y + h, x:x + w] = rgb
    else:
        frame[y:y + 2, x:x + w] = rgb
        frame[y + h - 2:y + h, x:x + w] = rgb
        frame[y:y + h, x:x + 2] = rgb
        frame[y:y + h, x + w - 2:x + w] = rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="artifacts/video/eye_before_after.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from brain.tectum import Eye
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False,
        eye=True, strike=True,
        render_mode="rgb_array", max_steps=100000, seed=args.seed,
        view_mode="hunt", camera_smoothing=0.85)

    # Two eyes, same optics, one difference. The env's own eye is left in place
    # and simply not read, so nothing about the environment changes.
    old_eye = Eye(fovy_deg=env.eye.retina.fovy_deg, pixels=env.eye.retina.pixels,
                  cells=env.eye.retina.cells, motion_window=OLD_WINDOW)
    new_eye = Eye(fovy_deg=env.eye.retina.fovy_deg, pixels=env.eye.retina.pixels,
                  cells=env.eye.retina.cells, motion_window=NEW_WINDOW)

    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])

    frames = []
    on_screen = old_hits = new_hits = 0
    try:
        env.reset(seed=args.seed)
        old_eye.reset(); new_eye.reset()
        height = float(env.prey.height_m)
        for i in range(args.steps):
            horizontal = float(np.linalg.norm(
                np.asarray(env.food_xy)[:2] - env._nose_xy()))
            action, creeping = approach_action(env, horizontal, i)
            _, _, term, trunc, info = env.step(action)

            # The image the eye reads, and the self-motion the env hands it.
            image = env._head_cam_image()
            self_motion = {
                "yaw_rate_deg_s": float(np.degrees(
                    env.walk_env._s("gyro_trunk")[2])),
                "forward_m_s": float(env.walk_env._s("vel_trunk")[0]),
            }
            dt = float(env.walk_env.model.opt.timestep
                       * env.walk_env.frame_skip)
            old = old_eye.step(image, dt, self_motion=self_motion)
            new = new_eye.step(image, dt, self_motion=self_motion)
            old_on = bool(old["prey_salience"] > 0.0)
            new_on = bool(new["prey_salience"] > 0.0)

            # The oracle: is the prey on the rendered image at all?
            data = env.walk_env.data
            px = project(data.cam_xpos[cam_id], data.cam_xmat[cam_id],
                         np.array([env.food_xy[0], env.food_xy[1], height]),
                         cam_fovy, pixels=PIX)
            visible = bool(px is not None and 0 <= px[0] < PIX and 0 <= px[1] < PIX)

            on_screen += visible
            old_hits += old_on and visible
            new_hits += new_on and visible

            frame = env.render()
            _text(frame, 10, 10, "SAME HUNT, TWO EYES - only the temporal window differs")
            _text(frame, 10, 30,
                  f"cricket ON SCREEN {on_screen}   "
                  f"mouth to prey {horizontal * 1000:4.0f} mm   "
                  f"{'CREEP' if creeping else 'WALK'}")
            _bar(frame, 10, 52, 22, 16, old_on, (200, 70, 60))
            _text(frame, 40, 54, f"OLD  window 1   spotted {old_hits}")
            _bar(frame, 10, 74, 22, 16, new_on, (70, 190, 120))
            _text(frame, 40, 76, f"NEW  window 8   spotted {new_hits}")
            if on_screen:
                _text(frame, 10, 98,
                      f"recall  old {100 * old_hits / on_screen:4.1f} %"
                      f"    new {100 * new_hits / on_screen:4.1f} %")
            frames.append(frame)
            if term or trunc:
                env.reset(seed=args.seed + i + 1)
                old_eye.reset(); new_eye.reset()
    finally:
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)

    summary = {
        "schema_version": 1,
        "generated_by": "tools/eye_before_after_video.py",
        "run": "one hunt, two eyes read on the same frames; the eye steers nothing",
        "seed": args.seed, "steps": args.steps,
        "old_motion_window": OLD_WINDOW, "new_motion_window": NEW_WINDOW,
        "frames_prey_on_screen": on_screen,
        "old_spotted": old_hits, "new_spotted": new_hits,
        "old_recall": round(old_hits / on_screen, 4) if on_screen else None,
        "new_recall": round(new_hits / on_screen, 4) if on_screen else None,
    }
    ev = REPO / "artifacts/evidence/session11/eye_before_after.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"cricket on screen : {on_screen} of {args.steps} frames")
    print(f"OLD window 1      : spotted {old_hits}"
          + (f"  ({100 * old_hits / on_screen:.1f} % recall)" if on_screen else ""))
    print(f"NEW window 8      : spotted {new_hits}"
          + (f"  ({100 * new_hits / on_screen:.1f} % recall)" if on_screen else ""))
    print(f"\nwritten: {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
