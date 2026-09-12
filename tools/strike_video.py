"""The mouth, opening on a cricket: the strike as a movement, filmed close.

WHAT IS SHOWN. An oracle-steered approach walks the animal into range (the
oracle is a measuring instrument here -- what is being filmed is the strike,
not the approach), and when the strike fires the camera is already tight on
the head. On screen every frame: the commanded and the ACTUAL gape, the head
pitch, the distance from the mouth to the prey, and whether the prey's centre
is inside the open mouth -- the geometric test that decides a catch (#292).

WHAT IS PUBLISHED IN IT. The gape profile and its timing: Delheusy, Brillet &
Bels 1995, target species, ~80 ms open-close, ~37 deg peak at ~47 ms, no
slow-open phase, ~27 mm head drop. The jaw actuator gain was DERIVED so the
generated body reaches that peak within one control step of the paper (#298).

WHAT IS NOT. Whether the animal actually catches anything. That is measured,
not filmed: see the earned capture rate in FAILURE_MAP #292 and after.

Usage:  python tools/strike_video.py [--seed 3] [--steps 1200]
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
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", default="artifacts/video/strike_mouth.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False,
        eye=True, strike=True, gaze_pitch_gain=2.0,
        render_mode="rgb_array", max_steps=100000, seed=args.seed,
        view_mode="hunt", camera_smoothing=0.85)
    env.reset(seed=args.seed)
    m = env.walk_env.model
    jq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jaw")]

    frames = []
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    strikes_seen = 0
    hold = 0                      # slow-motion hold counter around a strike
    try:
        for i in range(args.steps):
            dxy = np.asarray(env.food_xy) - env._nose_xy()
            hd = math.radians(((math.degrees(math.atan2(dxy[1], dxy[0]))
                                - trunk_yaw_deg(env)) + 180.0) % 360.0 - 180.0)
            a[0], a[1], a[2], a[3] = math.cos(hd), math.sin(hd), -0.2, 1.0
            _, _, term, trunc, info = env.step(a)

            gape_actual = math.degrees(env.walk_env.data.qpos[jq])
            gape_cmd = math.degrees(env.walk_env.jaw_rad)
            in_mouth = bool(env._jaw_ok and env.strike.active and env._prey_in_mouth())
            rng = 1000.0 * env._mouth_food_distance()
            active = env.strike.active
            if info["strikes"] > strikes_seen:
                strikes_seen = info["strikes"]

            frame = env.render()
            lines = [
                "THE STRIKE AS A MOVEMENT - jaw, head drop, and a geometric catch",
                f"mouth  commanded {gape_cmd:5.1f} deg   actual {gape_actual:5.1f} deg"
                f"   (published peak ~37 deg at ~47 ms of an ~80 ms cycle)",
                f"head pitch {math.degrees(env._gaze_pitch_rad):5.1f} deg   mouth to prey {rng:5.1f} mm"
                f"   strike trigger 20.3 mm",
                ("STRIKING" if active else "approach (oracle-steered, for the camera)")
                + f"   strikes {info['strikes']}   caught {info['strike_hits']}"
                + ("   PREY INSIDE THE MOUTH" if in_mouth else ""),
                "the catch is geometry: prey centre inside the open mouth, no dice roll",
            ]
            y = 10
            for line in lines:
                draw_text(frame, line, 10, y, scale=2)
                y += 20
            # slow motion through the strike so an 80 ms event is watchable
            repeat = 6 if (active or hold > 0) else 1
            if active:
                hold = 12
            elif hold > 0:
                hold -= 1
            for _ in range(repeat):
                frames.append(frame)
            if term or trunc:
                env.reset(seed=args.seed + i + 1)
    finally:
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)
    print(f"strikes filmed: {strikes_seen}   written: {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
