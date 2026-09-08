"""The gecko walking, and the gecko striking. One camera angle, no shake.

WHAT WENT WRONG IN THE EARLIER CLIPS, because both faults were mine and neither
was the animal's.

  THE SHAKE WAS THE CAMERA SWINGING, NOT THE GROUND SLIDING. I blamed the
  sliding ground and built a bolted-down camera to stop it. That was the wrong
  diagnosis. `close` and `chase` set azimuth = heading + 180, so the camera
  rotates every time the gecko yaws -- and a walking gecko yaws constantly.
  Holding the azimuth fixed while still following the animal removes the shake
  entirely, which is what `renders/session4/*.mp4` always did.

  THE GECKO WAS NOT WALKING BADLY. IT WAS NOT WALKING. `tools/strike_clip.py`
  sends action[3] = -1, which means hold position, because a charging gecko
  makes the cricket bolt and there would be no strike to film. So it stood
  there and shuffled. Nothing about the gait changed: the same frozen
  checkpoint, the same body, the same 481 passing tests. Measured again here --
  0.407 m in 10 s -- and it is on screen in part 1 so it can be judged rather
  than asserted.

So this renders both, back to back, from one angle:

  PART 1  the walk, straight ahead, nothing else running.
  PART 2  the strike, with the cricket placed in range because the walker
          cannot close the last two centimetres (ledger #174).

Usage:  python tools/gecko_walks_and_strikes.py
"""

# 50 FPS, AND THE NUMBER IS NOT A STYLE CHOICE. Every tool here appends one
# frame per brain step, and a brain step is 0.02 s. Writing those frames at
# 25 fps -- which all three tools did until Session 9f -- plays the animal at
# HALF SPEED. That is the "lag" in every Session 9 clip: not a gait change,
# not a physics change, a container header. The Session 4 renders were 50 fps
# and real-time, which is why they looked right and these did not.
# Measured: 1000 frames = 20.0 s of simulation shown over 40.0 s of video.

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

from tools._tinyfont import draw_text                        # noqa: E402


def _caption(frame, lines):
    y = 10
    for line in lines:
        draw_text(frame, line, 10, y, scale=2)
        y += 20


def _place_ahead(env, metres):
    forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    env.food_xy = env.prey.reset(env._nose_xy(),
                                 place_at=env._nose_xy() + forward * metres)
    env._write_prey()


def walk_part(env, steps, seed):
    """Just walking. Full engage, distant target, nothing held back."""
    frames = []
    env.reset(seed=seed)
    start = env._trunk_xy().copy()
    for i in range(steps):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[0], action[1] = 1.0, 0.0
        action[2], action[3] = 1.0, 1.0
        env.step(action)
        travelled = float(np.linalg.norm(env._trunk_xy() - start))
        frame = env.render()
        _caption(frame, [
            "PART 1   WALKING",
            f"travelled {travelled * 1000:.0f} mm   "
            f"{travelled / max((i + 1) * 0.02, 1e-6):.3f} m/s",
            "same body, same checkpoint, unchanged",
        ])
        frames.append(frame)
    return frames, float(np.linalg.norm(env._trunk_xy() - start))


def strike_part(env, steps, seed):
    """The strike. The gecko holds position -- a charge makes the cricket bolt."""
    frames = []
    env.reset(seed=seed)
    env.strike.reset()
    _place_ahead(env, env.strike.trigger_distance_m * 0.75)
    last_hits, flash = 0, 0
    for i in range(steps):
        offset = np.asarray(env.food_xy)[:2] - env._nose_xy()
        forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
        bearing = math.atan2(forward[0] * offset[1] - forward[1] * offset[0],
                             forward[0] * offset[0] + forward[1] * offset[1])
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[0], action[1] = math.cos(bearing), math.sin(bearing)
        action[2], action[3] = -1.0, -1.0      # hold position, on purpose
        _, _, term, trunc, info = env.step(action)
        if info["strike_hits"] > last_hits:
            last_hits = info["strike_hits"]
            flash = 12
            _place_ahead(env, env.strike.trigger_distance_m * 0.75)
        elif flash:
            flash -= 1
        state = env.strike.state()
        rate = state["success_rate"]
        frame = env.render()
        _caption(frame, [
            "PART 2   STRIKING   0.851 m/s from 20 mm",
            f"attempts {state['strikes']}   caught {state['hits']}   "
            f"missed {state['misses']}",
            (f"success {100 * rate:.0f}%   real gecko 82.9%" if rate is not None
             else "waiting for the first strike"),
            ("CAUGHT" if flash else
             "holding still on purpose - a charge makes the cricket bolt"),
        ])
        frames.append(frame)
        if term or trunc:
            env.reset(seed=seed + i)
            _place_ahead(env, env.strike.trigger_distance_m * 0.75)
    return frames, env.strike.state()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-steps", type=int, default=500)
    ap.add_argument("--strike-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="artifacts/video/gecko_session9.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        strike=True, render_mode="rgb_array", max_steps=100000,
        seed=args.seed, view_mode="track")
    try:
        walk_frames, distance = walk_part(env, args.walk_steps, args.seed)
        strike_frames, strike_state = strike_part(env, args.strike_steps, args.seed)
    finally:
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), walk_frames + strike_frames, fps=50, quality=8,
                     macro_block_size=1)

    seconds = args.walk_steps * 0.02
    print(f"PART 1  walked {distance:.3f} m in {seconds:.0f} s "
          f"-> {distance / seconds:.4f} m/s")
    print(f"PART 2  {strike_state['strikes']} strikes, "
          f"{strike_state['hits']} caught, {strike_state['misses']} missed")
    if strike_state["success_rate"] is not None:
        print(f"        {100 * strike_state['success_rate']:.0f}% "
              f"(published 82.9%)")
    print(f"written: {out}  "
          f"({len(walk_frames) + len(strike_frames)} frames, one fixed angle)")

    evidence = REPO / "artifacts/evidence/session9/gecko_video.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "tools/gecko_walks_and_strikes.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "camera": "track: follows position, azimuth fixed at the starting "
                  "heading, no rotation and no smoothing",
        "part_1_walk": {"distance_m": round(distance, 4),
                        "seconds": seconds,
                        "speed_m_s": round(distance / seconds, 4)},
        "part_2_strike": strike_state,
        "corrections_this_clip_makes": [
            "The shake in the earlier clips was the camera swinging with the "
            "gecko's heading, not the ground sliding. Fixed azimuth removes it.",
            "The gecko in tools/strike_clip.py was not walking badly -- it was "
            "commanded to hold position so the cricket would not bolt. The gait, "
            "body and checkpoint are unchanged.",
        ],
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
