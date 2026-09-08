"""Render the hunt, in two honest halves.

WHAT SESSION 9 ADDED, AND WHY A VIDEO IS THE TEST. Three things landed that had
never existed at the same time, and only running them together shows whether
they compose:

  * the prey MOVES. It used to travel exactly 0.0 mm over 1800 frames, because
    `FleeingPrey.step` only moved it inside the 0.075 m flee radius while the
    eye was asked to find it at 0.30-0.50 m (ledger #135).
  * there is a STRIKE. The walker moves at 0.055 m/s after a cricket that flees
    at 0.118. Pursuit is arithmetically impossible and the real animal does not
    attempt it: it creeps to about two centimetres and fires at 0.851 m/s.
  * the prey now notices HOW FAST it is being approached, so a stalk is a real
    behaviour rather than a slower way of doing the same thing.

They do not compose, and the video says so on screen. So it is in two parts:

  PART 1 -- THE HUNT. Everything running together. The prey wanders, the gecko
  approaches and creeps, the prey's flee radius collapses because it is being
  stalked rather than charged... and the gecko still never strikes. It closes
  to about 5 cm and oscillates there. The frozen walker was trained to arrive
  within 4 cm of a goal, and getting a mouth onto a 9 mm cricket is a precision
  it does not have. Its nose also trails its trunk by ~50 mm, so "the animal
  arrived" and "the mouth is at the prey" are 5 cm apart.

  PART 2 -- THE STRIKE. The prey is PLACED inside striking distance, which the
  approach cannot achieve. This is the strike module doing its job: it fires,
  and about one attempt in five misses, because the published capture rate on
  evasive crickets is 82.9 % and a model that never misses has been fitted to
  the harness rather than the animal.

WHAT NEITHER PART SHOWS. Vision. The approach is driven by the bearing to the
prey read out of the environment -- an oracle, captioned as one on every frame.
The eye still cannot find a cricket (ledger #134-136).

Usage:  python tools/hunt_video.py [--out artifacts/video/hunt_session9.mp4]
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

#: Below this horizontal range the approach switches from walking to creeping.
#: INVENTED. The published behaviour is named (`walk slow motion`, "mostly in
#: context of prey capture") but no distance at which a gecko starts creeping
#: has ever been measured.
STALK_RANGE_M = 0.15
#: Move one brain step in four while stalking. INVENTED; see above.
STALK_DUTY = 4


def _caption(frame, lines):
    y = 10
    for line in lines:
        draw_text(frame, line, 10, y, scale=2)
        y += 20
    return frame


def _approach_action(env, horizontal_m, step_index, stalk=True):
    """Head for the prey, creeping when close. Scripted, not learned.

    The bearing comes from the environment and is an ORACLE. Driving this with
    the eye would produce a video of an animal wandering, which is a true thing
    about the eye and a useless thing to film.
    """
    offset = np.asarray(env.food_xy)[:2] - env._nose_xy()
    forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    bearing = math.atan2(forward[0] * offset[1] - forward[1] * offset[0],
                         forward[0] * offset[0] + forward[1] * offset[1])
    # Aim PAST the prey by the walker's own arrival radius, or it stops short
    # of the animal it is trying to eat.
    trunk_range = float(np.linalg.norm(np.asarray(env.food_xy) - env._trunk_xy()))
    reach = float(env.walk_env.reach_dist)
    span = np.clip(trunk_range + reach, 0.05, 0.80)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = math.cos(bearing)
    action[1] = math.sin(bearing)
    action[2] = (span - 0.05) / 0.375 - 1.0
    creeping = stalk and horizontal_m < STALK_RANGE_M
    action[3] = 1.0 if (not creeping or step_index % STALK_DUTY == 0) else -1.0
    return action, creeping


def _horizontal(env):
    return float(np.linalg.norm(np.asarray(env.food_xy)[:2] - env._nose_xy()))


def part_one(env, steps, seed):
    """The hunt as it actually runs."""
    frames, closest = [], 9.0
    env.reset(seed=seed)
    for i in range(steps):
        h = _horizontal(env)
        closest = min(closest, h)
        action, creeping = _approach_action(env, h, i)
        _, _, term, trunc, info = env.step(action)
        prey = env.prey.state()
        _caption(env.render() if False else (frame := env.render()), [
            "PART 1  THE HUNT",
            f"{'CREEP' if creeping else 'WALK'}   mouth to prey {h * 1000:.0f} mm"
            f"   strike needs {env.strike.trigger_distance_m * 1000:.0f}",
            f"prey moved {prey['total_travel_m'] * 1000:.0f} mm    "
            f"flees at {prey['flee_radius_m'] * 1000:.0f} mm",
            f"strikes {info['strikes']}   -   approach uses an ORACLE, not the eye",
        ])
        frames.append(frame)
        if term or trunc:
            env.reset(seed=seed + i)
    return frames, closest


def part_two(env, steps, seed):
    """The strike, with the prey placed where the approach cannot put it."""
    frames = []
    env.reset(seed=seed)
    for i in range(steps):
        # Place the prey just inside striking distance, ahead of the mouth.
        if not env.strike.active and _horizontal(env) > env.strike.trigger_distance_m:
            forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
            spot = env._nose_xy() + forward * (env.strike.trigger_distance_m * 0.7)
            env.food_xy = env.prey.reset(env._nose_xy(), place_at=spot)
            env._write_prey()
        action, _ = _approach_action(env, _horizontal(env), i, stalk=False)
        _, _, term, trunc, info = env.step(action)
        rate = env.strike.state()["success_rate"]
        frame = env.render()
        _caption(frame, [
            "PART 2  THE STRIKE",
            "prey PLACED in range - the approach cannot get here",
            f"strikes {info['strikes']}   caught {info['strike_hits']}   "
            f"missed {env.strike.misses}",
            (f"success {100 * rate:.0f}%   published 82.9%" if rate is not None
             else "waiting for the first strike"),
        ])
        frames.append(frame)
        if term or trunc:
            env.reset(seed=seed + 500 + i)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunt-steps", type=int, default=700)
    ap.add_argument("--strike-steps", type=int, default=350)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="artifacts/video/hunt_session9.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        strike=True, render_mode="rgb_array", max_steps=100000, seed=args.seed,
        view_mode="hunt", camera_smoothing=0.85)
    try:
        hunt_frames, closest = part_one(env, args.hunt_steps, args.seed)
        hunt_state = {"closest_mouth_to_prey_mm": round(closest * 1000, 1),
                      "strikes": env.strike.strikes,
                      "prey_travel_mm": round(
                          env.prey.state()["total_travel_m"] * 1000, 1)}
        env.strike.reset()
        strike_frames = part_two(env, args.strike_steps, args.seed)
        strike_state = env.strike.state()
    finally:
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), hunt_frames + strike_frames, fps=50, quality=8,
                     macro_block_size=1)

    print("PART 1 -- the hunt")
    print(f"  prey travelled          : {hunt_state['prey_travel_mm']:.0f} mm "
          f"(was exactly 0.0 before this session)")
    print(f"  closest mouth to prey   : {hunt_state['closest_mouth_to_prey_mm']:.1f} mm")
    print(f"  strike needs            : {env.strike.trigger_distance_m * 1000:.1f} mm")
    print(f"  strikes                 : {hunt_state['strikes']}")
    print("\nPART 2 -- the strike, prey placed in range")
    print(f"  strikes                 : {strike_state['strikes']}")
    print(f"  caught                  : {strike_state['hits']}")
    print(f"  missed                  : {strike_state['misses']}")
    if strike_state["success_rate"] is not None:
        print(f"  success                 : {100 * strike_state['success_rate']:.0f}%"
              f"   (published 82.9%)")
    print(f"\nwritten: {out}  ({len(hunt_frames) + len(strike_frames)} frames)")

    evidence = REPO / "artifacts/evidence/session9/hunt_video.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "tools/hunt_video.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "part_1_the_hunt": hunt_state,
        "part_2_the_strike": strike_state,
        "what_it_shows": [
            "prey with undisturbed locomotion, where it used to travel 0.0 mm",
            "a flee radius that collapses when the predator creeps instead of charging",
            "strikes that fire and that miss at roughly the published rate",
        ],
        "what_it_does_not_show": [
            "vision. The approach is driven by the bearing to the prey read out "
            "of the environment, an oracle, captioned as one on every frame. The "
            "eye still cannot find a cricket -- ledger #134-136.",
            "a hunt that works. Part 1 never reaches striking distance. The "
            "frozen walker arrives within 4 cm of a goal and its nose trails its "
            "trunk by about 5 cm, so it cannot place a mouth on a 9 mm cricket. "
            "Part 2 places the prey there instead.",
            "a learned policy. Both parts are scripted.",
        ],
    }, indent=1), encoding="utf-8")
    print(f"evidence: {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
