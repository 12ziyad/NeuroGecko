"""A clean, single-angle clip of the strike. One camera, no shake, no teleporting.

WHY A SECOND VIDEO TOOL. `tools/hunt_video.py` shows the honest state of the
whole pipeline and is hard to watch: the camera tracks the trunk, so the animal
stays centred and the GROUND slides, which reads as camera shake and makes it
impossible to tell whether the gecko moved or the view did. Worse, its second
half re-placed the prey on EVERY frame the strike was not active, so the
cricket jittered around the mouth like a rendering fault instead of sitting
there like an animal.

This one fixes both, and the fixes are about honesty as much as about looks. A
clip where you cannot tell whether the animal moved is not evidence of
anything, and a prey that teleports every frame is not the prey the model
actually simulates -- it is an artefact of the filming.

  ONE CAMERA, BOLTED TO THE WORLD. `view_mode="static"` anchors the camera
  where the episode began and never moves it. The gecko walks across the frame
  and out of it, which is what actually happens.

  THE CRICKET IS PLACED ONCE PER HUNT. After a capture -- and only then -- a new
  one appears a short walk ahead, the way a respawn should look. Between
  captures it does what the model says it does: walks in bouts, and bolts if
  something charges it.

WHAT THIS STILL DOES NOT SHOW. Vision. The approach is driven by the bearing to
the prey read out of the environment, an oracle, captioned on every frame. And
the gecko cannot close the last two centimetres by itself (ledger #174), so the
cricket is respawned inside striking distance. That is stated on screen rather
than in a file nobody opens.

Usage:  python tools/strike_clip.py [--out artifacts/video/strike_clip.mp4]
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
    """Put a fresh cricket a set distance in front of the gecko's nose."""
    forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    spot = env._nose_xy() + forward * metres
    env.food_xy = env.prey.reset(env._nose_xy(), place_at=spot)
    env._write_prey()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="artifacts/video/strike_clip.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        strike=True, render_mode="rgb_array", max_steps=100000,
        seed=args.seed, view_mode="static")

    frames = []
    try:
        env.reset(seed=args.seed)
        # One cricket, placed once, inside striking distance. The gecko cannot
        # get here on its own -- see ledger #174 -- and the caption says so.
        _place_ahead(env, env.strike.trigger_distance_m * 0.75)
        last_hits = 0
        settle = 0

        for i in range(args.steps):
            # Face the prey; do not charge it. A charge would make the cricket
            # bolt, which is correct behaviour and the wrong thing to film here.
            offset = np.asarray(env.food_xy)[:2] - env._nose_xy()
            forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
            bearing = math.atan2(forward[0] * offset[1] - forward[1] * offset[0],
                                 forward[0] * offset[0] + forward[1] * offset[1])
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[0] = math.cos(bearing)
            action[1] = math.sin(bearing)
            action[2] = -1.0
            action[3] = -1.0        # hold position
            _, _, term, trunc, info = env.step(action)

            # A new cricket ONLY after one is eaten, and only after a pause, so
            # it reads as a respawn rather than a glitch.
            # Replace IMMEDIATELY on a capture. An earlier version waited 18
            # frames to make the respawn readable, and that was worse: the
            # eaten cricket stayed on screen at the old spot while the gecko
            # walked on, so the clip showed a cricket sitting behind the tail
            # of an animal that had already eaten it.
            if info["strike_hits"] > last_hits:
                last_hits = info["strike_hits"]
                settle = 10
                _place_ahead(env, env.strike.trigger_distance_m * 0.75)
            elif settle > 0:
                settle -= 1

            frame = env.render()
            state = env.strike.state()
            rate = state["success_rate"]
            _caption(frame, [
                "STRIKE   0.851 m/s   from 20 mm",
                f"attempts {state['strikes']}   caught {state['hits']}   "
                f"missed {state['misses']}",
                (f"success {100 * rate:.0f}%   real gecko 82.9%" if rate is not None
                 else "waiting for the first strike"),
                ("CAUGHT" if settle > 0 else
                 "cricket placed in range - the walk cannot get here yet"),
            ])
            frames.append(frame)
            if term or trunc:
                env.reset(seed=args.seed + i)
                _place_ahead(env, env.strike.trigger_distance_m * 0.75)
        state = env.strike.state()
    finally:
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)

    print(f"attempts : {state['strikes']}")
    print(f"caught   : {state['hits']}")
    print(f"missed   : {state['misses']}")
    if state["success_rate"] is not None:
        print(f"success  : {100 * state['success_rate']:.0f}%   (published 82.9%)")
    print(f"written  : {out}  ({len(frames)} frames, "
          f"{len(frames) / 50:.0f} s, one fixed camera)")

    evidence = REPO / "artifacts/evidence/session9/strike_clip.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "tools/strike_clip.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "camera": "static, world-anchored, no smoothing, no tracking",
        "strike": state,
        "what_it_does_not_show": [
            "vision -- the bearing is an oracle read from the environment",
            "an approach -- the cricket is placed inside striking distance "
            "because the frozen walker cannot close the last 2 cm (#174)",
        ],
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
