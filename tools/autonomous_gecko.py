"""The whole animal, running itself. Nothing outside it decides anything.

WHAT IS CONNECTED, and this is the first time all of it has been at once:

    brain 1  hypothalamus     energy against a setpoint -> hunger, fatigue
    brain 2  basal ganglia    Prescott 2024, six channels competing for release
    brain 4  eye              retina -> tectum -> a BEARING, and no position
             search           saccade-and-fixate, and a body that can stop
             evidence         several agreeing looks taken while the head is still
    brain 5  smell            the target-species chemical gate on defence
    brain 3b brainstem        behaviour + urgency -> effort
             programs         the seam: a behaviour name -> a motor command

WHAT IS SWITCHED OFF. `env.food_xy` steers nothing. The approach reads a bearing
the animal's own tectum committed to and nothing else. The animal is not told it
is hungry: its energy budget runs down and the basal ganglia notices.

WHAT IS STILL PRIVILEGED, captioned on screen every frame rather than mentioned
in a commit message: the STRIKE trigger reads the true range from the physics
engine, and the walker keeps its five privileged goal slots. Both are the next
dependencies, not achievements.

WHY THE ANIMAL STARTS HUNGRY. Measured (#268): the hypothalamus reports about
3292 HOURS of energy reserve, so across a whole rollout the energy deficit
reaches 0.0000 and the basal ganglia releases NOTHING on any step. A gecko is
simply not hungry on the timescale of a rollout. `meals_owed_at_start` says how
many meals behind the animal is when it emerges, which is what a nocturnal
forager coming out at dusk actually is. It is declared, reported, and INVENTED.

Usage:  python tools/autonomous_gecko.py [--steps 3000] [--seed 2]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from _tinyfont import draw_text                     # noqa: E402
from tectum_diagnosis import project                # noqa: E402

PIX = 64
#: The standing-still operating point for the eye, DERIVED in session 12 and
#: still applied here rather than in the library, because changing
#: `brain/tectum.py`'s defaults moves every gate that exercises the eye (#263).
MOTION_FLOOR_STILL = 0.08
FLOW_GAIN_YAW = 0.006


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--meals-owed", type=float, default=1.0)
    # WINDOW 3, NOT THE 8 THAT #230/#235 MEASURED AS OPTIMAL (#275). That
    # optimum was measured on a WALKING animal with the oracle aiming its head,
    # and it does not survive the animal standing still: A/B on the same seed,
    # window 3 gives 828 frames with prey on screen, 97 of 144 decisions right
    # and a 72.0 mm closest approach; window 8 gives 317, 0 of 176, and 124.2 mm.
    # A longer temporal baseline accumulates more of the body's residual sway
    # when the body is the only thing moving.
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="artifacts/video/autonomous_gecko.mp4")
    args = ap.parse_args()

    import mujoco
    from brain.tectum import Eye
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False,
        eye=True, strike=True, homeostasis=True,
        action_selection=True, behaviour_control=True, smell=True,
        meals_owed_at_start=args.meals_owed,
        gaze_pitch_gain=2.0,
        render_mode=None if args.no_video else "rgb_array",
        max_steps=100000, seed=args.seed,
        view_mode="hunt", camera_smoothing=0.85)
    env.eye = Eye(fovy_deg=env.eye.retina.fovy_deg, pixels=env.eye.retina.pixels,
                  cells=env.eye.retina.cells, motion_window=args.window)
    env.eye.tectum.motion_floor = MOTION_FLOOR_STILL
    env.eye.tectum.flow_gain_yaw = FLOW_GAIN_YAW

    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])

    env.reset(seed=args.seed)
    height = float(env.prey.height_m)

    frames = []
    behaviours, programs = Counter(), Counter()
    on_screen = moving = commits = commits_right = 0
    closest = 9.0
    try:
        for i in range(args.steps):
            _, _, term, trunc, info = env.autonomous_step()
            cmd = info["brain_command"]
            behaviours[info.get("behaviour")] += 1
            programs[cmd["program"]] += 1
            moving += float(info.get("locomotor_drive", 0.0)) > 0.0

            # ---- scoring only. Never steers anything. --------------------
            d = env.walk_env.data
            px = project(d.cam_xpos[cam_id], d.cam_xmat[cam_id],
                         np.array([env.food_xy[0], env.food_xy[1], height]),
                         cam_fovy, pixels=PIX)
            visible = bool(px is not None and 0 <= px[0] < PIX and 0 <= px[1] < PIX)
            on_screen += visible
            if env.evidence.commits > commits:
                commits = env.evidence.commits
                commits_right += visible
            rng = float(np.linalg.norm(np.asarray(env.food_xy) - env._nose_xy()))
            closest = min(closest, rng)

            if not args.no_video:
                frame = env.render()
                beh = info.get("behaviour") or "-- nothing released --"
                y = 10
                for line in [
                    "THE WHOLE BRAIN, RUNNING ITSELF - no oracle steers anything",
                    f"hunger {float(info['hunger']):.2f}"
                    f"   basal ganglia releases: {beh.upper()}"
                    f"   -> program: {cmd['program']}",
                    f"eye {'BELIEVED (head still)' if cmd['believed_eye'] else 'not trusted (moving)'}"
                    f"   needs {env.evidence.quorum} agreeing looks"
                    f"   decisions {commits} of which right {commits_right}",
                    f"body {'WALKING' if cmd['locomotor_drive'] > 0 else 'STOPPED'}"
                    f"   time moving {100.0 * moving / (i + 1):4.1f} %"
                    f"   tongue-flicks/min "
                    f"{(info.get('smell') or {}).get('tongue_flicks_per_min', 0):.1f}",
                    f"cricket really on screen {on_screen}/{i + 1}"
                    f"   mouth to prey {rng * 1000:4.0f} mm   best {closest * 1000:4.0f} mm",
                    "STILL PRIVILEGED: strike trigger reads true range; "
                    "walker keeps its 5 slots",
                ]:
                    draw_text(frame, line, 10, y, scale=2)
                    y += 20
                frames.append(frame)

            if term or trunc:
                env.reset(seed=args.seed + i + 1)
    finally:
        env.close()

    if frames:
        import imageio.v2 as imageio
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)

    summary = {
        "schema_version": 1,
        "generated_by": "tools/autonomous_gecko.py",
        "run": "hunger -> basal ganglia -> motor program -> body; no oracle",
        "connected": ["hypothalamus", "basal ganglia (Prescott 2024)", "eye",
                      "search + fixation evidence", "vomeronasal", "brainstem",
                      "motor programs"],
        "still_privileged": ["strike trigger reads true range",
                             "walker retains its five goal slots"],
        "seed": args.seed, "steps": args.steps,
        "meals_owed_at_start": args.meals_owed,
        "motion_floor": MOTION_FLOOR_STILL, "flow_gain_yaw": FLOW_GAIN_YAW,
        "behaviours_released": {str(k): v for k, v in behaviours.items()},
        "motor_programs": dict(programs),
        "fraction_of_time_moving": round(moving / max(args.steps, 1), 4),
        "frames_prey_on_screen": on_screen,
        "decisions": commits,
        "decisions_with_prey_really_on_screen": commits_right,
        "decision_precision": (round(commits_right / commits, 4)
                               if commits else None),
        "closest_approach_mm": round(closest * 1000, 2),
        "strike_trigger_mm": 20.3,
    }
    ev = REPO / f"artifacts/evidence/session12/autonomous_gecko_seed{args.seed}.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if frames:
        print(f"\nwritten: {REPO / args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
