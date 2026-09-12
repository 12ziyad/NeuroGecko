"""The hunt with the prey-position oracle disconnected. The animal looks.

WHAT IS AND IS NOT SWITCHED OFF, because a video is exactly where this gets
overstated and this project has shipped a privileged channel that was "supposed
to be off" three times already (#26, #33, #165).

  OFF -- the approach no longer reads `env.food_xy`. Every heading the animal
         takes comes from `info["prey_bearing_deg"]`, the tectum's own output,
         and from nothing else.
  OFF -- the gaze. The head pitches on the eye's reported elevation only, and
         yaws on the search module's own scan.
  OFF -- the decision threshold. It is set from the animal's own firing rate,
         not from a noise rate someone measured with ground truth.
  STILL ON -- the STRIKE trigger fires on the true range from the physics
         engine, and the walker still carries its five privileged slots. Both
         are captioned on screen every frame rather than mentioned in a commit
         message.

WHAT THE PREVIOUS VERSION OF THIS FILE MEASURED, and why almost none of it
survives. 900 steps, 118 "sightings", prey on the rendered image for ZERO of
them, closest approach 326 mm against a 20.3 mm strike. The diagnosis
(tools/oracle_free_diagnosis.py) found the cause was not the eye: the prey sat
BEHIND the camera on 581 of 600 frames. The animal was facing away from its
food 97 % of the time and no detector can help with that.

FOUR THINGS CHANGED, EACH ONE MEASURED, EACH ONE IN THE LEDGER.
  #244  the search heading is BODY relative, so the old alternating pattern
        could not turn around. Fixed: turns close on the trunk's own angle.
  #246  the head has +-70 deg of yaw that nothing had ever commanded. It now
        saccades and fixates rather than sweeping, because a smoothly sweeping
        head is self-motion and the eye reads it as the world.
  #247  the efference copy was computed from sensors on the TRUNK while the eye
        rides on the HEAD. The head's own yaw rate is 3-4x the trunk's. A gyro
        now sits where a vertebrate's semicircular canals sit -- in the skull,
        beside the eye.
  #249  the decision is made once per FIXATION, against a quorum sized from the
        animal's own report rate, with the null corrected for the fact that the
        test searches every bearing rather than one fixed in advance.

WHAT THE ANIMAL STILL CANNOT KNOW. The eye reports a BEARING and no RANGE. So
an oracle-free approach cannot aim at a point -- it can only walk along a
direction. The walker's goal distance is a fixed span ahead, INVENTED, declared
here, and constant.

Usage:  python tools/oracle_free_hunt_video.py [--steps 1500] [--seed 2]
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
from tectum_diagnosis import project                         # noqa: E402

from brain.search import FixationEvidence, SearchPattern     # noqa: E402

PIX = 64
#: Goal distance ahead, metres. INVENTED -- the eye reports no range, so the
#: animal walks along a bearing rather than toward a point.
SPAN_M = 0.35
#: Efference-copy gain for the uniform (rotational) term. Re-derived in session
#: 12 because its INPUT changed from the trunk gyro to the head gyro, whose
#: rates are 3-4x larger (#247). Chosen off a measured sweep on a stated
#: criterion: the highest recall whose false-alarm rate the per-fixation quorum
#: can still reject. Sweep in artifacts/evidence/session12/.
FLOW_GAIN_YAW = 0.006
#: Fixed downward head pitch held while searching, degrees. INVENTED.
SEARCH_PITCH_DEG = 10.0
#: Detection threshold on the local-motion map, for an animal that is STANDING
#: STILL. DERIVED, not invented: with the body frozen and the prey inside
#: 180 mm, the map's peak has median 0.1499 with the prey on the image and
#: 0.0751 with it off (artifacts/evidence/session12/), and this floor is where
#: the two distributions separate most -- 92.7 % recall at 48.5 % false alarms,
#: against 57.2 % at 20.3 % for the same animal walking.
#: The shipped default of 1e-4 is four hundred times below the noise and fires
#: on essentially every frame (#254).
#:
#: IT IS SIZED FOR AN ANIMAL THAT IS STILL. Walking, this floor gives 40.8 %
#: recall against 92.7 %, which is why the eye is consulted only during a
#: fixation -- not because a walking animal is blind (#259 corrects an earlier
#: claim that it was) but because a standing one is more than twice as good.
MOTION_FLOOR_STILL = 0.08


def trunk_yaw_deg(env):
    q = env.walk_env.data.qpos[3:7]
    return math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                   1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--gaze", type=float, default=2.0)
    ap.add_argument("--flow-gain-yaw", type=float, default=FLOW_GAIN_YAW)
    ap.add_argument("--motion-floor", type=float, default=MOTION_FLOOR_STILL)
    ap.add_argument("--blind", action="store_true",
                    help="THE CONTROL. Everything identical -- same search, "
                         "same freezes, same fixations, same number of "
                         "commitments -- except the committed bearing is "
                         "replaced by a random one. If the animal gets as "
                         "close this way, then it is finding its food by "
                         "wandering in a small arena and not by seeing")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="artifacts/video/oracle_free_hunt.mp4")
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
        gaze_pitch_gain=args.gaze,
        render_mode=None if args.no_video else "rgb_array",
        max_steps=100000, seed=args.seed,
        view_mode="hunt", camera_smoothing=0.85)
    env.eye = Eye(fovy_deg=env.eye.retina.fovy_deg, pixels=env.eye.retina.pixels,
                  cells=env.eye.retina.cells, motion_window=args.window)
    env.eye.tectum.flow_gain_yaw = float(args.flow_gain_yaw)
    env.eye.tectum.motion_floor = float(args.motion_floor)

    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])

    coin = np.random.default_rng(args.seed)
    search = SearchPattern()
    # Threshold set from the animal's own firing rate. Nothing privileged.
    evidence = FixationEvidence(adaptive=True,
                                fixate_steps=SearchPattern.FIXATE_STEPS)

    frames = []
    on_screen = reports = commits = commits_right = chasing = 0
    closest = 9.0
    commit_heading = None
    env.reset(seed=args.seed)
    height = float(env.prey.height_m)

    try:
        for i in range(args.steps):
            # ---- WHERE TO GO. The committed bearing if there is one, the
            # search otherwise. Nothing else is consulted.
            committed = evidence.last.get("committed_bearing_deg")
            if committed is not None and commit_heading is not None:
                # The eye's bearing is relative to the HEAD; the walker's
                # heading is relative to the TRUNK. The two differ by the head
                # yaw AT THE MOMENT THE DECISION WAS MADE, which is why it is
                # captured then and held rather than read now -- the head
                # recentres for the chase, and reading it live would swing the
                # animal by up to 70 degrees away from what it saw. Same class
                # of frame bug as #242.
                heading_deg, head_deg, moving = commit_heading, 0.0, True
                chasing += 1
            else:
                heading_deg, head_deg, moving = search.step(trunk_yaw_deg(env))
            env._gaze_yaw_rad = math.radians(head_deg)
            # A FIXATION HAS TO BE STILL IN BOTH AXES (#251). The first version
            # froze the head's yaw and left the pitch loop tracking, so the head
            # was still moving through every "still" look -- and because that
            # loop is driven by the eye, it chased its own false alarms and drove
            # the report rate from 20 % to 49.7 % of frames. The quorum, sized
            # off that rate, went to 12, which no real target could ever reach.
            # The pitch loop exists for the terminal approach (#241), where the
            # prey falls out of the bottom of the frame. It has no business
            # running during a search.
            # A SEARCH FIXATION HOLDS A FIXED DOWNWARD PITCH (#251). Not the
            # tracking loop -- that chases its own false alarms and drove the
            # report rate to 49.7 %. Not level either -- that put prey on the
            # screen on ZERO frames, because the cricket is on the ground and
            # the camera looks forward. A searching gecko holds its head angled
            # down at the substrate. The angle is INVENTED: it centres the
            # 100-400 mm band, where a target on the floor sits 4 to 14 deg
            # below a camera 25 mm off the ground.
            if committed is not None:
                env.gaze_pitch_gain = args.gaze
            else:
                env.gaze_pitch_gain = 0.0
                env._gaze_pitch_rad = math.radians(SEARCH_PITCH_DEG)

            heading = math.radians(heading_deg)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[0] = math.cos(heading)
            action[1] = math.sin(heading)
            action[2] = (SPAN_M - 0.05) / 0.375 - 1.0
            action[3] = 1.0 if moving else -1.0
            # THE OFF SWITCH (#253). `engage` above changes the goal distance
            # and nothing else -- it never stopped the animal. This does.
            env.walk_env.locomotor_drive = 1.0 if moving else 0.0
            _, _, term, trunc, info = env.step(action)

            # ---- WHAT THE EYE SAID ---------------------------------------
            bearing = info["prey_bearing_deg"]
            saw = bool(float(info["food_visible_frac"]) > 0.0 and bearing is not None)
            reports += saw
            looking = bool(search.last.get("looking")) and committed is None
            was = evidence.commits
            evidence.step(float(bearing) if saw else None, looking)

            # ---- WHAT WAS ACTUALLY TRUE (scoring only; never steers) ------
            d = env.walk_env.data
            tgt = np.array([env.food_xy[0], env.food_xy[1], height])
            px = project(d.cam_xpos[cam_id], d.cam_xmat[cam_id], tgt,
                         cam_fovy, pixels=PIX)
            visible = bool(px is not None and 0 <= px[0] < PIX and 0 <= px[1] < PIX)
            on_screen += visible
            rng = float(np.linalg.norm(np.asarray(env.food_xy) - env._nose_xy()))
            closest = min(closest, rng)
            if evidence.commits > was:
                commits += 1
                commits_right += visible
                commit_heading = (float(evidence.last["committed_bearing_deg"])
                                  + float(head_deg))
                if args.blind:
                    commit_heading = float(coin.uniform(-180.0, 180.0))
            if evidence.last.get("committed_bearing_deg") is None:
                commit_heading = None

            if not args.no_video:
                frame = env.render()
                state = ("CHASING a committed bearing"
                         if committed is not None else
                         f"SEARCHING - {search.last.get('state', '?')}"
                         + ("  (head still, LOOKING)" if looking
                            else "  (moving, not trusting the eye)"))
                y = 10
                for line in [
                    "ORACLE OFF - the gecko searches and decides on its EYE alone",
                    state,
                    f"eye fired {reports}x   its own report rate "
                    f"{evidence.last.get('report_rate')}   "
                    f"needs {evidence.last.get('quorum')} agreeing to commit",
                    f"decisions {commits}   of those, prey really on screen "
                    f"{commits_right}",
                    f"cricket really on screen {on_screen}/{i + 1}   "
                    f"mouth to prey {rng * 1000:4.0f} mm   best {closest * 1000:4.0f} mm",
                    f"head yaw {head_deg:+5.1f}   gaze down "
                    f"{info.get('gaze_pitch_deg', 0.0):4.1f}   "
                    f"strikes {info['strikes']}   caught {info['strike_hits']}",
                    "STILL PRIVILEGED: strike trigger reads true range; "
                    "walker keeps its 5 slots",
                ]:
                    draw_text(frame, line, 10, y, scale=2)
                    y += 20
                frames.append(frame)

            if term or trunc:
                env.reset(seed=args.seed + i + 1)
                search.reset(); evidence.reset(); commit_heading = None
    finally:
        env.close()

    if frames:
        import imageio.v2 as imageio
        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)

    summary = {
        "schema_version": 2,
        "generated_by": "tools/oracle_free_hunt_video.py",
        "run": ("CONTROL: committed bearings replaced by random ones"
                if args.blind else
                "prey-position oracle DISCONNECTED; search + per-fixation decision"),
        "still_privileged": ["strike trigger reads true range",
                             "walker retains its five goal slots"],
        "seed": args.seed, "steps": args.steps,
        "motion_window": args.window, "gaze_pitch_gain": args.gaze,
        "flow_gain_yaw": float(args.flow_gain_yaw),
        "motion_floor": float(args.motion_floor),
        "eye_raw_reports": reports,
        "decisions": commits,
        "decisions_with_prey_really_on_screen": commits_right,
        "decision_precision": (round(commits_right / commits, 4)
                               if commits else None),
        "steps_chasing": chasing,
        "frames_prey_on_screen": on_screen,
        "closest_approach_mm": round(closest * 1000, 2),
        "strike_trigger_mm": 20.3,
        "final_quorum": evidence.quorum,
        "final_report_rate": evidence.last.get("report_rate"),
    }
    ev = REPO / "artifacts/evidence/session12/oracle_free_hunt.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    if frames:
        print(f"\nwritten: {REPO / args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
