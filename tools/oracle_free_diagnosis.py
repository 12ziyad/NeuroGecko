"""Why the prey is never on screen when the oracle is off. No tuning, just facts.

The oracle-free run reports `frames_prey_on_screen = 0`. There are only a few
ways that can be true, and they call for completely different fixes, so this
measures which one it is instead of guessing:

  AZIMUTH   the prey is to the side, outside the camera's horizontal half-angle.
            -> a SEARCH problem. The animal is not pointing at it.
  ELEVATION the prey is below (or above) the vertical half-angle.
            -> a GAZE problem. The head is at the wrong pitch.
  RANGE     it is on the image but too few pixels to ever be differenced.
            -> an OPTICS problem, and the one this session already worked on.

Also logs WHERE the false alarms sit. If they cluster at one bearing they will
walk straight through any quorum -- which is what `PreyEvidence` assumes they do
not do, and that assumption is now under test.

Nothing here steers. The controller is whatever the caller passes in.
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

from tectum_diagnosis import project  # noqa: E402

PIX = 64
#: Cricket body length, metres. PUBLISHED, from the project's own registry
#: row for the prey item; used only to report angular size, never to steer.
PREY_LEN_M = 0.015
#: How close a reported bearing must be to the prey's true bearing to count as
#: having FOUND the prey, degrees. INVENTED, and generous: the prey subtends
#: about 2.8 deg at 380 mm, so 10 deg is roughly three body widths of slack.
AIM_TOL_DEG = 10.0


def camera_frame_angles(cam_xpos, cam_xmat, target_xyz):
    """Signed (azimuth, elevation) of a world point in camera frame, degrees.

    MuJoCo cameras look down -z of their own frame, with +x right and +y up.
    """
    R = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    rel = np.asarray(target_xyz, dtype=float) - np.asarray(cam_xpos, dtype=float)
    local = R.T @ rel
    x, y, z = float(local[0]), float(local[1]), float(local[2])
    forward = -z
    if forward <= 1e-9:
        return None, None, float(np.linalg.norm(rel))
    az = math.degrees(math.atan2(x, forward))
    el = math.degrees(math.atan2(y, forward))
    return az, el, float(np.linalg.norm(rel))


def _wrap180(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def _fixation_stats(log):
    """Does a real target's bearing move differently from an artefact's?

    A cricket walks; the arena edge does not. With the head held still the
    difference should show up as bearing SPREAD across one fixation -- and if
    it does not, persistence is the wrong discriminator and so is everything
    built on it.
    """
    def block(rows):
        rows = [r for r in rows if len(r["bearings"]) >= 2]
        if not rows:
            return {"fixations": 0}
        spread = [float(np.max(r["bearings"]) - np.min(r["bearings"])) for r in rows]
        std = [float(np.std(r["bearings"])) for r in rows]
        return {"fixations": len(rows),
                "median_bearing_spread_deg": round(float(np.median(spread)), 2),
                "median_bearing_std_deg": round(float(np.median(std)), 2),
                "median_reports_per_fixation": round(float(np.median(
                    [len(r["bearings"]) for r in rows])), 1)}
    with_prey = [r for r in log if r["visible"] > 0]
    without = [r for r in log if r["visible"] == 0]
    return {"fixations_total": len(log),
            "fixations_with_prey_on_image": len(with_prey),
            "prey_present": block(with_prey),
            "prey_absent": block(without)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--gaze", type=float, default=2.0)
    ap.add_argument("--freeze", action="store_true",
                    help="with --controller oracle: face the prey but do not "
                         "approach. Isolates whether the eye is detecting the "
                         "PREY'S motion or the ANIMAL'S")
    ap.add_argument("--noise-rate", type=float, default=0.19,
                    help="MEASURED per-frame false-alarm rate the fixation "
                         "quorum is sized from")
    ap.add_argument("--motion-floor", type=float, default=None,
                    help="override Tectum.motion_floor -- the level a cell's "
                         "frame-to-frame change must clear to count as motion")
    ap.add_argument("--flow-gain-yaw", type=float, default=None,
                    help="override Tectum.flow_gain_yaw; the efference-copy "
                         "gain, INVENTED, re-derived in session 12 because its "
                         "input changed from the trunk gyro to the head gyro")
    ap.add_argument("--controller", default="search",
                    choices=["search", "still", "walk", "oracle"],
                    help="'oracle' steers on the TRUE bearing. It is a "
                         "measuring instrument, not a proposed controller: it "
                         "is the only way to sweep the close ranges the animal "
                         "cannot yet reach on its own, and what is being "
                         "measured is still the eye's own reports")
    args = ap.parse_args()

    import mujoco
    from brain.search import FixationEvidence, PreyEvidence, SearchPattern
    from brain.tectum import Eye
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False,
        eye=True, strike=True, homeostasis=True,
        gaze_pitch_gain=args.gaze,
        render_mode=None, max_steps=100000, seed=args.seed)
    env.eye = Eye(fovy_deg=env.eye.retina.fovy_deg, pixels=env.eye.retina.pixels,
                  cells=env.eye.retina.cells, motion_window=args.window)

    if args.motion_floor is not None:
        env.eye.tectum.motion_floor = float(args.motion_floor)
    if args.flow_gain_yaw is not None:
        env.eye.tectum.flow_gain_yaw = float(args.flow_gain_yaw)

    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])
    half_v = cam_fovy / 2.0
    half_h = math.degrees(math.atan(math.tan(math.radians(half_v)) * 1.0))

    env.reset(seed=args.seed)
    height = float(env.prey.height_m)
    search = SearchPattern()
    evidence = PreyEvidence()
    # Quorum DERIVED from the false-alarm rate this same tool measured, not
    # chosen. Re-derive it whenever that rate changes.
    fix = FixationEvidence.from_noise_rate(args.noise_rate,
                                           fixate_steps=SearchPattern.FIXATE_STEPS)

    def trunk_yaw_deg():
        q = env.walk_env.data.qpos[3:7]
        return math.degrees(math.atan2(
            2.0 * (q[0] * q[3] + q[1] * q[2]),
            1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))

    state_counts = Counter()
    by_state = {}   # state -> [frames, on_image, true_reports, false_reports]
    n = 0
    on_image = behind = az_out = el_out = 0
    both_out = 0
    az_abs, el_vals, ranges, sizes = [], [], [], []
    false_bearings, true_bearings = [], []
    reports = committed = committed_true = 0
    loose_true = aimed_true = 0
    fix_commits = fix_commits_true = 0
    # Recall broken out by RANGE. The animal strikes at 20.3 mm and the
    # published approach is 2-30 cm; measuring one number averaged over a
    # 40 cm arena says nothing about whether the eye works where it matters.
    RANGE_EDGES = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 9.0]
    by_range = {}
    # Recall as a function of ECCENTRICITY -- how far off the optical axis the
    # prey sits. If the eye only finds what is already near the middle, then a
    # search that inspects the world in 22 deg steps is asking it the wrong
    # question, and the fix is a second look that centres a candidate before
    # deciding on it.
    ECC_EDGES = [0, 5, 10, 15, 20, 25, 30, 36]
    by_ecc = {}
    fixation_log = []
    _prev_look = False
    fix_bearing_err = []
    gaze = []

    env_flow_gain = float(env.eye.tectum.flow_gain_yaw)
    _mf = float(env.eye.tectum.motion_floor)
    try:
        for i in range(args.steps):
            if args.controller == "still":
                heading_deg, head_deg, moving = 0.0, 0.0, False
            elif args.controller == "oracle":
                dxy = np.asarray(env.food_xy) - env._nose_xy()
                world = math.degrees(math.atan2(dxy[1], dxy[0]))
                heading_deg = _wrap180(world - trunk_yaw_deg())
                head_deg, moving = 0.0, not args.freeze
            elif args.controller == "walk":
                # The old behaviour, kept as the control: walk straight ahead
                # with a fixed head. This is what produced 581/600 behind.
                heading_deg, head_deg, moving = 0.0, 0.0, True
            else:
                heading_deg, head_deg, moving = search.step(trunk_yaw_deg())
            env._gaze_yaw_rad = math.radians(head_deg)

            heading = math.radians(heading_deg)
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action[0] = math.cos(heading)
            action[1] = math.sin(heading)
            action[2] = (0.35 - 0.05) / 0.375 - 1.0
            action[3] = 1.0 if moving else -1.0
            # THE OFF SWITCH (#253). `engage` above changes the goal distance
            # and nothing else -- it never stopped the animal. This does.
            env.walk_env.locomotor_drive = 1.0 if moving else 0.0
            _, _, term, trunc, info = env.step(action)
            state = search.last.get("state", args.controller)
            if state == "scan":
                state = "scan.fixate" if search.last.get("looking") else "scan.saccade"
            state_counts[state] += 1
            row = by_state.setdefault(state, [0, 0, 0, 0])
            row[0] += 1

            _range_row = _ecc_row = None
            d = env.walk_env.data
            tgt = np.array([env.food_xy[0], env.food_xy[1], height])
            az, el, rng = camera_frame_angles(d.cam_xpos[cam_id],
                                              d.cam_xmat[cam_id], tgt)
            gaze.append(float(info.get("gaze_pitch_deg", 0.0)))
            n += 1
            if az is None:
                behind += 1
            else:
                az_abs.append(abs(az)); el_vals.append(el); ranges.append(rng)
                a_out, e_out = abs(az) > half_h, abs(el) > half_v
                px = project(d.cam_xpos[cam_id], d.cam_xmat[cam_id], tgt,
                             cam_fovy, pixels=PIX)
                vis = bool(px is not None and 0 <= px[0] < PIX and 0 <= px[1] < PIX)
                if vis:
                    on_image += 1
                    row[1] += 1
                rb = next(k for k in range(len(RANGE_EDGES) - 1)
                          if rng < RANGE_EDGES[k + 1])
                key = f"{int(RANGE_EDGES[rb]*1000)}-{int(RANGE_EDGES[rb+1]*1000)}mm"
                rr = by_range.setdefault(key, [0, 0, 0, 0])
                rr[0] += 1
                rr[1] += int(vis)
                _range_row = rr
                if vis:
                    _eb = next(k for k in range(len(ECC_EDGES) - 1)
                               if abs(az) < ECC_EDGES[k + 1])
                    _ek = f"{ECC_EDGES[_eb]:2d}-{ECC_EDGES[_eb+1]:2d}deg"
                    _ecc_row = by_ecc.setdefault(_ek, [0, 0])
                    _ecc_row[0] += 1
                if vis:
                    # Angular size of the prey at this range, for the RANGE case.
                    sizes.append(2 * math.degrees(math.atan(
                        PREY_LEN_M / 2.0 / max(rng, 1e-6))))
                if a_out and e_out:
                    both_out += 1
                elif a_out:
                    az_out += 1
                elif e_out:
                    el_out += 1

            b = info.get("prey_bearing_deg")
            if float(info["food_visible_frac"]) > 0.0 and b is not None:
                reports += 1
                # A TRUE POSITIVE IS THE EYE POINTING AT THE PREY, not the
                # eye firing while the prey happens to be somewhere on the
                # image (#252). Every recall number this project has quoted
                # used the looser definition, which counts a report aimed 60
                # deg away from the cricket as a hit so long as the cricket was
                # in frame. Both are reported here so the gap is visible.
                on_img = (az is not None and abs(az) <= half_h
                          and abs(el) <= half_v)
                aimed = bool(on_img and abs(float(b) - az) <= AIM_TOL_DEG)
                if on_img:
                    loose_true += 1
                if aimed:
                    aimed_true += 1
                    if _range_row is not None:
                        _range_row[2] += 1
                    if _ecc_row is not None:
                        _ecc_row[1] += 1
                real = aimed
                if az is not None and not real:
                    _rb = next(k for k in range(len(RANGE_EDGES) - 1)
                               if rng < RANGE_EDGES[k + 1])
                    _k = f"{int(RANGE_EDGES[_rb]*1000)}-{int(RANGE_EDGES[_rb+1]*1000)}mm"
                    by_range.setdefault(_k, [0, 0, 0, 0])[3] += 1
                (true_bearings if real else false_bearings).append(float(b))
                row[2 if real else 3] += 1
            # Per-fixation record: the bearings reported inside one look, and
            # whether the prey was really on the image during it. This is the
            # measurement that decides what discriminates a cricket from a
            # static scene edge -- persistence does not (#248).
            looking_now = bool(search.last.get("looking"))
            if looking_now and not _prev_look:
                cur_fix = {"bearings": [], "visible": 0, "frames": 0}
                fixation_log.append(cur_fix)
            _prev_look = looking_now
            if looking_now and fixation_log:
                cur = fixation_log[-1]
                cur["frames"] += 1
                cur["visible"] += int(az is not None and abs(az) <= half_h
                                      and abs(el) <= half_v)
                if b is not None and float(info["food_visible_frac"]) > 0:
                    cur["bearings"].append(float(b))

            rep = (float(b) if (b is not None
                                and float(info["food_visible_frac"]) > 0)
                   else None)
            was = fix.commits
            cf = fix.step(rep, bool(search.last.get("looking")))
            if fix.commits > was:
                fix_commits += 1
                if az is not None and abs(az) <= half_h and abs(el) <= half_v:
                    fix_commits_true += 1
                    fix_bearing_err.append(abs(float(cf) - az))
            c = evidence.step(rep)
            if c is not None:
                committed += 1
                if az is not None and abs(az) <= half_h and abs(el) <= half_v:
                    committed_true += 1
            if term or trunc:
                env.reset(seed=args.seed + i + 1)
                search.reset(); evidence.reset(); fix.reset()
    finally:
        env.close()

    def hist(vals, width=20):
        c = Counter(int(round(v / width)) * width for v in vals)
        return {str(k): c[k] for k in sorted(c)}

    out = {
        "schema_version": 1,
        "generated_by": "tools/oracle_free_diagnosis.py",
        "question": "with the oracle off, WHY is the prey never on the image?",
        "controller": args.controller + ("+freeze" if args.freeze else ""),
        "seed": args.seed, "steps": n,
        "flow_gain_yaw": float(env_flow_gain),
        "motion_floor": _mf,
        "camera_half_angle_deg": {"horizontal": round(half_h, 2),
                                  "vertical": round(half_v, 2)},
        "frames": {
            "prey_on_image": on_image,
            "behind_camera": behind,
            "outside_azimuth_only": az_out,
            "outside_elevation_only": el_out,
            "outside_both": both_out,
        },
        "prey_geometry": {
            "median_abs_azimuth_deg": round(float(np.median(az_abs)), 2) if az_abs else None,
            "median_elevation_deg": round(float(np.median(el_vals)), 2) if el_vals else None,
            "median_range_mm": round(float(np.median(ranges)) * 1000, 1) if ranges else None,
            "median_angular_size_deg_when_on_image":
                round(float(np.median(sizes)), 3) if sizes else None,
            "abs_azimuth_histogram_deg": hist(az_abs),
        },
        "eye_reports": {
            "definitions": {
                "aimed_at_prey": f"report within {AIM_TOL_DEG} deg of the prey's "
                                 f"true bearing -- the eye FOUND it",
                "prey_merely_on_image": "the older, looser definition every "
                                        "previous recall number in this project "
                                        "used",
            },
            "reports_aimed_at_prey": aimed_true,
            "reports_while_prey_merely_on_image": loose_true,
            "total": reports,
            "true_positives": len(true_bearings),
            "false_alarms": len(false_bearings),
            "false_alarm_rate_per_frame": round(len(false_bearings) / n, 4) if n else None,
            "false_alarm_bearing_histogram_deg": hist(false_bearings),
        },
        "accumulator": {
            "steps_committed": committed,
            "steps_committed_while_prey_actually_visible": committed_true,
        },
        "by_range": {k: {"frames": v[0], "prey_on_image": v[1],
                         "true_reports": v[2], "false_reports": v[3],
                         "on_image_fraction": round(v[1] / v[0], 4) if v[0] else None,
                         "aimed_recall_when_on_image": round(v[2] / v[1], 4) if v[1] else None}
                     for k, v in sorted(by_range.items(),
                                        key=lambda kv: int(kv[0].split("-")[0]))},
        "by_eccentricity": {k: {"frames_prey_on_image": v[0],
                                "aimed_reports": v[1],
                                "aimed_recall": round(v[1] / v[0], 4) if v[0] else None}
                            for k, v in sorted(by_ecc.items())},
        "per_fixation": _fixation_stats(fixation_log),
        "fixation_evidence": {
            "quorum_derivation": fix.derivation,
            "fixations": fix.fixations,
            "commit_events": fix_commits,
            "commit_events_with_prey_really_visible": fix_commits_true,
            "commit_precision": (round(fix_commits_true / fix_commits, 4)
                                 if fix_commits else None),
            "median_bearing_error_deg_when_right":
                round(float(np.median(fix_bearing_err)), 2) if fix_bearing_err else None,
        },
        "search_state_counts": dict(state_counts),
        "by_state": {k: {"frames": v[0], "prey_on_image": v[1],
                         "true_reports": v[2], "false_reports": v[3],
                         "false_alarms_per_frame": round(v[3] / v[0], 4) if v[0] else None,
                         "recall_when_on_image": round(v[2] / v[1], 4) if v[1] else None}
                     for k, v in sorted(by_state.items())},
        "gaze_pitch_deg": {
            "median": round(float(np.median(gaze)), 2),
            "min": round(float(np.min(gaze)), 2),
            "max": round(float(np.max(gaze)), 2),
        },
    }
    ev = REPO / f"artifacts/evidence/session11/oracle_free_diagnosis_{args.controller}.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
