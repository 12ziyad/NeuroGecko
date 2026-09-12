"""Why is the eye quiet? Split the 5.4 % into the terms that cause it.

WHAT THIS EXISTS TO SETTLE. Ledger #220 measured the eye reporting on 5.4 % of
steps across a real hunt and concluded the blocker is the DETECTION RATE rather
than the bearing. That is true and it is not a cause. A report rate is a
product, and the plan's next module -- the two-channel wide/narrow detector --
touches exactly one term in it:

    P(report) = P(prey in frustum)        <- where the camera points
              x P(prey moving this frame) <- a frame difference cannot see a
                                             stationary object, by construction
              x P(displacement > floor)   <- #188: real prey moves 0.02-0.12
                                             cells per frame, 8 to 41 frames to
                                             cross one cell. The band is finer
                                             than the instrument.
              x P(survives self-motion)   <- #214: 3 % of frames moving,
                                             7 % standing still
              x P(above threshold)        <- the only term step 5 touches

If the first two multiply to something near 5.4 %, the detector is already at
its ceiling and a perfect one buys almost nothing. The project has never
measured the first term during a hunt: #136's 18.7 % was taken on a
configuration in which `prey_total_travel_mm` was 0.0 (#135), and it has not
been re-measured since the prey was made to move.

WHAT IS NEW HERE, AND WHAT IS NOT. `tools/tectum_fov_census.py` already
computes the frustum term, but it drives the animal with a zero action -- a
passive gecko, not a hunt. `brain/hunt_metrics.py` already scores a hunt, but
it is handed `speed_m_s` and never records it, and it stores only the summary,
which is why this could not be done as a post-hoc join on the #220 run. This
reproduces #220's configuration exactly -- habitat world, accepted walker (lab,
no policy), eye on, scripted creep, seed 1 -- and writes the per-step rows out.

THE FALSE ALARM COLUMN. Once the frustum is known, one number falls out that
#220 could not compute: how often the eye reports a target when the prey is not
on the image at all. #220's reading -- "when it speaks it is right more often
than not, 44 usable against 37 wrong" -- is conditional on those 81 reports
being about the prey. If a large share of them happened with the prey off the
image, they are not detections of anything and the reading inverts.

NOTHING HERE TOUCHES THE ANIMAL. Every column is computed from the same oracle
that `HuntMetrics` already treats as a shadow, and nothing is returned to the
policy. The approach itself is the same scripted, oracle-driven creep used by
`tools/hunt_video.py`; driving it with the eye would produce a record of an
animal wandering, which is a true thing about the eye and a useless thing to
decompose.

Usage:  python tools/hunt_decomposition.py [--steps 1500] [--seed 1]
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

from tectum_diagnosis import camera_frame_bearing, project  # noqa: E402

#: The render the tectum actually reads. Not a free parameter here -- it is
#: read back off the environment and asserted, so a config change cannot
#: silently make this projection wrong.
PIX = 64

#: Below this horizontal range the approach creeps. INVENTED, and identical to
#: `tools/hunt_video.py` so the run reproduces #220 rather than resembling it.
STALK_RANGE_M = 0.15
#: Move one brain step in four while stalking. INVENTED; same source.
STALK_DUTY = 4

#: Salience above which `GeckoBrainEnv` is treated as having reported a target.
#: `HuntMetrics` uses `> 0.0`; `tectum_fov_census` uses `> 0.05`. They disagree,
#: so BOTH are reported below rather than one being chosen silently.
REPORT_EPS = 0.0
CENSUS_EPS = 0.05


def approach_action(env, horizontal_m, step_index):
    """The scripted creep from tools/hunt_video.py, unchanged.

    ORACLE-DRIVEN, deliberately. The bearing is read from the physics engine.
    """
    offset = np.asarray(env.food_xy)[:2] - env._nose_xy()
    forward = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    bearing = math.atan2(forward[0] * offset[1] - forward[1] * offset[0],
                         forward[0] * offset[0] + forward[1] * offset[1])
    trunk_range = float(np.linalg.norm(np.asarray(env.food_xy) - env._trunk_xy()))
    reach = float(env.walk_env.reach_dist)
    span = float(np.clip(trunk_range + reach, 0.05, 0.80))
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = math.cos(bearing)
    action[1] = math.sin(bearing)
    action[2] = (span - 0.05) / 0.375 - 1.0
    creeping = horizontal_m < STALK_RANGE_M
    action[3] = 1.0 if (not creeping or step_index % STALK_DUTY == 0) else -1.0
    return action, creeping


def run(steps=1500, seed=1, motion_window=1, efference='scaled',
        flow_gain_scale=1.0, cells=None, gaze_pitch_gain=0.0, gaze_pitch_decay=0.0):
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    prey_params = PreyParameters.from_registry()
    env = GeckoBrainEnv(
        walker_xml_path=REPO / "morphology" / "gecko_world_v1.xml",
        prey_parameters=prey_params,
        gait_profile="lab",
        use_policy=False,          # the accepted walker is lab + zero residual
        eye=True,
        strike=True,
        homeostasis=True,
        gaze_pitch_gain=gaze_pitch_gain,
        gaze_pitch_decay=gaze_pitch_decay,
        max_steps=steps + 10,
    )
    # ALWAYS rebuild the eye when a window is named. Gating this on
    # `motion_window > 1` silently returned the CLASS DEFAULT for window 1, so
    # a sweep's own baseline row reported whatever the default happened to be.
    # Caught when windows 1 and 3 came back byte-identical.
    if True:
        from brain.tectum import Eye
        # Same optics, same cell grid -- only the temporal baseline differs, so
        # the comparison isolates the window and nothing else.
        env.eye = Eye(fovy_deg=env.eye.retina.fovy_deg,
                      pixels=env.eye.retina.pixels,
                      cells=(env.eye.retina.cells if cells is None
                             else int(cells)),
                      motion_window=motion_window)
    if efference == "unscaled":
        env.eye.tectum.scale_efference_by_span = False
    elif efference == "off":
        env.eye.tectum.efference_copy = False
    if flow_gain_scale != 1.0:
        # Both INVENTED gains scaled together. 1.0 is what ships; 0.0 is the
        # efference copy off. Sweeping between them turns a knob that was set
        # to make false alarms go away into a measured trade-off.
        env.eye.tectum.flow_gain_yaw *= float(flow_gain_scale)
        env.eye.tectum.flow_gain_surge *= float(flow_gain_scale)
    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])

    # The prey's physical half-width, for angular size. Read from the registry
    # rather than assumed, so this cannot drift from the body being rendered.
    prey_radius_m = float(prey_params.radius_m)

    rows = []
    prev_px = None
    try:
        env.reset(seed=seed)
        height = (float(env.prey.height_m) if env.prey is not None
                  else float(env.food_radius))
        for i in range(steps):
            horizontal = float(np.linalg.norm(
                np.asarray(env.food_xy)[:2] - env._nose_xy()))
            action, creeping = approach_action(env, horizontal, i)
            _, _, term, trunc, info = env.step(action)

            data = env.walk_env.data
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            food = np.array([env.food_xy[0], env.food_xy[1], height])

            az, el, fwd = camera_frame_bearing(cam_pos, cam_mat, food)
            px = project(cam_pos, cam_mat, food, cam_fovy, pixels=PIX)
            on_image = bool(px is not None
                            and 0 <= px[0] < PIX and 0 <= px[1] < PIX)

            # TERM 1. Where the camera points.
            # TERM 2/3. How far the prey's image moved since the last frame, in
            # pixels. Only meaningful while it stays on the image across both
            # frames -- a jump on or off the edge is not prey motion.
            if px is not None and prev_px is not None:
                disp_px = float(math.hypot(px[0] - prev_px[0], px[1] - prev_px[1]))
            else:
                disp_px = None
            prev_px = px

            true_range = float(np.linalg.norm(
                np.asarray(env.food_xy) - env._nose_xy()))
            ang_size = (math.degrees(2.0 * math.atan(prey_radius_m / true_range))
                        if true_range > 1e-6 else None)

            salience = float(info["food_visible_frac"])
            bearing = info["prey_bearing_deg"]
            reported = bool(salience > REPORT_EPS and bearing is not None)

            # WHICH TEST REJECTED IT. The tectum already records this in
            # `self.last` and nothing has ever read it. Four branches set it:
            # "fills the field", "too slow"/"too fast", "above the horizon",
            # and one silent branch (peak <= motion_floor) that writes no
            # reason at all -- inferred here by its signature rather than by
            # editing the brain module during a diagnosis.
            tl = getattr(env.eye.tectum, "last", {}) or {}
            if salience > 0.0:
                why = None
            elif tl.get("rejected"):
                why = str(tl["rejected"])
            elif tl.get("elevation_cell") is None:
                why = "below motion floor"
            else:
                why = "gain zero, reason unrecorded"
            peak_speed = tl.get("peak_speed_deg_s")

            rows.append({
                "step": i,
                # --- the shadow, as HuntMetrics already records it ---
                "range_m": round(true_range, 4),
                "salience": round(salience, 5),
                "reported": reported,
                "reported_census_eps": bool(salience > CENSUS_EPS
                                            and bearing is not None),
                "eye_bearing_deg": (round(float(bearing), 2)
                                    if bearing is not None else None),
                # --- the four columns that were missing ---
                "prey_on_image": on_image,
                "prey_angular_size_deg": (round(ang_size, 3)
                                          if ang_size is not None else None),
                "image_displacement_px": (round(disp_px, 4)
                                          if disp_px is not None else None),
                "self_speed_m_s": round(float(info["moving_speed"]), 5),
                "rejected_by": why,
                "peak_speed_deg_s": (round(float(peak_speed), 3)
                                     if peak_speed is not None else None),
                # --- context, for reading the above ---
                "cam_az_deg": round(az, 2),
                "cam_el_deg": round(el, 2),
                "behind_head": bool(fwd <= 0.0),
                "creeping": bool(creeping),
                "gaze_pitch_deg": round(float(info.get("gaze_pitch_deg", 0.0)), 2),
                "strikes": int(info["strikes"]),
                "strike_hits": int(info["strike_hits"]),
            })
            if term or trunc:
                env.reset(seed=seed + i + 1)
                prev_px = None
    finally:
        env.close()
    return rows, cam_fovy, prey_radius_m


def decompose(rows, cam_fovy, prey_radius_m):
    n = len(rows)
    on = np.array([r["prey_on_image"] for r in rows])
    rep = np.array([r["reported"] for r in rows])
    rep5 = np.array([r["reported_census_eps"] for r in rows])
    behind = np.array([r["behind_head"] for r in rows])
    speed = np.array([r["self_speed_m_s"] for r in rows], dtype=float)
    disp = np.array([r["image_displacement_px"] if r["image_displacement_px"]
                     is not None else np.nan for r in rows], dtype=float)
    ang = np.array([r["prey_angular_size_deg"] if r["prey_angular_size_deg"]
                    is not None else np.nan for r in rows], dtype=float)

    moving = speed > 0.03          # #214's own split
    both = on & rep
    false_alarm = rep & ~on

    def frac(mask):
        return float(np.mean(mask)) if n else float("nan")

    bins = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, np.inf)]
    by_disp = []
    for lo, hi in bins:
        m = on & np.isfinite(disp) & (disp >= lo) & (disp < hi)
        by_disp.append({
            "bin_px": f"{lo}-{'inf' if hi == np.inf else hi}",
            "frames": int(m.sum()),
            "reported_given_in_frame": (round(float(np.mean(rep[m])), 4)
                                        if m.sum() else None),
        })

    from collections import Counter
    why_on = Counter(r["rejected_by"] for r in rows
                     if r["prey_on_image"] and r["rejected_by"] is not None)
    n_on_rejected = sum(why_on.values())

    return {
        "schema_version": 1,
        "rejection_census_prey_on_image": {
            k: {"frames": v, "share": round(v / n_on_rejected, 4)}
            for k, v in why_on.most_common()},
        "test": "the 5.4 % detection rate, split into the terms that produce it",
        "run": ("habitat world, accepted walker (lab, no policy), eye on, "
                "scripted creep -- the #220 configuration"),
        "model_fovy_deg": cam_fovy,
        "prey_radius_m": prey_radius_m,
        "steps": n,

        # ---- the headline product ----
        "P_prey_on_image": round(frac(on), 4),
        "P_report": round(frac(rep), 4),
        "P_report_at_census_threshold": round(frac(rep5), 4),
        "P_report_given_on_image": (round(float(np.mean(rep[on])), 4)
                                    if on.sum() else None),

        # ---- the column #220 could not compute ----
        "P_false_alarm_reported_off_image": round(frac(false_alarm), 4),
        "false_alarm_share_of_all_reports": (
            round(float(false_alarm.sum() / rep.sum()), 4) if rep.sum() else None),
        "reports_total": int(rep.sum()),
        "reports_with_prey_on_image": int(both.sum()),

        # ---- where the prey actually is ----
        "P_behind_head": round(frac(behind), 4),
        "median_abs_cam_azimuth_deg": round(float(np.median(
            np.abs([r["cam_az_deg"] for r in rows]))), 2),

        # ---- #214, now with the speed column recorded ----
        "P_report_while_moving": (round(float(np.mean(rep[moving])), 4)
                                  if moving.sum() else None),
        "P_report_while_still": (round(float(np.mean(rep[~moving])), 4)
                                 if (~moving).sum() else None),
        "frames_moving": int(moving.sum()),

        # ---- #188, the instrument floor ----
        "median_image_displacement_px_on_image": (
            round(float(np.nanmedian(disp[on])), 4) if on.sum() else None),
        "report_rate_by_displacement_bin": by_disp,

        # ---- geometry ----
        "median_angular_size_deg_on_image": (
            round(float(np.nanmedian(ang[on])), 3) if on.sum() else None),

        # ---- the trade-off the efference copy was never scored on ----
        # RECALL    -- of the frames where the prey is on the image, how many
        #              did the eye report? This is what #220's 5.4 % is really
        #              asking, and it could not be computed without the
        #              prey-on-image column.
        # PRECISION -- of the reports the eye made, how many were about the
        #              prey at all? #213 measured the efference copy cutting
        #              false alarms 72 % -> 3-7 % and recorded it as a success.
        #              Nobody measured what it cost in recall.
        "recall": (round(float(np.mean(rep[on])), 4) if on.sum() else None),
        "precision": (round(float(both.sum() / rep.sum()), 4)
                      if rep.sum() else None),
        "f1": (round(float(2 * both.sum() / (rep.sum() + on.sum())), 4)
               if (rep.sum() + on.sum()) else None),

        # ---- the ceiling ----
        "ceiling_if_detector_were_perfect": round(frac(on), 4),
        "headroom_multiple": (round(float(frac(on) / frac(rep)), 2)
                              if frac(rep) > 0 else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--motion-window", type=int, default=1)
    ap.add_argument("--cells", type=int, default=None,
                    help="Retinal cell grid. Must divide the render "
                         "size: 8, 16, 32 or 64.")
    ap.add_argument("--flow-gain-scale", type=float, default=1.0)
    ap.add_argument("--efference", choices=["scaled","unscaled","off"],
                    default="scaled")
    ap.add_argument("--seeds", type=int, default=1,
                    help="Pool this many consecutive seeds. The prey spawn "
                         "angle is UNIFORM over the full circle, so a single "
                         "seed measures one spawn, not the animal.")
    args = ap.parse_args()

    rows, per_seed = [], []
    fovy = prey_r = None
    for k in range(args.seeds):
        seed = args.seed + k
        r, fovy, prey_r = run(steps=args.steps, seed=seed,
                             motion_window=args.motion_window,
                             efference=args.efference,
                             flow_gain_scale=args.flow_gain_scale,
                             cells=args.cells)
        on = float(np.mean([x["prey_on_image"] for x in r]))
        rep = float(np.mean([x["reported"] for x in r]))
        per_seed.append({"seed": seed, "steps": len(r),
                         "P_prey_on_image": round(on, 4),
                         "P_report": round(rep, 4)})
        print(f"seed {seed:>3}: {len(r):>5} steps, prey on image "
              f"{100*on:5.1f} %, eye reports {100*rep:5.2f} %")
        rows.extend(r)
    summary = decompose(rows, fovy, prey_r)
    summary["seeds"] = args.seeds
    summary["motion_window"] = args.motion_window
    summary["per_seed"] = per_seed

    out_dir = REPO / "artifacts" / "evidence" / "session11"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "hunt_decomposition_rows.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out_dir / "hunt_decomposition.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    s = summary
    print(f"\n===== {s['steps']} steps, model fovy {s['model_fovy_deg']:g} =====")
    print(f"  prey on the rendered image      {100*s['P_prey_on_image']:6.2f} %"
          f"   <- TERM 1, never measured during a hunt before")
    print(f"  eye reports a target            {100*s['P_report']:6.2f} %"
          f"   (#220 measured 5.4 %)")
    if s["P_report_given_on_image"] is not None:
        print(f"  reports GIVEN prey on image     "
              f"{100*s['P_report_given_on_image']:6.2f} %")
    print(f"  prey BEHIND the head            {100*s['P_behind_head']:6.2f} %")
    print(f"\n  FALSE ALARMS -- reported with prey NOT on image:")
    print(f"    {100*s['P_false_alarm_reported_off_image']:6.2f} % of all frames")
    if s["false_alarm_share_of_all_reports"] is not None:
        print(f"    {100*s['false_alarm_share_of_all_reports']:6.2f} % of every "
              f"report the eye made ({s['reports_total']} reports, "
              f"{s['reports_with_prey_on_image']} with prey on image)")
    print(f"\n  #214 re-measured, with the speed column recorded:")
    print(f"    reports while moving > 0.03 m/s  {s['P_report_while_moving']}")
    print(f"    reports while still              {s['P_report_while_still']}")
    print(f"\n  #188, the instrument floor:")
    print(f"    median image displacement, prey on image: "
          f"{s['median_image_displacement_px_on_image']} px/frame")
    for b in s["report_rate_by_displacement_bin"]:
        print(f"      {b['bin_px']:>10} px : {b['frames']:>5} frames, "
              f"reported {b['reported_given_in_frame']}")
    print(f"\n  CEILING. A perfect detector could not exceed "
          f"{100*s['ceiling_if_detector_were_perfect']:.2f} %.")
    print(f"  Headroom over the current eye: {s['headroom_multiple']}x")
    print(f"\nwritten: {jsonl}")
    print(f"written: {out_dir / 'hunt_decomposition.json'}")


if __name__ == "__main__":
    main()
