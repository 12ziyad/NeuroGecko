"""Where the detection threshold should sit, measured instead of chosen.

`Tectum.motion_floor` is the level a cell's local motion has to clear before the
eye will call it a target. It shipped at 1e-4, INVENTED, and had never been
checked against the noise it exists to reject. It turns out to sit roughly four
hundred times BELOW that noise: with the camera motionless and nothing in the
scene moving, the gated quantity exceeds 1e-4 on 100 % of frames and the eye
fires on about two frames in three (#254).

This measures the distribution of the exact quantity the threshold gates --
`Tectum._peak_local`, the peak of the local-motion map after surround
subtraction -- separately for frames where the prey really is on the image and
frames where it is not, and prints the recall and false-alarm rate that every
candidate threshold would produce.

IT MEASURES TWO CONDITIONS, AND THE DIFFERENCE BETWEEN THEM IS THE POINT.

  WALKING   median peak 0.0589 with the prey on the image, 0.0000 with it off.
            Best operating point 57.2 % recall at 20.3 % false alarms.
  STANDING  with `locomotor_drive` at 0.0, which did not exist before #253:
            0.1499 against 0.0751 -- the SIGNAL is 2.5x larger. Best operating
            point 92.7 % recall at 48.5 % false alarms.

  An earlier single-seed run of this had the walking noise EXCEEDING the
  walking signal, and that was reported as a general fact before it was
  checked across seeds. It is not true (#259). What is true is the 2.5x.

So the threshold is not one number. It is one number FOR AN ANIMAL THAT IS
HOLDING STILL, and the eye should not be consulted otherwise. That is a claim
about behaviour, not about optics, and it is the reason `brain/search.py`
freezes before it looks.

The oracle appears here as a MEASURING INSTRUMENT: the prey's true position is
used to label frames and to walk the animal into range, never to steer anything
whose performance is being reported.

Usage:  python tools/motion_floor_derivation.py [--trials 20]
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

from tectum_diagnosis import project  # noqa: E402

PIX = 64
FLOORS = (1e-4, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
#: Steps after the gait stops before the eye is read. DERIVED -- see #258 and
#: SearchPattern.SETTLE_STEPS; the body sways for about seventy steps.
SETTLE = 70


def trunk_yaw_deg(env):
    q = env.walk_env.data.qpos[3:7]
    return math.degrees(math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                                   1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--hold", type=int, default=160)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--flow-gain-yaw", type=float, default=0.006)
    ap.add_argument("--approach-to", type=float, default=0.18)
    args = ap.parse_args()

    import mujoco
    from brain.tectum import Eye
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False,
        eye=True, strike=True, gaze_pitch_gain=2.0,
        render_mode=None, max_steps=100000, seed=2)
    env.eye = Eye(fovy_deg=env.eye.retina.fovy_deg, pixels=env.eye.retina.pixels,
                  cells=env.eye.retina.cells, motion_window=args.window)
    env.eye.tectum.flow_gain_yaw = float(args.flow_gain_yaw)

    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])
    action = np.zeros(env.action_space.shape, dtype=np.float32)

    walk = {"on": [], "off": []}
    stand = {"on": [], "off": []}

    def sample(bucket, height):
        d = env.walk_env.data
        px = project(d.cam_xpos[cam_id], d.cam_xmat[cam_id],
                     np.array([env.food_xy[0], env.food_xy[1], height]),
                     cam_fovy, pixels=PIX)
        vis = bool(px is not None and 0 <= px[0] < PIX and 0 <= px[1] < PIX)
        bucket["on" if vis else "off"].append(
            float(getattr(env.eye.tectum, "_peak_local", 0.0)))

    try:
        for trial in range(args.trials):
            env.reset(seed=300 + trial)
            env.walk_env.locomotor_drive = 1.0
            height = float(env.prey.height_m)
            for _ in range(900):
                dxy = np.asarray(env.food_xy) - env._nose_xy()
                world = math.degrees(math.atan2(dxy[1], dxy[0]))
                hd = math.radians((world - trunk_yaw_deg(env) + 180.0) % 360.0 - 180.0)
                action[0], action[1] = math.cos(hd), math.sin(hd)
                action[2], action[3] = -0.2, 1.0
                env.step(action)
                sample(walk, height)
                if float(np.linalg.norm(np.asarray(env.food_xy)
                                        - env._nose_xy())) < args.approach_to:
                    break
            env.walk_env.locomotor_drive = 0.0
            for k in range(args.hold):
                env.step(action)
                if k >= SETTLE:
                    sample(stand, height)
    finally:
        env.close()

    def block(b):
        on, off = np.array(b["on"]), np.array(b["off"])
        rows = []
        for f in FLOORS:
            rows.append({
                "floor": f,
                "recall": round(float(np.mean(on > f)), 4) if on.size else None,
                "false_alarm": round(float(np.mean(off > f)), 4) if off.size else None,
                "separation": (round(float(np.mean(on > f) - np.mean(off > f)), 4)
                               if on.size and off.size else None)})
        return {"n_prey_on_image": int(on.size), "n_prey_off_image": int(off.size),
                "median_peak_prey_on_image": round(float(np.median(on)), 4) if on.size else None,
                "median_peak_prey_off_image": round(float(np.median(off)), 4) if off.size else None,
                "sweep": rows}

    out = {"schema_version": 1,
           "generated_by": "tools/motion_floor_derivation.py",
           "question": "where should Tectum.motion_floor sit?",
           "shipped_default": 1e-4,
           "trials": args.trials, "settle_steps": SETTLE,
           "walking": block(walk), "standing_still": block(stand),
           "note": "The oracle labels frames and walks the animal into range. "
                   "It steers nothing whose performance is reported here."}
    ev = REPO / "artifacts/evidence/session12/motion_floor_derivation.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for name in ("walking", "standing_still"):
        b = out[name]
        print(f"\n{name.upper()}   prey on image n={b['n_prey_on_image']}, "
              f"off n={b['n_prey_off_image']}   "
              f"median peak {b['median_peak_prey_on_image']} vs "
              f"{b['median_peak_prey_off_image']}")
        for r in b["sweep"]:
            if r["recall"] is None or r["false_alarm"] is None:
                continue
            print(f"   floor {r['floor']:7.4f}   recall {100*r['recall']:5.1f} %"
                  f"   false alarm {100*r['false_alarm']:5.1f} %"
                  f"   separation {100*r['separation']:+6.1f}")
    print(f"\nwritten: {ev}")


if __name__ == "__main__":
    main()
