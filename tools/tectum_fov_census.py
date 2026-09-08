"""How often is the prey in the field of view at all, over many episodes?

Also checks two things the single-episode dump cannot:

  * the SIGN CONVENTION of tools/tectum_field_test.py against the tectum's
    own. The test's true_bearing is +ve LEFT (atan2 of a z-up cross product);
    Retina.azimuth_of is +ve RIGHT ("Negative is left of the optical axis").
    If those disagree the correlation is negated before anything else happens.
  * whether the tectum's bearing tracks the prey on the SUBSET of frames
    where the prey is provably on the image -- the only frames on which any
    detector could possibly be right.
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

PIX = 64


def trunk_bearing_plus_left(env, food_xy):
    """Exactly what tools/tectum_field_test.py calls the truth."""
    data = env.walk_env.data
    forward = data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    offset = np.asarray(food_xy) - np.asarray(env._nose_xy())
    return math.degrees(math.atan2(
        forward[0] * offset[1] - forward[1] * offset[0],
        forward[0] * offset[0] + forward[1] * offset[1]))


def episode(seed, steps=150, cells=16, world=False, eye_fovy=70.0):
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from brain.tectum import Eye

    kwargs = dict(homeostasis=True, eye=True)
    if world:
        from envs.prey import PreyParameters
        kwargs["walker_xml_path"] = REPO / "morphology" / "gecko_world_v1.xml"
        kwargs["prey_parameters"] = PreyParameters.from_registry()
    env = GeckoBrainEnv(**kwargs)
    env.eye = Eye(fovy_deg=eye_fovy, pixels=PIX, cells=cells)
    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])
    out = []
    try:
        env.reset(seed=seed)
        height = float(env.prey.height_m) if env.prey is not None else float(env.food_radius)
        prev = np.asarray(env.food_xy, dtype=float).copy()
        moved_total = 0.0
        for _ in range(steps):
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
            data = env.walk_env.data
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            food = np.array([env.food_xy[0], env.food_xy[1], height])
            az, el, fwd = camera_frame_bearing(cam_pos, cam_mat, food)
            px = project(cam_pos, cam_mat, food, cam_fovy)
            moved = float(np.linalg.norm(np.asarray(env.food_xy) - prev))
            moved_total += moved
            prev = np.asarray(env.food_xy, dtype=float).copy()
            out.append({
                "az_cam": az, "el_cam": el, "fwd": fwd,
                "on_image": bool(px is not None and 0 <= px[0] < PIX
                                 and 0 <= px[1] < PIX),
                "in70": bool(fwd > 0 and abs(az) <= 35.0 and abs(el) <= 35.0),
                "in120": bool(fwd > 0 and abs(az) <= 60.0 and abs(el) <= 60.0),
                "trunk_bearing_plus_left": trunk_bearing_plus_left(env, env.food_xy),
                "reported": info["prey_bearing_deg"],
                "salience": float(info["food_visible_frac"]),
                "dist": float(np.linalg.norm(np.asarray(env.food_xy) - env._nose_xy())),
                "moved": moved,
            })
            if term or trunc:
                break
    finally:
        env.close()
    return out, cam_fovy, moved_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    every = []
    fovy = None
    total_prey_travel = 0.0
    for seed in range(1, args.seeds + 1):
        rows, fovy, moved = episode(seed, steps=args.steps, world=args.world)
        total_prey_travel += moved
        every.extend(rows)
        print(f"seed {seed:>3}: {len(rows):>4} frames, "
              f"in70 {100*np.mean([r['in70'] for r in rows]):5.1f} %, "
              f"in120 {100*np.mean([r['in120'] for r in rows]):5.1f} %, "
              f"prey travelled {moved*1000:.2f} mm")

    n = len(every)
    in70 = np.array([r["in70"] for r in every])
    in120 = np.array([r["in120"] for r in every])
    on_img = np.array([r["on_image"] for r in every])
    az = np.array([r["az_cam"] for r in every])
    label = "prey world (fovy 70)" if args.world else "field test config"
    print(f"\n===== {label} -- model fovy {fovy:g}, "
          f"{args.seeds} seeds, {n} frames =====")
    print(f"prey within a 70 deg field (+/-35 az, +/-35 el): "
          f"{in70.sum()}/{n} = {100*in70.mean():.1f} %")
    print(f"prey within a 120 deg field:                     "
          f"{in120.sum()}/{n} = {100*in120.mean():.1f} %")
    print(f"prey actually projects onto the rendered image:   "
          f"{on_img.sum()}/{n} = {100*on_img.mean():.1f} %")
    print(f"|azimuth| median {np.median(np.abs(az)):.1f} deg; "
          f"fraction beyond 90 deg (behind the head): "
          f"{100*np.mean(np.abs(az) > 90):.1f} %")
    print(f"total distance the prey MOVED across all episodes: "
          f"{total_prey_travel*1000:.3f} mm")

    rep = np.array([r["reported"] for r in every], dtype=float)
    tru_left = np.array([r["trunk_bearing_plus_left"] for r in every], dtype=float)
    ok = np.isfinite(rep) & np.isfinite(tru_left)
    print("\nSIGN CONVENTION")
    print(f"  corr(reported, trunk bearing +ve LEFT   [what the test uses]) = "
          f"{np.corrcoef(rep[ok], tru_left[ok])[0,1]:+.4f}")
    print(f"  corr(reported, camera azimuth  +ve RIGHT [what the eye means]) = "
          f"{np.corrcoef(rep[ok], az[ok])[0,1]:+.4f}")
    sub = ok & on_img
    if sub.sum() > 6:
        print(f"\nRESTRICTED to the {int(sub.sum())} frames where the prey is "
              f"provably on the image:")
        print(f"  corr(reported, camera azimuth +ve RIGHT) = "
              f"{np.corrcoef(rep[sub], az[sub])[0,1]:+.4f}")
        print(f"  mean |reported - camera azimuth| = "
              f"{np.mean(np.abs(rep[sub] - az[sub])):.1f} deg")
    fired = np.array([r["salience"] for r in every]) > 0.05
    print(f"\nthe tectum reports a target on {100*fired.mean():.1f} % of frames; "
          f"the prey is on the image on {100*on_img.mean():.1f} % of them.")
    print(f"  fires while the prey is NOT on the image: "
          f"{100*np.mean(fired & ~on_img):.1f} % of all frames")

    out = REPO / ("artifacts/evidence/session8/tectum_fov_census"
                  + ("_world" if args.world else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model_fovy_deg": fovy,
        "seeds": args.seeds,
        "frames": n,
        "in_70_deg_fraction": float(in70.mean()),
        "in_120_deg_fraction": float(in120.mean()),
        "on_image_fraction": float(on_img.mean()),
        "prey_total_travel_mm": total_prey_travel * 1000.0,
        "corr_reported_vs_trunk_plus_left": float(np.corrcoef(rep[ok], tru_left[ok])[0, 1]),
        "corr_reported_vs_camera_plus_right": float(np.corrcoef(rep[ok], az[ok])[0, 1]),
    }, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
