"""What is the tectum actually locking onto, and is the prey even in frame?

Diagnostic only, no fix. Measures, per frame:

  * the food's TRUE azimuth and elevation in the HEAD CAMERA's own frame,
    using the camera's ACTUAL fovy read out of the model -- so "inside the
    field of view" is a geometric fact rather than an assumption;
  * whether the food projects onto the 64x64 image at all, and onto which
    pixel;
  * where the tectum's motion peak actually is, and how far that is from the
    food's own pixel;
  * how much the food MOVED in the world this step.

Run with --world to use the generated prey world (fovy 70, real prey geom,
FleeingPrey) instead of the default walker body the field test uses.
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

PIX = 64


def camera_frame_bearing(cam_pos, cam_mat, point):
    """Azimuth (+ right) and elevation (+ up), degrees, in the camera frame.

    MuJoCo camera convention: looks down its own -z, +x right, +y up.
    """
    rel = np.asarray(point, dtype=float) - np.asarray(cam_pos, dtype=float)
    local = np.asarray(cam_mat, dtype=float).reshape(3, 3).T @ rel
    x, y, z = local
    forward = -z
    az = math.degrees(math.atan2(x, forward))
    el = math.degrees(math.atan2(y, math.hypot(x, forward)))
    return az, el, forward


def project(cam_pos, cam_mat, point, fovy_deg, pixels=PIX):
    rel = np.asarray(point, dtype=float) - np.asarray(cam_pos, dtype=float)
    local = np.asarray(cam_mat, dtype=float).reshape(3, 3).T @ rel
    x, y, z = local
    if -z <= 1e-9:
        return None
    focal_px = (pixels / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    col = focal_px * (x / -z) + pixels / 2.0 - 0.5
    row = pixels / 2.0 - 0.5 - focal_px * (y / -z)
    return col, row


def run(steps=150, seed=1, cells=16, world=False, eye_fovy=70.0):
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

    # Capture the retinal maps the tectum actually saw.
    captured = {}
    original_step = env.eye.retina.step

    def recording_step(image):
        out = original_step(image)
        captured["motion"] = np.array(out["motion"], copy=True)
        captured["image"] = np.array(image, copy=True)
        return out

    env.eye.retina.step = recording_step

    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])

    rows = []
    try:
        env.reset(seed=seed)
        food_height = float(env.prey.height_m) if env.prey is not None \
            else float(env.food_radius)
        prev_food = np.array(env.food_xy, dtype=float).copy()
        for i in range(steps):
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
            data = env.walk_env.data
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            food_xyz = np.array([env.food_xy[0], env.food_xy[1], food_height],
                                dtype=float)
            az, el, fwd = camera_frame_bearing(cam_pos, cam_mat, food_xyz)
            px = project(cam_pos, cam_mat, food_xyz, cam_fovy)
            nose = env._nose_xy()
            dist = float(np.linalg.norm(food_xyz[:2] - nose))
            ang_rad = math.degrees(math.atan(env.food_radius / max(dist, 1e-6)))

            motion = captured["motion"]
            peak_row, peak_col = np.unravel_index(int(np.argmax(motion)),
                                                  motion.shape)
            k = PIX / cells
            peak_px_col = (peak_col + 0.5) * k
            peak_px_row = (peak_row + 0.5) * k
            food_cell_col = None if px is None else int(np.clip(px[0] // k, 0, cells - 1))
            food_cell_row = None if px is None else int(np.clip(px[1] // k, 0, cells - 1))
            on_image = bool(px is not None and 0 <= px[0] < PIX and 0 <= px[1] < PIX)

            row = {
                "i": i,
                "az_cam_deg": az,
                "el_cam_deg": el,
                "in_fov": bool(abs(az) <= cam_fovy / 2 and abs(el) <= cam_fovy / 2
                               and fwd > 0),
                "px_col": None if px is None else float(px[0]),
                "px_row": None if px is None else float(px[1]),
                "on_image": on_image,
                "dist_m": dist,
                "food_angular_radius_deg": ang_rad,
                "food_moved_m": float(np.linalg.norm(
                    np.asarray(env.food_xy, dtype=float) - prev_food)),
                "salience": float(info["food_visible_frac"]),
                "reported_bearing_deg": info["prey_bearing_deg"],
                "peak_cell": [int(peak_row), int(peak_col)],
                "peak_px": [float(peak_px_row), float(peak_px_col)],
                "peak_value": float(motion.max()),
                "motion_median": float(np.median(motion)),
                "food_cell": None if px is None else [food_cell_row, food_cell_col],
                "motion_at_food_cell": (None if not on_image else
                                        float(motion[food_cell_row, food_cell_col])),
                "moving_speed": float(info["moving_speed"]),
            }
            if on_image:
                row["peak_err_px"] = float(math.hypot(peak_px_col - px[0],
                                                      peak_px_row - px[1]))
            rows.append(row)
            prev_food = np.asarray(env.food_xy, dtype=float).copy()
            if term or trunc:
                break
    finally:
        env.close()
    return rows, cam_fovy


def report(rows, cam_fovy, label):
    n = len(rows)
    in_fov = [r for r in rows if r["in_fov"]]
    on_image = [r for r in rows if r["on_image"]]
    moved = [r for r in rows if r["food_moved_m"] > 1e-9]
    print(f"\n===== {label} =====")
    print(f"model head_cam fovy: {cam_fovy:g} deg    frames: {n}")
    print(f"prey inside the camera's real field of view: "
          f"{len(in_fov)}/{n} = {100.0*len(in_fov)/n:.1f} %")
    print(f"prey projects onto the 64x64 image:          "
          f"{len(on_image)}/{n} = {100.0*len(on_image)/n:.1f} %")
    print(f"frames on which the prey MOVED at all:       {len(moved)}/{n}")
    az = np.abs([r["az_cam_deg"] for r in rows])
    el = np.array([r["el_cam_deg"] for r in rows])
    d = np.array([r["dist_m"] for r in rows])
    ar = np.array([r["food_angular_radius_deg"] for r in rows])
    sp = np.array([r["moving_speed"] for r in rows])
    print(f"|azimuth|  median {np.median(az):6.1f}  min {az.min():6.1f}  max {az.max():6.1f} deg")
    print(f"elevation  median {np.median(el):6.1f}  min {el.min():6.1f}  max {el.max():6.1f} deg")
    print(f"nose-prey  median {np.median(d):6.3f}  min {d.min():6.3f}  max {d.max():6.3f} m")
    print(f"prey angular radius median {np.median(ar):.2f} deg")
    print(f"gecko speed median {np.median(sp):.4f} m/s")
    if on_image:
        err = np.array([r["peak_err_px"] for r in on_image])
        print(f"on the {len(on_image)} frames the prey IS on the image, the "
              f"tectum's peak lands {np.median(err):.1f} px away (median), "
              f"min {err.min():.1f} px")
        hits = int(np.sum(err <= 4.0))
        print(f"   peak within 4 px (one cell) of the prey: {hits}/{len(on_image)}")
        mv = np.array([r["motion_at_food_cell"] for r in on_image])
        pv = np.array([r["peak_value"] for r in on_image])
        print(f"   motion at the prey's own cell / peak motion: "
              f"median {np.median(mv/np.maximum(pv,1e-12)):.3f}")
    rowsn = np.array([r["peak_cell"][0] for r in rows])
    colsn = np.array([r["peak_cell"][1] for r in rows])
    cells = max(rowsn.max(), colsn.max()) + 1
    print("where the peak lands, by row (0 = top of image):")
    print("   " + "  ".join(f"{c}:{int(np.sum(rowsn==c))}"
                            for c in range(cells) if np.sum(rowsn == c)))
    print("where the peak lands, by column (0 = left):")
    print("   " + "  ".join(f"{c}:{int(np.sum(colsn==c))}"
                            for c in range(cells) if np.sum(colsn == c)))
    bottom_half = float(np.mean(rowsn >= cells // 2))
    print(f"fraction of peaks in the LOWER half of the image (the floor): "
          f"{bottom_half:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", action="store_true")
    ap.add_argument("--eye-fovy", type=float, default=70.0)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    rows, cam_fovy = run(steps=args.steps, world=args.world,
                         eye_fovy=args.eye_fovy)
    report(rows, cam_fovy,
           "generated prey world" if args.world else "field test's own config")
    out = REPO / ("artifacts/evidence/session8/tectum_diagnosis"
                  + ("_world" if args.world else "") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
