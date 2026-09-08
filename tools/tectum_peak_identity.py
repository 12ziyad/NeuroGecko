"""What is the peak, and why does the whole-field self-motion test never fire?

For each frame of ordinary walking, this records:

  * `filling` -- the statistic the tectum uses to decide "this is the world
    going past, not an object". It rejects when filling > 0.25.
  * the elevation of the peak cell in the camera's own frame, so the peak can
    be compared to the HORIZON (elevation 0) and to the floor.
  * the head's angular velocity between frames, which is the thing generating
    the flow.
  * what the peak pixel is actually looking at: the depth of the first geom
    along that ray, and its name, by ray-casting through the model.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PIX = 64
CELLS = 16


def main():
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters
    from brain.tectum import Eye

    env = GeckoBrainEnv(homeostasis=True, eye=True,
                        walker_xml_path=REPO / "morphology" / "gecko_world_v1.xml",
                        prey_parameters=PreyParameters.from_registry())
    env.eye = Eye(fovy_deg=70.0, pixels=PIX, cells=CELLS)
    captured = {}
    original = env.eye.retina.step

    def rec(image):
        out = original(image)
        captured["motion"] = np.array(out["motion"], copy=True)
        return out
    env.eye.retina.step = rec

    model = env.walk_env.model
    data = env.walk_env.data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    focal_px = (PIX / 2.0) / math.tan(math.radians(70.0) / 2.0)

    fillings, rejected, elevations, azimuths, hits = [], 0, [], [], []
    head_rate = []
    prev_mat = None
    try:
        env.reset(seed=1)
        for i in range(150):
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
            motion = captured["motion"]
            raw_peak = float(motion.max())
            filling = float(np.mean(motion > 0.5 * raw_peak)) if raw_peak > 1e-4 else 0.0
            fillings.append(filling)
            if filling > 0.25:
                rejected += 1
            r, c = np.unravel_index(int(np.argmax(motion)), motion.shape)
            # centre of that cell in pixels -> angles
            px_c = (c + 0.5) * (PIX / CELLS) - PIX / 2.0
            px_r = (r + 0.5) * (PIX / CELLS) - PIX / 2.0
            az = math.degrees(math.atan(px_c / focal_px))
            el = math.degrees(math.atan(-px_r / focal_px))
            azimuths.append(az)
            elevations.append(el)

            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
            # ray through the peak pixel, in world coordinates
            dir_cam = np.array([px_c / focal_px, -px_r / focal_px, -1.0])
            dir_cam /= np.linalg.norm(dir_cam)
            dir_world = cam_mat @ dir_cam
            geomid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(model, data, cam_pos, dir_world,
                                 None, 1, -1, geomid)
            gid = int(geomid[0])
            name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    if gid >= 0 else None)
            if name is None and gid >= 0:
                bid = int(model.geom_bodyid[gid])
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            hits.append((name if name is not None else "sky(no hit)", float(dist)))

            if prev_mat is not None:
                rel = prev_mat.T @ cam_mat
                ang = math.degrees(math.acos(
                    max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0))))
                head_rate.append(ang)
            prev_mat = cam_mat
            if term or trunc:
                break
    finally:
        env.close()

    f = np.array(fillings)
    print(f"frames {len(f)}")
    print(f"'fills the field' statistic: median {np.median(f):.3f}, "
          f"max {f.max():.3f}; the rejection threshold is 0.25")
    print(f"frames rejected as self-motion: {rejected}/{len(f)}"
          f" = {100.0*rejected/len(f):.1f} %")
    el = np.array(elevations)
    az = np.array(azimuths)
    print(f"peak elevation: median {np.median(el):+.1f} deg "
          f"(0 = the horizon), min {el.min():+.1f}, max {el.max():+.1f}")
    print(f"peak azimuth:   median {np.median(az):+.1f} deg, "
          f"|az| > 25 deg (outer third of the field) on "
          f"{100.0*np.mean(np.abs(az) > 25):.1f} % of frames")
    hr = np.array(head_rate)
    print(f"head rotation between frames: median {np.median(hr):.2f} deg "
          f"(at ~50 Hz that is {np.median(hr)*50:.0f} deg/s of gaze motion)")
    counts = collections.Counter(n for n, _ in hits)
    print("what the peak pixel is actually looking at:")
    for name, k in counts.most_common(8):
        print(f"   {name:<24} {k:>4} frames  = {100.0*k/len(hits):5.1f} %")
    d = np.array([x for _, x in hits if x > 0])
    if d.size:
        print(f"depth along the peak ray: median {np.median(d):.3f} m")

    out = REPO / "artifacts/evidence/session8/tectum_peak_identity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "filling_median": float(np.median(f)),
        "filling_max": float(f.max()),
        "self_motion_rejection_rate": rejected / len(f),
        "peak_elevation_median_deg": float(np.median(el)),
        "peak_azimuth_median_deg": float(np.median(az)),
        "peak_outer_third_fraction": float(np.mean(np.abs(az) > 25)),
        "head_rotation_per_frame_deg_median": float(np.median(hr)),
        "peak_ray_targets": dict(counts),
    }, indent=1), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
