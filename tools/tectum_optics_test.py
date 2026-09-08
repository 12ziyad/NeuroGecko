"""Is the missing optical low-pass load-bearing?

GeckoBrainEnv accepts `eye_render_pixels`, stores it in `_eye_render_px`, and
never reads it again: `_head_cam_image` always renders at camera_width, which
is 64. `Retina._receptors` only applies the blur when the incoming frame is at
`render_pixels` AND `render_pixels != pixels`, so on every env frame the
optics are skipped and the floor texture is sampled straight off a 64 px
render. This renders the SAME frames at 256 px, applies the blur the retina
was written to apply, and re-runs the identical tectum on both.

Prey is parked 0.15 m dead ahead where it occupies ~60 rendered pixels, so
this is the most generous geometry available, not a hard case.
"""
from __future__ import annotations

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

CELLS = 16


def main(steps=60, render_hi=256):
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters
    from brain.tectum import Eye

    env = GeckoBrainEnv(homeostasis=True, eye=False,
                        walker_xml_path=REPO / "morphology" / "gecko_world_v1.xml",
                        prey_parameters=PreyParameters.from_registry())
    model = env.walk_env.model
    data = env.walk_env.data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    hi = mujoco.Renderer(model, render_hi, render_hi)

    eye_lo = Eye(fovy_deg=70.0, pixels=64, cells=CELLS)
    eye_hi = Eye(fovy_deg=70.0, pixels=64, cells=CELLS, render_pixels=render_hi)
    assert eye_hi.retina.render_pixels == render_hi
    print(f"optical blur width at {render_hi} px render: "
          f"{eye_hi.retina.optical_limit_px():.2f} render pixels")

    lo_hits = hi_hits = 0
    lo_floor = hi_floor = 0
    rows = []
    try:
        env.reset(seed=1)
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        for i in range(steps):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            fwd = data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            xy = env._nose_xy() + fwd * 0.15
            env.prey.position = xy.copy()
            env.food_xy = xy.copy()
            env._write_prey()

            img_lo = env._head_cam_image()
            hi.update_scene(data, camera="head_cam")
            img_hi = np.asarray(hi.render())[:, :, :3]

            s_lo, b_lo = eye_lo.tectum.step(eye_lo.retina.step(img_lo))
            s_hi, b_hi = eye_hi.tectum.step(eye_hi.retina.step(img_hi))

            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            px = project(cam_pos, cam_mat,
                         np.array([xy[0], xy[1], env.prey.height_m]), 70.0, 64)
            if px is None or not (0 <= px[0] < 64 and 0 <= px[1] < 64):
                continue
            k = 64 / CELLS
            pc, pr = int(px[0] // k), int(px[1] // k)
            for eye, tag in ((eye_lo, "lo"), (eye_hi, "hi")):
                cell = eye.tectum.last.get("elevation_cell")
            m_lo = eye_lo.tectum.last
            m_hi = eye_hi.tectum.last
            # recompute the peak column from the reported bearing
            lo_ok = _peak_cell(eye_lo) == (pr, pc)
            hi_ok = _peak_cell(eye_hi) == (pr, pc)
            lo_hits += lo_ok
            hi_hits += hi_ok
            rows.append({"i": i, "prey_cell": [pr, pc],
                         "lo_peak": list(_peak_cell(eye_lo)),
                         "hi_peak": list(_peak_cell(eye_hi)),
                         "lo_bearing": b_lo, "hi_bearing": b_hi,
                         "true_az_deg": math.degrees(math.atan(
                             (px[0] - 32 + 0.5) /
                             ((64 / 2) / math.tan(math.radians(70) / 2))))})
    finally:
        hi.close()
        env.close()

    n = len(rows)
    print(f"\nframes with the prey on the image: {n}")
    print(f"peak lands on the prey's cell, 64 px render, NO optics : "
          f"{lo_hits}/{n} = {100.0*lo_hits/max(n,1):.1f} %")
    print(f"peak lands on the prey's cell, {render_hi} px render + optics: "
          f"{hi_hits}/{n} = {100.0*hi_hits/max(n,1):.1f} %")
    tru = np.array([r["true_az_deg"] for r in rows])
    lo = np.array([r["lo_bearing"] for r in rows])
    hh = np.array([r["hi_bearing"] for r in rows])
    if n > 6:
        print(f"corr(bearing, true azimuth)  no optics {np.corrcoef(lo, tru)[0,1]:+.3f}"
              f"   with optics {np.corrcoef(hh, tru)[0,1]:+.3f}")
    out = REPO / "artifacts/evidence/session8/tectum_optics_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"written: {out}")


def _peak_cell(eye):
    last = eye.tectum.last
    row = last.get("elevation_cell")
    bearing = last.get("bearing_deg", 0.0)
    if row is None:
        return (-1, -1)
    focal_px = (eye.retina.pixels / 2.0) / math.tan(
        math.radians(eye.retina.fovy_deg) / 2.0)
    centre_px = math.tan(math.radians(bearing)) * focal_px
    col = int(round((centre_px + eye.retina.pixels / 2.0)
                    / (eye.retina.pixels / eye.retina.cells) - 0.5))
    return (int(row), int(np.clip(col, 0, eye.retina.cells - 1)))


if __name__ == "__main__":
    main()
