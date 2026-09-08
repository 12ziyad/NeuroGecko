"""Is the SELECTION RULE the cause, rather than the front end?

Tectum.step subtracts a SCALAR -- np.median(motion) -- and then takes a global
argmax. The docstring says the detector is "centre-surround in SPACE as well",
but a scalar median is not a surround: it removes the average level of the flow
field and leaves its spatial STRUCTURE intact, and the structure is the whole
problem, because self-motion flow is strong at the horizon and at the periphery
and weak in the middle.

This changes NOTHING in brain/tectum.py. It reads the same motion maps and
scores four selection rules side by side, so the difference between them is a
measurement rather than a proposal:

  raw            argmax(motion)                         -- no rejection at all
  scalar         argmax(motion - median(motion))        -- AS SHIPPED
  surround       argmax(motion - boxblur(motion, w))    -- a real centre-surround
  z              argmax((motion - blur)/ (blurred spread + eps))

Ground truth is the prey's own cell, computed by projecting its world position
through the camera. Prey is parked 0.15 m dead ahead: the generous case.
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
PIX = 64


def box(plane, w):
    from brain.retina import _blur
    return _blur(plane, w)


def rules(motion, w=3):
    out = {}
    out["raw"] = motion
    out["scalar"] = np.clip(motion - float(np.median(motion)), 0.0, None)
    sur = box(motion, w)
    out["surround"] = np.clip(motion - sur, 0.0, None)
    spread = box(np.abs(motion - sur), w)
    out["z"] = np.clip((motion - sur) / (spread + 1e-3), 0.0, None)
    return out


def main(steps=90, distance=0.15, seed=1):
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters
    from brain.retina import Retina

    env = GeckoBrainEnv(homeostasis=True, eye=False,
                        walker_xml_path=REPO / "morphology" / "gecko_world_v1.xml",
                        prey_parameters=PreyParameters.from_registry())
    model = env.walk_env.model
    data = env.walk_env.data
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    retina = Retina(fovy_deg=70.0, pixels=PIX, cells=CELLS)
    names = ("raw", "scalar", "surround", "z")
    hits = {n: 0 for n in names}
    err = {n: [] for n in names}
    truth = []
    reported = {n: [] for n in names}
    n_frames = 0
    try:
        env.reset(seed=seed)
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        for i in range(steps):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            fwd = data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            xy = env._nose_xy() + fwd * distance
            env.prey.position = xy.copy()
            env.food_xy = xy.copy()
            env._write_prey()
            img = env._head_cam_image()
            motion = retina.step(img)["motion"]
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            px = project(cam_pos, cam_mat,
                         np.array([xy[0], xy[1], env.prey.height_m]), 70.0, PIX)
            if px is None or not (0 <= px[0] < PIX and 0 <= px[1] < PIX):
                continue
            k = PIX / CELLS
            pc, pr = int(px[0] // k), int(px[1] // k)
            focal = (PIX / 2.0) / math.tan(math.radians(70.0) / 2.0)
            true_az = math.degrees(math.atan((px[0] - PIX / 2.0 + 0.5) / focal))
            truth.append(true_az)
            n_frames += 1
            maps = rules(motion)
            for name in names:
                m = maps[name]
                r, c = np.unravel_index(int(np.argmax(m)), m.shape)
                hits[name] += int((r, c) == (pr, pc))
                err[name].append(math.hypot(c - pc, r - pr))
                reported[name].append(retina.azimuth_of(int(c)))
    finally:
        env.close()

    print(f"\nprey 0.15 m dead ahead, {n_frames} frames, gecko walking")
    print(f"{'rule':>10}{'peak on prey':>15}{'cell error':>13}{'corr':>9}")
    result = {}
    t = np.array(truth)
    for name in ("raw", "scalar", "surround", "z"):
        e = np.array(err[name])
        rep = np.array(reported[name])
        corr = float(np.corrcoef(rep, t)[0, 1]) if n_frames > 6 else float("nan")
        label = name + (" *shipped*" if name == "scalar" else "")
        print(f"{label:>10}{hits[name]:>8}/{n_frames:<6}"
              f"{np.median(e):>13.1f}{corr:>9.3f}")
        result[name] = {"hits": hits[name], "frames": n_frames,
                        "median_cell_error": float(np.median(e)),
                        "corr_bearing_vs_truth": corr}
    p = REPO / "artifacts/evidence/session8/tectum_selection_probe.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"written: {p}")


if __name__ == "__main__":
    main()
