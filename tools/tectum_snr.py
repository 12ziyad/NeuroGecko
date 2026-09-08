"""The one ratio that decides it: prey motion energy vs self-motion energy.

Prey parked 0.15 m dead ahead (a ~60-pixel target, the most generous geometry
the arena allows). Every frame is rendered twice, with the prey and with it
deleted, so the prey's OWN contribution to the temporal-contrast map is
isolated exactly rather than estimated.

  signal     = |motion_with_prey - motion_without_prey| at the prey's cell
  background = motion_without_prey at that same cell   (self-motion flow)
  ceiling    = max(motion_without_prey) anywhere       (what the peak competes with)
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
AWAY = np.array([50.0, 50.0])


def main(steps=60, distance=0.15, freeze=False):
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
    r_on = Retina(fovy_deg=70.0, pixels=PIX, cells=CELLS)
    r_off = Retina(fovy_deg=70.0, pixels=PIX, cells=CELLS)

    sig, bg, ceil = [], [], []
    try:
        env.reset(seed=1)
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        for i in range(steps):
            if not freeze:
                env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            fwd = data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            xy = env._nose_xy() + fwd * distance
            env.prey.position = xy.copy()
            env.food_xy = xy.copy()
            env._write_prey()
            img_on = env._head_cam_image()
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            env.prey.position = AWAY.copy()
            env.food_xy = AWAY.copy()
            env._write_prey()
            img_off = env._head_cam_image()

            m_on = r_on.step(img_on)["motion"]
            m_off = r_off.step(img_off)["motion"]
            px = project(cam_pos, cam_mat,
                         np.array([xy[0], xy[1], env.prey.height_m]), 70.0, PIX)
            if px is None or not (0 <= px[0] < PIX and 0 <= px[1] < PIX):
                continue
            k = PIX / CELLS
            pc, pr = int(px[0] // k), int(px[1] // k)
            sig.append(abs(float(m_on[pr, pc]) - float(m_off[pr, pc])))
            bg.append(float(m_off[pr, pc]))
            ceil.append(float(m_off.max()))
    finally:
        env.close()

    s = np.array(sig); b = np.array(bg); c = np.array(ceil)
    tag = "BODY FROZEN" if freeze else "gecko walking"
    print(f"\n=== {tag}, prey static at {distance:.2f} m dead ahead, "
          f"{len(s)} frames ===")
    print(f"prey's own motion contribution at its cell : median {np.median(s):.5f}")
    print(f"self-motion flow at that same cell         : median {np.median(b):.5f}")
    print(f"strongest self-motion cell anywhere        : median {np.median(c):.5f}")
    print(f"local  signal/background ratio             : "
          f"{np.median(s)/max(np.median(b),1e-12):.3f}")
    print(f"GLOBAL signal/peak ratio (what argmax sees): "
          f"{np.median(s)/max(np.median(c),1e-12):.3f}")
    print(f"frames where the prey's contribution EXCEEDS the strongest "
          f"self-motion cell: {int(np.sum(s > c))}/{len(s)}")
    return {"tag": tag, "n": int(len(s)),
            "signal_median": float(np.median(s)),
            "background_median": float(np.median(b)),
            "ceiling_median": float(np.median(c)),
            "global_ratio": float(np.median(s) / max(np.median(c), 1e-12)),
            "frames_signal_beats_ceiling": int(np.sum(s > c))}


if __name__ == "__main__":
    out = [main(freeze=False, distance=0.15),
           main(freeze=False, distance=0.30),
           main(freeze=True, distance=0.15)]
    p = REPO / "artifacts/evidence/session8/tectum_snr.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten: {p}")
