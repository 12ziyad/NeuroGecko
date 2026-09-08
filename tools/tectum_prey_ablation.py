"""Does the prey contribute ANYTHING to the map the tectum reads?

Three experiments, all in the generated prey world (fovy 70, real prey geom).

A. PREY PRESENT vs PREY DELETED, on the same walking trajectory, prey placed
   directly ahead where it is guaranteed to be in view. Two retinas are run
   side by side so each keeps its own previous frame. If the tectum's peak and
   bearing are identical with and without the prey, the tectum is provably not
   seeing it, and no threshold change can fix that.

B. THE SAME, with the prey driven laterally at the published cricket escape
   speed, 0.118 m/s. This is the only condition in which a motion detector
   could see it.

C. BODY FROZEN, prey moving. No self-motion at all. This is the control that
   says whether the detector works when the only thing moving is the prey.
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
from tectum_diagnosis import camera_frame_bearing, project  # noqa: E402

PIX = 64
CELLS = 16
AWAY = np.array([50.0, 50.0])  # far outside the arena; effectively deleted


def make_env():
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters
    return GeckoBrainEnv(homeostasis=True, eye=False,
                         walker_xml_path=REPO / "morphology" / "gecko_world_v1.xml",
                         prey_parameters=PreyParameters.from_registry())


def eye_pair():
    from brain.tectum import Eye
    return (Eye(fovy_deg=70.0, pixels=PIX, cells=CELLS),
            Eye(fovy_deg=70.0, pixels=PIX, cells=CELLS))


def place_prey(env, xy):
    env.prey.position = np.asarray(xy, dtype=float).copy()
    env.food_xy = env.prey.position.copy()
    env._write_prey()


def ahead_of_nose(env, distance):
    import mujoco  # noqa: F401
    data = env.walk_env.data
    fwd = data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    return env._nose_xy() + fwd * distance


def experiment(label, steps=60, prey_speed=0.0, freeze_body=False,
               distance=0.15, seed=1, flee_radially=False):
    import mujoco
    env = make_env()
    eye_on, eye_off = eye_pair()
    model = env.walk_env.model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_fovy = float(model.cam_fovy[cam_id])
    rows = []
    try:
        env.reset(seed=seed)
        # settle a few steps so the walker is in a normal gait cycle
        if not freeze_body:
            for _ in range(10):
                env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        prey_xy = ahead_of_nose(env, distance)
        # lateral escape direction, perpendicular to the animal's facing
        fwd = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
        lateral = np.array([-fwd[1], fwd[0]]) / (np.linalg.norm(fwd) + 1e-12)
        # walk_env.dt already folds in frame_skip; one brain step is
        # brain_steps_per_action of those.
        dt_guess = float(env.walk_env.dt) * env.brain_steps_per_action

        for i in range(steps):
            if not freeze_body:
                env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                dt = dt_guess
            else:
                dt = 0.02
            if freeze_body:
                # hold the prey at a fixed offset from the (unmoving) nose
                base = prey_xy
            else:
                base = ahead_of_nose(env, distance)
            if flee_radially:
                # what FleeingPrey ACTUALLY does: run directly away from the
                # predator. Along the line of sight, so it produces looming
                # rather than retinal displacement.
                moving_xy = base + (fwd / (np.linalg.norm(fwd) + 1e-12)) \
                    * prey_speed * dt * i
            else:
                moving_xy = base + lateral * prey_speed * dt * i

            # --- frame WITH the prey
            place_prey(env, moving_xy)
            if freeze_body:
                mujoco.mj_forward(model, env.walk_env.data)
            img_on = env._head_cam_image()
            data = env.walk_env.data
            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            px = project(cam_pos, cam_mat,
                         np.array([moving_xy[0], moving_xy[1], env.prey.height_m]),
                         cam_fovy)
            az, el, fwd_c = camera_frame_bearing(
                cam_pos, cam_mat,
                np.array([moving_xy[0], moving_xy[1], env.prey.height_m]))

            # --- the SAME frame with the prey deleted
            place_prey(env, AWAY)
            if freeze_body:
                mujoco.mj_forward(model, env.walk_env.data)
            img_off = env._head_cam_image()
            place_prey(env, moving_xy)

            s_on, b_on = None, None
            out_on = eye_on.retina.step(img_on)
            s_on, b_on = eye_on.tectum.step(out_on)
            out_off = eye_off.retina.step(img_off)
            s_off, b_off = eye_off.tectum.step(out_off)

            m_on = out_on["motion"]
            m_off = out_off["motion"]
            pk_on = np.unravel_index(int(np.argmax(m_on)), m_on.shape)
            pk_off = np.unravel_index(int(np.argmax(m_off)), m_off.shape)
            # how many rendered pixels actually differ between the two frames
            diff_px = int(np.sum(np.any(img_on.astype(int) != img_off.astype(int),
                                        axis=2)))
            rows.append({
                "i": i,
                "prey_px": None if px is None else [float(px[0]), float(px[1])],
                "prey_az_deg": az, "prey_el_deg": el,
                "prey_pixels_in_frame": diff_px,
                "salience_with_prey": float(s_on),
                "salience_without_prey": float(s_off),
                "bearing_with_prey": float(b_on),
                "bearing_without_prey": float(b_off),
                "peak_with_prey": [int(pk_on[0]), int(pk_on[1])],
                "peak_without_prey": [int(pk_off[0]), int(pk_off[1])],
                "same_peak": bool(pk_on == pk_off),
                "motion_max": float(m_on.max()),
            })
    finally:
        env.close()

    n = len(rows)
    same = sum(r["same_peak"] for r in rows)
    px_seen = np.array([r["prey_pixels_in_frame"] for r in rows])
    print(f"\n===== {label} =====")
    print(f"frames {n};  prey occupies {px_seen.mean():.1f} rendered pixels "
          f"on average (min {px_seen.min()}, max {px_seen.max()}) out of 4096")
    onframe = [r for r in rows if r["prey_pixels_in_frame"] > 0]
    print(f"prey actually rendered on {len(onframe)}/{n} frames")
    print(f"tectum peak IDENTICAL with and without the prey: {same}/{n}"
          f" = {100.0*same/max(n,1):.1f} %")
    if onframe:
        db = np.array([abs(r["bearing_with_prey"] - r["bearing_without_prey"])
                       for r in onframe])
        ds = np.array([abs(r["salience_with_prey"] - r["salience_without_prey"])
                       for r in onframe])
        print(f"on the frames the prey is rendered: "
              f"median |bearing change| {np.median(db):.3f} deg, "
              f"median |salience change| {np.median(ds):.4f}")
        # did the tectum's peak land on the prey?
        hits = 0
        for r in onframe:
            if r["prey_px"] is None:
                continue
            k = PIX / CELLS
            pc = int(np.clip(r["prey_px"][0] // k, 0, CELLS - 1))
            pr = int(np.clip(r["prey_px"][1] // k, 0, CELLS - 1))
            if [pr, pc] == r["peak_with_prey"]:
                hits += 1
        print(f"tectum peak lands ON the prey's own cell: {hits}/{len(onframe)}")
    return rows


def main():
    out = {}
    out["A_static_prey_walking_gecko"] = experiment(
        "A. prey static, 0.15 m dead ahead, gecko walking", prey_speed=0.0)
    out["B_fleeing_prey_walking_gecko"] = experiment(
        "B. prey fleeing laterally at 0.118 m/s, gecko walking", prey_speed=0.118)
    out["B2_radial_flee_walking_gecko"] = experiment(
        "B2. prey fleeing DIRECTLY AWAY at 0.118 m/s (what FleeingPrey does), "
        "gecko walking", prey_speed=0.118, flee_radially=True)
    out["C_moving_prey_frozen_body"] = experiment(
        "C. body frozen, prey moving at 0.118 m/s (control)",
        prey_speed=0.118, freeze_body=True)
    path = REPO / "artifacts/evidence/session8/tectum_prey_ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
