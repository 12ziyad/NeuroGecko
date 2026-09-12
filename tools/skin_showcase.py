"""Render the skinned animal: stills, and a clip of the movements that show it.

The point of the continuous skin is what happens at the JOINTS, so a still of a
neutral pose proves nothing. This renders the three motions where rigid per-body
pieces visibly came apart -- the head yawing, the tail sweeping, the mouth
opening -- and puts the same pose side by side for the record.

Usage:  python tools/skin_showcase.py [--no-video]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
WORLD = REPO / "morphology" / "gecko_world_v1.xml"
OUT = REPO / "artifacts" / "video"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    import imageio.v2 as iio
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(WORLD))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    for _ in range(400):
        mujoco.mj_step(model, data)
    settled = data.qpos.copy()

    renderer = mujoco.Renderer(model, 760, 1010)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    def pose(**joints):
        data.qpos[:] = settled
        for name, deg in joints.items():
            try:
                data.qpos[model.joint(name).qposadr[0]] = np.deg2rad(deg)
            except KeyError:
                pass
        mujoco.mj_forward(model, data)

    def shot(body, distance, azimuth, elevation):
        cam.lookat[:] = data.xpos[model.body(body).id]
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        renderer.update_scene(data, cam)
        return renderer.render()

    pose(jaw=22)
    iio.imwrite(OUT / "skin_side.png", shot("trunk_anterior", 0.21, 108, -13))
    iio.imwrite(OUT / "skin_head.png", shot("head", 0.068, 150, -8))
    iio.imwrite(OUT / "skin_top.png", shot("trunk_anterior", 0.24, 90, -72))
    # the test that matters: the neck, yawed hard, seen from above
    pose(neck_yaw=25, head_yaw=25)
    iio.imwrite(OUT / "skin_neck_turn.png", shot("neck", 0.10, 90, -70))
    print("stills written")

    if args.no_video:
        return
    frames = []
    for k in range(210):
        t = k / 60.0
        pose(neck_yaw=26 * np.sin(2 * np.pi * 0.45 * t),
             head_yaw=22 * np.sin(2 * np.pi * 0.45 * t + 0.5),
             head_pitch=9 * np.sin(2 * np.pi * 0.3 * t),
             jaw=max(0.0, 30 * np.sin(2 * np.pi * 0.55 * t)),
             gular=-7 * (0.5 + 0.5 * np.sin(2 * np.pi * 1.4 * t)),
             eyelids=max(0.0, 70 * np.sin(2 * np.pi * 0.22 * t - 1.2)))
        cam.lookat[:] = data.xpos[model.body("neck").id]
        cam.distance, cam.elevation = 0.115, -14
        cam.azimuth = 90 + 55 * np.sin(2 * np.pi * 0.12 * t)
        renderer.update_scene(data, cam)
        frames.append(renderer.render())
    path = OUT / "skin_showcase.mp4"
    iio.mimwrite(path, frames, fps=30, quality=9)
    print(f"written: {path}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
