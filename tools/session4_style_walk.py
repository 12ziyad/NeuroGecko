"""Reproduce the Session 4 walking clip: accepted walker, textured world, 50 fps.

WHY THIS EXISTS. The Session 4 renders (`renders/session4/*.mp4`) were made by a
script that was never committed -- only `utils/render_trace_video.py` is in the
history, and it cannot produce them: it replays a recorded trace against the
archived BODY file, so it draws the bare default floor and no prey, and it
disables shadows. Put side by side with `prey_escape_demo.mp4` the differences
are obvious and none of them are the animal:

    Session 4                       trace replay
    fine floor texture              coarse default checkerboard
    prey visible                    no prey
    shadow under the animal         shadows disabled
    gecko small in frame            gecko large in frame

So this reproduces the SETUP rather than the file: the accepted walker driving
the real body inside `morphology/gecko_world_v1.xml` -- the no-cheat world, with
its 2 cm floor texture and its prey -- rendered at 50 fps in real time with
shadows on and the camera framed as Session 4 framed it.

WHAT MAKES IT THE ACCEPTED WALKER. `gait_profile="lab"`. That is the profile the
4/6 gate evidence was measured under (artifacts/evidence/lab_frozen/report.json)
and it is NOT what GeckoBrainEnv runs -- the brain env never passes the
parameter, so GeckoWalkEnv's "legacy" default wins. Same body and the same
frozen checkpoint (model sha256 77b7a99d...), different CPG scaffolding under
it. That difference is the subject of its own ledger entry; this tool simply
names the profile instead of inheriting one.

Usage:  python tools/session4_style_walk.py [--seconds 30]
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

#: One frame per control step, and a control step is 0.02 s. Writing these at
#: 25 fps is what made every Session 9 clip play at half speed (ledger #176).
FPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distance", type=float, default=0.50)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--out", default="artifacts/video/session4_style_walk.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from envs.gecko_walk_env import GeckoWalkEnv

    env = GeckoWalkEnv(xml_path="morphology/gecko_world_v1.xml",
                       gait_profile="lab", privileged_target=True)
    norm = VecNormalize.load("models/v4_5b_speed_polish_1m/vecnormalize.pkl",
                             DummyVecEnv([lambda: env]))
    norm.training = False
    policy = PPO.load("models/v4_5b_speed_polish_1m/final.zip", device="cpu")

    renderer = mujoco.Renderer(env.model, 480, 640)
    options = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = args.distance
    camera.elevation = args.elevation

    def push_target_away(e):
        """Keep the goal far ahead so the walk is not cut short by arriving.

        The episode ends when the animal reaches its target, and with a nearby
        goal that happened repeatedly inside a 30 s clip -- each reset teleports
        the trunk, which both jumps the picture and inflates any distance
        measured by summing per-step displacement. Session 4's clip is described
        as the gecko walking "in a straight line for 60 s", which is what a goal
        it never reaches looks like.
        """
        forward = e.data.xmat[e._trunk].reshape(3, 3)[:, 0][:2]
        e.target = e.data.xpos[e._trunk][:2] + forward * 5.0
        _, e._prev_dist, _ = e._target_egocentric()

    frames, path, azimuth, resets = [], 0.0, None, 0
    obs, _ = env.reset(seed=args.seed)
    push_target_away(env)
    previous = env.data.xpos[env._trunk][:2].copy()
    try:
        for i in range(int(args.seconds / env.dt)):
            action, _ = policy.predict(
                norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :]),
                deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            push_target_away(env)

            here = env.data.xpos[env._trunk][:2].copy()
            path += float(np.linalg.norm(here - previous))
            previous = here

            if azimuth is None:
                # Fixed once, from the heading, so the camera never rotates --
                # a camera that swings with the animal's yaw is what read as
                # shake in the earlier clips (ledger, session 9d).
                forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0]
                azimuth = math.degrees(math.atan2(forward[1], forward[0])) + 138.0
            camera.azimuth = azimuth
            camera.lookat[:] = env.data.xpos[env._trunk]

            renderer.update_scene(env.data, camera=camera, scene_option=options)
            # Shadows ON. The trace replay disables them; Session 4 has them,
            # and the contact shadow is most of what makes a foot look planted.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            frames.append(renderer.render())

            if terminated or truncated:
                resets += 1
                obs, _ = env.reset(seed=args.seed + i)
                push_target_away(env)
                previous = env.data.xpos[env._trunk][:2].copy()
    finally:
        renderer.close()
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=FPS, quality=8, macro_block_size=1)

    seconds = len(frames) * 0.02
    print(f"frames        : {len(frames)}  ({seconds:.1f} s at {FPS} fps = real time)")
    print(f"path walked   : {path:.3f} m  ->  {path / seconds:.4f} m/s along its track")
    print(f"episode resets: {resets}   (each one jumps the picture)")
    print(f"gait profile  : lab  (the profile the 4/6 gates were measured under)")
    print(f"world         : morphology/gecko_world_v1.xml  (textured floor, prey)")
    print(f"written       : {out}")

    evidence = REPO / "artifacts/evidence/session9/session4_style_walk.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "tools/session4_style_walk.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "reproduces": "the Session 4 render setup, not the Session 4 file. The "
                      "script that made renders/session4/*.mp4 was never "
                      "committed.",
        "gait_profile": "lab",
        "checkpoint": "models/v4_5b_speed_polish_1m",
        "world": "morphology/gecko_world_v1.xml",
        "fps": FPS,
        "real_time": True,
        "shadows": True,
        "path_m": round(path, 4),
        "speed_along_track_m_s": round(path / seconds, 4),
        "episode_resets": resets,
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
