"""Rhythm alone versus rhythm plus the trained add-on, side by side.

Same body, same world, same seed, same camera, same frames. The ONLY difference
between the two halves is whether `models/v4_5b_speed_polish_1m` is loaded on top
of the hand-written CPG.

WHY IT IS WORTH FILMING. This project decided in Session 4 that the hand-written
base IS the walker -- #71 states it, and #11 records that 2 x 3.01 M training
steps produced a residual scoring 3/6 against the base's 4/6. That decision has
been in the ledger ever since and I still loaded the checkpoint into every clip
I rendered in Session 9. Putting them beside each other is the cheapest way to
make the decision visible rather than merely recorded.

The pairing matters as much as the policy: measured across all four
combinations, only `lab` + policy limps (front-foot duty 0.203 / 0.591, gap
0.388). The same policy on `legacy` is balanced. So this films the broken pair,
which is what was actually on screen all session.

Usage:  python tools/rhythm_vs_addon.py [--seconds 20] [--profile lab]
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

from tools._tinyfont import draw_text                        # noqa: E402

#: One frame per 0.02 s control step. 25 fps would be half speed (#176).
FPS = 50
CHECKPOINT = "models/v4_5b_speed_polish_1m"


def run(profile, use_policy, seconds, seed):
    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.gecko_walk_env import GeckoWalkEnv

    env = GeckoWalkEnv(xml_path="morphology/gecko_world_v1.xml",
                       gait_profile=profile, control_mode="cpg_residual",
                       residual_scale=0.25, privileged_target=True)
    policy = norm = None
    if use_policy:
        norm = VecNormalize.load(f"{CHECKPOINT}/vecnormalize.pkl",
                                 DummyVecEnv([lambda: env]))
        norm.training = False
        policy = PPO.load(f"{CHECKPOINT}/final.zip", device="cpu")

    renderer = mujoco.Renderer(env.model, 480, 640)
    options = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.elevation = 0.42, -10.0

    frames, azimuth, smooth, path = [], None, None, 0.0
    duty = {"fl_duty": [], "fr_duty": []}
    obs, _ = env.reset(seed=seed)
    previous = env.data.xpos[env._trunk][:2].copy()
    try:
        for i in range(int(seconds / env.dt)):
            if use_policy:
                a, _ = policy.predict(
                    norm.normalize_obs(np.asarray(obs, dtype=np.float32)[None, :]),
                    deterministic=True)
                action = a[0]
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0][:2]
            env.target = env.data.xpos[env._trunk][:2] + forward * 5.0
            try:
                _, env._prev_dist, _ = env._target_egocentric()
            except Exception:
                pass
            obs, _, terminated, truncated, info = env.step(action)

            here = env.data.xpos[env._trunk][:2].copy()
            path += float(np.linalg.norm(here - previous))
            previous = here
            for k in duty:
                duty[k].append(float(info.get(k, np.nan)))

            if azimuth is None:
                f3 = env.data.xmat[env._trunk].reshape(3, 3)[:, 0]
                azimuth = math.degrees(math.atan2(f3[1], f3[0])) + 90.0
            camera.azimuth = azimuth
            target = np.array([here[0], here[1], 0.02])
            smooth = target.copy() if smooth is None else smooth + (target - smooth) * .10
            camera.lookat[:] = smooth

            renderer.update_scene(env.data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            frames.append(renderer.render())

            if terminated or truncated:
                obs, _ = env.reset(seed=seed + i)
                previous = env.data.xpos[env._trunk][:2].copy()
    finally:
        renderer.close()
        env.close()

    fl = float(np.nanmean(duty["fl_duty"]))
    fr = float(np.nanmean(duty["fr_duty"]))
    return frames, {"front_left": round(fl, 3), "front_right": round(fr, 3),
                    "gap": round(abs(fl - fr), 3),
                    "path_m": round(path, 3),
                    "speed_m_s": round(path / (len(frames) * 0.02), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--profile", default="lab", choices=("lab", "legacy"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/video/rhythm_vs_addon.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio

    left, ls = run(args.profile, False, args.seconds, args.seed)
    right, rs = run(args.profile, True, args.seconds, args.seed)

    out_frames = []
    for a, b in zip(left, right):
        a, b = a.copy(), b.copy()
        draw_text(a, "RHYTHM ALONE", 10, 10, scale=2)
        draw_text(a, "accepted walker  4 of 6 gates", 10, 32, scale=2)
        draw_text(a, f"front feet {ls['front_left']:.3f} / {ls['front_right']:.3f}"
                     f"   even", 10, 54, scale=2)
        draw_text(b, "RHYTHM + TRAINED ADD-ON", 10, 10, scale=2)
        draw_text(b, "rejected in Session 4  3 of 6 gates", 10, 32, scale=2)
        draw_text(b, f"front feet {rs['front_left']:.3f} / {rs['front_right']:.3f}"
                     f"   uneven", 10, 54, scale=2)
        pair = np.concatenate([a, b], axis=1)
        draw_text(pair, "same body  same world  same seed  same camera",
                  10, 452, scale=2)
        out_frames.append(pair)

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), out_frames, fps=FPS, quality=8, macro_block_size=1)

    print(f"profile: {args.profile}\n")
    print(f"{'':22s}{'FL':>8s}{'FR':>8s}{'gap':>8s}{'path m':>9s}{'m/s':>9s}")
    print(f"{'rhythm alone':22s}{ls['front_left']:8.3f}{ls['front_right']:8.3f}"
          f"{ls['gap']:8.3f}{ls['path_m']:9.3f}{ls['speed_m_s']:9.4f}")
    print(f"{'rhythm + add-on':22s}{rs['front_left']:8.3f}{rs['front_right']:8.3f}"
          f"{rs['gap']:8.3f}{rs['path_m']:9.3f}{rs['speed_m_s']:9.4f}")
    print(f"\nwritten: {out}  ({len(out_frames)} frames at {FPS} fps = real time)")

    ev = REPO / "artifacts/evidence/session9/rhythm_vs_addon.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "tools/rhythm_vs_addon.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "gait_profile": args.profile,
        "only_difference": f"whether {CHECKPOINT} is loaded on top of the CPG",
        "rhythm_alone": ls,
        "rhythm_plus_addon": rs,
        "gates": {"rhythm_alone": "4 of 6 (accepted, ledger #71)",
                  "rhythm_plus_addon": "3 of 6 (rejected, ledger #11, "
                                       "after 2 x 3.01 M training steps)"},
        "caveat": "The add-on is only this bad when paired with `lab`. On `legacy` "
                  "its front duty is 0.483/0.500, balanced (#184). The gate result "
                  "-- 3/6 against 4/6 -- is the durable reason it was rejected; the "
                  "visible limp needs this specific pairing.",
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
