"""Film the accepted walker: the hand-written base, zero residual, no policy.

WHAT I HAD BEEN FILMING INSTEAD, ALL SESSION. Every clip I rendered loaded
`models/v4_5b_speed_polish_1m` and ran it as a residual on top of the CPG. That
is the TRAINED policy, and this project's own ledger rejected it:

    #11  "A learned residual can fix the forefoot" -- REFUTED.
         2 x 3.01 M steps; best trained 3/6 against base 4/6.
    #71  "The accepted walker IS the open-loop base: 19 strides,
         1.1892 Hz measured against 1.1888 commanded, duty 0.733."

So the accepted 4/6 walker is the hand-written controller with the policy
switched OFF. The evidence directories say the same thing in their own protocol
field -- `lab_zero_residual/report.json` records controller "zero residual with
contact reflex". I read that line more than once this session and still kept
loading the checkpoint.

WHAT IT COSTS TO GET THIS WRONG, measured. Front-foot duty factor, fraction of
the step each front foot carries load, against a published 0.70 target:

    trained residual, lab       FL 0.203   FR 0.591     asymmetric ~3x
    trained residual, legacy    FL 0.483   FR 0.500
    ZERO RESIDUAL, lab          FL 0.432   FR 0.454     balanced
    ZERO RESIDUAL, legacy       FL 0.463   FR 0.491     balanced

The trained policy is what limps. One front foot carries load for a fifth of the
step while the other carries it for well over half, and that asymmetry is the
push-then-drag rhythm the user kept pointing at and I kept explaining away --
first as a camera artefact, then as a frame-rate artefact, then as an
under-actuated forelimb they would have to live with.

The forelimb IS under-actuated (#18, one free joint against the hindlimb's two)
and that is a real body limit. But it is not what made the animal limp in my
clips. The policy was.

Usage:  python tools/accepted_walker_clip.py [--profile lab] [--seconds 20]
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

#: One frame per control step, and a control step is 0.02 s. 25 fps here is
#: half speed -- ledger #176.
FPS = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="lab", choices=("lab", "legacy"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/video/accepted_walker_clip.mp4")
    args = ap.parse_args()

    import imageio.v2 as imageio
    import mujoco

    from envs.gecko_walk_env import GeckoWalkEnv

    env = GeckoWalkEnv(xml_path="morphology/gecko_world_v1.xml",
                       gait_profile=args.profile, control_mode="cpg_residual",
                       residual_scale=0.25, privileged_target=True)

    renderer = mujoco.Renderer(env.model, 480, 640)
    options = mujoco.MjvOption()
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance, camera.elevation = 0.42, -10.0

    frames, azimuth, smooth, path = [], None, None, 0.0
    duty = {"fl_duty": [], "fr_duty": []}
    env.reset(seed=args.seed)
    previous = env.data.xpos[env._trunk][:2].copy()
    try:
        for i in range(int(args.seconds / env.dt)):
            forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0][:2]
            env.target = env.data.xpos[env._trunk][:2] + forward * 5.0
            try:
                _, env._prev_dist, _ = env._target_egocentric()
            except Exception:
                pass
            # ZERO RESIDUAL. No policy. This is the controller the gates
            # accepted, and switching it on is what made the animal limp.
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))

            here = env.data.xpos[env._trunk][:2].copy()
            path += float(np.linalg.norm(here - previous))
            previous = here
            for k in duty:
                duty[k].append(float(info.get(k, np.nan)))

            if azimuth is None:
                f3 = env.data.xmat[env._trunk].reshape(3, 3)[:, 0]
                azimuth = math.degrees(math.atan2(f3[1], f3[0])) + 90.0
            camera.azimuth = azimuth
            # Follow a smoothed path, not the trunk: the trunk bobs once per
            # stride and a camera locked to it shakes the whole world (#181).
            target = np.array([here[0], here[1], 0.02])
            smooth = target.copy() if smooth is None else smooth + (target - smooth) * 0.10
            camera.lookat[:] = smooth

            renderer.update_scene(env.data, camera=camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            frame = renderer.render()
            fl = float(np.nanmean(duty["fl_duty"])) if duty["fl_duty"] else float("nan")
            fr = float(np.nanmean(duty["fr_duty"])) if duty["fr_duty"] else float("nan")
            draw_text(frame, "ACCEPTED WALKER   hand-written base, NO policy", 10, 10, scale=2)
            draw_text(frame, f"front feet {fl:.3f} / {fr:.3f}   balanced", 10, 32, scale=2)
            frames.append(frame)

            if terminated or truncated:
                env.reset(seed=args.seed + i)
                previous = env.data.xpos[env._trunk][:2].copy()
    finally:
        renderer.close()
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=FPS, quality=8, macro_block_size=1)

    fl = float(np.nanmean(duty["fl_duty"]))
    fr = float(np.nanmean(duty["fr_duty"]))
    seconds = len(frames) * 0.02
    print(f"profile        : {args.profile}, ZERO residual (no policy)")
    print(f"front duty     : FL {fl:.3f}   FR {fr:.3f}   (target 0.70, "
          f"asymmetry {abs(fl - fr):.3f})")
    print(f"path           : {path:.3f} m in {seconds:.1f} s -> {path / seconds:.4f} m/s")
    print(f"written        : {out}  ({len(frames)} frames at {FPS} fps = real time)")

    evidence = REPO / "artifacts/evidence/session9/accepted_walker_clip.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "tools/accepted_walker_clip.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "controller": "zero residual with contact reflex -- the hand-written base",
        "gait_profile": args.profile,
        "front_duty_left": round(fl, 4),
        "front_duty_right": round(fr, 4),
        "front_duty_asymmetry": round(abs(fl - fr), 4),
        "path_m": round(path, 4),
        "speed_m_s": round(path / seconds, 4),
        "why_no_policy": "FAILURE_MAP #11 and #71: the trained residual scored 3/6 "
                         "against the base's 4/6, and the accepted walker IS the "
                         "open-loop base. Every clip earlier this session loaded "
                         "the trained checkpoint by mistake, and its front-foot "
                         "asymmetry (0.203 vs 0.591 under lab) is the limp the "
                         "user kept reporting.",
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
