"""
watch_policy.py  --  roll out a trained policy and print metrics.

    python utils/watch_policy.py --run v3_cpg_sanity_200k --episodes 3
    python utils/watch_policy.py --run v3_cpg_sanity_200k --episodes 3 --render-video
"""
from __future__ import annotations
import argparse, sys, os, platform, contextlib, math
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# Only force EGL on headless Linux; on Windows/desktop let MuJoCo pick its default.
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from envs.gecko_walk_env import GeckoWalkEnv, _GAIT_FEET

_FOOT_INDEX = {foot: i for i, foot in enumerate(_GAIT_FEET)}


def _ensure_renderer(env):
    import mujoco
    if env._renderer is None:
        env._renderer = mujoco.Renderer(env.model, 480, 640)
    return env._renderer


def _scene_obj(renderer):
    return getattr(renderer, "scene", getattr(renderer, "_scene", None))


def _target_marker_xyz(env):
    target = getattr(env, "target", None)
    if target is None:
        return None
    return np.array([float(target[0]), float(target[1]), 0.045], dtype=np.float64)


def _add_scene_sphere(renderer, pos, radius=0.025, rgba=(1.0, 0.1, 0.05, 1.0)):
    import mujoco
    scene = _scene_obj(renderer)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_scene_capsule(renderer, start, end, radius=0.006, rgba=(1.0, 0.7, 0.05, 0.9)):
    import mujoco
    scene = _scene_obj(renderer)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    try:
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            float(radius),
            np.asarray(start, dtype=np.float64),
            np.asarray(end, dtype=np.float64),
        )
        geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
        scene.ngeom += 1
    except Exception:
        return


def _add_target_marker(env, renderer):
    target_xyz = _target_marker_xyz(env)
    if target_xyz is None:
        return
    trunk = env.data.xpos[env._trunk].copy()
    trunk_ground = np.array([trunk[0], trunk[1], target_xyz[2]], dtype=np.float64)
    _add_scene_capsule(renderer, trunk_ground, target_xyz)
    _add_scene_sphere(renderer, target_xyz)


def _wide_camera_lookat(env, lookahead):
    trunk = env.data.xpos[env._trunk].copy()
    lookat = trunk.copy()
    target = getattr(env, "target", None)
    if target is not None:
        delta = np.array([target[0] - trunk[0], target[1] - trunk[1]], dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            lookat[:2] += delta / norm * float(lookahead)
            return lookat
    forward = env.data.xmat[env._trunk].reshape(3, 3)[:, 0]
    lookat[:2] += forward[:2] * float(lookahead)
    return lookat


def _render_frame(env, camera_mode, args):
    import mujoco
    renderer = _ensure_renderer(env)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    if camera_mode == "wide":
        distance = max(float(args.camera_distance), 1e-6)
        height = max(float(args.camera_height), 0.0)
        cam.lookat[:] = _wide_camera_lookat(env, args.camera_lookahead)
        cam.distance = distance
        cam.azimuth = 130
        cam.elevation = -math.degrees(math.asin(min(height / distance, 0.95)))
    else:
        cam.lookat[:] = env.data.xpos[env._trunk]
        cam.distance, cam.azimuth, cam.elevation = 0.34, 130, -18
    renderer.update_scene(env.data, camera=cam)
    if args.show_target:
        _add_target_marker(env, renderer)
    return renderer.render()


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def _window_participation(window, feet):
    if len(window) == 0:
        return False
    arr = np.asarray(window, dtype=bool)
    for foot in feet:
        series = arr[:, _FOOT_INDEX[foot]]
        if not (np.any(series) and np.any(~series)):
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="v3_cpg_sanity_200k")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--render-video", action="store_true")
    p.add_argument("--camera-mode", choices=["close", "wide", "both"], default="close")
    p.add_argument("--camera-distance", type=float, default=3.0)
    p.add_argument("--camera-height", type=float, default=0.9)
    p.add_argument("--camera-lookahead", type=float, default=0.8)
    p.add_argument("--show-target", action="store_true")
    p.add_argument("--participation-window", type=int, default=50,
                   help="steps used for hind/all-foot stance+swing participation metrics")
    p.add_argument("--control-mode", choices=["raw", "cpg_residual"], default="raw")
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--front-stance-press", type=float, default=0.40)
    p.add_argument("--front-swing-lift", type=float, default=0.40)
    p.add_argument("--contact-thresh", type=float, default=0.0564)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    out = REPO / "models" / args.run

    base = GeckoWalkEnv(
        render_mode="rgb_array" if args.render_video else None,
        control_mode=args.control_mode,
        residual_scale=args.residual_scale,
        front_stance_press=args.front_stance_press,
        front_swing_lift=args.front_swing_lift,
        contact_thresh=args.contact_thresh,
    )
    venv = DummyVecEnv([lambda: base])
    vn = out / "vecnormalize.pkl"
    if vn.exists():
        venv = VecNormalize.load(str(vn), venv); venv.training = False; venv.norm_reward = False
    ckpt = out / "best_model"
    model = PPO.load(str(ckpt) if (out / "best_model.zip").exists() else str(out / "final"),
                     device="cpu")

    render_views = []
    if args.render_video:
        render_views = ["close", "wide"] if args.camera_mode == "both" else [args.camera_mode]
    frames_by_view = {view: [] for view in render_views}
    reached, falls, rets = 0, 0, []
    final_distances = []
    forward_speeds = []
    gait_matches = []
    belly_contacts = []
    slips = []
    progresses = []
    hind_pushes = []
    front_pair_sync = []
    front_pair_hops = []
    hind_participation_windows = []
    foot_participation_windows = []
    trunk_heights = []
    body_bounces = []
    feet_seen_down = np.zeros(len(_GAIT_FEET), dtype=bool)
    feet_seen_up = np.zeros(len(_GAIT_FEET), dtype=bool)
    foot_duty_sum = np.zeros(len(_GAIT_FEET), dtype=float)
    foot_duty_n = 0
    for ep in range(args.episodes):
        obs = venv.reset(); done = [False]; ret = 0.0; info = [{}]
        episode_contacts = []
        episode_trunk_heights = []
        while not done[0]:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = venv.step(act); ret += float(r[0])
            step_info = info[0]
            reached += int(step_info.get("reached", 0.0) > 0.5)
            forward_speeds.append(float(step_info.get("forward_speed", 0.0)))
            gait_matches.append(float(step_info.get("gait_match", 0.0)))
            belly_contacts.append(float(step_info.get("belly_contact", 0.0)))
            slips.append(float(step_info.get("slip", 0.0)))
            progresses.append(float(step_info.get("progress", 0.0)))
            hind_pushes.append(float(step_info.get("hind_push", 0.0)))
            forces = step_info.get("foot_contact_forces")
            contacts = None
            if forces is not None:
                contacts = np.asarray(forces, dtype=np.float32) > args.contact_thresh
            elif step_info.get("foot_contacts") is not None:
                contacts = np.asarray(step_info["foot_contacts"], dtype=np.float32) > 0.5
            if contacts is not None:
                feet_seen_down |= contacts
                feet_seen_up |= ~contacts
                foot_duty_sum += contacts.astype(float)
                foot_duty_n += 1
                episode_contacts.append(contacts.copy())
                fl = contacts[_FOOT_INDEX["FL"]]
                fr = contacts[_FOOT_INDEX["FR"]]
                front_pair_sync.append(float(fl == fr))
                front_pair_hops.append(float((not fl) and (not fr)))
                window = episode_contacts[-max(args.participation_window, 1):]
                hind_participation_windows.append(
                    float(_window_participation(window, ("HL", "HR")))
                )
                foot_participation_windows.append(
                    float(_window_participation(window, _GAIT_FEET))
                )
            trunk_height = float(base.data.xpos[base._trunk][2])
            trunk_heights.append(trunk_height)
            episode_trunk_heights.append(trunk_height)
            if args.render_video:
                for view in render_views:
                    frames_by_view[view].append(_render_frame(base, view, args))
        rets.append(ret)
        final_distances.append(float(info[0].get("distance", info[0].get("dist", np.nan))))
        falls += int(info[0].get("fallen", 0.0) > 0.5)
        if episode_trunk_heights:
            body_bounces.append(float(np.std(episode_trunk_heights)))
    feet_participating = feet_seen_down & feet_seen_up
    active_labels = [foot for foot, ok in zip(_GAIT_FEET, feet_participating) if ok]
    foot_duty = foot_duty_sum / max(foot_duty_n, 1)
    print(f"contact_threshold={args.contact_thresh:.4f} control_mode={args.control_mode}")
    print(f"episodes={args.episodes} mean_return={np.mean(rets):.1f}")
    print(f"reached_count={reached} falls_count={falls}")
    print(f"final_distance_mean={_mean_or_nan(final_distances):.4f} "
          f"final_distance_last={final_distances[-1]:.4f}")
    print(f"mean_forward_speed={_mean_or_nan(forward_speeds):.4f} "
          f"gait_match={_mean_or_nan(gait_matches):.3f}")
    print(f"belly_contact_rate={_mean_or_nan(belly_contacts):.3f} "
          f"slip_metric={_mean_or_nan(slips):.4f} "
          f"progress_metric={_mean_or_nan(progresses):.4f}")
    print(f"front_pair_sync_rate={_mean_or_nan(front_pair_sync):.3f} "
          f"front_pair_hop_rate={_mean_or_nan(front_pair_hops):.3f}")
    print(f"hind_participation={_mean_or_nan(hind_participation_windows):.3f} "
          f"hind_push={_mean_or_nan(hind_pushes):.3f} "
          f"foot_participation={_mean_or_nan(foot_participation_windows):.3f}")
    print("foot_duty=" + " ".join(
        f"{foot}:{foot_duty[_FOOT_INDEX[foot]]:.3f}" for foot in _GAIT_FEET
    ))
    print(f"body_bounce={_mean_or_nan(body_bounces):.5f} "
          f"trunk_height={_mean_or_nan(trunk_heights):.5f} "
          f"forward_speed={_mean_or_nan(forward_speeds):.4f}")
    print(f"all_4_feet_participate={bool(np.all(feet_participating))} "
          f"active_feet={','.join(active_labels) if active_labels else 'none'}")
    if args.render_video:
        try:
            import imageio.v2 as imageio
            outdir = REPO / "renders"; outdir.mkdir(exist_ok=True)

            def write_video(path, frames):
                if not frames:
                    return
                imageio.mimwrite(path, frames, fps=args.fps, quality=8)
                print("video ->", path)

            if args.camera_mode == "both":
                close_path = outdir / f"{args.run}_close.mp4"
                wide_path = outdir / f"{args.run}_wide.mp4"
                both_path = outdir / f"{args.run}_both_views.mp4"
                write_video(close_path, frames_by_view.get("close", []))
                write_video(wide_path, frames_by_view.get("wide", []))
                write_video(
                    both_path,
                    frames_by_view.get("close", []) + frames_by_view.get("wide", []),
                )
            elif args.camera_mode == "wide":
                write_video(outdir / f"{args.run}_wide.mp4", frames_by_view.get("wide", []))
            else:
                write_video(outdir / f"{args.run}.mp4", frames_by_view.get("close", []))
        except Exception as e:
            print("video write failed:", str(e)[:120])


if __name__ == "__main__":
    main()
