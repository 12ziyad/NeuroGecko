from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import PPO

from brain.actor_critic import BrainActorCriticPolicy  # noqa: F401
from envs.gecko_brain_env import GeckoBrainEnv


def _mean_or_nan(values) -> float:
    return float(np.mean(values)) if values else float("nan")


def _load_train_config(run_dir: Path) -> dict:
    path = run_dir / "train_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _check_obs_mode(
    train_config: dict,
    use_privileged_food: bool,
    privileged_food_scale: float,
) -> None:
    if not train_config:
        return

    trained_privileged = bool(train_config.get("use_privileged_food", False))

    if trained_privileged and not use_privileged_food:
        print(
            "[watch] INFO: Model was trained WITH privileged food (curriculum mode). "
            "Running pure eval — the privileged channel is zeroed out. "
            "This is the correct real-world test. "
            "Pass --use-privileged-food to reproduce training conditions."
        )
    elif not trained_privileged and use_privileged_food:
        print(
            "[watch] WARNING: Model was trained WITHOUT privileged food. "
            "Its extractor ignores the privileged channel entirely. "
            "--use-privileged-food has no effect on this model. "
            "Run without --use-privileged-food to avoid confusion."
        )
    elif trained_privileged and use_privileged_food:
        scale = privileged_food_scale
        print(
            f"[watch] INFO: Running with privileged food (scale={scale}). "
            "This matches training conditions — NOT a pure evaluation."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained Brain V1 model")
    parser.add_argument("--brain-run", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--walker-run", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--food-spawn-angle-deg",
        type=float,
        default=180.0,
        help="Food spawn half-angle in body frame. 180.0 keeps full-circle spawn.",
    )
    parser.add_argument(
        "--eat-radius",
        type=float,
        default=0.10,
        help="Mouth/nose distance threshold for eating in the Brain env.",
    )
    parser.add_argument(
        "--view",
        choices=["fixed", "chase", "close"],
        default="close",
        help="Viewer camera mode. 'close' = stable close follow (default), "
             "'chase' = wider follow, 'fixed' = static azimuth.",
    )
    parser.add_argument(
        "--camera-smoothing",
        type=float,
        default=0.85,
        help="EMA smoothing for the viewer camera (0=none, 0.85=default). "
             "Reduces jitter in chase/close modes.",
    )
    parser.add_argument(
        "--use-privileged-food",
        action="store_true",
        help=(
            "Expose real egocentric food direction/distance to the policy. "
            "Must match the observation mode the model was trained with. "
            "Default (no flag) = pure visual eval."
        ),
    )
    parser.add_argument(
        "--privileged-food-scale",
        type=float,
        default=1.0,
        help="Scale for the privileged food vector. Only active with --use-privileged-food.",
    )
    parser.add_argument(
        "--privileged-food-dropout-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of zeroing the privileged food vector each step. "
            "1.0 = always zero (pure visual eval equivalent). "
            "Default 0.0 = no dropout."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic (non-deterministic) action sampling. Default is deterministic.",
    )
    args = parser.parse_args()

    run_dir = REPO / "models" / "brain" / args.brain_run
    model_path = run_dir / "final.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained brain model: {model_path}")

    train_config = _load_train_config(run_dir)
    walker_run = args.walker_run or train_config.get("walker_run", "v4_5b_speed_polish_1m")
    obs_mode = train_config.get("observation_mode", "unknown")

    print("=" * 60)
    print(f"[watch] brain_run    = {args.brain_run}")
    print(f"[watch] walker_run   = {walker_run}")
    print(f"[watch] train_obs    = {obs_mode}")
    print(f"[watch] food_spawn_angle_deg = {args.food_spawn_angle_deg}")
    print(f"[watch] eat_radius   = {args.eat_radius}")
    priv_label = f"YES (scale={args.privileged_food_scale})" if args.use_privileged_food else "NO  (pure visual eval)"
    print(f"[watch] privileged   = {priv_label}")
    if train_config.get("privileged_food_taper_enabled", False):
        print(
            "[watch] train_scale_taper = "
            f"{train_config.get('privileged_food_start_scale')} -> "
            f"{train_config.get('privileged_food_end_scale')} "
            f"over {train_config.get('privileged_food_taper_steps')} steps"
        )
    if train_config.get("privileged_food_dropout_taper_enabled", False):
        print(
            "[watch] train_drop_taper  = "
            f"{train_config.get('privileged_food_start_dropout')} -> "
            f"{train_config.get('privileged_food_end_dropout')} "
            f"over {train_config.get('privileged_food_dropout_taper_steps')} steps"
        )
    dropout_label = f"{args.privileged_food_dropout_prob:.3f}" if args.privileged_food_dropout_prob > 0.0 else "off"
    print(f"[watch] dropout_prob = {dropout_label}")
    action_mode = "stochastic" if args.stochastic else "deterministic"
    print(f"[watch] action_mode  = {action_mode}")
    print(f"[watch] view         = {args.view}  smoothing={args.camera_smoothing}")
    print("=" * 60)

    _check_obs_mode(train_config, args.use_privileged_food, args.privileged_food_scale)

    privileged_target = float(args.privileged_food_scale) if args.use_privileged_food else 0.0

    env = GeckoBrainEnv(
        walker_run=walker_run,
        max_steps=args.steps,
        seed=args.seed,
        privileged_target=privileged_target,
        privileged_food_dropout_prob=float(args.privileged_food_dropout_prob),
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        render_mode="rgb_array" if args.render_video else None,
        view_mode=args.view,
        camera_smoothing=args.camera_smoothing,
    )
    model = PPO.load(str(model_path), device="cpu")
    frames = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            eat_count = 0
            falls = 0
            food_distances = [float(info.get("food_dist", np.nan))]
            mouth_food_distances = []
            if "mouth_food_dist" in info:
                mouth_food_distances.append(float(info["mouth_food_dist"]))
            belly_contacts = []
            hungers = []
            min_mouth_food_dist = float("inf")
            engage_vals = []
            food_visible_fracs = []

            for _ in range(args.steps):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, _, terminated, truncated, info = env.step(action)
                eat_count += int(bool(info.get("ate", False)))
                falls += int(bool(info.get("fallen", False)))
                food_distances.append(float(info.get("food_dist", np.nan)))
                mfd = info.get("mouth_food_dist", None)
                if mfd is not None:
                    mfd_f = float(mfd)
                    mouth_food_distances.append(mfd_f)
                    if mfd_f < min_mouth_food_dist:
                        min_mouth_food_dist = mfd_f
                belly_contacts.append(float(info.get("belly_contact", 0.0)))
                hungers.append(float(info.get("hunger", 0.0)))

                # engage: action[3] in [-1,1] -> [0,1]
                try:
                    raw_engage = float(np.asarray(action).flat[3])
                    engage_val = max(0.0, min(1.0, (raw_engage + 1.0) * 0.5))
                    engage_vals.append(engage_val)
                except Exception:
                    pass

                # food visible fraction: prefer env info field, fallback to green-pixel heuristic
                if "food_visible_frac" in info:
                    food_visible_fracs.append(float(info["food_visible_frac"]))
                else:
                    try:
                        img = obs["image"]
                        img_np = np.asarray(img)
                        if img_np.ndim == 4:
                            img_np = img_np[0]
                        if img_np.ndim == 3 and img_np.shape[0] == 3:
                            img_np = img_np.transpose(1, 2, 0)
                        if img_np.dtype != np.uint8:
                            if img_np.max() <= 1.0:
                                img_np = (img_np * 255.0).astype(np.uint8)
                            else:
                                img_np = img_np.astype(np.uint8)
                        r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]
                        green_mask = (g > 120) & (g > r.astype(np.int32) + 30) & (g > b.astype(np.int32) + 30)
                        food_visible_fracs.append(float(green_mask.mean()))
                    except Exception:
                        food_visible_fracs.append(float("nan"))

                if args.render_video:
                    frames.append(env.render())
                if terminated or truncated:
                    break

            if min_mouth_food_dist == float("inf"):
                min_mouth_food_dist = float("nan")

            mean_engage = _mean_or_nan(engage_vals)
            engage_gt06_frac = (
                float(np.mean([v > 0.6 for v in engage_vals])) if engage_vals else float("nan")
            )
            food_visible_frac = _mean_or_nan(
                [v for v in food_visible_fracs if not np.isnan(v)]
            )

            episode_parts = [
                f"episode={ep + 1}",
                f"eat_count={eat_count}",
                f"final_food_dist={food_distances[-1]:.4f}",
                f"mean_food_dist={_mean_or_nan(food_distances):.4f}",
            ]
            if mouth_food_distances:
                episode_parts.extend([
                    f"final_mouth_food_dist={mouth_food_distances[-1]:.4f}",
                    f"mean_mouth_food_dist={_mean_or_nan(mouth_food_distances):.4f}",
                ])
            episode_parts.extend([
                f"falls={falls}",
                f"belly_contact_rate={_mean_or_nan(belly_contacts):.3f}",
                f"mean_hunger={_mean_or_nan(hungers):.3f}",
                f"min_mouth_food_dist={min_mouth_food_dist:.4f}",
                f"mean_engage={mean_engage:.4f}",
                f"engage_gt0.6_frac={engage_gt06_frac:.4f}",
                f"food_visible_frac={food_visible_frac:.6f}",
            ])
            print(" ".join(episode_parts))

        if args.render_video:
            import imageio.v2 as imageio

            out_dir = REPO / "renders"
            out_dir.mkdir(exist_ok=True)
            safe_run = args.brain_run.replace("/", "_").replace("\\", "_")
            video_path = out_dir / f"trained_brain_{safe_run}.mp4"
            imageio.mimwrite(video_path, frames, fps=args.fps, quality=8)
            print("video ->", video_path)

            downloads = Path.home() / "Downloads"
            downloads.mkdir(exist_ok=True)
            copied_path = downloads / video_path.name
            shutil.copy2(video_path, copied_path)
            print("download copy ->", copied_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
