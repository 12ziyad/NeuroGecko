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
    priv_label = f"YES (scale={args.privileged_food_scale})" if args.use_privileged_food else "NO  (pure visual eval)"
    print(f"[watch] privileged   = {priv_label}")
    if train_config.get("privileged_food_taper_enabled", False):
        print(
            "[watch] train_taper  = "
            f"{train_config.get('privileged_food_start_scale')} -> "
            f"{train_config.get('privileged_food_end_scale')} "
            f"over {train_config.get('privileged_food_taper_steps')} steps"
        )
    print(f"[watch] view         = {args.view}  smoothing={args.camera_smoothing}")
    print("=" * 60)

    _check_obs_mode(train_config, args.use_privileged_food, args.privileged_food_scale)

    privileged_target = float(args.privileged_food_scale) if args.use_privileged_food else 0.0

    env = GeckoBrainEnv(
        walker_run=walker_run,
        max_steps=args.steps,
        seed=args.seed,
        privileged_target=privileged_target,
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
            belly_contacts = []
            hungers = []

            for _ in range(args.steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                eat_count += int(bool(info.get("ate", False)))
                falls += int(bool(info.get("fallen", False)))
                food_distances.append(float(info.get("food_dist", np.nan)))
                belly_contacts.append(float(info.get("belly_contact", 0.0)))
                hungers.append(float(info.get("hunger", 0.0)))

                if args.render_video:
                    frames.append(env.render())
                if terminated or truncated:
                    break

            print(
                f"episode={ep + 1} "
                f"eat_count={eat_count} "
                f"final_food_dist={food_distances[-1]:.4f} "
                f"mean_food_dist={_mean_or_nan(food_distances):.4f} "
                f"falls={falls} "
                f"belly_contact_rate={_mean_or_nan(belly_contacts):.3f} "
                f"mean_hunger={_mean_or_nan(hungers):.3f}"
            )

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
