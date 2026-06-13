from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from envs.gecko_brain_env import GeckoBrainEnv


def _mean_or_nan(values) -> float:
    return float(np.mean(values)) if values else float("nan")


def _random_action(rng: np.random.Generator, env: GeckoBrainEnv) -> np.ndarray:
    action = rng.uniform(-1.0, 1.0, size=4).astype(np.float32)
    if rng.random() < 0.70:
        action[3] = rng.uniform(0.0, 1.0)
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def _episode_action(mode: str, rng: np.random.Generator, env: GeckoBrainEnv) -> np.ndarray:
    if mode == "oracle":
        return env.oracle_action(engage=1.0)
    return _random_action(rng, env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["random", "oracle"], default="random")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--privileged-target", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        max_steps=args.steps,
        render_mode="rgb_array" if args.render_video else None,
        seed=args.seed,
        privileged_target=args.privileged_target,
    )
    frames = []

    try:
        for ep in range(args.episodes):
            _, info = env.reset(seed=args.seed + ep)
            episode_return = 0.0
            eat_count = 0
            falls = 0
            food_distances = [float(info.get("food_dist", np.nan))]
            belly_contacts = []
            hungers = []
            engages = []

            for _ in range(args.steps):
                action = _episode_action(args.mode, rng, env)
                _, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                eat_count += int(bool(info.get("ate", False)))
                falls += int(bool(info.get("fallen", False)))
                food_distances.append(float(info.get("food_dist", np.nan)))
                belly_contacts.append(float(info.get("belly_contact", 0.0)))
                hungers.append(float(info.get("hunger", 0.0)))
                engages.append(float(info.get("engage", 0.0)))

                if args.render_video:
                    frames.append(env.render())
                if terminated or truncated:
                    break

            print(
                f"episode={ep + 1} "
                f"return={episode_return:.3f} "
                f"eat_count={eat_count} "
                f"final_food_dist={food_distances[-1]:.4f} "
                f"mean_food_dist={_mean_or_nan(food_distances):.4f} "
                f"falls={falls} "
                f"belly_contact_rate={_mean_or_nan(belly_contacts):.3f} "
                f"mean_hunger={_mean_or_nan(hungers):.3f} "
                f"mean_engage={_mean_or_nan(engages):.3f}"
            )

        if args.render_video:
            try:
                import imageio.v2 as imageio

                out_dir = REPO / "renders"
                out_dir.mkdir(exist_ok=True)
                safe_run = args.walker_run.replace("/", "_").replace("\\", "_")
                out_path = out_dir / f"brain_{args.mode}_{safe_run}.mp4"
                imageio.mimwrite(out_path, frames, fps=args.fps, quality=8)
                print("video ->", out_path)
            except Exception as exc:
                print("video write failed:", str(exc)[:160])
    finally:
        env.close()


if __name__ == "__main__":
    main()
