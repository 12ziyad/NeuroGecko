from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from envs.gecko_brain_env import GeckoBrainEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect oracle-labeled BC dataset")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--num-transitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noisy-fraction",
        type=float,
        default=0.7,
        help="Fraction of steps using noisy oracle action (0.0–1.0). Default 0.7.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.15,
        help="Gaussian noise std applied to oracle action in noisy steps.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="Probability of fully random action instead of noisy oracle. Default 0.10.",
    )
    parser.add_argument("--food-spawn-angle-deg", type=float, default=180.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode before forced reset.",
    )
    args = parser.parse_args()

    out_dir = REPO / "data" / "oracle_bc"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.dataset_name}.npz"
    meta_path = out_dir / f"{args.dataset_name}_meta.json"

    print("=" * 60)
    print(f"[collect] dataset_name    = {args.dataset_name}")
    print(f"[collect] walker_run      = {args.walker_run}")
    print(f"[collect] num_transitions = {args.num_transitions}")
    print(f"[collect] noisy_fraction  = {args.noisy_fraction}")
    print(f"[collect] noise_std       = {args.noise_std}")
    print(f"[collect] epsilon         = {args.epsilon}")
    print(f"[collect] food_radius     = {args.food_radius}")
    print(f"[collect] eat_radius      = {args.eat_radius}")
    print(f"[collect] output          = {npz_path}")
    print("=" * 60)

    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        max_steps=args.max_steps,
        seed=args.seed,
        privileged_target=1.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )

    proprio_dim = env.observation_space["proprio"].shape[0]
    N = args.num_transitions
    rng = np.random.default_rng(args.seed)

    obs_images = np.empty((N, 64, 64, 3), dtype=np.uint8)
    obs_proprios = np.empty((N, proprio_dim), dtype=np.float32)
    obs_drives = np.empty((N, 6), dtype=np.float32)
    obs_prev_actions = np.empty((N, 4), dtype=np.float32)
    obs_privileged = np.empty((N, 5), dtype=np.float32)
    labels = np.empty((N, 4), dtype=np.float32)

    collected = 0
    episodes = 0
    log_interval = max(1, N // 20)

    obs, _ = env.reset(seed=args.seed)
    try:
        while collected < N:
            # label: always clean oracle for current state
            label = env.oracle_action()

            obs_images[collected] = obs["image"]
            obs_proprios[collected] = obs["proprio"]
            obs_drives[collected] = obs["drives"]
            obs_prev_actions[collected] = obs["prev_action"]
            obs_privileged[collected] = obs["privileged"]
            labels[collected] = label
            collected += 1

            if collected % log_interval == 0:
                print(f"[collect] {collected}/{N}  episodes={episodes}")

            # choose action to step with (label is ALWAYS clean oracle above)
            if rng.random() < args.noisy_fraction:
                if rng.random() < args.epsilon:
                    step_action = rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
                else:
                    noise = (rng.standard_normal(4) * args.noise_std).astype(np.float32)
                    step_action = np.clip(label + noise, -1.0, 1.0)
            else:
                step_action = label.copy()

            obs, _, terminated, truncated, _ = env.step(step_action)
            if terminated or truncated:
                episodes += 1
                obs, _ = env.reset(seed=args.seed + episodes)
    finally:
        env.close()

    print(f"[collect] done: {collected} transitions, {episodes + 1} episodes")

    np.savez_compressed(
        npz_path,
        obs_image=obs_images[:collected],
        obs_proprio=obs_proprios[:collected],
        obs_drives=obs_drives[:collected],
        obs_prev_action=obs_prev_actions[:collected],
        obs_privileged=obs_privileged[:collected],
        actions=labels[:collected],
    )
    print(f"[collect] dataset  -> {npz_path}")

    meta = {
        "dataset_name": args.dataset_name,
        "walker_run": args.walker_run,
        "num_transitions": int(collected),
        "proprio_dim": int(proprio_dim),
        "obs_image_shape": [64, 64, 3],
        "obs_drives_shape": [6],
        "obs_prev_action_shape": [4],
        "obs_privileged_shape": [5],
        "action_dim": 4,
        "noisy_fraction": float(args.noisy_fraction),
        "noise_std": float(args.noise_std),
        "epsilon": float(args.epsilon),
        "food_spawn_angle_deg": float(args.food_spawn_angle_deg),
        "eat_radius": float(args.eat_radius),
        "food_radius": float(args.food_radius),
        "max_steps_per_episode": int(args.max_steps),
        "privileged_target": 1.0,
        "privileged_food_dropout_prob": 0.0,
        "seed": int(args.seed),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[collect] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
