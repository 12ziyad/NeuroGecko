from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from brain.agwm import BrainV1Config, make_brain_ppo, recurrent_ppo_available
from envs.gecko_brain_env import GeckoBrainEnv


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _make_env_fn(args):
    privileged_target = float(args.privileged_food_scale) if args.use_privileged_food else 0.0

    def thunk():
        env = GeckoBrainEnv(
            walker_run=args.walker_run,
            max_steps=args.episode_steps,
            seed=args.seed,
            privileged_target=privileged_target,
            render_mode=None,
        )
        return Monitor(env)

    return thunk


def _make_vec_env(args):
    thunks = [_make_env_fn(args) for _ in range(args.num_envs)]
    if platform.system() == "Linux" and args.num_envs > 1:
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
            return SubprocVecEnv(thunks), "SubprocVecEnv"
        except Exception as exc:
            print(f"[warn] SubprocVecEnv failed ({exc}), falling back to DummyVecEnv")
    return DummyVecEnv(thunks), "DummyVecEnv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain V1 PPO trainer")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--run-name", type=str, default="brain_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=1000)
    parser.add_argument(
        "--use-privileged-food",
        action="store_true",
        help="Expose egocentric food direction/distance to the policy observation (curriculum mode).",
    )
    parser.add_argument(
        "--privileged-food-scale",
        type=float,
        default=1.0,
        help="Scale applied to the privileged food vector (default 1.0). Only active with --use-privileged-food.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel envs. Uses SubprocVecEnv on Linux when > 1, DummyVecEnv on Windows/fallback.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=128,
        help="PPO rollout steps collected per env before each update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO minibatch size. Must be <= n_steps * num_envs.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show tqdm progress bar during training (requires tqdm).",
    )
    args = parser.parse_args()

    rollout_size = args.n_steps * args.num_envs
    if args.batch_size > rollout_size:
        parser.error(
            f"--batch-size {args.batch_size} > n_steps*num_envs ({args.n_steps}*{args.num_envs}={rollout_size}). "
            "Reduce --batch-size or increase --n-steps / --num-envs."
        )

    out_dir = REPO / "models" / "brain" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_mode = "privileged" if args.use_privileged_food else "pure"
    config = BrainV1Config(use_privileged=args.use_privileged_food)

    train_config = {
        "run_name": args.run_name,
        "walker_run": args.walker_run,
        "total_steps": int(args.total_steps),
        "seed": int(args.seed),
        "episode_steps": int(args.episode_steps),
        "brain_action": ["target_dir_x", "target_dir_y", "target_distance", "engage"],
        "brain_action_dim": 4,
        "algo": "PPO",
        "recurrent_ppo_available": recurrent_ppo_available(),
        "architecture": config.to_json_dict(),
        "use_privileged_food": bool(args.use_privileged_food),
        "privileged_food_scale": float(args.privileged_food_scale) if args.use_privileged_food else 0.0,
        "observation_mode": obs_mode,
        "num_envs": int(args.num_envs),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
    }
    config_path = out_dir / "train_config.json"
    config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    device = "cuda" if _cuda_available() else "cpu"
    env, vec_type = _make_vec_env(args)

    print("=" * 60)
    print(f"[brain train] run         = {args.run_name}")
    print(f"[brain train] obs_mode    = {obs_mode}", end="")
    if args.use_privileged_food:
        print(f"  (privileged_scale={args.privileged_food_scale})")
    else:
        print()
    print(f"[brain train] num_envs    = {args.num_envs}  vec={vec_type}")
    print(f"[brain train] n_steps     = {args.n_steps}  batch_size={args.batch_size}")
    print(f"[brain train] total_steps = {args.total_steps}")
    print(f"[brain train] device      = {device}")
    print("=" * 60)

    try:
        model = make_brain_ppo(
            env,
            config,
            verbose=1,
            seed=args.seed,
            device=device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )
        model.learn(total_timesteps=int(args.total_steps), progress_bar=args.progress_bar)
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        print(f"brain model  -> {final_path}")
        print(f"train config -> {config_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
