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


def _make_env(args):
    def thunk():
        env = GeckoBrainEnv(
            walker_run=args.walker_run,
            max_steps=args.episode_steps,
            seed=args.seed,
            privileged_target=0.0,
            render_mode=None,
        )
        return Monitor(env)

    return thunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--run-name", type=str, default="brain_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-steps", type=int, default=1000)
    args = parser.parse_args()

    out_dir = REPO / "models" / "brain" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BrainV1Config(use_privileged=False)
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
        "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
    }
    config_path = out_dir / "train_config.json"
    config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    env = DummyVecEnv([_make_env(args)])
    try:
        model = make_brain_ppo(
            env,
            config,
            verbose=1,
            seed=args.seed,
            device="cpu",
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )
        model.learn(total_timesteps=int(args.total_steps), progress_bar=False)
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        print(f"brain model -> {final_path}")
        print(f"train config -> {config_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
