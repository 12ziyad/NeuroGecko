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

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from brain.agwm import BrainV1Config, make_brain_ppo, recurrent_ppo_available
from envs.gecko_brain_env import GeckoBrainEnv


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_brain_run_path(run_name: str) -> Path:
    candidates = [
        REPO / "models" / "brain" / run_name,
        REPO / "models" / run_name,
    ]
    for path in candidates:
        if (path / "final.zip").exists():
            return path

    attempted = "\n".join(f"  - {path / 'final.zip'}" for path in candidates)
    raise FileNotFoundError(
        f"Missing brain checkpoint for '{run_name}'. Tried:\n{attempted}"
    )


def _taper_enabled(args) -> bool:
    return (
        args.privileged_food_start_scale is not None
        or args.privileged_food_end_scale is not None
        or int(args.privileged_food_taper_steps) > 0
    )


def _dropout_taper_enabled(args) -> bool:
    return (
        args.privileged_food_start_dropout is not None
        or args.privileged_food_end_dropout is not None
        or int(args.privileged_food_dropout_taper_steps) > 0
    )


def _validate_privileged_food_args(parser: argparse.ArgumentParser, args) -> bool:
    taper_enabled = _taper_enabled(args)

    if int(args.privileged_food_taper_steps) < 0:
        parser.error("--privileged-food-taper-steps must be >= 0.")

    if taper_enabled:
        if not args.use_privileged_food:
            parser.error("Privileged food taper requires --use-privileged-food.")
        if args.privileged_food_start_scale is None:
            parser.error("--privileged-food-start-scale is required when using a taper.")
        if args.privileged_food_end_scale is None:
            parser.error("--privileged-food-end-scale is required when using a taper.")

    return taper_enabled


def _validate_privileged_food_dropout_args(parser: argparse.ArgumentParser, args) -> bool:
    dropout_enabled = _dropout_taper_enabled(args)

    if int(args.privileged_food_dropout_taper_steps) < 0:
        parser.error("--privileged-food-dropout-taper-steps must be >= 0.")

    if dropout_enabled:
        if not args.use_privileged_food:
            parser.error("Privileged food dropout taper requires --use-privileged-food.")
        if args.privileged_food_start_dropout is None:
            parser.error("--privileged-food-start-dropout is required when using dropout taper.")
        if args.privileged_food_end_dropout is None:
            parser.error("--privileged-food-end-dropout is required when using dropout taper.")

    return dropout_enabled


def _initial_privileged_food_scale(args, taper_enabled: bool) -> float:
    if not args.use_privileged_food:
        return 0.0
    if taper_enabled:
        return float(args.privileged_food_start_scale)
    return float(args.privileged_food_scale)


def _privileged_food_scale_at_step(
    step: int,
    start_scale: float,
    end_scale: float,
    taper_steps: int,
) -> float:
    if taper_steps <= 0:
        return float(start_scale if step <= 0 else end_scale)
    alpha = min(max(float(step) / float(taper_steps), 0.0), 1.0)
    return float(start_scale + alpha * (end_scale - start_scale))


class PrivilegedFoodTaperCallback(BaseCallback):
    def __init__(
        self,
        start_scale: float,
        end_scale: float,
        taper_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.start_scale = float(start_scale)
        self.end_scale = float(end_scale)
        self.taper_steps = int(taper_steps)
        if self.taper_steps > 0:
            self.log_interval = max(1, min(10_000, self.taper_steps // 10 or 1))
        else:
            self.log_interval = 1
        self._last_logged_step: int | None = None

    def _scale(self, step: int) -> float:
        return _privileged_food_scale_at_step(
            step,
            self.start_scale,
            self.end_scale,
            self.taper_steps,
        )

    def _set_scale(self, step: int, force_print: bool = False) -> None:
        scale = self._scale(step)
        self.training_env.env_method("set_privileged_food_scale", scale)
        self.logger.record("curriculum/privileged_food_scale", scale)

        should_print = force_print
        if self._last_logged_step is None:
            should_print = True
        elif step - self._last_logged_step >= self.log_interval:
            should_print = True
        elif self.taper_steps > 0 and step >= self.taper_steps > self._last_logged_step:
            should_print = True

        if should_print:
            print(f"[curriculum] step={step} privileged_food_scale={scale:.6f}")
            self._last_logged_step = step

    def _on_training_start(self) -> None:
        self._set_scale(0, force_print=True)

    def _on_step(self) -> bool:
        self._set_scale(int(self.num_timesteps))
        return True


class PrivilegedFoodDropoutCallback(BaseCallback):
    def __init__(
        self,
        start_prob: float,
        end_prob: float,
        taper_steps: int,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.start_prob = float(start_prob)
        self.end_prob = float(end_prob)
        self.taper_steps = int(taper_steps)
        if self.taper_steps > 0:
            self.log_interval = max(1, min(10_000, self.taper_steps // 10 or 1))
        else:
            self.log_interval = 1
        self._last_logged_step: int | None = None

    def _prob(self, step: int) -> float:
        if self.taper_steps <= 0:
            return float(self.start_prob if step <= 0 else self.end_prob)
        alpha = min(max(float(step) / float(self.taper_steps), 0.0), 1.0)
        return float(self.start_prob + alpha * (self.end_prob - self.start_prob))

    def _set_prob(self, step: int, force_print: bool = False) -> None:
        prob = self._prob(step)
        self.training_env.env_method("set_privileged_food_dropout_prob", prob)
        self.logger.record("curriculum/privileged_food_dropout", prob)

        should_print = force_print
        if self._last_logged_step is None:
            should_print = True
        elif step - self._last_logged_step >= self.log_interval:
            should_print = True
        elif self.taper_steps > 0 and step >= self.taper_steps > self._last_logged_step:
            should_print = True

        if should_print:
            print(f"[curriculum] step={step} privileged_food_dropout={prob:.6f}")
            self._last_logged_step = step

    def _on_training_start(self) -> None:
        self._set_prob(0, force_print=True)

    def _on_step(self) -> bool:
        self._set_prob(int(self.num_timesteps))
        return True


def _make_env_fn(args):
    privileged_target = float(args._initial_privileged_food_scale)
    privileged_food_dropout_prob = float(args._initial_privileged_food_dropout_prob)

    def thunk():
        env = GeckoBrainEnv(
            walker_run=args.walker_run,
            max_steps=args.episode_steps,
            seed=args.seed,
            privileged_target=privileged_target,
            privileged_food_dropout_prob=privileged_food_dropout_prob,
            food_spawn_angle_deg=float(args.food_spawn_angle_deg),
            eat_radius=float(args.eat_radius),
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
        "--resume-brain-run",
        "--init-from-brain-run",
        dest="resume_brain_run",
        type=str,
        default=None,
        help="Continue training from an existing brain run checkpoint.",
    )
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
        "--privileged-food-start-scale",
        type=float,
        default=None,
        help="Starting scale for privileged food curriculum taper.",
    )
    parser.add_argument(
        "--privileged-food-end-scale",
        type=float,
        default=None,
        help="Final scale for privileged food curriculum taper.",
    )
    parser.add_argument(
        "--privileged-food-taper-steps",
        type=int,
        default=0,
        help="Steps over which privileged food scale linearly decays to the end scale.",
    )
    parser.add_argument(
        "--privileged-food-start-dropout",
        type=float,
        default=None,
        help="Starting dropout probability for privileged food curriculum (0.0 = no dropout).",
    )
    parser.add_argument(
        "--privileged-food-end-dropout",
        type=float,
        default=None,
        help="Final dropout probability for privileged food curriculum (1.0 = always zero out).",
    )
    parser.add_argument(
        "--privileged-food-dropout-taper-steps",
        type=int,
        default=0,
        help="Steps over which privileged food dropout probability linearly ramps to end value.",
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
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25000,
        help="Save an intermediate checkpoint every N timesteps. 0 to disable.",
    )
    args = parser.parse_args()
    taper_enabled = _validate_privileged_food_args(parser, args)
    args._initial_privileged_food_scale = _initial_privileged_food_scale(args, taper_enabled)
    dropout_taper_enabled = _validate_privileged_food_dropout_args(parser, args)
    args._initial_privileged_food_dropout_prob = (
        float(args.privileged_food_start_dropout) if dropout_taper_enabled else 0.0
    )
    args._effective_checkpoint_freq = (
        max(int(args.checkpoint_freq) // max(int(args.num_envs), 1), 1)
        if int(args.checkpoint_freq) > 0 else 0
    )
    resumed_from_path = (
        _resolve_brain_run_path(args.resume_brain_run)
        if args.resume_brain_run is not None
        else None
    )

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
        "privileged_food_scale": float(args._initial_privileged_food_scale),
        "privileged_food_taper_enabled": bool(taper_enabled),
        "privileged_food_start_scale": (
            float(args.privileged_food_start_scale) if taper_enabled else None
        ),
        "privileged_food_end_scale": (
            float(args.privileged_food_end_scale) if taper_enabled else None
        ),
        "privileged_food_taper_steps": (
            int(args.privileged_food_taper_steps) if taper_enabled else 0
        ),
        "privileged_food_dropout_taper_enabled": bool(dropout_taper_enabled),
        "privileged_food_start_dropout": (
            float(args.privileged_food_start_dropout) if dropout_taper_enabled else None
        ),
        "privileged_food_end_dropout": (
            float(args.privileged_food_end_dropout) if dropout_taper_enabled else None
        ),
        "privileged_food_dropout_taper_steps": (
            int(args.privileged_food_dropout_taper_steps) if dropout_taper_enabled else 0
        ),
        "observation_mode": obs_mode,
        "num_envs": int(args.num_envs),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "continued_training": bool(resumed_from_path is not None),
        "resumed_from_run": args.resume_brain_run if resumed_from_path is not None else None,
        "resumed_from_path": str(resumed_from_path) if resumed_from_path is not None else None,
        "food_spawn_angle_deg": float(args.food_spawn_angle_deg),
        "eat_radius": float(args.eat_radius),
        "checkpoint_freq": int(args.checkpoint_freq),
        "checkpoint_freq_env_calls": int(args._effective_checkpoint_freq),
        "notes": "Brain V1 trains only the high-level 4D target/engage channel.",
    }
    config_path = out_dir / "train_config.json"
    config_path.write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    device = "cuda" if _cuda_available() else "cpu"
    env, vec_type = _make_vec_env(args)
    if resumed_from_path is not None:
        vec_path = resumed_from_path / "vecnormalize.pkl"
        if vec_path.exists():
            env = VecNormalize.load(str(vec_path), env)
            env.training = True
            vec_type = f"VecNormalize({vec_type})"
        else:
            print(
                f"[warn] resumed brain run has no vecnormalize.pkl at {vec_path}; "
                "continuing with fresh brain env normalization path."
            )

    print("=" * 60)
    print(f"[brain train] run         = {args.run_name}")
    if resumed_from_path is not None:
        print(f"[brain train] resumed_from = {resumed_from_path}")
        print("[brain train] continued_training = true")
    print(f"[brain train] obs_mode    = {obs_mode}", end="")
    if args.use_privileged_food:
        print(f"  (privileged_scale={args._initial_privileged_food_scale})")
    else:
        print()
    if taper_enabled:
        print(
            "[brain train] scale_taper = "
            f"{args.privileged_food_start_scale} -> {args.privileged_food_end_scale} "
            f"over {args.privileged_food_taper_steps} steps"
        )
    if dropout_taper_enabled:
        print(
            "[brain train] drop_taper  = "
            f"{args.privileged_food_start_dropout} -> {args.privileged_food_end_dropout} "
            f"over {args.privileged_food_dropout_taper_steps} steps"
        )
    print(f"[brain train] num_envs    = {args.num_envs}  vec={vec_type}")
    print(f"[brain train] n_steps     = {args.n_steps}  batch_size={args.batch_size}")
    print(f"[brain train] food_spawn_angle_deg = {args.food_spawn_angle_deg}")
    print(f"[brain train] eat_radius = {args.eat_radius}")
    print(f"[brain train] total_steps = {args.total_steps}")
    if args._effective_checkpoint_freq > 0:
        print(
            f"[brain train] checkpoint  = every {args.checkpoint_freq} total steps "
            f"({args._effective_checkpoint_freq} env calls x {args.num_envs} envs)"
        )
    print(f"[brain train] device      = {device}")
    if callable(getattr(env, "save", None)):
        print("[brain train] vecnormalize = active; will save brain vecnormalize.pkl")
    else:
        print("[brain train] vecnormalize = not active for brain env; no brain vecnormalize.pkl will be saved")
    print("=" * 60)

    try:
        if resumed_from_path is not None:
            model = PPO.load(str(resumed_from_path / "final.zip"), env=env, device=device)
        else:
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
        callbacks = []
        if taper_enabled:
            callbacks.append(PrivilegedFoodTaperCallback(
                start_scale=float(args.privileged_food_start_scale),
                end_scale=float(args.privileged_food_end_scale),
                taper_steps=int(args.privileged_food_taper_steps),
            ))
        if dropout_taper_enabled:
            callbacks.append(PrivilegedFoodDropoutCallback(
                start_prob=float(args.privileged_food_start_dropout),
                end_prob=float(args.privileged_food_end_dropout),
                taper_steps=int(args.privileged_food_dropout_taper_steps),
            ))
        if args._effective_checkpoint_freq > 0:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(CheckpointCallback(
                save_freq=args._effective_checkpoint_freq,
                save_path=str(ckpt_dir),
                name_prefix="ckpt",
                verbose=1,
            ))
        callback = CallbackList(callbacks) if callbacks else None
        model.learn(
            total_timesteps=int(args.total_steps),
            callback=callback,
            progress_bar=args.progress_bar,
            reset_num_timesteps=resumed_from_path is None,
        )
        final_path = out_dir / "final.zip"
        model.save(str(final_path))
        if callable(getattr(env, "save", None)):
            vec_out = out_dir / "vecnormalize.pkl"
            env.save(str(vec_out))
            print(f"brain vecnormalize -> {vec_out}")
        print(f"brain model  -> {final_path}")
        print(f"train config -> {config_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
