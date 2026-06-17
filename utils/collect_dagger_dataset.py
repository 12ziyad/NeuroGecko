from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from brain.bc_actor import BrainBCActor, build_obs_space
from envs.gecko_brain_env import GeckoBrainEnv


def _load_train_config(run_dir: Path) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing train_config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_bc_model(run_dir: Path, train_config: dict) -> BrainBCActor:
    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing BC model: {model_path}")

    algo = str(train_config.get("algo", "")).lower()
    if algo and algo != "behavior_cloning":
        raise ValueError(
            f"Expected behavior_cloning train_config, got algo={train_config.get('algo')!r}"
        )

    proprio_dim = int(train_config.get("proprio_dim", 0))
    if proprio_dim <= 0:
        raise ValueError("train_config missing positive 'proprio_dim'")

    model = BrainBCActor(
        build_obs_space(proprio_dim),
        image_features_dim=int(train_config.get("image_features_dim", 128)),
        body_features_dim=int(train_config.get("body_features_dim", 96)),
        fused_features_dim=int(train_config.get("fused_features_dim", 256)),
        use_privileged=bool(train_config.get("use_privileged_food", True)),
        action_dim=int(train_config.get("action_dim", 4)),
    )
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model


def _validate_mix(args: argparse.Namespace) -> None:
    values = {
        "policy_fraction": args.policy_fraction,
        "noisy_policy_fraction": args.noisy_policy_fraction,
        "oracle_fraction": args.oracle_fraction,
        "epsilon": args.epsilon,
        "noise_std": args.noise_std,
    }
    for name, value in values.items():
        if float(value) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if float(args.epsilon) > 1.0:
        raise ValueError("--epsilon must be <= 1.0")
    non_random = (
        float(args.policy_fraction)
        + float(args.noisy_policy_fraction)
        + float(args.oracle_fraction)
    )
    if non_random <= 0.0 and float(args.epsilon) < 1.0:
        raise ValueError(
            "At least one of --policy-fraction, --noisy-policy-fraction, "
            "or --oracle-fraction must be positive when epsilon < 1.0"
        )


def _choose_step_action(
    rng: np.random.Generator,
    model_action: np.ndarray,
    oracle_action: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    if rng.random() < float(args.epsilon):
        return rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32), "random"

    policy_weight = float(args.policy_fraction)
    noisy_weight = float(args.noisy_policy_fraction)
    oracle_weight = float(args.oracle_fraction)
    total = policy_weight + noisy_weight + oracle_weight
    draw = rng.random() * total

    if draw < policy_weight:
        return np.asarray(model_action, dtype=np.float32).copy(), "policy"
    draw -= policy_weight
    if draw < noisy_weight:
        noise = (rng.standard_normal(4) * float(args.noise_std)).astype(np.float32)
        action = np.clip(np.asarray(model_action, dtype=np.float32) + noise, -1.0, 1.0)
        return action.astype(np.float32), "noisy_policy"
    return np.asarray(oracle_action, dtype=np.float32).copy(), "oracle"


def _effective_mix(args: argparse.Namespace) -> dict[str, float]:
    epsilon = float(args.epsilon)
    policy = float(args.policy_fraction)
    noisy = float(args.noisy_policy_fraction)
    oracle = float(args.oracle_fraction)
    total = max(policy + noisy + oracle, 1e-12)
    non_random = 1.0 - epsilon
    return {
        "policy": non_random * policy / total,
        "noisy_policy": non_random * noisy / total,
        "oracle": non_random * oracle / total,
        "random": epsilon,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect DAgger/recovery BC data from a trained BC brain policy"
    )
    parser.add_argument("--brain-run", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--num-transitions", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-fraction", type=float, default=0.70)
    parser.add_argument("--noisy-policy-fraction", type=float, default=0.20)
    parser.add_argument("--oracle-fraction", type=float, default=0.10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--walker-run", type=str, default=None)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    _validate_mix(args)
    if int(args.num_transitions) <= 0:
        raise ValueError("--num-transitions must be positive")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")

    brain_run_dir = REPO / "models" / "brain" / args.brain_run
    train_config = _load_train_config(brain_run_dir)
    model = _load_bc_model(brain_run_dir, train_config)

    walker_run = args.walker_run or train_config.get("walker_run", "v4_5b_speed_polish_1m")
    out_dir = REPO / "data" / "oracle_bc"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.dataset_name}.npz"
    meta_path = out_dir / f"{args.dataset_name}_meta.json"
    effective = _effective_mix(args)

    print("=" * 60)
    print(f"[dagger] dataset_name           = {args.dataset_name}")
    print(f"[dagger] source_brain_run       = {args.brain_run}")
    print(f"[dagger] walker_run             = {walker_run}")
    print(f"[dagger] num_transitions        = {args.num_transitions}")
    print(f"[dagger] policy_fraction        = {args.policy_fraction}")
    print(f"[dagger] noisy_policy_fraction  = {args.noisy_policy_fraction}")
    print(f"[dagger] oracle_fraction        = {args.oracle_fraction}")
    print(f"[dagger] epsilon                = {args.epsilon}")
    print(f"[dagger] effective_mix          = {effective}")
    print(f"[dagger] noise_std              = {args.noise_std}")
    print(f"[dagger] food_spawn_angle_deg   = {args.food_spawn_angle_deg}")
    print(f"[dagger] eat_radius             = {args.eat_radius}")
    print(f"[dagger] food_radius            = {args.food_radius}")
    print(f"[dagger] output                 = {npz_path}")
    print("=" * 60)

    env = GeckoBrainEnv(
        walker_run=walker_run,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        privileged_target=1.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )

    proprio_dim = int(env.observation_space["proprio"].shape[0])
    expected_proprio_dim = int(train_config["proprio_dim"])
    if proprio_dim != expected_proprio_dim:
        env.close()
        raise ValueError(
            "Env/model proprio_dim mismatch: "
            f"env={proprio_dim} train_config={expected_proprio_dim}"
        )

    N = int(args.num_transitions)
    rng = np.random.default_rng(int(args.seed))
    obs_images = np.empty((N, 64, 64, 3), dtype=np.uint8)
    obs_proprios = np.empty((N, proprio_dim), dtype=np.float32)
    obs_drives = np.empty((N, 6), dtype=np.float32)
    obs_prev_actions = np.empty((N, 4), dtype=np.float32)
    obs_privileged = np.empty((N, 5), dtype=np.float32)
    labels = np.empty((N, 4), dtype=np.float32)

    collected = 0
    episodes = 0
    log_interval = max(1, N // 20)
    action_counts = {
        "policy": 0,
        "noisy_policy": 0,
        "oracle": 0,
        "random": 0,
    }

    obs, _ = env.reset(seed=int(args.seed))
    try:
        while collected < N:
            label = np.asarray(env.oracle_action(), dtype=np.float32)

            obs_images[collected] = obs["image"]
            obs_proprios[collected] = obs["proprio"]
            obs_drives[collected] = obs["drives"]
            obs_prev_actions[collected] = obs["prev_action"]
            obs_privileged[collected] = obs["privileged"]
            labels[collected] = label

            model_action = np.asarray(model.predict(obs), dtype=np.float32)
            step_action, action_kind = _choose_step_action(rng, model_action, label, args)
            action_counts[action_kind] += 1

            collected += 1
            if collected % log_interval == 0:
                print(
                    f"[dagger] {collected}/{N} episodes={episodes} "
                    f"mix_counts={action_counts}"
                )

            obs, _, terminated, truncated, _ = env.step(step_action)
            if terminated or truncated:
                episodes += 1
                obs, _ = env.reset(seed=int(args.seed) + episodes)
    finally:
        env.close()

    print(f"[dagger] done: {collected} transitions, {episodes + 1} episodes")
    print(f"[dagger] action_counts={action_counts}")

    np.savez_compressed(
        npz_path,
        obs_image=obs_images[:collected],
        obs_proprio=obs_proprios[:collected],
        obs_drives=obs_drives[:collected],
        obs_prev_action=obs_prev_actions[:collected],
        obs_privileged=obs_privileged[:collected],
        actions=labels[:collected],
    )
    print(f"[dagger] dataset  -> {npz_path}")

    meta = {
        "dataset_name": args.dataset_name,
        "source_brain_run": args.brain_run,
        "walker_run": walker_run,
        "num_transitions": int(collected),
        "proprio_dim": int(proprio_dim),
        "obs_image_shape": [64, 64, 3],
        "obs_drives_shape": [6],
        "obs_prev_action_shape": [4],
        "obs_privileged_shape": [5],
        "action_dim": 4,
        "food_spawn_angle_deg": float(args.food_spawn_angle_deg),
        "eat_radius": float(args.eat_radius),
        "food_radius": float(args.food_radius),
        "max_steps_per_episode": int(args.max_steps),
        "privileged_target": 1.0,
        "privileged_food_dropout_prob": 0.0,
        "policy_fraction": float(args.policy_fraction),
        "noisy_policy_fraction": float(args.noisy_policy_fraction),
        "oracle_fraction": float(args.oracle_fraction),
        "epsilon": float(args.epsilon),
        "noise_std": float(args.noise_std),
        "seed": int(args.seed),
        "effective_action_mix": effective,
        "action_counts": {k: int(v) for k, v in action_counts.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[dagger] metadata -> {meta_path}")


if __name__ == "__main__":
    main()
