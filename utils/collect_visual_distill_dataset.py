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
from utils.audit_camera_visibility import food_mask_proxy


DEFAULT_TEACHER = "brain_v1_patch37b_dagger_200k_seed1"


def _load_train_config(run_dir: Path) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing teacher train_config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_teacher(run_dir: Path, train_config: dict) -> BrainBCActor:
    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing Patch37B teacher final.pt. Expected: "
            f"{model_path}"
        )

    algo = str(train_config.get("algo", "behavior_cloning")).lower()
    if algo not in {"behavior_cloning", "visual_distillation"}:
        raise ValueError(
            "Visual distillation collector expects a .pt BrainBCActor teacher; "
            f"got algo={train_config.get('algo')!r}"
        )

    proprio_dim = int(train_config.get("proprio_dim", 0))
    action_dim = int(train_config.get("action_dim", 4))
    if proprio_dim <= 0:
        raise ValueError("Teacher train_config missing positive proprio_dim")
    if action_dim != 4:
        raise ValueError(f"Teacher action_dim must be 4, got {action_dim}")

    model = BrainBCActor(
        build_obs_space(proprio_dim),
        image_features_dim=int(train_config.get("image_features_dim", 128)),
        body_features_dim=int(train_config.get("body_features_dim", 96)),
        fused_features_dim=int(train_config.get("fused_features_dim", 256)),
        use_privileged=bool(train_config.get("use_privileged_food", True)),
        action_dim=4,
    )
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model


def _assert_action_space_4d(env: GeckoBrainEnv) -> None:
    if tuple(env.action_space.shape) != (4,):
        raise RuntimeError(
            "Patch38A requires 4D brain action space "
            "[target_dir_x, target_dir_y, target_distance, engage]; "
            f"got {env.action_space.shape}"
        )


def _teacher_action(model: BrainBCActor, obs: dict[str, np.ndarray]) -> np.ndarray:
    action = np.asarray(model.predict(obs), dtype=np.float32).reshape(-1)
    if action.shape != (4,):
        raise RuntimeError(f"Teacher emitted non-4D action shape {action.shape}")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _target_food(env: GeckoBrainEnv) -> np.ndarray:
    ego, dist, _ = env.food_egocentric()
    return np.array([float(ego[0]), float(ego[1]), float(dist)], dtype=np.float32)


def _food_visible(image: np.ndarray, visible_pixel_threshold: int) -> float:
    return float(int(food_mask_proxy(image).sum()) >= int(visible_pixel_threshold))


def _step_action(
    rng: np.random.Generator,
    teacher_action: np.ndarray,
    random_fraction: float,
    noise_std: float,
) -> np.ndarray:
    if rng.random() < float(random_fraction):
        return rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
    if float(noise_std) > 0.0:
        noise = (rng.standard_normal(4) * float(noise_std)).astype(np.float32)
        return np.clip(teacher_action + noise, -1.0, 1.0).astype(np.float32)
    return teacher_action.copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect visual student observations labeled by the Patch37B teacher."
    )
    parser.add_argument("--teacher-brain-run", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--num-transitions", type=int, default=10_000)
    parser.add_argument("--walker-run", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--visible-pixel-threshold",
        type=int,
        default=3,
        help="Green-proxy pixel count required to mark food_visible=1.",
    )
    parser.add_argument(
        "--random-action-fraction",
        type=float,
        default=0.0,
        help="Optional exploration fraction for stepping only. Labels remain teacher actions.",
    )
    parser.add_argument(
        "--teacher-action-noise-std",
        type=float,
        default=0.0,
        help="Optional Gaussian noise on step actions only. Labels remain clean teacher actions.",
    )
    args = parser.parse_args()

    if int(args.num_transitions) <= 0:
        raise ValueError("--num-transitions must be positive")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    if not (0.0 <= float(args.random_action_fraction) <= 1.0):
        raise ValueError("--random-action-fraction must be in [0, 1]")
    if float(args.teacher_action_noise_std) < 0.0:
        raise ValueError("--teacher-action-noise-std must be non-negative")

    teacher_dir = REPO / "models" / "brain" / args.teacher_brain_run
    teacher_config = _load_train_config(teacher_dir)
    teacher = _load_teacher(teacher_dir, teacher_config)
    walker_run = args.walker_run or teacher_config.get("walker_run", "v4_5b_speed_polish_1m")

    out_dir = REPO / "data" / "visual_distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.dataset_name}.npz"
    meta_path = out_dir / f"{args.dataset_name}_meta.json"

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
    _assert_action_space_4d(env)

    proprio_dim = int(env.observation_space["proprio"].shape[0])
    teacher_proprio_dim = int(teacher_config.get("proprio_dim", proprio_dim))
    if proprio_dim != teacher_proprio_dim:
        env.close()
        raise ValueError(
            "Env/teacher proprio_dim mismatch: "
            f"env={proprio_dim} teacher={teacher_proprio_dim}"
        )

    N = int(args.num_transitions)
    obs_images = np.empty((N, 64, 64, 3), dtype=np.uint8)
    obs_proprios = np.empty((N, proprio_dim), dtype=np.float32)
    obs_drives = np.empty((N, 6), dtype=np.float32)
    obs_prev_actions = np.empty((N, 4), dtype=np.float32)
    obs_privileged = np.zeros((N, 5), dtype=np.float32)
    labels = np.empty((N, 4), dtype=np.float32)
    target_food = np.empty((N, 3), dtype=np.float32)
    food_visible = np.empty((N,), dtype=np.float32)
    episode_ids = np.empty((N,), dtype=np.int32)
    step_ids = np.empty((N,), dtype=np.int32)

    rng = np.random.default_rng(int(args.seed))
    collected = 0
    episode = 0
    step_id = 0
    log_interval = max(1, N // 20)

    print("=" * 72)
    print(f"[visual distill collect] teacher_brain_run={args.teacher_brain_run}")
    print(f"[visual distill collect] walker_run={walker_run}")
    print(f"[visual distill collect] dataset_name={args.dataset_name}")
    print(f"[visual distill collect] num_transitions={N}")
    print("[visual distill collect] student obs_privileged will be all zeros")
    print("[visual distill collect] NOTE: food_visible uses green color proxy mask.")
    print("=" * 72)

    obs, _ = env.reset(seed=int(args.seed))
    try:
        while collected < N:
            action = _teacher_action(teacher, obs)
            image = np.asarray(obs["image"], dtype=np.uint8)

            obs_images[collected] = image
            obs_proprios[collected] = obs["proprio"]
            obs_drives[collected] = obs["drives"]
            obs_prev_actions[collected] = obs["prev_action"]
            labels[collected] = action
            target_food[collected] = _target_food(env)
            food_visible[collected] = _food_visible(image, int(args.visible_pixel_threshold))
            episode_ids[collected] = int(episode)
            step_ids[collected] = int(step_id)

            collected += 1
            if collected % log_interval == 0:
                print(f"[visual distill collect] {collected}/{N} episodes={episode + 1}")

            step_action = _step_action(
                rng,
                action,
                float(args.random_action_fraction),
                float(args.teacher_action_noise_std),
            )
            obs, _, terminated, truncated, _ = env.step(step_action)
            step_id += 1
            if terminated or truncated:
                episode += 1
                step_id = 0
                obs, _ = env.reset(seed=int(args.seed) + episode)
    finally:
        env.close()

    if not np.allclose(obs_privileged[:collected], 0.0):
        raise RuntimeError("Patch38A violation: student obs_privileged is nonzero")
    if labels[:collected].ndim != 2 or labels[:collected].shape[1] != 4:
        raise RuntimeError(f"Patch38A violation: labels shape is {labels[:collected].shape}")

    np.savez_compressed(
        npz_path,
        obs_image=obs_images[:collected],
        obs_proprio=obs_proprios[:collected],
        obs_drives=obs_drives[:collected],
        obs_prev_action=obs_prev_actions[:collected],
        obs_privileged=obs_privileged[:collected],
        actions=labels[:collected],
        target_food=target_food[:collected],
        food_visible=food_visible[:collected],
        episode_id=episode_ids[:collected],
        step_id=step_ids[:collected],
    )

    meta = {
        "dataset_name": args.dataset_name,
        "observation_mode": "visual",
        "use_privileged_food_student": False,
        "use_privileged_food": False,
        "teacher_brain_run": args.teacher_brain_run,
        "teacher_algo": teacher_config.get("algo", "unknown"),
        "teacher_use_privileged_food": bool(teacher_config.get("use_privileged_food", True)),
        "walker_run": walker_run,
        "food_spawn_angle_deg": float(args.food_spawn_angle_deg),
        "eat_radius": float(args.eat_radius),
        "food_radius": float(args.food_radius),
        "max_steps_per_episode": int(args.max_steps),
        "seed": int(args.seed),
        "num_transitions": int(collected),
        "proprio_dim": int(proprio_dim),
        "obs_image_shape": [64, 64, 3],
        "obs_drives_shape": [6],
        "obs_prev_action_shape": [4],
        "obs_privileged_shape": [5],
        "action_dim": 4,
        "target_food_shape": [3],
        "food_visible_shape": [],
        "visible_pixel_threshold": int(args.visible_pixel_threshold),
        "food_visible_fraction": float(np.mean(food_visible[:collected])) if collected else 0.0,
        "random_action_fraction": float(args.random_action_fraction),
        "teacher_action_noise_std": float(args.teacher_action_noise_std),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[visual distill collect] done: {collected} transitions, episodes={episode + 1}")
    print(f"[visual distill collect] dataset  -> {npz_path}")
    print(f"[visual distill collect] metadata -> {meta_path}")
    print(f"[visual distill collect] food_visible_fraction={meta['food_visible_fraction']:.6f}")
    print("[visual distill collect] PASS: student obs_privileged is all zeros")


if __name__ == "__main__":
    main()
