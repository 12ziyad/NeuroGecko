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


DEFAULT_STUDENT = "brain_v1_patch38b_visual_50k_seed0"
DEFAULT_TEACHER = "brain_v1_patch37b_dagger_200k_seed1"
OBS_KEYS = ("image", "proprio", "drives", "prev_action", "privileged")


def _load_train_config(run_dir: Path, label: str) -> dict:
    config_path = run_dir / "train_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {label} train_config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_actor(run_dir: Path, train_config: dict, *, label: str, force_visual_student: bool) -> BrainBCActor:
    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {label} final.pt: {model_path}")

    algo = str(train_config.get("algo", "behavior_cloning")).lower()
    if algo not in {"behavior_cloning", "visual_distillation"}:
        raise ValueError(f"{label} must be a .pt BrainBCActor run, got algo={algo!r}")

    proprio_dim = int(train_config.get("proprio_dim", 0))
    action_dim = int(train_config.get("action_dim", 4))
    if proprio_dim <= 0:
        raise ValueError(f"{label} train_config missing positive proprio_dim")
    if action_dim != 4:
        raise ValueError(f"{label} action_dim must be 4, got {action_dim}")

    use_privileged = bool(train_config.get("use_privileged_food", True))
    if force_visual_student:
        if algo != "visual_distillation":
            raise ValueError(f"Student must be visual_distillation, got algo={algo!r}")
        if train_config.get("observation_mode") != "visual":
            raise ValueError("Student train_config must declare observation_mode='visual'")
        if use_privileged:
            raise ValueError("Patch38C violation: visual student train_config uses privileged food")
        use_privileged = False

    model = BrainBCActor(
        build_obs_space(proprio_dim),
        image_features_dim=int(train_config.get("image_features_dim", 128)),
        body_features_dim=int(train_config.get("body_features_dim", 96)),
        fused_features_dim=int(train_config.get("fused_features_dim", 256)),
        use_privileged=use_privileged,
        action_dim=4,
    )
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model


def _zero_privileged_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in OBS_KEYS:
        value = np.asarray(obs[key])
        out[key] = value.copy()
    out["privileged"] = np.zeros_like(out["privileged"], dtype=np.float32)
    return out


def _predict_4d(model: BrainBCActor, obs: dict[str, np.ndarray], label: str) -> np.ndarray:
    action = np.asarray(model.predict(obs), dtype=np.float32).reshape(-1)
    if action.shape != (4,):
        raise RuntimeError(f"{label} emitted non-4D action shape {action.shape}")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _action_metrics(student_action: np.ndarray, teacher_action: np.ndarray) -> tuple[float, float, float]:
    student = np.asarray(student_action, dtype=np.float32)
    teacher = np.asarray(teacher_action, dtype=np.float32)
    denom = (float(np.linalg.norm(student[:2])) * float(np.linalg.norm(teacher[:2]))) + 1e-8
    dir_cos = float(np.dot(student[:2], teacher[:2]) / denom)
    dist_err = float(abs(student[2] - teacher[2]))
    engage_err = float(abs(student[3] - teacher[3]))
    return dir_cos, dist_err, engage_err


def _food_visible(image: np.ndarray, visible_pixel_threshold: int) -> float:
    return float(int(food_mask_proxy(image).sum()) >= int(visible_pixel_threshold))


def _safe_random_action(rng: np.random.Generator, student_action: np.ndarray) -> np.ndarray:
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    distance = np.clip(float(student_action[2]) + rng.normal(0.0, 0.25), -1.0, 1.0)
    engage = np.clip(max(float(student_action[3]), 0.0) + rng.normal(0.0, 0.15), -1.0, 1.0)
    return np.array([direction[0], direction[1], distance, engage], dtype=np.float32)


def _choose_step_action(
    rng: np.random.Generator,
    student_action: np.ndarray,
    teacher_action: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    weights = np.array(
        [
            float(args.student_action_fraction),
            float(args.teacher_action_fraction),
            float(args.explore_action_fraction),
        ],
        dtype=np.float64,
    )
    weights = weights / float(weights.sum())
    draw = rng.random()
    if draw < weights[0]:
        return student_action.copy(), "student"
    if draw < weights[0] + weights[1]:
        return teacher_action.copy(), "teacher"
    if args.explore_mode == "random_safe":
        return _safe_random_action(rng, student_action), "random_safe"
    noise = (rng.standard_normal(4) * float(args.noise_std)).astype(np.float32)
    return np.clip(student_action + noise, -1.0, 1.0).astype(np.float32), "noisy_student"


def _validate_mix(args: argparse.Namespace) -> None:
    values = [
        float(args.student_action_fraction),
        float(args.teacher_action_fraction),
        float(args.explore_action_fraction),
    ]
    if any(value < 0.0 for value in values):
        raise ValueError("Action mix fractions must be non-negative")
    if sum(values) <= 0.0:
        raise ValueError("At least one action mix fraction must be positive")
    if float(args.noise_std) < 0.0:
        raise ValueError("--noise-std must be non-negative")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect visual DAgger recovery data from visual-student closed-loop states."
    )
    parser.add_argument("--student-brain-run", type=str, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher-brain-run", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--num-transitions", type=int, default=20_000)
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--visible-pixel-threshold", type=int, default=3)
    parser.add_argument("--student-action-fraction", type=float, default=0.85)
    parser.add_argument("--teacher-action-fraction", type=float, default=0.10)
    parser.add_argument("--explore-action-fraction", type=float, default=0.05)
    parser.add_argument("--explore-mode", choices=["noisy_student", "random_safe"], default="noisy_student")
    parser.add_argument("--noise-std", type=float, default=0.15)
    args = parser.parse_args()

    if int(args.num_transitions) <= 0:
        raise ValueError("--num-transitions must be positive")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")
    _validate_mix(args)

    student_dir = REPO / "models" / "brain" / args.student_brain_run
    teacher_dir = REPO / "models" / "brain" / args.teacher_brain_run
    student_config = _load_train_config(student_dir, "student")
    teacher_config = _load_train_config(teacher_dir, "teacher")
    student = _load_actor(student_dir, student_config, label="student", force_visual_student=True)
    teacher = _load_actor(teacher_dir, teacher_config, label="teacher", force_visual_student=False)

    out_dir = REPO / "data" / "visual_distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.dataset_name}.npz"
    meta_path = out_dir / f"{args.dataset_name}_meta.json"

    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        privileged_target=1.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )
    if tuple(env.action_space.shape) != (4,):
        env.close()
        raise RuntimeError(f"Patch38C requires 4D action space, got {env.action_space.shape}")

    proprio_dim = int(env.observation_space["proprio"].shape[0])
    for label, config in (("student", student_config), ("teacher", teacher_config)):
        expected = int(config.get("proprio_dim", proprio_dim))
        if expected != proprio_dim:
            env.close()
            raise ValueError(f"Env/{label} proprio_dim mismatch: env={proprio_dim} {label}={expected}")

    N = int(args.num_transitions)
    obs_images = np.empty((N, 64, 64, 3), dtype=np.uint8)
    obs_proprios = np.empty((N, proprio_dim), dtype=np.float32)
    obs_drives = np.empty((N, 6), dtype=np.float32)
    obs_prev_actions = np.empty((N, 4), dtype=np.float32)
    obs_privileged = np.zeros((N, 5), dtype=np.float32)
    labels = np.empty((N, 4), dtype=np.float32)
    food_visible = np.empty((N,), dtype=np.float32)
    episode_ids = np.empty((N,), dtype=np.int32)
    step_ids = np.empty((N,), dtype=np.int32)

    rng = np.random.default_rng(int(args.seed))
    action_source_counts = {
        "student": 0,
        "teacher": 0,
        "noisy_student": 0,
        "random_safe": 0,
    }
    dir_cosines: list[float] = []
    dist_errors: list[float] = []
    engage_errors: list[float] = []
    eat_count = 0
    min_mouth_food_dist = float("inf")
    collected = 0
    episode = 0
    step_id = 0
    log_interval = max(1, N // 20)

    print("=" * 72)
    print(f"[visual dagger] student_brain_run={args.student_brain_run}")
    print(f"[visual dagger] teacher_brain_run={args.teacher_brain_run}")
    print(f"[visual dagger] walker_run={args.walker_run}")
    print(f"[visual dagger] dataset_name={args.dataset_name}")
    print(
        "[visual dagger] action_mix="
        f"student:{args.student_action_fraction} "
        f"teacher:{args.teacher_action_fraction} "
        f"explore:{args.explore_action_fraction} ({args.explore_mode})"
    )
    print("[visual dagger] student obs_privileged will be all zeros")
    print("[visual dagger] NOTE: food_visible uses green color proxy mask.")
    print("=" * 72)

    obs, info = env.reset(seed=int(args.seed))
    if "mouth_food_dist" in info:
        min_mouth_food_dist = min(min_mouth_food_dist, float(info["mouth_food_dist"]))
    try:
        while collected < N:
            student_obs = _zero_privileged_obs(obs)
            student_action = _predict_4d(student, student_obs, "student")
            teacher_action = _predict_4d(teacher, obs, "teacher")
            dir_cos, dist_err, engage_err = _action_metrics(student_action, teacher_action)
            dir_cosines.append(dir_cos)
            dist_errors.append(dist_err)
            engage_errors.append(engage_err)

            image = np.asarray(student_obs["image"], dtype=np.uint8)
            obs_images[collected] = image
            obs_proprios[collected] = student_obs["proprio"]
            obs_drives[collected] = student_obs["drives"]
            obs_prev_actions[collected] = student_obs["prev_action"]
            labels[collected] = teacher_action
            food_visible[collected] = _food_visible(image, int(args.visible_pixel_threshold))
            episode_ids[collected] = int(episode)
            step_ids[collected] = int(step_id)

            step_action, source = _choose_step_action(rng, student_action, teacher_action, args)
            action_source_counts[source] += 1

            collected += 1
            if collected % log_interval == 0:
                print(
                    f"[visual dagger] {collected}/{N} episodes={episode + 1} "
                    f"mix={action_source_counts}"
                )

            obs, _, terminated, truncated, info = env.step(step_action)
            eat_count += int(bool(info.get("ate", False)))
            mfd = info.get("mouth_food_dist")
            if mfd is not None:
                min_mouth_food_dist = min(min_mouth_food_dist, float(mfd))
            step_id += 1
            if terminated or truncated:
                episode += 1
                step_id = 0
                obs, info = env.reset(seed=int(args.seed) + episode)
                if "mouth_food_dist" in info:
                    min_mouth_food_dist = min(min_mouth_food_dist, float(info["mouth_food_dist"]))
    finally:
        env.close()

    if min_mouth_food_dist == float("inf"):
        min_mouth_food_dist = float("nan")
    if not np.allclose(obs_privileged[:collected], 0.0):
        raise RuntimeError("Patch38C violation: saved obs_privileged is nonzero")
    if labels[:collected].ndim != 2 or labels[:collected].shape[1] != 4:
        raise RuntimeError(f"Patch38C violation: labels/actions shape is {labels[:collected].shape}")

    np.savez_compressed(
        npz_path,
        obs_image=obs_images[:collected],
        obs_proprio=obs_proprios[:collected],
        obs_drives=obs_drives[:collected],
        obs_prev_action=obs_prev_actions[:collected],
        obs_privileged=obs_privileged[:collected],
        actions=labels[:collected],
        food_visible=food_visible[:collected],
        episode_id=episode_ids[:collected],
        step_id=step_ids[:collected],
    )

    food_visible_fraction = float(np.mean(food_visible[:collected])) if collected else 0.0
    mean_dir_cos = float(np.mean(dir_cosines)) if dir_cosines else float("nan")
    mean_dist_err = float(np.mean(dist_errors)) if dist_errors else float("nan")
    mean_engage_err = float(np.mean(engage_errors)) if engage_errors else float("nan")
    meta = {
        "dataset_name": args.dataset_name,
        "dataset_type": "visual_dagger_recovery",
        "observation_mode": "visual",
        "train_obs": "visual",
        "use_privileged_food_student": False,
        "use_privileged_food": False,
        "student_brain_run": args.student_brain_run,
        "teacher_brain_run": args.teacher_brain_run,
        "teacher_algo": teacher_config.get("algo", "unknown"),
        "teacher_use_privileged_food": bool(teacher_config.get("use_privileged_food", True)),
        "walker_run": args.walker_run,
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
        "food_visible_shape": [],
        "visible_pixel_threshold": int(args.visible_pixel_threshold),
        "food_visible_fraction": food_visible_fraction,
        "student_action_fraction": float(args.student_action_fraction),
        "teacher_action_fraction": float(args.teacher_action_fraction),
        "explore_action_fraction": float(args.explore_action_fraction),
        "explore_mode": args.explore_mode,
        "noise_std": float(args.noise_std),
        "action_source_counts": {k: int(v) for k, v in action_source_counts.items()},
        "mean_student_teacher_dir_cos": mean_dir_cos,
        "mean_student_teacher_dist_error": mean_dist_err,
        "mean_student_teacher_engage_error": mean_engage_err,
        "eat_count": int(eat_count),
        "min_mouth_food_dist": float(min_mouth_food_dist),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[visual dagger] total_transitions={collected}")
    print(f"[visual dagger] episodes={episode + 1}")
    print(f"[visual dagger] action_source_counts={action_source_counts}")
    print(f"[visual dagger] food_visible_fraction={food_visible_fraction:.6f}")
    print(f"[visual dagger] mean_student_teacher_dir_cos={mean_dir_cos:.6f}")
    print(f"[visual dagger] mean_dist_error={mean_dist_err:.6f}")
    print(f"[visual dagger] mean_engage_error={mean_engage_err:.6f}")
    print(f"[visual dagger] eat_count={eat_count}")
    print(f"[visual dagger] min_mouth_food_dist={min_mouth_food_dist:.6f}")
    print(f"[visual dagger] dataset  -> {npz_path}")
    print(f"[visual dagger] metadata -> {meta_path}")
    print("[visual dagger] PASS: saved obs_privileged is all zeros")
    print("[visual dagger] PASS: labels/actions are 4D")


if __name__ == "__main__":
    main()
