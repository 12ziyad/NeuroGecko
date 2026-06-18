from __future__ import annotations

import argparse
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
from utils.collect_visual_dagger_dataset import (
    DEFAULT_TEACHER,
    _action_metrics,
    _food_visible,
    _load_actor,
    _load_train_config,
    _predict_4d,
    _zero_privileged_obs,
)


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop visual-student diagnostics against the privileged teacher. "
            "Teacher is used only for diagnostics labels, never for student actions."
        )
    )
    parser.add_argument("--student-brain-run", type=str, required=True)
    parser.add_argument("--teacher-brain-run", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--visible-pixel-threshold", type=int, default=3)
    args = parser.parse_args()

    student_dir = REPO / "models" / "brain" / args.student_brain_run
    teacher_dir = REPO / "models" / "brain" / args.teacher_brain_run
    student_config = _load_train_config(student_dir, "student")
    teacher_config = _load_train_config(teacher_dir, "teacher")
    student = _load_actor(student_dir, student_config, label="student", force_visual_student=True)
    teacher = _load_actor(teacher_dir, teacher_config, label="teacher", force_visual_student=False)

    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        max_steps=int(args.steps),
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

    print("=" * 72)
    print(f"[debug visual] student_brain_run={args.student_brain_run}")
    print(f"[debug visual] teacher_brain_run={args.teacher_brain_run}")
    print("[debug visual] student privileged food OFF; teacher diagnostics only")
    print("=" * 72)

    dir_cosines: list[float] = []
    dist_errors: list[float] = []
    engage_errors: list[float] = []
    food_visible: list[float] = []
    eat_count = 0
    falls = 0
    min_mouth_food_dist = float("inf")
    total_steps = 0

    try:
        for ep in range(int(args.episodes)):
            obs, info = env.reset(seed=int(args.seed) + ep)
            if "mouth_food_dist" in info:
                min_mouth_food_dist = min(min_mouth_food_dist, float(info["mouth_food_dist"]))

            for _ in range(int(args.steps)):
                student_obs = _zero_privileged_obs(obs)
                student_action = _predict_4d(student, student_obs, "student")
                teacher_action = _predict_4d(teacher, obs, "teacher")
                dcos, derr, eerr = _action_metrics(student_action, teacher_action)
                dir_cosines.append(dcos)
                dist_errors.append(derr)
                engage_errors.append(eerr)
                food_visible.append(_food_visible(student_obs["image"], int(args.visible_pixel_threshold)))

                obs, _, terminated, truncated, info = env.step(student_action)
                total_steps += 1
                eat_count += int(bool(info.get("ate", False)))
                falls += int(bool(info.get("fallen", False)))
                mfd = info.get("mouth_food_dist")
                if mfd is not None:
                    min_mouth_food_dist = min(min_mouth_food_dist, float(mfd))
                if terminated or truncated:
                    break
    finally:
        env.close()

    if min_mouth_food_dist == float("inf"):
        min_mouth_food_dist = float("nan")

    print(f"[debug visual] compared_steps={total_steps}")
    print(f"[debug visual] closed_loop_mean_dir_cos={_mean_or_nan(dir_cosines):.6f}")
    print(f"[debug visual] closed_loop_dist_err={_mean_or_nan(dist_errors):.6f}")
    print(f"[debug visual] closed_loop_engage_err={_mean_or_nan(engage_errors):.6f}")
    print(f"[debug visual] food_visible_frac={_mean_or_nan(food_visible):.6f}")
    print(f"[debug visual] eat_count={eat_count}")
    print(f"[debug visual] falls={falls}")
    print(f"[debug visual] min_mouth_food_dist={min_mouth_food_dist:.6f}")
    print("[debug visual] PASS: diagnostics did not feed privileged food to student action")


if __name__ == "__main__":
    main()
