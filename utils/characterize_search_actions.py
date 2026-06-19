from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from envs.gecko_brain_env import GeckoBrainEnv


@dataclass(frozen=True)
class SearchPattern:
    name: str
    action: np.ndarray
    commit_frames: int = 1
    alternate: bool = False


def _heading(env: GeckoBrainEnv) -> float:
    rot = env._trunk_rot()
    forward = rot[:, 0]
    return float(math.atan2(forward[1], forward[0]))


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _trunk_xy(env: GeckoBrainEnv) -> np.ndarray:
    return env.walk_env.data.xpos[env.walk_env._trunk][:2].copy()


def _default_patterns(commit_frames: int) -> list[SearchPattern]:
    engage_off = -1.0
    return [
        SearchPattern(
            name="near_in_place_yaw_left",
            action=np.array([0.0, 1.0, -0.75, engage_off], dtype=np.float32),
        ),
        SearchPattern(
            name="near_in_place_yaw_right",
            action=np.array([0.0, -1.0, -0.75, engage_off], dtype=np.float32),
        ),
        SearchPattern(
            name="constant_curvature_left_arc",
            action=np.array([0.35, 1.0, -0.35, engage_off], dtype=np.float32),
        ),
        SearchPattern(
            name="constant_curvature_right_arc",
            action=np.array([0.35, -1.0, -0.35, engage_off], dtype=np.float32),
        ),
        SearchPattern(
            name="alternating_sweep",
            action=np.array([0.20, 1.0, -0.50, engage_off], dtype=np.float32),
            commit_frames=max(1, int(commit_frames)),
            alternate=True,
        ),
    ]


def _pattern_action(pattern: SearchPattern, step_idx: int) -> np.ndarray:
    action = pattern.action.copy()
    if pattern.alternate and (step_idx // max(1, pattern.commit_frames)) % 2 == 1:
        action[1] *= -1.0
    return action


def _run_pattern(args: argparse.Namespace, pattern: SearchPattern, seed: int) -> dict[str, float | int | str]:
    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        max_steps=int(args.steps),
        seed=int(seed),
        privileged_target=0.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )
    if tuple(env.action_space.shape) != (4,):
        env.close()
        raise RuntimeError(f"Patch39A requires 4D brain action space, got {env.action_space.shape}")

    yaw_start = 0.0
    yaw_end = 0.0
    xy_start = np.zeros(2, dtype=np.float64)
    xy_end = np.zeros(2, dtype=np.float64)
    belly_contacts: list[float] = []
    speeds: list[float] = []
    falls = 0
    steps_run = 0
    dt = 0.5

    try:
        env.reset(seed=seed)
        dt = float(env.walk_env.dt)
        yaw_start = _heading(env)
        xy_start = _trunk_xy(env)
        for step_idx in range(int(args.steps)):
            action = _pattern_action(pattern, step_idx)
            obs, _, terminated, truncated, info = env.step(action)
            privileged = np.asarray(obs.get("privileged", np.zeros(1)), dtype=np.float32)
            if privileged.size and not np.allclose(privileged, 0.0):
                raise RuntimeError("Patch39A violation: characterization env emitted privileged obs")
            belly_contacts.append(float(info.get("belly_contact", 0.0)))
            if "moving_speed" in info:
                speeds.append(float(info["moving_speed"]))
            elif "walker_forward_speed" in info:
                speeds.append(float(info["walker_forward_speed"]))
            falls += int(bool(info.get("fallen", False)))
            steps_run += 1
            if terminated or truncated:
                break
        yaw_end = _heading(env)
        xy_end = _trunk_xy(env)
    finally:
        env.close()

    yaw_delta = _wrap_angle(yaw_end - yaw_start)
    duration = max(steps_run, 1) * max(dt, 1e-9)
    translation = float(np.linalg.norm(xy_end - xy_start))
    belly_rate = float(np.mean(belly_contacts)) if belly_contacts else 0.0
    min_speed = float(np.min(speeds)) if speeds else float("nan")
    max_speed = float(np.max(speeds)) if speeds else float("nan")
    stable = falls == 0 and belly_rate <= float(args.max_belly_contact_rate)
    return {
        "pattern": pattern.name,
        "steps_run": int(steps_run),
        "yaw_delta": float(yaw_delta),
        "mean_yaw_rate": float(yaw_delta / duration),
        "translation_distance": translation,
        "falls": int(falls),
        "belly_contact_rate": belly_rate,
        "min_speed": min_speed,
        "max_speed": max_speed,
        "final_heading": float(yaw_end),
        "stability": "PASS" if stable else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Characterize frozen-walker response to scripted 4D search actions."
    )
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--commit-frames", type=int, default=30)
    parser.add_argument("--max-belly-contact-rate", type=float, default=0.05)
    args = parser.parse_args()

    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")

    print("=" * 88)
    print("[search characterize] privileged food OFF")
    print("[search characterize] oracle OFF")
    print("[search characterize] action_dim=4")
    print(f"[search characterize] walker_run={args.walker_run}")
    print(f"[search characterize] steps={args.steps}")
    print("=" * 88)
    print(
        "pattern steps yaw_delta mean_yaw_rate translation_distance "
        "falls belly_contact_rate min_speed max_speed final_heading stability"
    )

    for idx, pattern in enumerate(_default_patterns(int(args.commit_frames))):
        result = _run_pattern(args, pattern, int(args.seed) + idx)
        print(
            "{pattern} {steps_run} {yaw_delta:.6f} {mean_yaw_rate:.6f} "
            "{translation_distance:.6f} {falls} {belly_contact_rate:.6f} "
            "{min_speed:.6f} {max_speed:.6f} {final_heading:.6f} {stability}".format(**result)
        )


if __name__ == "__main__":
    main()
