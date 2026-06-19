from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
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


DEFAULT_APPROACH_RUN = "brain_v1_patch38c_visual_dagger_80k_seed0"
OBS_KEYS = ("image", "proprio", "drives", "prev_action", "privileged")


class GreenPixelFoodDetector:
    detector_type = "green_pixel_proxy"

    def __init__(self, visible_pixel_threshold: int = 3, confidence_pixels: int = 24):
        self.visible_pixel_threshold = int(visible_pixel_threshold)
        self.confidence_pixels = max(1, int(confidence_pixels))

    def confidence(self, obs: dict[str, np.ndarray]) -> float:
        pixels = int(food_mask_proxy(np.asarray(obs["image"], dtype=np.uint8)).sum())
        return float(np.clip(pixels / float(self.confidence_pixels), 0.0, 1.0))

    def visible(self, obs: dict[str, np.ndarray]) -> bool:
        pixels = int(food_mask_proxy(np.asarray(obs["image"], dtype=np.uint8)).sum())
        return pixels >= self.visible_pixel_threshold


@dataclass
class SearchPolicy:
    pattern: str
    dir_x: float
    dir_y: float
    target_distance: float
    commit_frames: int
    step_count: int = 0

    def action(self) -> np.ndarray:
        sign = 1.0
        if self.pattern == "alternating_sweep":
            sign = 1.0 if (self.step_count // max(1, self.commit_frames)) % 2 == 0 else -1.0
        direction = np.array([self.dir_x, sign * self.dir_y], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([0.0, 1.0], dtype=np.float32)
        else:
            direction = direction / norm
        self.step_count += 1
        return np.array(
            [
                float(direction[0]),
                float(direction[1]),
                float(np.clip(self.target_distance, -1.0, 1.0)),
                -1.0,
            ],
            dtype=np.float32,
        )


def _load_train_config(run_dir: Path) -> dict:
    path = run_dir / "train_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing train_config.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_visual_actor(run_name: str) -> tuple[BrainBCActor, dict]:
    run_dir = REPO / "models" / "brain" / run_name
    train_config = _load_train_config(run_dir)
    algo = str(train_config.get("algo", "")).lower()
    if algo != "visual_distillation":
        raise ValueError(f"Approach model must be visual_distillation, got algo={algo!r}")
    if train_config.get("observation_mode") != "visual":
        raise ValueError("Approach model train_config must declare observation_mode='visual'")
    if bool(train_config.get("use_privileged_food", True)):
        raise ValueError("Patch39A violation: approach model uses privileged food")
    action_dim = int(train_config.get("action_dim", 0))
    if action_dim != 4:
        raise ValueError(f"Patch39A requires action_dim=4, got {action_dim}")
    proprio_dim = int(train_config.get("proprio_dim", 0))
    if proprio_dim <= 0:
        raise ValueError("Approach model train_config missing positive proprio_dim")

    model_path = run_dir / "final.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing approach model final.pt: {model_path}")
    model = BrainBCActor(
        build_obs_space(proprio_dim),
        image_features_dim=int(train_config.get("image_features_dim", 128)),
        body_features_dim=int(train_config.get("body_features_dim", 96)),
        fused_features_dim=int(train_config.get("fused_features_dim", 256)),
        use_privileged=False,
        action_dim=4,
    )
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model, train_config


def _zero_privileged_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in OBS_KEYS:
        out[key] = np.asarray(obs[key]).copy()
    out["privileged"] = np.zeros_like(out["privileged"], dtype=np.float32)
    return out


def _assert_zero_privileged(obs: dict[str, np.ndarray], context: str) -> None:
    privileged = np.asarray(obs.get("privileged", np.zeros(1)), dtype=np.float32)
    if privileged.size and not np.allclose(privileged, 0.0):
        raise RuntimeError(
            f"Patch39A violation: nonzero privileged obs at {context}; "
            f"max_abs={float(np.max(np.abs(privileged))):.8f}"
        )


def _predict_approach(model: BrainBCActor, obs: dict[str, np.ndarray]) -> np.ndarray:
    safe_obs = _zero_privileged_obs(obs)
    _assert_zero_privileged(safe_obs, "approach model input")
    action = np.asarray(model.predict(safe_obs), dtype=np.float32).reshape(-1)
    if action.shape != (4,):
        raise RuntimeError(f"Approach model emitted non-4D action shape {action.shape}")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _episode_metrics_template() -> dict[str, object]:
    return {
        "eat_count": 0,
        "falls": 0,
        "visible_frames": 0,
        "search_success": False,
        "time_to_first_visible": None,
        "search_steps": 0,
        "approach_steps": 0,
        "state_transition_count": 0,
        "hungers": [],
        "min_mouth_food_dist": float("inf"),
        "steps": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scripted visual search + Patch38C approach/eat.")
    parser.add_argument("--approach-brain-run", type=str, default=DEFAULT_APPROACH_RUN)
    parser.add_argument("--walker-run", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--visible-pixel-threshold", type=int, default=3)
    parser.add_argument("--confidence-pixels", type=int, default=24)
    parser.add_argument("--visible-enter-frames", type=int, default=3)
    parser.add_argument("--invisible-exit-frames", type=int, default=15)
    parser.add_argument("--search-pattern", choices=["always_left_arc", "alternating_sweep"], default="always_left_arc")
    parser.add_argument("--search-dir-x", type=float, default=0.35)
    parser.add_argument("--search-dir-y", type=float, default=1.0)
    parser.add_argument("--search-target-distance", type=float, default=-0.50)
    parser.add_argument("--search-commit-frames", type=int, default=30)
    args = parser.parse_args()

    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    if int(args.visible_enter_frames) <= 0:
        raise ValueError("--visible-enter-frames must be positive")
    if int(args.invisible_exit_frames) <= 0:
        raise ValueError("--invisible-exit-frames must be positive")

    model, train_config = _load_visual_actor(args.approach_brain_run)
    walker_run = args.walker_run or train_config.get("walker_run", "v4_5b_speed_polish_1m")
    detector = GreenPixelFoodDetector(
        visible_pixel_threshold=int(args.visible_pixel_threshold),
        confidence_pixels=int(args.confidence_pixels),
    )

    print("=" * 88)
    print("FINAL/VISUAL SEARCH MODE")
    print("privileged food OFF")
    print("oracle OFF")
    print("action_dim=4")
    print(f"detector type={detector.detector_type}")
    print(f"visibility threshold={detector.visible_pixel_threshold}")
    print(f"approach model run name={args.approach_brain_run}")
    print(
        "search pattern parameters="
        f"pattern:{args.search_pattern} dir_x:{args.search_dir_x} dir_y:{args.search_dir_y} "
        f"target_distance:{args.search_target_distance} commit_frames:{args.search_commit_frames} "
        "engage:0"
    )
    print(f"food_spawn_angle_deg={args.food_spawn_angle_deg}")
    print("=" * 88)

    aggregate: list[dict[str, object]] = []
    for ep in range(int(args.episodes)):
        env = GeckoBrainEnv(
            walker_run=walker_run,
            max_steps=int(args.steps),
            seed=int(args.seed) + ep,
            privileged_target=0.0,
            privileged_food_dropout_prob=0.0,
            food_spawn_angle_deg=float(args.food_spawn_angle_deg),
            eat_radius=float(args.eat_radius),
            food_radius=float(args.food_radius),
            render_mode=None,
        )
        if tuple(env.action_space.shape) != (4,):
            env.close()
            raise RuntimeError(f"Patch39A requires 4D action space, got {env.action_space.shape}")

        metrics = _episode_metrics_template()
        search_policy = SearchPolicy(
            pattern=args.search_pattern,
            dir_x=float(args.search_dir_x),
            dir_y=float(args.search_dir_y),
            target_distance=float(args.search_target_distance),
            commit_frames=int(args.search_commit_frames),
        )
        state = "SEARCH"
        visible_streak = 0
        invisible_streak = 0

        try:
            obs, info = env.reset(seed=int(args.seed) + ep)
            _assert_zero_privileged(obs, f"episode {ep + 1} reset")
            if "mouth_food_dist" in info:
                metrics["min_mouth_food_dist"] = min(
                    float(metrics["min_mouth_food_dist"]), float(info["mouth_food_dist"])
                )

            for step_idx in range(int(args.steps)):
                visible_now = detector.visible(obs)
                if visible_now:
                    visible_streak += 1
                    invisible_streak = 0
                    metrics["visible_frames"] = int(metrics["visible_frames"]) + 1
                    if not bool(metrics["search_success"]):
                        metrics["search_success"] = True
                        metrics["time_to_first_visible"] = step_idx
                else:
                    invisible_streak += 1
                    visible_streak = 0

                prev_state = state
                if state == "SEARCH" and visible_streak >= int(args.visible_enter_frames):
                    state = "APPROACH"
                elif state == "APPROACH" and invisible_streak >= int(args.invisible_exit_frames):
                    state = "SEARCH"
                if state != prev_state:
                    metrics["state_transition_count"] = int(metrics["state_transition_count"]) + 1

                if state == "APPROACH":
                    action = _predict_approach(model, obs)
                    metrics["approach_steps"] = int(metrics["approach_steps"]) + 1
                else:
                    action = search_policy.action()
                    if float(action[3]) != -1.0:
                        raise RuntimeError("Patch39A violation: search engage must be 0")
                    metrics["search_steps"] = int(metrics["search_steps"]) + 1

                if action.shape != (4,):
                    raise RuntimeError(f"Patch39A violation: action shape is {action.shape}")
                obs, _, terminated, truncated, info = env.step(action)
                _assert_zero_privileged(obs, f"episode {ep + 1} step {step_idx + 1}")
                metrics["steps"] = int(metrics["steps"]) + 1
                metrics["eat_count"] = int(metrics["eat_count"]) + int(bool(info.get("ate", False)))
                metrics["falls"] = int(metrics["falls"]) + int(bool(info.get("fallen", False)))
                metrics["hungers"].append(float(info.get("hunger", 0.0)))
                mfd = info.get("mouth_food_dist")
                if mfd is not None:
                    metrics["min_mouth_food_dist"] = min(
                        float(metrics["min_mouth_food_dist"]), float(mfd)
                    )
                if terminated or truncated:
                    break
        finally:
            env.close()

        steps = max(int(metrics["steps"]), 1)
        search_frac = int(metrics["search_steps"]) / float(steps)
        approach_frac = int(metrics["approach_steps"]) / float(steps)
        food_visible_frac = int(metrics["visible_frames"]) / float(steps)
        if float(metrics["min_mouth_food_dist"]) == float("inf"):
            metrics["min_mouth_food_dist"] = float("nan")
        handoff_success = bool(metrics["search_success"]) and int(metrics["eat_count"]) > 0
        episode_summary = {
            "episode": ep + 1,
            "eat_count": int(metrics["eat_count"]),
            "falls": int(metrics["falls"]),
            "food_visible_frac": food_visible_frac,
            "search_success": bool(metrics["search_success"]),
            "time_to_first_visible": metrics["time_to_first_visible"],
            "handoff_success": handoff_success,
            "search_frac": search_frac,
            "approach_frac": approach_frac,
            "state_transition_count": int(metrics["state_transition_count"]),
            "mean_hunger": _mean_or_nan(metrics["hungers"]),
            "min_mouth_food_dist": float(metrics["min_mouth_food_dist"]),
        }
        aggregate.append(episode_summary)
        print(
            "episode={episode} eat_count={eat_count} falls={falls} "
            "food_visible_frac={food_visible_frac:.6f} search_success={search_success} "
            "time_to_first_visible={time_to_first_visible} handoff_success={handoff_success} "
            "search_frac={search_frac:.6f} approach_frac={approach_frac:.6f} "
            "state_transition_count={state_transition_count} mean_hunger={mean_hunger:.6f} "
            "min_mouth_food_dist={min_mouth_food_dist:.6f}".format(**episode_summary)
        )

    visible_times = [
        float(item["time_to_first_visible"])
        for item in aggregate
        if item["time_to_first_visible"] is not None
    ]
    search_successes = [bool(item["search_success"]) for item in aggregate]
    handoff_candidates = [item for item in aggregate if bool(item["search_success"])]
    summary = {
        "angle": float(args.food_spawn_angle_deg),
        "episodes": int(args.episodes),
        "episodes_with_eat": int(sum(int(item["eat_count"]) > 0 for item in aggregate)),
        "total_eats": int(sum(int(item["eat_count"]) for item in aggregate)),
        "search_success_rate": float(np.mean(search_successes)) if search_successes else 0.0,
        "mean_time_to_first_visible": float(np.mean(visible_times)) if visible_times else float("nan"),
        "handoff_success_rate": (
            float(np.mean([bool(item["handoff_success"]) for item in handoff_candidates]))
            if handoff_candidates
            else float("nan")
        ),
        "falls": int(sum(int(item["falls"]) for item in aggregate)),
        "mean_search_frac": _mean_or_nan([float(item["search_frac"]) for item in aggregate]),
        "mean_approach_frac": _mean_or_nan([float(item["approach_frac"]) for item in aggregate]),
        "transition_count": int(sum(int(item["state_transition_count"]) for item in aggregate)),
    }
    print(
        "aggregate angle={angle:.1f} episodes={episodes} episodes_with_eat={episodes_with_eat} "
        "total_eats={total_eats} search_success_rate={search_success_rate:.6f} "
        "mean_time_to_first_visible={mean_time_to_first_visible:.6f} "
        "handoff_success_rate={handoff_success_rate:.6f} falls={falls} "
        "mean_search_frac={mean_search_frac:.6f} mean_approach_frac={mean_approach_frac:.6f} "
        "transition_count={transition_count}".format(**summary)
    )
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
