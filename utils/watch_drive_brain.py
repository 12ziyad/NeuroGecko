from __future__ import annotations

import argparse
import json
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
from utils.watch_search_brain import (
    DEFAULT_APPROACH_RUN,
    GreenPixelFoodDetector,
    SearchPolicy,
    _load_visual_actor,
    _predict_approach,
)


BEHAVIORS = (
    "REST",
    "EXPLORE",
    "SEARCH_FOOD",
    "APPROACH_FOOD",
    "EAT",
    "RECOVER",
    "FREEZE_AVOID",
)
FORAGE_BEHAVIORS = {"SEARCH_FOOD", "APPROACH_FOOD"}


@dataclass
class DriveSnapshot:
    hunger: float
    energy: float
    fear: float
    curiosity: float
    danger: float
    target_interest: float


@dataclass
class DriveHomeostat:
    hunger: float
    energy: float
    fear: float
    curiosity: float
    danger: float
    target_interest: float
    hunger_rise_per_step: float
    eat_hunger_drop: float
    eat_energy_gain: float
    rest_energy_recover_per_step: float
    explore_energy_cost_per_step: float
    search_energy_cost_per_step: float
    approach_energy_cost_per_step: float
    baseline_energy_decay_per_step: float

    @classmethod
    def from_obs(cls, obs: dict[str, np.ndarray], args: argparse.Namespace) -> "DriveHomeostat":
        drives = _drive_snapshot(obs)
        hunger = drives.hunger if args.initial_hunger is None else float(args.initial_hunger)
        energy = drives.energy if args.initial_energy is None else float(args.initial_energy)
        return cls(
            hunger=float(np.clip(hunger, 0.0, 1.0)),
            energy=float(np.clip(energy, 0.0, 1.0)),
            fear=drives.fear,
            curiosity=drives.curiosity,
            danger=drives.danger,
            target_interest=drives.target_interest,
            hunger_rise_per_step=float(args.hunger_rise_per_step),
            eat_hunger_drop=float(args.eat_hunger_drop),
            eat_energy_gain=float(args.eat_energy_gain),
            rest_energy_recover_per_step=float(args.rest_energy_recover_per_step),
            explore_energy_cost_per_step=float(args.explore_energy_cost_per_step),
            search_energy_cost_per_step=float(args.search_energy_cost_per_step),
            approach_energy_cost_per_step=float(args.approach_energy_cost_per_step),
            baseline_energy_decay_per_step=float(args.baseline_energy_decay_per_step),
        )

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.hunger,
                self.energy,
                self.fear,
                self.curiosity,
                self.danger,
                self.target_interest,
            ],
            dtype=np.float32,
        )

    def apply_to_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {key: np.asarray(value).copy() for key, value in obs.items()}
        out["drives"] = self.vector()
        return out

    def update(self, behavior: str, *, ate: bool, info: dict) -> None:
        self.hunger += self.hunger_rise_per_step
        self.curiosity += 0.25 * self.hunger_rise_per_step
        if ate:
            self.hunger -= self.eat_hunger_drop
            self.energy += self.eat_energy_gain
            self.curiosity -= 0.10

        if behavior == "REST":
            self.energy += self.rest_energy_recover_per_step
        elif behavior == "EXPLORE":
            self.energy -= self.explore_energy_cost_per_step
        elif behavior == "SEARCH_FOOD":
            self.energy -= self.search_energy_cost_per_step
        elif behavior == "APPROACH_FOOD":
            self.energy -= self.approach_energy_cost_per_step

        moving_speed = float(info.get("moving_speed", 0.0))
        moving_cost = 0.0005 * float(np.clip(moving_speed / 0.25, 0.0, 1.0))
        self.energy -= moving_cost + self.baseline_energy_decay_per_step

        danger_signal = float(np.clip(info.get("danger", 0.0), 0.0, 1.0))
        self.danger += 0.15 * (danger_signal - self.danger)
        self.fear += 0.05 * self.danger
        if danger_signal <= 1e-6:
            self.fear -= 0.02

        self.hunger = float(np.clip(self.hunger, 0.0, 1.0))
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        self.curiosity = float(np.clip(self.curiosity, 0.0, 1.0))
        self.danger = float(np.clip(self.danger, 0.0, 1.0))
        self.fear = float(np.clip(self.fear, 0.0, 1.0))
        self.target_interest = float(
            np.clip(0.70 * self.hunger + 0.25 * self.curiosity - 0.35 * self.fear, 0.0, 1.0)
        )


def _drive_snapshot(obs: dict[str, np.ndarray]) -> DriveSnapshot:
    drives = np.asarray(obs["drives"], dtype=np.float32).reshape(-1)
    if drives.size < 6:
        raise RuntimeError(f"Expected drives vector with at least 6 values, got {drives.shape}")
    return DriveSnapshot(
        hunger=float(drives[0]),
        energy=float(drives[1]),
        fear=float(drives[2]),
        curiosity=float(drives[3]),
        danger=float(drives[4]),
        target_interest=float(drives[5]),
    )


def _rest_action() -> np.ndarray:
    return np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32)


def _assert_action_4d(action: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape != (4,):
        raise RuntimeError(f"Patch40A violation: {context} action shape is {arr.shape}")
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def _assert_zero_privileged(obs: dict[str, np.ndarray], context: str) -> None:
    privileged = np.asarray(obs.get("privileged", np.zeros(1)), dtype=np.float32)
    if privileged.size and not np.allclose(privileged, 0.0):
        raise RuntimeError(
            f"Patch40A violation: nonzero privileged observation at {context}; "
            f"max_abs={float(np.max(np.abs(privileged))):.8f}"
        )


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _behavior_fractions(counts: dict[str, int], steps: int) -> dict[str, float]:
    denom = float(max(steps, 1))
    return {name: float(counts.get(name, 0) / denom) for name in BEHAVIORS}


class DriveArbiter:
    def __init__(
        self,
        *,
        detector: GreenPixelFoodDetector,
        approach_model,
        force_behavior: str | None,
        hungry_threshold: float,
        sated_threshold: float,
        energy_critical_threshold: float,
        energy_rest_exit_threshold: float,
        visible_enter_frames: int,
        invisible_exit_frames: int,
        search_policy: SearchPolicy,
        explore_policy: SearchPolicy,
    ):
        self.detector = detector
        self.approach_model = approach_model
        self.force_behavior = force_behavior
        self.hungry_threshold = float(hungry_threshold)
        self.sated_threshold = float(sated_threshold)
        self.energy_critical_threshold = float(energy_critical_threshold)
        self.energy_rest_exit_threshold = float(energy_rest_exit_threshold)
        self.visible_enter_frames = int(visible_enter_frames)
        self.invisible_exit_frames = int(invisible_exit_frames)
        self.search_policy = search_policy
        self.explore_policy = explore_policy

        self.hungry_latched = False
        self.rest_latched = False
        self.food_locked = False
        self.visible_streak = 0
        self.invisible_streak = 0
        self.time_since_food_seen: int | None = None
        self.current_behavior = "EXPLORE"
        self.behavior_dwell_steps = 0
        self.behavior_transition_count = 0

    def _update_visibility(self, obs: dict[str, np.ndarray]) -> tuple[bool, float]:
        visible = self.detector.visible(obs)
        confidence = self.detector.confidence(obs)
        if visible:
            self.visible_streak += 1
            self.invisible_streak = 0
            self.time_since_food_seen = 0
        else:
            self.invisible_streak += 1
            self.visible_streak = 0
            if self.time_since_food_seen is None:
                self.time_since_food_seen = 1
            else:
                self.time_since_food_seen += 1

        if self.visible_streak >= self.visible_enter_frames:
            self.food_locked = True
        elif self.invisible_streak >= self.invisible_exit_frames:
            self.food_locked = False
        return visible, confidence

    def _update_drive_latches(self, drives: DriveSnapshot) -> None:
        if self.hungry_latched:
            if drives.hunger <= self.sated_threshold:
                self.hungry_latched = False
        elif drives.hunger >= self.hungry_threshold:
            self.hungry_latched = True

        if self.rest_latched:
            if drives.energy >= self.energy_rest_exit_threshold:
                self.rest_latched = False
        elif drives.energy <= self.energy_critical_threshold:
            self.rest_latched = True

    def choose(self, obs: dict[str, np.ndarray]) -> tuple[str, np.ndarray, bool, float]:
        _assert_zero_privileged(obs, "drive arbiter input")
        drives = _drive_snapshot(obs)
        visible, confidence = self._update_visibility(obs)
        self._update_drive_latches(drives)

        if self.force_behavior is not None:
            behavior = self.force_behavior
        elif self.rest_latched:
            behavior = "REST"
        elif self.hungry_latched:
            behavior = "APPROACH_FOOD" if self.food_locked else "SEARCH_FOOD"
        else:
            behavior = "EXPLORE"

        # RECOVER and FREEZE_AVOID are deliberately inactive in Patch40A. The
        # current env exposes belly/fall/danger scalars, but no recovery or
        # predator-specific policy has been grounded yet.
        if behavior != self.current_behavior:
            self.current_behavior = behavior
            self.behavior_dwell_steps = 0
            self.behavior_transition_count += 1
        self.behavior_dwell_steps += 1

        if behavior == "REST":
            action = _rest_action()
        elif behavior == "EXPLORE":
            action = self.explore_policy.action()
        elif behavior == "SEARCH_FOOD":
            action = self.search_policy.action()
        elif behavior == "APPROACH_FOOD":
            if self.approach_model is None:
                raise RuntimeError("APPROACH_FOOD requested but no approach model was loaded")
            action = _predict_approach(self.approach_model, obs)
        elif behavior in {"RECOVER", "FREEZE_AVOID"}:
            action = _rest_action()
        else:
            raise RuntimeError(f"Unknown behavior {behavior!r}")

        action = _assert_action_4d(action, behavior)
        if behavior in {"REST", "EXPLORE", "SEARCH_FOOD"} and float(action[3]) != -1.0:
            raise RuntimeError(f"Patch40A violation: {behavior} must have engage=0")
        return behavior, action, visible, confidence


def _make_controller(args: argparse.Namespace, approach_model) -> DriveArbiter:
    detector = GreenPixelFoodDetector(
        visible_pixel_threshold=int(args.visible_pixel_threshold),
        confidence_pixels=int(args.confidence_pixels),
    )
    search_policy = SearchPolicy(
        pattern=args.search_pattern,
        dir_x=float(args.search_dir_x),
        dir_y=float(args.search_dir_y),
        target_distance=float(args.search_target_distance),
        commit_frames=int(args.search_commit_frames),
    )
    explore_policy = SearchPolicy(
        pattern=args.explore_pattern,
        dir_x=float(args.explore_dir_x),
        dir_y=float(args.explore_dir_y),
        target_distance=float(args.explore_target_distance),
        commit_frames=int(args.explore_commit_frames),
    )
    force_behavior = None if args.force_behavior == "AUTO" else args.force_behavior
    return DriveArbiter(
        detector=detector,
        approach_model=approach_model,
        force_behavior=force_behavior,
        hungry_threshold=float(args.hungry_threshold),
        sated_threshold=float(args.sated_threshold),
        energy_critical_threshold=float(args.energy_critical_threshold),
        energy_rest_exit_threshold=float(args.energy_rest_exit_threshold),
        visible_enter_frames=int(args.visible_enter_frames),
        invisible_exit_frames=int(args.invisible_exit_frames),
        search_policy=search_policy,
        explore_policy=explore_policy,
    )


def _run_episode(
    args: argparse.Namespace,
    approach_model,
    walker_run: str,
    episode_idx: int,
) -> dict[str, object]:
    env = GeckoBrainEnv(
        walker_run=walker_run,
        max_steps=int(args.steps),
        seed=int(args.seed) + episode_idx,
        privileged_target=0.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )
    if tuple(env.action_space.shape) != (4,):
        env.close()
        raise RuntimeError(f"Patch40A requires 4D action space, got {env.action_space.shape}")

    controller = _make_controller(args, approach_model)
    behavior_counts = {name: 0 for name in BEHAVIORS}
    visible_frames = 0
    hunger_values: list[float] = []
    energy_values: list[float] = []
    hunger_after_eat_values: list[float] = []
    food_conf_values: list[float] = []
    search_or_approach_after_eat_seen = False
    cycle_phase = "await_eat"
    closed_loop_cycles = 0
    eat_count = 0
    falls = 0
    min_mouth_food_dist = float("inf")
    initial_xy = np.zeros(2, dtype=np.float64)
    final_xy = np.zeros(2, dtype=np.float64)
    first_behavior: str | None = None

    try:
        obs, info = env.reset(seed=int(args.seed) + episode_idx)
        homeostat = DriveHomeostat.from_obs(obs, args)
        obs = homeostat.apply_to_obs(obs)
        _assert_zero_privileged(obs, f"episode {episode_idx + 1} reset")
        initial_xy = env.walk_env.data.xpos[env.walk_env._trunk][:2].copy()
        if "mouth_food_dist" in info:
            min_mouth_food_dist = min(min_mouth_food_dist, float(info["mouth_food_dist"]))

        for step_idx in range(int(args.steps)):
            behavior, action, visible, confidence = controller.choose(obs)
            selected_behavior = behavior
            if first_behavior is None:
                first_behavior = selected_behavior
            if visible:
                visible_frames += 1
            food_conf_values.append(float(confidence))

            obs, _, terminated, truncated, info = env.step(action)
            homeostat.update(selected_behavior, ate=bool(info.get("ate", False)), info=info)
            obs = homeostat.apply_to_obs(obs)
            _assert_zero_privileged(obs, f"episode {episode_idx + 1} step {step_idx + 1}")
            drives = _drive_snapshot(obs)
            hunger_values.append(drives.hunger)
            energy_values.append(drives.energy)
            falls += int(bool(info.get("fallen", False)))
            ate = bool(info.get("ate", False))
            if ate:
                eat_count += 1
                behavior_counts["EAT"] += 1
                hunger_after_eat_values.append(drives.hunger)
                cycle_phase = "await_nonforage"
            else:
                behavior_counts[selected_behavior] += 1

            mfd = info.get("mouth_food_dist")
            if mfd is not None:
                min_mouth_food_dist = min(min_mouth_food_dist, float(mfd))

            if cycle_phase == "await_nonforage" and selected_behavior not in FORAGE_BEHAVIORS:
                cycle_phase = "await_reforage"
            elif cycle_phase == "await_reforage" and selected_behavior in FORAGE_BEHAVIORS:
                closed_loop_cycles += 1
                search_or_approach_after_eat_seen = True
                cycle_phase = "await_eat"

            if int(args.trace_every) > 0 and (step_idx % int(args.trace_every) == 0):
                print(
                    "trace episode={ep} step={step} current_behavior={behavior} "
                    "behavior_dwell_steps={dwell} hunger={hunger:.4f} energy={energy:.4f} "
                    "time_since_food_seen={tsfs} visible={visible} confidence={conf:.4f}".format(
                        ep=episode_idx + 1,
                        step=step_idx,
                        behavior=controller.current_behavior,
                        dwell=controller.behavior_dwell_steps,
                        hunger=drives.hunger,
                        energy=drives.energy,
                        tsfs=controller.time_since_food_seen,
                        visible=visible,
                        conf=confidence,
                    )
                )

            if terminated or truncated:
                break

        final_xy = env.walk_env.data.xpos[env.walk_env._trunk][:2].copy()
    finally:
        env.close()

    steps = max(sum(behavior_counts.values()), 1)
    if min_mouth_food_dist == float("inf"):
        min_mouth_food_dist = float("nan")
    fractions = _behavior_fractions(behavior_counts, steps)
    final_hunger = hunger_values[-1] if hunger_values else float("nan")
    final_energy = energy_values[-1] if energy_values else float("nan")
    summary = {
        "episode": int(episode_idx + 1),
        "steps": int(steps),
        "first_behavior": first_behavior or "unknown",
        "current_behavior": controller.current_behavior,
        "behavior_dwell_steps": int(controller.behavior_dwell_steps),
        "behavior_transition_count": int(controller.behavior_transition_count),
        "behavior_counts": {key: int(value) for key, value in behavior_counts.items()},
        "behavior_fractions": fractions,
        "hunger": float(final_hunger),
        "energy": float(final_energy),
        "mean_hunger": _mean_or_nan(hunger_values),
        "mean_energy": _mean_or_nan(energy_values),
        "time_since_food_seen": controller.time_since_food_seen,
        "food_visible_frac": float(visible_frames / float(steps)),
        "mean_food_confidence": _mean_or_nan(food_conf_values),
        "eat_count": int(eat_count),
        "falls": int(falls),
        "min_mouth_food_dist": float(min_mouth_food_dist),
        "closed_loop_cycles": int(closed_loop_cycles),
        "search_or_approach_after_eat_seen": bool(search_or_approach_after_eat_seen),
        "rest_drift": float(np.linalg.norm(final_xy - initial_xy)),
        "energy_recovered": bool(
            len(energy_values) >= 2 and max(energy_values) > energy_values[0] + 1e-6
        ),
        "hunger_after_eat_mean": _mean_or_nan(hunger_after_eat_values),
    }
    return summary


def _aggregate(summaries: list[dict[str, object]]) -> dict[str, object]:
    total_steps = max(sum(int(item["steps"]) for item in summaries), 1)
    behavior_counts = {name: 0 for name in BEHAVIORS}
    for item in summaries:
        counts = item["behavior_counts"]
        for name in BEHAVIORS:
            behavior_counts[name] += int(counts.get(name, 0))
    behavior_fractions = _behavior_fractions(behavior_counts, total_steps)
    return {
        "episodes": len(summaries),
        "steps": int(total_steps),
        "eat_count": int(sum(int(item["eat_count"]) for item in summaries)),
        "episodes_with_eat": int(sum(int(item["eat_count"]) > 0 for item in summaries)),
        "falls": int(sum(int(item["falls"]) for item in summaries)),
        "behavior_counts": behavior_counts,
        "behavior_fractions": behavior_fractions,
        "transition_count": int(sum(int(item["behavior_transition_count"]) for item in summaries)),
        "closed_loop_cycles": int(sum(int(item["closed_loop_cycles"]) for item in summaries)),
        "food_visible_frac": _mean_or_nan([float(item["food_visible_frac"]) for item in summaries]),
        "mean_hunger": _mean_or_nan([float(item["mean_hunger"]) for item in summaries]),
        "final_hunger": float(summaries[-1]["hunger"]) if summaries else float("nan"),
        "mean_energy": _mean_or_nan([float(item["mean_energy"]) for item in summaries]),
        "final_energy": float(summaries[-1]["energy"]) if summaries else float("nan"),
        "min_mouth_food_dist": float(
            np.nanmin([float(item["min_mouth_food_dist"]) for item in summaries])
        )
        if summaries
        else float("nan"),
        "mean_rest_drift": _mean_or_nan([float(item["rest_drift"]) for item in summaries]),
        "energy_recovered_any": bool(any(bool(item["energy_recovered"]) for item in summaries)),
        "episodes_started_rest": int(
            sum(str(item.get("first_behavior")) == "REST" for item in summaries)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grounded drive-based GeckoBrain arbiter.")
    parser.add_argument("--approach-brain-run", type=str, default=DEFAULT_APPROACH_RUN)
    parser.add_argument("--walker-run", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument("--initial-hunger", type=float, default=None)
    parser.add_argument("--initial-energy", type=float, default=None)
    parser.add_argument("--hungry-threshold", type=float, default=0.45)
    parser.add_argument("--sated-threshold", type=float, default=0.25)
    parser.add_argument("--energy-critical-threshold", type=float, default=0.25)
    parser.add_argument("--energy-rest-exit-threshold", type=float, default=0.45)
    parser.add_argument("--hunger-rise-per-step", type=float, default=0.0005)
    parser.add_argument("--eat-hunger-drop", type=float, default=0.65)
    parser.add_argument("--eat-energy-gain", type=float, default=0.08)
    parser.add_argument("--rest-energy-recover-per-step", type=float, default=0.0020)
    parser.add_argument("--explore-energy-cost-per-step", type=float, default=0.00025)
    parser.add_argument("--search-energy-cost-per-step", type=float, default=0.00080)
    parser.add_argument("--approach-energy-cost-per-step", type=float, default=0.00100)
    parser.add_argument("--baseline-energy-decay-per-step", type=float, default=0.0)
    parser.add_argument("--visible-pixel-threshold", type=int, default=3)
    parser.add_argument("--confidence-pixels", type=int, default=24)
    parser.add_argument("--visible-enter-frames", type=int, default=3)
    parser.add_argument("--invisible-exit-frames", type=int, default=15)
    parser.add_argument("--search-pattern", choices=["always_left_arc", "alternating_sweep"], default="always_left_arc")
    parser.add_argument("--search-dir-x", type=float, default=0.35)
    parser.add_argument("--search-dir-y", type=float, default=1.0)
    parser.add_argument("--search-target-distance", type=float, default=-0.50)
    parser.add_argument("--search-commit-frames", type=int, default=30)
    parser.add_argument("--explore-pattern", choices=["always_left_arc", "alternating_sweep"], default="alternating_sweep")
    parser.add_argument("--explore-dir-x", type=float, default=0.55)
    parser.add_argument("--explore-dir-y", type=float, default=0.55)
    parser.add_argument("--explore-target-distance", type=float, default=-0.75)
    parser.add_argument("--explore-commit-frames", type=int, default=80)
    parser.add_argument(
        "--force-behavior",
        choices=["AUTO", "REST", "EXPLORE", "SEARCH_FOOD", "APPROACH_FOOD"],
        default="AUTO",
    )
    parser.add_argument("--trace-every", type=int, default=0)
    args = parser.parse_args()

    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive")
    if int(args.steps) <= 0:
        raise ValueError("--steps must be positive")
    if float(args.sated_threshold) > float(args.hungry_threshold):
        raise ValueError("--sated-threshold must be <= --hungry-threshold")
    if float(args.energy_rest_exit_threshold) < float(args.energy_critical_threshold):
        raise ValueError("--energy-rest-exit-threshold must be >= --energy-critical-threshold")

    needs_model = args.force_behavior in ("AUTO", "APPROACH_FOOD")
    approach_model = None
    train_config: dict[str, object] = {}
    if needs_model:
        approach_model, train_config = _load_visual_actor(args.approach_brain_run)
    walker_run = args.walker_run or train_config.get("walker_run", "v4_5b_speed_polish_1m")

    print("=" * 88)
    print("FINAL/VISUAL DRIVE MODE")
    print("privileged food OFF")
    print("oracle OFF")
    print("action_dim=4")
    print(f"approach model run name={args.approach_brain_run if needs_model else 'not_loaded'}")
    print("detector type=green_pixel_proxy")
    print(f"visibility threshold={args.visible_pixel_threshold}")
    print(
        "drive thresholds="
        f"hungry:{args.hungry_threshold} sated:{args.sated_threshold} "
        f"energy_critical:{args.energy_critical_threshold} "
        f"energy_rest_exit:{args.energy_rest_exit_threshold}"
    )
    print(
        "drive rates="
        f"hunger_rise:{args.hunger_rise_per_step} eat_hunger_drop:{args.eat_hunger_drop} "
        f"rest_energy_recover:{args.rest_energy_recover_per_step} "
        f"explore_cost:{args.explore_energy_cost_per_step} "
        f"search_cost:{args.search_energy_cost_per_step} "
        f"approach_cost:{args.approach_energy_cost_per_step}"
    )
    print(
        "search params="
        f"pattern:{args.search_pattern} dir_x:{args.search_dir_x} dir_y:{args.search_dir_y} "
        f"target_distance:{args.search_target_distance} commit:{args.search_commit_frames}"
    )
    print(
        "explore params="
        f"pattern:{args.explore_pattern} dir_x:{args.explore_dir_x} dir_y:{args.explore_dir_y} "
        f"target_distance:{args.explore_target_distance} commit:{args.explore_commit_frames}"
    )
    print("RECOVER inactive: no grounded recovery action beyond fall termination is wired in Patch40A")
    print("FREEZE_AVOID inactive: no predator/avoidance signal is present in this environment")
    print("=" * 88)

    summaries = []
    for ep in range(int(args.episodes)):
        summary = _run_episode(args, approach_model, str(walker_run), ep)
        summaries.append(summary)
        fractions = summary["behavior_fractions"]
        print(
            "episode={episode} first_behavior={first_behavior} current_behavior={current_behavior} "
            "behavior_dwell_steps={behavior_dwell_steps} behavior_transition_count={behavior_transition_count} "
            "REST={REST:.4f} EXPLORE={EXPLORE:.4f} SEARCH_FOOD={SEARCH_FOOD:.4f} "
            "APPROACH_FOOD={APPROACH_FOOD:.4f} EAT={EAT:.4f} RECOVER={RECOVER:.4f} "
            "FREEZE_AVOID={FREEZE_AVOID:.4f} hunger={hunger:.4f} energy={energy:.4f} "
            "time_since_food_seen={time_since_food_seen} food_visible_frac={food_visible_frac:.6f} "
            "eat_count={eat_count} falls={falls} min_mouth_food_dist={min_mouth_food_dist:.6f} "
            "closed_loop_cycles={closed_loop_cycles} rest_drift={rest_drift:.6f}".format(
                **summary,
                **fractions,
            )
        )

    aggregate = _aggregate(summaries)
    print(
        "aggregate episodes={episodes} steps={steps} eats={eat_count} falls={falls} "
        "transition_count={transition_count} closed_loop_cycles={closed_loop_cycles} "
        "food_visible_frac={food_visible_frac:.6f} mean_hunger={mean_hunger:.6f} "
        "final_hunger={final_hunger:.6f} mean_energy={mean_energy:.6f} "
        "final_energy={final_energy:.6f} min_mouth_food_dist={min_mouth_food_dist:.6f} "
        "mean_rest_drift={mean_rest_drift:.6f}".format(**aggregate)
    )
    print("SUMMARY_JSON=" + json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
