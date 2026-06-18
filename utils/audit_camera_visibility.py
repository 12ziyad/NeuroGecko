from __future__ import annotations

import argparse
import json
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


def food_mask_proxy(image: np.ndarray) -> np.ndarray:
    """Green food marker proxy for the rendered head camera image."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")
    rgb = img[..., :3].astype(np.int32)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (g > 120) & (g > r + 30) & (g > b + 30)


def _save_frame(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(str(path))
        return
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), np.asarray(image, dtype=np.uint8))
    except Exception as exc:
        raise RuntimeError(f"Failed to save debug frame {path}") from exc


def _assert_action_space_4d(env: GeckoBrainEnv) -> None:
    if tuple(env.action_space.shape) != (4,):
        raise RuntimeError(
            "Patch38A requires 4D brain action space "
            "[target_dir_x, target_dir_y, target_distance, engage]; "
            f"got {env.action_space.shape}"
        )


def _step_action(env: GeckoBrainEnv, policy: str, rng: np.random.Generator) -> np.ndarray:
    if policy == "oracle":
        return np.asarray(env.oracle_action(), dtype=np.float32)
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
    if policy == "idle":
        return np.array([1.0, 0.0, -1.0, -1.0], dtype=np.float32)
    raise ValueError(f"Unknown rollout policy: {policy}")


def _summary(pixel_counts: np.ndarray, visible_threshold: int) -> dict[str, float | int]:
    visible = pixel_counts >= int(visible_threshold)
    return {
        "total_samples": int(pixel_counts.size),
        "visible_samples": int(visible.sum()),
        "visible_fraction": float(visible.mean()) if pixel_counts.size else 0.0,
        "mean_visible_pixels": float(np.mean(pixel_counts)) if pixel_counts.size else 0.0,
        "median_visible_pixels": float(np.median(pixel_counts)) if pixel_counts.size else 0.0,
        "min_visible_pixels": int(np.min(pixel_counts)) if pixel_counts.size else 0,
        "max_visible_pixels": int(np.max(pixel_counts)) if pixel_counts.size else 0,
        "visible_pixel_threshold": int(visible_threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether the GeckoBrain head camera can see the food marker."
    )
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--food-spawn-angle-deg", type=float, default=60.0)
    parser.add_argument("--eat-radius", type=float, default=0.10)
    parser.add_argument("--food-radius", type=float, default=0.035)
    parser.add_argument(
        "--rollout-policy",
        choices=["oracle", "random", "idle"],
        default="oracle",
        help="Policy used only to move through camera states during the audit.",
    )
    parser.add_argument(
        "--visible-pixel-threshold",
        type=int,
        default=3,
        help="Minimum green-proxy pixels required to count a sample as food-visible.",
    )
    parser.add_argument(
        "--warn-visible-fraction",
        type=float,
        default=0.10,
        help="Print a warning when visible_fraction is below this value.",
    )
    parser.add_argument("--save-debug-frames", action="store_true")
    parser.add_argument("--debug-frame-count", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO / "renders" / "camera_audit"),
    )
    args = parser.parse_args()

    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be positive")
    if int(args.max_steps) <= 0:
        raise ValueError("--max-steps must be positive")

    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(int(args.seed))
    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        privileged_target=0.0,
        privileged_food_dropout_prob=0.0,
        food_spawn_angle_deg=float(args.food_spawn_angle_deg),
        eat_radius=float(args.eat_radius),
        food_radius=float(args.food_radius),
        render_mode=None,
    )
    _assert_action_space_4d(env)

    pixel_counts = np.empty(int(args.num_samples), dtype=np.int32)
    saved = 0
    episode = 0
    obs, _ = env.reset(seed=int(args.seed))

    print("=" * 72)
    print("[camera audit] NOTE: using green color segmentation as a proxy mask.")
    print("[camera audit] Exact MuJoCo object-id segmentation is not used by this audit.")
    print(f"[camera audit] walker_run={args.walker_run}")
    print(f"[camera audit] food_spawn_angle_deg={args.food_spawn_angle_deg}")
    print(f"[camera audit] eat_radius={args.eat_radius} food_radius={args.food_radius}")
    print(f"[camera audit] rollout_policy={args.rollout_policy}")
    print("=" * 72)

    try:
        for sample_idx in range(int(args.num_samples)):
            image = np.asarray(obs["image"], dtype=np.uint8)
            mask = food_mask_proxy(image)
            pixels = int(mask.sum())
            pixel_counts[sample_idx] = pixels

            if args.save_debug_frames and saved < int(args.debug_frame_count):
                frame_path = out_dir / f"sample_{sample_idx:05d}_green_pixels_{pixels:04d}.png"
                _save_frame(image, frame_path)
                saved += 1

            action = _step_action(env, args.rollout_policy, rng)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                episode += 1
                obs, _ = env.reset(seed=int(args.seed) + episode)
    finally:
        env.close()

    result = _summary(pixel_counts, int(args.visible_pixel_threshold))
    summary_path = out_dir / "camera_visibility_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        "total_samples={total_samples} visible_samples={visible_samples} "
        "visible_fraction={visible_fraction:.6f}".format(**result)
    )
    print(
        "mean_visible_pixels={mean_visible_pixels:.3f} "
        "median_visible_pixels={median_visible_pixels:.3f} "
        "min_visible_pixels={min_visible_pixels} "
        "max_visible_pixels={max_visible_pixels}".format(**result)
    )
    print(f"summary_json={summary_path}")
    if args.save_debug_frames:
        print(f"debug_frames_dir={out_dir}")

    if result["visible_fraction"] < float(args.warn_visible_fraction):
        print(
            "WARNING: camera food visibility is very low under the proxy mask "
            f"(visible_fraction={result['visible_fraction']:.6f})."
        )
    elif result["mean_visible_pixels"] < float(args.visible_pixel_threshold):
        print(
            "WARNING: mean visible pixels is below the visibility threshold; "
            "food may be too small or often out of frame."
        )
    else:
        print("PASS: camera visibility proxy found food in enough samples for preflight.")


if __name__ == "__main__":
    main()
