"""Save one 64x64 head-camera frame (enlarged) to ~/Downloads for visual food-marker verification."""
from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from envs.gecko_brain_env import GeckoBrainEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one head-camera frame to Downloads for food visibility check")
    parser.add_argument("--walker-run", type=str, default="v4_5b_speed_polish_1m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=8, help="Integer upscale factor (default 8 → 512x512)")
    parser.add_argument("--out", type=str, default=None, help="Override output path (default: ~/Downloads/brain_headcam_food_debug.png)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else Path.home() / "Downloads" / "brain_headcam_food_debug.png"

    print(f"[dump_brain_camera] walker_run = {args.walker_run}")
    print(f"[dump_brain_camera] seed       = {args.seed}")

    env = GeckoBrainEnv(
        walker_run=args.walker_run,
        seed=args.seed,
        render_mode=None,
        privileged_target=1.0,
    )
    try:
        env.reset(seed=args.seed)
        raw = env._head_cam_image()  # HxWx3 uint8, food sphere injected
    finally:
        env.close()

    h, w = raw.shape[:2]
    upscale = max(1, int(args.scale))
    img = Image.fromarray(raw, mode="RGB")
    img_big = img.resize((w * upscale, h * upscale), resample=Image.NEAREST)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_big.save(str(out_path))

    print(f"[dump_brain_camera] raw size   = {w}x{h}")
    print(f"[dump_brain_camera] saved {w*upscale}x{h*upscale} -> {out_path}")

    unique_colors = len(np.unique(raw.reshape(-1, 3), axis=0))
    green_pixels = int(np.sum(
        (raw[:, :, 1].astype(int) > 150)
        & (raw[:, :, 0].astype(int) < 80)
        & (raw[:, :, 2].astype(int) < 80)
    ))
    print(f"[dump_brain_camera] unique colors={unique_colors}  green_pixels(food)={green_pixels}")
    if green_pixels > 0:
        print("[dump_brain_camera] PASS: food marker visible in head camera")
    else:
        print("[dump_brain_camera] WARN: no obvious green pixels found — food may be out of frame")


if __name__ == "__main__":
    main()
