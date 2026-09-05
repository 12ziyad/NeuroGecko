"""Replay recorded generalized positions for a human movie, without physics steps.

Every displayed pose is an existing trace sample. mj_forward derives render
transforms; this module never calls mj_step and never executes a controller.
Camera options/scene effects are local to this human renderer. Policy inputs,
model dynamics, source XML, source trace, and earlier videos are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.checkpoints import atomic_json, sha256_file


def load_replay(trace_path: Path, xml_path: Path):
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    metadata, samples = trace["metadata"], trace["samples"]
    if metadata.get("schema_version") != 1:
        raise ValueError("Only version1 recorded gait traces are supported")
    if metadata.get("xml_sha256") != sha256_file(xml_path):
        raise ValueError("XML hash differs from the recorded trace; supply its exact archived XML")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    times = np.asarray(samples["time_s"], dtype=float)
    qpos = np.asarray(samples["qpos"], dtype=float)
    qvel = np.asarray(samples["qvel"], dtype=float) if "qvel" in samples else None
    if times.ndim != 1 or times.size < 2 or qpos.shape != (times.size, model.nq):
        raise ValueError("Trace times/qpos have incompatible dimensions")
    if not np.isfinite(times).all() or not np.isfinite(qpos).all() or not (np.diff(times) > 0).all():
        raise ValueError("Trace must contain finite poses and strictly increasing times")
    if qvel is not None and (qvel.shape != (times.size, model.nv) or not np.isfinite(qvel).all()):
        raise ValueError("Recorded velocity array is malformed")
    return model, metadata, times, qpos, qvel


def frame_indices(times, fps, duration):
    """Select nearest actual samples; do not interpolate or synthesize qpos."""
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not (np.diff(times) > 0).all() or not math.isfinite(fps) or fps <= 0
            or not math.isfinite(duration) or duration <= 0
            or duration > times[-1]-times[0]+1e-8):
        raise ValueError("Use finite increasing samples and an in-range positive duration/rate")
    count = int(round(duration * fps))
    if count < 1 or not math.isclose(count/fps, duration, abs_tol=1e-8):
        raise ValueError("Duration must be a positive whole number of video frames")
    requested = times[0] + np.arange(count, dtype=float)/fps
    right = np.clip(np.searchsorted(times, requested), 0, len(times)-1)
    left = np.maximum(right-1, 0)
    return np.where(np.abs(times[left]-requested) <= np.abs(times[right]-requested), left, right)


def caption_image(frame, text, font):
    if not text:
        return frame
    picture = Image.fromarray(frame)
    draw = ImageDraw.Draw(picture)
    width, height = picture.size
    draw.rectangle((0, 0, width, 48), fill=(18, 25, 31))
    draw.text((20, 13), text, font=font, fill=(245, 247, 250))
    return np.asarray(picture)


def render_replay(trace_path, xml_path, output, preview, *, fps=25,
                  width=960, height=540, max_wall_seconds=120., caption=""):
    if fps <= 0 or width < 128 or height < 128 or width % 2 or height % 2:
        raise ValueError("Use positive fps and even video dimensions >=128")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be finite and positive")
    trace_path, xml_path = Path(trace_path).resolve(strict=True), Path(xml_path).resolve(strict=True)
    output, preview = Path(output).resolve(), Path(preview).resolve()
    if output.suffix.lower() != '.mp4' or preview.suffix.lower() != '.png':
        raise ValueError("Use an .mp4 video and .png preview")
    if any(path.exists() for path in (output, preview, output.with_suffix('.json'))):
        raise FileExistsError("Refusing to overwrite an existing video, preview or evidence sidecar")
    started = time.monotonic()
    trace_hash, xml_hash = sha256_file(trace_path), sha256_file(xml_path)
    model, metadata, times, qpos, qvel = load_replay(trace_path, xml_path)
    duration = float(times[-1]-times[0])
    if duration > 60:
        raise ValueError("This presentation helper is bounded to at most60 recorded seconds")
    indices = frame_indices(times, fps, duration)
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    trunk = model.body("trunk_middle").id
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth = 135.
    camera.elevation = -23.
    camera.distance = .40  # Human presentation choice; not an animal measurement.
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[0] = options.geomgroup[1] = 1
    options.sitegroup[:] = 0
    options.jointgroup[:] = 0
    options.tendongroup[:] = 0
    options.actuatorgroup[:] = 0
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = ImageFont.truetype(str(font_path), 19) if font_path.exists() else ImageFont.load_default()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.parent.mkdir(parents=True, exist_ok=True)
    rendering = None
    writer = None
    evidence = {
        "kind": "recorded_qpos_human_replay", "complete": False,
        "trace_path": str(trace_path), "trace_sha256": trace_hash,
        "xml_path": str(xml_path), "xml_sha256": xml_hash,
        "controller_recorded": metadata.get("controller"),
        "gait_profile_recorded": metadata.get("gait_profile"),
        "fps": fps, "duration_s": duration, "planned_frames": len(indices),
        "rendered_frames": 0, "physics_steps_executed": 0,
        "sampling": "nearest existing recorded qpos; no interpolation/controller/optimization",
        "human_render": {"sites": False, "collision_geoms": False,
                         "shadows": False, "reflections": False, "body_visible": True},
        "caption": caption,
    }
    try:
        rendering = mujoco.Renderer(model, height, width)
        writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_params=["-threads", "1"])
        for frame_number, sample_index in enumerate(indices):
            if time.monotonic()-started >= max_wall_seconds:
                raise TimeoutError("Replay rendering exceeded its wall-time limit")
            data.qpos[:] = qpos[sample_index]
            data.qvel[:] = qvel[sample_index] if qvel is not None else 0.
            data.time = times[sample_index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[trunk]
            rendering.update_scene(data, camera=camera, scene_option=options)
            rendering.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            rendering.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            frame = caption_image(rendering.render(), caption, font)
            writer.append_data(frame)
            evidence["rendered_frames"] += 1
            if frame_number == len(indices)//2:
                imageio.imwrite(preview, frame)
            if frame_number % fps == 0:
                print(f"[replay] frame={frame_number}/{len(indices)}", flush=True)
        writer.close()
        writer = None
        if sha256_file(trace_path) != trace_hash or sha256_file(xml_path) != xml_hash:
            raise RuntimeError("Replay source changed while rendering")
        evidence["complete"] = True
        evidence["video_sha256"] = sha256_file(output)
        evidence["source_files_unchanged"] = True
    finally:
        if writer is not None:
            writer.close()
        if rendering is not None:
            rendering.close()
        evidence["elapsed_wall_seconds"] = time.monotonic()-started
        atomic_json(output.with_suffix(".json"), evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-wall-seconds", type=float, default=120)
    parser.add_argument("--caption", default="NEW BODY - UNTUNED GAIT | recorded simulation")
    args = parser.parse_args()
    render_replay(args.trace, args.xml, args.output, args.preview,
                  fps=args.fps, max_wall_seconds=args.max_wall_seconds, caption=args.caption)


if __name__ == "__main__":
    main()
