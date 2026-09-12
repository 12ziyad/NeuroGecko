"""One run of the animal, filmed twice and logged once.

WHAT THIS IS FOR. Every clip this project has made shows the animal from the
outside. This one shows the same animal three ways at the same instant: the
body in its world, the physics body underneath it in the identical pose from
the identical camera, and the brain state that produced both. Nothing is
re-simulated between the passes -- `renderer.update_scene` is called twice on
the SAME `env.data`, so frame n of the skeleton is frame n of the skin, not a
second run that happens to look similar.

WHAT IS REAL HERE AND WHAT IS A PRESENTATION CHOICE.

  REAL: every number in `telemetry.json`. The gates are what the basal ganglia
  released that step, the arousal is what `brain/arousal.py` produced, the body
  temperature is what the thermostat computed from the ground the animal was
  standing on, the speed is measured trunk displacement.

  A CHOICE, AND DECLARED: the clock runs compressed. A day is 86400 s and a
  watchable clip is tens of seconds, so `homeostasis_time_compression` is set
  high enough to cross dusk inside the clip. It is the SAME declared knob the
  hypothalamus already uses, applied to the same simulated clock, and it scales
  metabolism identically -- a few compressed hours against a starvation reserve
  measured in months, so nothing is distorted by it. The animal's behaviour is
  its own.

  ALSO A CHOICE: the camera follows the animal and is shared by both passes.

THE WALKER IS ASSERTED, NOT ASSUMED. `GeckoBrainEnv` defaults to the REJECTED
pairing -- `gait_profile="legacy"` with the frozen trained residual on top -- and
an entire session of clips was filmed on it before anyone noticed (#338). This
refuses to run unless `info["accepted_walker"]` is True.

Usage:
    python tools/telemetry_run.py [--steps 1200] [--start-h 10.5] [--compress 600]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WORLD = "morphology/gecko_world_furnished_v1.xml"
FPS = 50


def _option(mujoco, *, skin, visual, collision, joints=False):
    """One render pass's visibility. Group 1 is the visual surface, 3 the
    collision body; the skin is a separate list and sits in skingroup 1."""
    o = mujoco.MjvOption()
    o.geomgroup[:] = 0
    if visual:
        o.geomgroup[1] = 1
    if collision:
        o.geomgroup[3] = 1
    o.skingroup[:] = 0
    if skin:
        o.skingroup[1] = 1
    if joints:
        o.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--start-h", type=float, default=10.5,
                    help="time of day to begin at; the clock's dusk is 12.0")
    ap.add_argument("--compress", type=float, default=600.0,
                    help="simulated-time multiplier, declared in the output")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--outdir", default="artifacts/telemetry")
    args = ap.parse_args()

    import mujoco
    from brain.gecko_selector import CHANNELS
    from envs.gecko_brain_env import GeckoBrainEnv

    env = GeckoBrainEnv(
        walker_xml_path=WORLD, seed=args.seed,
        gait_profile="lab", use_policy=False,          # THE ACCEPTED WALKER
        behaviour_control=True, homeostasis=True, action_selection=True,
        eye=True, smell=True, circadian=True,
        time_of_day_h=args.start_h,
        homeostasis_time_compression=args.compress,
        meals_owed_at_start=0.90,
    )
    env.reset()
    probe = env.step(np.zeros(env.action_space.shape, dtype=np.float32))[4]
    if not probe.get("accepted_walker"):
        raise SystemExit(
            "refusing to film a rejected walker: "
            f"gait_profile={probe.get('gait_profile')!r}, "
            f"use_policy={probe.get('use_policy')!r}")
    env.reset()

    model = env.walk_env.model
    renderer = mujoco.Renderer(model, args.height, args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)

    world_pass = _option(mujoco, skin=True, visual=True, collision=False)
    bones_pass = _option(mujoco, skin=False, visual=False, collision=True,
                         joints=True)

    world_frames, bones_frames, trace = [], [], []
    smooth = None

    for i in range(args.steps):
        info = env.autonomous_step()[4]

        # ---- the shared camera. Both passes use it, so the two images are the
        # same view of the same instant rather than two similar runs.
        look = env.walk_env.data.xpos[env.walk_env._trunk].copy()
        smooth = look.copy() if smooth is None else smooth + (look - smooth) * 0.08
        camera.lookat[:] = smooth
        camera.distance = 0.44
        camera.elevation = -18.0
        camera.azimuth = 112.0 + 18.0 * math.sin(2 * math.pi * 0.035 * (i / FPS))

        renderer.update_scene(env.walk_env.data, camera=camera,
                              scene_option=world_pass)
        world_frames.append(renderer.render().copy())
        renderer.update_scene(env.walk_env.data, camera=camera,
                              scene_option=bones_pass)
        bones_frames.append(renderer.render().copy())

        gates = [float(x) for x in env.selector.gates()]
        sal = [float(x) for x in env.selector.last_salience]
        clock = info.get("clock", {})
        trace.append({
            "i": i,
            "t": round(i / FPS, 3),
            "hour": round(float(clock.get("time_of_day_h", 0.0)), 4),
            "arousal": round(float(info.get("arousal", 1.0)), 5),
            "asleep": bool(clock.get("asleep", False)),
            "behaviour": info.get("behaviour"),
            "program": info.get("program"),
            "gates": [round(g, 5) for g in gates],
            "salience": [round(s, 5) for s in sal],
            "body_C": round(float(env.homeostasis.body_temperature_C), 3),
            "hunger": round(float(env.homeostasis.vector()[0]), 5),
            "speed_m_s": round(float(info.get("moving_speed", 0.0)), 5),
            "eyelid_deg": round(math.degrees(float(env.walk_env.eyelid_rad)), 2),
        })

    out = REPO / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    for name, frames in (("world", world_frames), ("bones", bones_frames)):
        path = out / f"{name}.mp4"
        imageio.mimwrite(str(path), frames, fps=FPS, quality=6,
                         macro_block_size=1)
        print(f"written: {path}  ({len(frames)} frames)")

    meta = {
        "generated_by": "tools/telemetry_run.py",
        "world": WORLD,
        "seed": args.seed,
        "fps": FPS,
        "steps": args.steps,
        "channels": list(CHANNELS),
        "accepted_walker": True,
        "gait_profile": "lab",
        "use_policy": False,
        "release_floor_salience": 0.199878,
        "declared_choices": {
            "homeostasis_time_compression": args.compress,
            "why": "a day is 86400 s and a watchable clip is tens of seconds, "
                   "so the clock is compressed to cross dusk inside the clip. "
                   "It is the same declared knob the hypothalamus already uses "
                   "and it scales metabolism identically.",
            "camera": "follows the trunk; shared by both render passes",
        },
        "trace": trace,
    }
    (out / "telemetry.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"written: {out / 'telemetry.json'}  ({len(trace)} rows)")

    woke = next((r for r in trace if not r["asleep"]), None)
    print(f"\nstart {trace[0]['hour']:.2f} h  ->  end {trace[-1]['hour']:.2f} h")
    print(f"arousal {trace[0]['arousal']:.3f} -> {trace[-1]['arousal']:.3f}")
    if woke:
        print(f"woke at step {woke['i']} ({woke['t']:.1f} s, {woke['hour']:.2f} h)")
    seen = sorted({r["behaviour"] for r in trace if r["behaviour"]})
    print(f"behaviours released: {seen}")


if __name__ == "__main__":
    main()
