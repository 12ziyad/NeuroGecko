"""The body's S-curve and the tail's counter-curve, at two phases, from above.

WHAT IS SHOWN. The same animal walking twice on the same seed: first with the
shipped tail phase (`tail_phase_lag` = 0.15), then with the phase that meets
the published rule (0.76). Filmed from above so the curve is the thing you see.

WHAT IS PUBLISHED. The RULE, not a number: in this species the trunk carries a
standing wave that becomes a caudally travelling wave down the tail, and the
tail base flexes toward the PROTRACTED hindlimb (Jagnandan & Higham 2017,
qualitative, target species). No tail amplitude, frequency or phase angle has
been published for any gecko. Measured on this body (#299): shipped 0.15 puts
the tail base 0.40 of a stride off that rule and bending the same way as the
trunk; 0.76 puts it on the rule to +0.02 of a stride.

WHAT IS NOT DECIDED HERE. `tail_phase_lag` is part of the accepted walker's
parameter set, so changing it re-opens the walking gate. This clip is evidence
for that decision, not the decision.

Usage:  python tools/tail_phase_video.py [--seed 1] [--steps 350]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from _tinyfont import draw_text  # noqa: E402


def film(lag, seed, steps, label, phase_note):
    import mujoco
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters

    env = GeckoBrainEnv(
        prey_parameters=PreyParameters.from_registry(),
        walker_xml_path="morphology/gecko_world_v1.xml",
        gait_profile="lab", use_policy=False, eye=True,
        render_mode="rgb_array", max_steps=100000, seed=seed,
        view_mode="hunt", camera_smoothing=0.85)
    env.walk_env.cpg.tail_phase_lag = float(lag)
    env.reset(seed=seed)
    env.walk_env.locomotor_drive = 1.0
    m = env.walk_env.model
    J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[0], a[2], a[3] = 1.0, -0.2, 1.0
    frames = []
    try:
        for _ in range(120):                      # settle into the gait first
            env.step(a)
        for i in range(steps):
            env.step(a)
            d = env.walk_env.data
            sp = math.degrees(d.qpos[J("spine_lat_2")])
            tb = math.degrees(d.qpos[J("tail_yaw_1")])
            frame = env.render()
            y = 10
            for line in [
                f"TAIL PHASE {lag:.2f}  -  {label}",
                f"trunk bend {sp:+5.1f} deg    tail base {tb:+5.1f} deg"
                f"    {'SAME way' if sp * tb > 0 else 'OPPOSITE'}",
                phase_note,
                "published rule (Jagnandan & Higham 2017): tail base flexes toward the forward hindlimb",
            ]:
                draw_text(frame, line, 10, y, scale=2)
                y += 20
            frames.append(frame)
    finally:
        env.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--out", default="artifacts/video/tail_phase.mp4")
    args = ap.parse_args()
    import imageio.v2 as imageio
    frames = film(0.15, args.seed, args.steps, "SHIPPED",
                  "measured: tail base 0.40 of a stride OFF the rule, curving WITH the trunk (#299)")
    frames += film(0.76, args.seed, args.steps, "ON THE PUBLISHED RULE",
                   "measured: tail base +0.02 of a stride from the rule, a travelling wave (#299)")
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)
    print(f"written: {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
