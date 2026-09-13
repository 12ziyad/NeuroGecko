"""The knee, before and after, walking side by side.

WHY THIS EXISTS. Two keepers on r/leopardgeckos said the hind legs were the
wrong shape and one asked, precisely, whether the joint pointed forward "unlike
hands -- ours have the opposite way". #408 measured that they were right: the
elbow bowed caudally, which is correct, and the knee bowed caudally too, which
no tetrapod does. #413 fixed it by taking the other branch of the two-link
chain, so the knee turned cranial and the foot did not move at all.

A number in a ledger does not answer a keeper looking at a render. This films
BOTH bodies -- the committed one from before the fix, from its own git
worktree with its own meshes, and the one shipping now -- walking the same
controller from the same seed, from a low side angle where a knee is actually
visible, and puts them in one frame.

The clip runs at the animal's own speed. Two control steps per frame at 25 fps
is real time for this controller; four was the mistake that made an earlier clip
look like a sped-up lizard.

Usage:  python tools/knee_before_after_video.py [--seconds 12]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OLD_TREE = REPO.parent / "GeckoBrain_old"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools._tinyfont import draw_text  # noqa: E402

FPS = 25
STEPS_PER_FRAME = 2          # real time for this controller, not 2x (#407 note)
H, W = 470, 430              # inside the body XML's 480x640 offscreen buffer


def build(xml_path, seed):
    """One walker on one body, in the accepted zero-residual configuration."""
    import mujoco
    from common.gait_config import get_gait_profile
    from envs.gecko_walk_env import GeckoWalkEnv

    env = GeckoWalkEnv(xml_path=pathlib.Path(xml_path),
                       gait_profile=get_gait_profile("lab"),
                       control_mode="cpg_residual", reset_noise=0,
                       residual_scale=.25, max_steps=100000,
                       hind_stance_compensation=True)
    env.reset(seed=seed)
    env.target = env.data.xpos[env._trunk, :2] + np.array([10., 0.])
    _, env._prev_dist, _ = env._target_egocentric()
    return env


class SideShot:
    """Locked to the animal's left flank, low, where the knee is visible."""

    def __init__(self, env):
        import mujoco
        self._mj = mujoco
        self.env = env
        self.r = mujoco.Renderer(env.model, H, W)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 0.195         # hip, knee, foot, and enough flank for context
        self.cam.elevation = -13.0        # looking slightly down, so the body does not occlude the leg
        self.opt = mujoco.MjvOption()
        self.smooth = None
        self.az = None
        self._pelvis = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def frame(self):
        d = self.env.data
        # The HIP, not the trunk centre. The knee is the subject, and the two
        # animals walk at slightly different speeds, so a camera locked to a
        # shared point would let one drift out of its panel.
        hip = d.xpos[self._pelvis]
        if self.az is None:
            fwd = d.xmat[self.env._trunk].reshape(3, 3)[:, 0]
            self.az = math.degrees(math.atan2(fwd[1], fwd[0])) + 90.
        self.cam.azimuth = self.az
        target = np.array([hip[0], hip[1], 0.010])
        self.smooth = target.copy() if self.smooth is None else \
            self.smooth + (target - self.smooth) * .12
        self.cam.lookat[:] = self.smooth
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        self.r.scene.flags[self._mj.mjtRndFlag.mjRND_SHADOW] = 1
        return self.r.render()

    def close(self):
        self.r.close()


def label(panel, title, subtitle, colour):
    band = np.full((54, panel.shape[1], 3), 18, np.uint8)
    band[:3, :] = colour
    draw_text(band, title, 10, 10, scale=2)
    draw_text(band, subtitle, 10, 32, scale=1)
    return np.vstack([band, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", default="artifacts/video/knee_before_after.mp4")
    args = ap.parse_args()

    old_xml = OLD_TREE / "morphology" / "gecko_body_lab_v2.xml"
    if not old_xml.exists():
        raise SystemExit(f"Need the pre-fix worktree at {OLD_TREE}. "
                         f"git worktree add {OLD_TREE} 04fcc95")

    # The OLD body must be compiled with the OLD meshes, which live beside it in
    # that worktree; MuJoCo resolves meshdir relative to the XML, so simply
    # pointing at the file in the worktree is enough.
    old = build(old_xml, args.seed)
    new = build(REPO / "morphology" / "gecko_body_lab_v2.xml", args.seed)

    shots = SideShot(old), SideShot(new)
    frames = []
    n = int(args.seconds * FPS)
    zero_old = np.zeros(old.action_space.shape, dtype=np.float32)
    zero_new = np.zeros(new.action_space.shape, dtype=np.float32)
    for _ in range(n):
        for _ in range(STEPS_PER_FRAME):
            old.step(zero_old)
            new.step(zero_new)
        a = label(shots[0].frame(), "BEFORE", "knee bows BACKWARD, 3.2 mm", (190, 70, 60))
        b = label(shots[1].frame(), "AFTER", "knee bows FORWARD, 1.35 mm", (70, 170, 110))
        frame = np.hstack([a, np.full((a.shape[0], 4, 3), 40, np.uint8), b])
        draw_text(frame, "same controller, same seed, same speed -- only the joint changed",
                  10, frame.shape[0] - 20, scale=1)
        frames.append(frame)
    for s in shots:
        s.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimwrite(out, frames, fps=FPS, quality=8, macro_block_size=1)
    print(f"wrote {out}  ({len(frames)} frames, {len(frames)/FPS:.1f} s at {FPS} fps)")


if __name__ == "__main__":
    main()
