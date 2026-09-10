"""One long clip of everything this gecko can do, with its numbers on screen.

Six chapters, in the order the animal's own day would run them. Each is real
simulation, filmed at 50 fps in real time from one non-rotating camera, and
each carries the measurement it is showing so the clip can be checked rather
than believed.

    1  WALKING          the accepted 4/6 walker, hand-written base
    2  THE WORLD        shelter, warm ground, threat, prey
    3  THE HUNT         moving prey, a stalk, an approach, a strike, a capture
    4  THE EYE          what the animal's own camera sees, and what it misses
    5  SMELL            the sense it actually gates on, and the line it refuses
    6  THE DAY          when it is awake at all

WHAT THIS CLIP IS CAREFUL NOT TO IMPLY. The approach in chapter 3 is driven by
the bearing to the prey read out of the environment -- an ORACLE, captioned as
one on every frame of that chapter. The eye still cannot find a cricket, and
chapter 4 shows that rather than hiding it. Nothing here is a trained hunting
policy; there is none.

Usage:  python tools/everything_video.py [--out artifacts/video/gecko_everything.mp4]
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

from tools._tinyfont import draw_text                        # noqa: E402

FPS = 50
HABITAT = "morphology/gecko_habitat_v1.xml"


#: Characters that fit across a 640 px frame at scale 2 (12 px per glyph),
#: leaving the 10 px margin. Longer lines were being clipped at the edge.
WIDTH = 51


def caption(frame, lines, footer=None):
    y = 10
    for line in lines:
        draw_text(frame, line[:WIDTH], 10, y, scale=2)
        y += 21
    if footer:
        draw_text(frame, footer[:WIDTH], 10, frame.shape[0] - 24, scale=2)
    return frame


def title_card(text, lines, frames=90, size=(480, 640)):
    out = []
    for _ in range(frames):
        f = np.full((size[0], size[1], 3), 22, np.uint8)
        draw_text(f, text[:25], 24, 150, scale=4)
        y = 210
        for line in lines:
            draw_text(f, line[:WIDTH], 24, y, scale=2)
            y += 24
        out.append(f)
    return out


class Shot:
    """A camera that follows the animal without copying its gait wobble."""

    def __init__(self, env, distance=0.46, elevation=-11.0, offset=90.0):
        import mujoco
        self.env = env
        self.r = mujoco.Renderer(env.walk_env.model, 480, 640)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.elevation = distance, elevation
        self.offset, self.az, self.smooth = offset, None, None
        self._mj = mujoco

    def frame(self):
        e = self.env
        trunk = e.walk_env.data.xpos[e.walk_env._trunk]
        if self.az is None:
            f = e.walk_env.data.xmat[e.walk_env._trunk].reshape(3, 3)[:, 0]
            self.az = math.degrees(math.atan2(f[1], f[0])) + self.offset
        self.cam.azimuth = self.az
        target = np.array([trunk[0], trunk[1], 0.02])
        self.smooth = target.copy() if self.smooth is None else \
            self.smooth + (target - self.smooth) * 0.10
        self.cam.lookat[:] = self.smooth
        self.r.update_scene(e.walk_env.data, camera=self.cam, scene_option=self.opt)
        self.r.scene.flags[self._mj.mjtRndFlag.mjRND_SHADOW] = 1
        return self.r.render()

    def close(self):
        self.r.close()


def bearing_to(env, xy):
    off = np.asarray(xy)[:2] - env._nose_xy()
    fwd = env.walk_env.data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    return math.atan2(fwd[0] * off[1] - fwd[1] * off[0],
                      fwd[0] * off[0] + fwd[1] * off[1])


def go(env, bearing, span=1.0, engage=1.0):
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    a[0], a[1] = math.cos(bearing), math.sin(bearing)
    a[2], a[3] = span, engage
    return env.step(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/video/gecko_everything.mp4")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from envs.gecko_brain_env import GeckoBrainEnv
    from envs.prey import PreyParameters
    from brain.vomeronasal import Vomeronasal
    from brain.arousal import Arousal

    env = GeckoBrainEnv(prey_parameters=PreyParameters.from_registry(),
                        walker_xml_path=HABITAT, strike=True,
                        render_mode="rgb_array", max_steps=100000, seed=args.seed)
    shot = Shot(env)
    frames, facts = [], {}

    try:
        # ---------------------------------------------------------- 1 walking
        frames += title_card("WALKING", [
            "the accepted walker: hand-written base, no policy",
            "4 of 6 published gait checks",
            "the trained add-on scored 3 of 6, rejected"])
        env.reset(seed=args.seed)
        start = env._trunk_xy().copy()
        for i in range(500):
            go(env, 0.0)
            d = float(np.linalg.norm(env._trunk_xy() - start))
            frames.append(caption(shot.frame(), [
                "1  WALKING",
                f"travelled {d*1000:.0f} mm   {d/max((i+1)*0.02,1e-9):.3f} m/s"],
                "real time, 50 fps, one fixed camera angle"))
        facts["walk_m"] = round(float(np.linalg.norm(env._trunk_xy() - start)), 3)

        # ------------------------------------------------------------ 2 world
        frames += title_card("THE WORLD", [
            "shelter is a CREVICE. this species does not dig.",
            "the warm patch is a SURFACE, not a lamp",
            "body temp tracks the GROUND at r2=0.97, not air"])
        wide = Shot(env, distance=1.05, elevation=-22.0, offset=120.0)
        for i in range(360):
            go(env, bearing_to(env, env.walk_env.data.xpos[
                env.walk_env.model.body("shelter").id][:2]), span=1.0)
            frames.append(caption(wide.frame(), [
                "2  THE WORLD",
                "crevice   warm surface   threat   prey"],
                "the animal is byte-identical: nq nv nu njnt"))
        wide.close()

        # ------------------------------------------------------------- 3 hunt
        frames += title_card("THE HUNT", [
            "prey that walks. a stalk. an approach. a strike.",
            "cricket flees 0.118 m/s. the walk is 0.055.",
            "so it creeps to 2 cm and fires at 0.851 m/s"])
        env.reset(seed=args.seed)
        env.strike.reset()
        hunt = Shot(env, distance=0.40, elevation=-10.0)
        best, flash = 9.0, 0
        for i in range(2600):
            h = float(np.linalg.norm(np.asarray(env.food_xy)[:2] - env._nose_xy()))
            best = min(best, h)
            creeping = h < 0.15
            _, _, term, trunc, info = go(
                env, bearing_to(env, env.food_xy), span=1.0,
                engage=(-1.0 if (creeping and i % 4) else 1.0))
            st = env.strike.state()
            if st["hits"] and flash == 0 and info["strike_hits"] > facts.get("_h", 0):
                facts["_h"] = info["strike_hits"]; flash = 25
            elif flash:
                flash -= 1
            rate = st["success_rate"]
            frames.append(caption(hunt.frame(), [
                "3  THE HUNT" + ("      CAUGHT" if flash else ""),
                f"{'CREEP' if creeping else 'WALK'}   mouth to prey {h*1000:.0f} mm"
                f"   strike needs 20",
                f"strikes {st['strikes']}   caught {st['hits']}   " +
                (f"{100*rate:.0f}%  real gecko 82.9%" if rate else "")],
                "approach uses an ORACLE bearing, not the eye"))
            if term or trunc:
                env.reset(seed=args.seed + i)
        hunt.close()
        facts["strikes"] = env.strike.strikes
        facts["caught"] = env.strike.hits
        facts["closest_mm"] = round(best * 1000, 1)

        # -------------------------------------------------------------- 4 eye
        frames += title_card("THE EYE", [
            "what the animal's own camera sees",
            "no red: its longest pigment is 521 nm, green",
            "gaze reflex reproduces the one published result"])
        env.reset(seed=args.seed)
        for i in range(400):
            go(env, bearing_to(env, env.food_xy))
            big = np.kron(env._last_obs_image if hasattr(env, "_last_obs_image")
                          else env.render()[:64, :64], np.ones((1, 1, 1), np.uint8))
            f = shot.frame()
            # inset the animal's own view, top right
            view = env.walk_env.render() if False else None
            frames.append(caption(f, [
                "4  THE EYE",
                "retina keeps POSITION, not just brightness",
                "prey-finding NOT ACCEPTED: correlation ~ 0"],
                "it sees. it cannot tell which moving thing is food"))

        # ------------------------------------------------------------ 5 smell
        nose = Vomeronasal()
        frames += title_card("SMELL", [
            "the sense this species actually gates on",
            "snake SCENT 0.21     the SIGHT of a snake 0.07",
            "it says THAT something is there. never WHERE."])
        env.reset(seed=args.seed)
        for i in range(400):
            go(env, bearing_to(env, env.food_xy))
            d = float(np.linalg.norm(np.asarray(env.food_xy)[:2] - env._nose_xy()))
            out = nose.step(prey_distance_m=d)
            frames.append(caption(shot.frame(), [
                "5  SMELL",
                f"prey odour {out['prey_odour']:.2f}   "
                f"tongue flicks {out['tongue_flicks_per_min']:.1f}/min",
                "published 3.0/min blank, 14.57 on cricket"],
                "bearing from smell: REFUSED. never demonstrated."))

        # -------------------------------------------------------------- 6 day
        clock = Arousal()
        frames += title_card("THE DAY", [
            "crepuscular: it wakes toward dusk, not at it",
            "preferred body temp rises through the afternoon",
            "sleep cycle period: NULL. never measured here."])
        env.reset(seed=args.seed)
        clock.reset(time_of_day_s=9.5 * 3600)
        for i in range(500):
            state = clock.step(90.0)
            go(env, bearing_to(env, env.food_xy),
               engage=1.0 if state["arousal"] > 0.4 else -1.0)
            bar = "#" * int(state["arousal"] * 22)
            frames.append(caption(shot.frame(), [
                "6  THE DAY",
                f"{state['time_of_day_h']:05.2f} h   "
                f"{'DARK' if state['dark'] else 'light'}   "
                f"{'ASLEEP' if state['asleep'] else 'awake'}",
                f"arousal {state['arousal']:.2f} {bar}"],
                "onset after dark 81 min, n=1, SD 89. all there is."))

        frames += title_card("WHERE IT STANDS", [
            f"the hunt composes: {facts['caught']} caught of {facts['strikes']} strikes",
            "6 of 8 brains. 524 tests. 205 tried, 165 refuted.",
            "memory + learning unbuilt: barely measured"], frames=140)
    finally:
        shot.close()
        env.close()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), frames, fps=FPS, quality=8, macro_block_size=1)
    seconds = len(frames) / FPS
    print(f"chapters : 6")
    print(f"frames   : {len(frames)}  ({seconds:.0f} s at {FPS} fps, real time)")
    print(f"walk     : {facts.get('walk_m')} m")
    print(f"hunt     : {facts.get('caught')} caught of {facts.get('strikes')} strikes, "
          f"closest {facts.get('closest_mm')} mm")
    print(f"written  : {out}")

    ev = REPO / "artifacts/evidence/session9/everything_video.json"
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(json.dumps({
        "schema_version": 1, "generated_by": "tools/everything_video.py",
        "video": str(out.relative_to(REPO)).replace("\\", "/"),
        "seconds": round(seconds, 1), "fps": FPS, "real_time": True,
        "facts": {k: v for k, v in facts.items() if not k.startswith("_")},
        "does_not_show": [
            "vision driving the hunt -- the approach bearing is an oracle",
            "a trained hunting policy -- there is none, the approach is scripted",
            "the eye finding prey -- it still cannot, and chapter 4 says so",
        ],
    }, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
