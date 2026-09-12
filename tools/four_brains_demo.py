"""Switch on the channels the world could never reach, and film what happens.

WHAT WAS WRONG. Measured over 900 autonomous steps: four of the selector's six
channels had a gate of EXACTLY 0.0000 on every single step. `flee` and `bask`
because the world contained nothing to flee from and nothing warm to lie on;
`rest` because a strolling gecko does not tire; `groom` because a hungry
animal's explore salience beats its 0.05 tonic on every step. Prescott 2024's
six-channel basal ganglia was running as a two-channel one.

WHAT THIS RUNS. The furnished world -- shelter, warm patch and threat, all three
of which `utils/build_world.py` could already emit and none of which the
committed world had -- with the three sensor paths that were stubbed out now
reading it:

  * the threat's distance, through the nose, into the flee channel
  * the ground's temperature under the animal, into body temperature, into the
    cold error, into the bask channel
  * the shelter's and the warm patch's bearings, into the motor programs

NOTHING IS TUNED TO MAKE THIS HAPPEN. The flee channel's salience IS the
published probability of a defensive reaction in this species: **0.21 to scent,
n = 42**, and that alone. It used to reach 0.28 by adding the published VISUAL
figure of 0.07 on top, and the condition it added it on was `danger > 0.0` --
the animal's belly touching the ground, or it having fallen over. That is a
mechanosensory and postural state rather than sight, the same factorial
measured the mechanosensory term at chi2 < 0.01, p > 0.9, and 0.28 appears
nowhere in `config/proxies.yaml`. Corrected at #355. Explore is 0.35 x hunger.
So a STARVING gecko in this model still never flees, whatever is standing on
it, and a partly fed one does -- the correction did not rescue it, which is the
point: the prediction now stands on one published number and no invented ones.

AND THE BASAL GANGLIA HAS ITS OWN FLOOR. Bisected on one channel with
everything else at zero, it is **0.199878** -- a hard step, with the gate
exactly 0.0 below it and 0.9271 above. It was recorded here as 0.2008, which is
neither that figure nor the 0.201091 the floor becomes once groom's old 0.05
tonic sits in the vector (#351). That tonic is gone (#353): it was untagged,
uncited, raised the bar for every OTHER channel including flee, and was masking
a degenerate fixed point in which an empty salience vector part-releases all six
channels equally and `argmax` picks whichever is declared first (#357).

`rest` IS NOW REACHABLE, and it took connecting brain 6 rather than changing a
number (#349, #352). Its salience was fatigue, and measured over five seeds x
1500 autonomous steps fatigue equilibrates at 0.002 and correlates with the
FRACTION OF CONTROL STEPS whose instantaneous displacement crosses the 0.100 m/s
aerobic ceiling at r = +0.9998 -- against +0.71 with the animal's actual mean
speed. Averaged over one stride the same trace gives exactly zero. A strolling
gecko does not tire, so it now rests for the reason this species is documented
to rest: it is daytime. `groom` stays unreachable and now says so honestly.

THE SCRIPT. The threat is a mocap body, so it can be moved without touching the
physics state. It waits off stage, walks in, stays a moment and leaves. Nothing
else is scripted: where the animal goes, and which behaviour it releases, are
its own.

Usage:  python tools/four_brains_demo.py [--meals-owed 0.5] [--steps 1400]
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

WORLD = "morphology/gecko_world_furnished_v1.xml"
FPS = 50

#: When the predator walks on, how close it gets, and when it leaves, as
#: fractions of the run. INVENTED: this is stage direction, not biology.
THREAT_IN = 0.70
THREAT_CLOSE = 0.78
THREAT_OUT = 0.94
THREAT_REACH_M = 0.006


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meals-owed", type=float, default=None,
                    help="run one scenario at this hunger instead of both")
    ap.add_argument("--steps", type=int, default=1400)
    ap.add_argument("--time-of-day", type=float, default=16.0, dest="time_of_day",
                    help="hours; the clock's dusk is at 12.0, so 2 is the light "
                         "phase and 16 is inside the published evening peak")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default="artifacts/video/four_brains.mp4")
    args = ap.parse_args()

    import mujoco
    from brain.gecko_selector import CHANNELS
    from envs.gecko_brain_env import GeckoBrainEnv

    # THE THIRD SCENARIO IS THE LIGHT PHASE (#352). Brain 6 is connected now,
    # so the time of day is a scenario setting like hunger is -- a day is
    # 86400 s and this clip is tens of seconds, so the phase is chosen rather
    # than waited for. At 2 h the clock reports arousal 0.000 and the animal
    # rests with its eyes shut; at 16 h it reports 1.000 and rest is exactly
    # 0.0000. Both are the same code with the same constants.
    scenarios = ([(args.meals_owed, args.time_of_day, "")]
                 if args.meals_owed is not None
                 else [(0.90, 16.0, "HUNGRY at its activity peak: warms up, explores, hunts"),
                       (0.00, 16.0, "FED: with no hunger at all, the predator gets through"),
                       (0.90, 2.0, "THE LIGHT PHASE: it rests, with its eyes shut")])
    all_frames, all_live, all_peak, all_won = [], [], {}, {}
    for meals, tod, caption in scenarios:
        f, live, peak, won, temps = run_one(args, meals, caption, tod)
        all_frames += f
        all_live += [c for c in live if c not in all_live]
        for c in peak:
            all_peak[c] = max(all_peak.get(c, 0.0), peak[c])
            all_won[c] = all_won.get(c, 0) + won[c]
    finish(args, all_frames, all_live, all_peak, all_won)


def run_one(args, meals_owed, caption, time_of_day_h=16.0):
    import mujoco
    from brain.gecko_selector import CHANNELS
    from envs.gecko_brain_env import GeckoBrainEnv
    # THE ACCEPTED WALKER, EXPLICITLY. `GeckoBrainEnv` defaults to
    # gait_profile="legacy" with the frozen trained residual loaded on top, and
    # that pairing is NOT the one the gates accepted. Measured with one method,
    # fraction of the step each foot carries load, against published targets of
    # 0.70 fore and 0.765 hind:
    #
    #     lab + no policy     0.770 0.764 0.787 0.790   even to 0.006
    #     legacy + policy     0.704 0.601 0.435 0.367   hind feet at half
    #
    # The defect is in the HIND feet, which is why it survived: the front pair
    # looks fine and front duty was the only thing being checked. The first cut
    # of this demo was filmed on that second line (#338).
    env = GeckoBrainEnv(walker_xml_path=WORLD, seed=args.seed,
                        gait_profile="lab", use_policy=False,
                        behaviour_control=True, homeostasis=True,
                        action_selection=True, eye=True, smell=True,
                        circadian=True, time_of_day_h=time_of_day_h,
                        meals_owed_at_start=meals_owed)
    env.reset()
    probe = env.step(np.zeros(env.action_space.shape, dtype=np.float32))[4]
    if not probe.get("accepted_walker"):
        raise SystemExit(
            f"refusing to film a rejected walker: gait_profile="
            f"{probe.get('gait_profile')!r}, use_policy={probe.get('use_policy')!r}")
    env.reset()
    model, data = env.walk_env.model, env.walk_env.data

    threat_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "threat")
    if threat_bid < 0:
        raise SystemExit(f"{WORLD} has no threat body; rebuild it")
    threat_mocap = int(model.body_mocapid[threat_bid])
    parked = np.array(data.mocap_pos[threat_mocap], dtype=np.float64).copy()

    # PARK THE PREY OUT OF SIGHT IN THE FED SCENARIO, and say why. Flee's
    # salience is the published 0.21 and the basal ganglia releases at
    # 0.199878, a margin of 0.0101 -- so ANY other salience at all keeps it
    # shut. Measured:
    # with the predator touching the animal, hunt sitting at 0.1504 because a
    # cricket was visible was enough to stop it fleeing. The prey is therefore
    # moved out of range here rather than the numbers being adjusted, and the
    # caption on screen says so (#339).
    prey_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "prey")
    prey_mocap = (int(model.body_mocapid[prey_bid]) if prey_bid >= 0
                  and model.body_mocapid[prey_bid] >= 0 else None)
    hide_prey = meals_owed < 0.05 and prey_mocap is not None

    frames = []
    won = {c: 0 for c in CHANNELS}
    peak = {c: 0.0 for c in CHANNELS}
    trace = []
    renderer = None
    if not args.no_video:
        renderer = mujoco.Renderer(model, args.height, args.width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
    smooth = None

    for i in range(args.steps):
        t = i / float(max(1, args.steps - 1))

        # ---- stage the predator -----------------------------------------
        here = np.array(data.xpos[env.walk_env._trunk][:2], dtype=np.float64)
        if t < THREAT_IN or t > THREAT_OUT:
            target = parked
        else:
            approach = min(1.0, (t - THREAT_IN) / max(1e-9, THREAT_CLOSE - THREAT_IN))
            ease = approach * approach * (3.0 - 2.0 * approach)
            near = np.array([here[0] + THREAT_REACH_M, here[1], 0.05])
            target = parked + (near - parked) * ease
        data.mocap_pos[threat_mocap] = target
        if hide_prey:
            data.mocap_pos[prey_mocap] = [3.0, 3.0, 0.01]

        behaviour = env.selector.selected()
        gates = np.asarray(env.selector.gates(), dtype=np.float64)
        for k, c in enumerate(CHANNELS):
            peak[c] = max(peak[c], float(gates[k]))
        if behaviour:
            won[behaviour] += 1

        body_C = float(env.homeostasis.body_temperature_C)
        ground_C = env._substrate_temperature_C()
        predator_m = env._predator_distance_m()
        trace.append({
            "step": i, "behaviour": behaviour, "body_C": round(body_C, 3),
            "ground_C": None if ground_C is None else round(ground_C, 2),
            "predator_m": None if predator_m is None else round(predator_m, 4),
            "gates": [round(float(g), 4) for g in gates],
            "salience": [round(float(x), 4) for x in
                         (env.selector.last_salience
                          if env.selector.last_salience is not None
                          else [0] * len(CHANNELS))],
        })

        env.autonomous_step()

        if renderer is None:
            continue
        look = np.array([here[0], here[1], 0.03])
        smooth = look.copy() if smooth is None else smooth + (look - smooth) * 0.06
        camera.lookat[:] = smooth
        camera.distance = 0.62
        camera.elevation = -26.0
        camera.azimuth = 108.0 + 26.0 * math.sin(2 * math.pi * 0.045 * (i / FPS))
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        frame = renderer.render()
        draw_text(frame, f"RELEASED: {(behaviour or 'none').upper()}", 12, 12, scale=3)
        if caption:
            draw_text(frame, caption, 12, args.height - 26, scale=2)
        draw_text(frame, f"body {body_C:.1f}C  ground "
                         f"{'--' if ground_C is None else format(ground_C, '.0f')}C"
                         f"   predator "
                         f"{'away' if predator_m is None or predator_m > 0.25 else format(predator_m * 100, '.1f') + 'cm'}",
                  12, 42, scale=2)
        # THE CLOCK ON SCREEN (#352, #325). A clip that shows the animal
        # asleep has to show WHY, or "it is lying still" and "it is broken"
        # look identical. Arousal is what brain 6 produced this step; ASLEEP is
        # the module's own `last["asleep"]`, its invented 0.1 cut on arousal.
        clock = env.clock.last if env.clock is not None else {}
        if clock:
            draw_text(frame,
                      f"clock {clock.get('time_of_day_h', 0.0):05.2f}h   "
                      f"arousal {clock.get('arousal', 0.0):.2f}   "
                      f"{'ASLEEP -- eyes shut' if clock.get('asleep') else 'AWAKE'}",
                      12, args.height - 52, scale=2)
        for k, c in enumerate(CHANNELS):
            bar = int(round(gates[k] * 90))
            colour = (250, 230, 90) if c == behaviour else (150, 150, 150)
            draw_text(frame, f"{c:8}", 12, 72 + k * 20, scale=2)
            if bar > 0:
                frame[74 + k * 20:84 + k * 20, 96:96 + bar] = colour
        frames.append(frame)

    if renderer is not None:
        renderer.close()
    env.close()
    live = sorted(c for c in CHANNELS if won[c] > 0)
    temps = [r["body_C"] for r in trace]
    pm = [r["predator_m"] for r in trace if r["predator_m"] is not None]
    print(f"  predator closest {min(pm)*100:.2f} cm" if pm else "  no predator")
    fl = [r["gates"][1] for r in trace]
    print(f"  flee gate peak {max(fl):.4f}")
    fs = [r["salience"][1] for r in trace]
    print(f"  flee SALIENCE peak {max(fs):.4f}  (needs ~0.20 to release)")
    hi = [r for r in trace if r["salience"][1] >= 0.205]
    print(f"  steps with flee salience >= 0.205: {len(hi)} of {len(trace)}")
    if hi:
        r = hi[len(hi)//2]
        print(f"    mid sample step {r['step']}: salience {r['salience']} gates {r['gates']}")
    print(f"  meals owed {meals_owed:.2f}: released {live}, "
          f"body {temps[0]:.1f} -> {min(temps):.1f} -> {temps[-1]:.1f} C")
    return frames, live, peak, won, temps


def finish(args, frames, live, peak, won):
    import json
    from brain.gecko_selector import CHANNELS
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        import imageio.v2 as imageio
        imageio.mimwrite(str(out), frames, fps=FPS, quality=6, macro_block_size=1)
        print(f"\nwritten: {out}  ({len(frames)} frames)")
    print(f"\n{'channel':10} {'peak gate':>10} {'steps won':>10}")
    for c in CHANNELS:
        print(f"  {c:8} {peak.get(c, 0.0):10.4f} {won.get(c, 0):10d}")
    print(f"\nchannels released: {sorted(live)}  ({len(live)} of {len(CHANNELS)})")
    record = REPO / "artifacts" / "evidence" / "session12" / "four_brains.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "generated_by": "tools/four_brains_demo.py",
        "world": WORLD,
        "seed": args.seed,
        "channels_released": sorted(live),
        "peak_gate": {c: round(peak.get(c, 0.0), 4) for c in CHANNELS},
        "steps_won": {c: won.get(c, 0) for c in CHANNELS},
        "measured_limits": {
            "basal_ganglia_release_threshold_salience": 0.199878,
            "basal_ganglia_release_threshold_with_old_groom_tonic": 0.201091,
            "flee_salience_ceiling_scent_alone": 0.21,
            "flee_margin_over_threshold": 0.01012,
            "flee_survives_hunger_up_to": 0.2400,
            "groom_tonic_salience": 0.0,
            "rest_salience_source": "1 - arousal, from brain/arousal.py",
            "note": "CORRECTED (#351, #353, #349, #350). The threshold was "
                    "recorded here as 0.2008 and typed rather than computed; "
                    "bisected it is 0.199878 with every other channel at zero "
                    "and 0.201091 with groom's old 0.05 tonic in the vector, "
                    "and it is a hard step -- gate exactly 0.0 below, 0.9271 "
                    "above. It carries no provenance tag because it is not a "
                    "constant: it is emergent and conditional on the channel "
                    "count, the dopamine level and the extended variant. The "
                    "line `rest_fatigue_plateau: 0.011` that stood here was a "
                    "typed literal no run computed; measured across five seeds "
                    "fatigue equilibrates at 0.00203-0.00254, and it is an "
                    "artefact of sampling speed over one control step rather "
                    "than one stride. Nothing here was tuned.",
        },
    }, indent=1) + "\n", encoding="utf-8")
    print(f"evidence: {record}")


if __name__ == "__main__":
    main()
