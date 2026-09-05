#!/usr/bin/env python3
"""CMA-ES fit of the lab base controller to the six Gate 2 checks.

Why this exists: eight single-parameter interventions were tried by hand and all
failed to move realised limb phase (hind stance height, fore stance height,
actuator bandwidth with and without force headroom, spine amplitude, fore/hind
gearing, contact softness, commanded-delay calibration). Commanding a different
touchdown delay does not shift realised phase by that amount -- the plant is a
coupled nonlinear system, so single-parameter reasoning does not apply. Joint
search over the interacting parameters is the appropriate tool.

This searches CONTROLLER parameters only. It never touches the body, the locked
1.1888 Hz cadence, the published stance ratios, or any published target. The
objective is the Gate 2 checks themselves, so a pass here is a pass on the same
measurements, not on a surrogate.

Scoring is a normalised distance to each check's target band: zero inside the
band, growing linearly outside it. Speed and net/path are one-sided. A rollout
that falls, or that produces no complete strides, scores a large constant.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SITES = ("footzone_hind_L", "footzone_fore_L", "footzone_hind_R", "footzone_fore_R")
TOUCH = ("touch_hind_L", "touch_fore_L", "touch_hind_R", "touch_fore_R")
FEET = ("HL", "FL", "HR", "FR")
FREQ = 1.1888
CONTACT_THRESHOLD_N = 0.03501960784313725
DEBOUNCE_S = 0.040
GOAL_DISTANCE_M = 5.0

# name, low, high, default. Controller-only; nothing published is searched.
SPACE = [
    ("hind_fa_amplitude",         0.40, 1.00, 0.98),
    ("fore_hind_amplitude_ratio", 0.20, 1.20, 0.54),
    ("front_stance_press",       -0.40, 0.40, 0.05),
    ("front_stance_press_fr",    -0.40, 0.40, 0.05),
    ("front_swing_lift",         -0.90, 0.10, -0.40),
    ("hind_lift_multiplier",     -2.50, 0.00, -1.3333333333333333),
    ("spine_amp",                 0.00, 1.00, 0.30),
    ("tail_amp",                  0.00, 0.80, 0.15),
    ("tail_phase_lag",            0.00, 0.50, 0.15),
    ("fl_touchdown_delay",        0.05, 0.95, 0.435),
]
LAB_KEYS = {"hind_fa_amplitude", "fore_hind_amplitude_ratio", "front_stance_press",
            "front_stance_press_fr", "front_swing_lift", "hind_lift_multiplier"}

# check, low, high  (inf on a side means one-sided)
GATES = [
    ("forward_speed",     0.04, math.inf),
    ("net_over_path",     0.50, math.inf),
    ("hind_swing_load", -math.inf, 0.10),
    ("front_stance_load", 0.65, math.inf),
    ("hind_duty",         0.73, 0.83),
    ("limb_phase",        0.405, 0.465),
]
# scale used to normalise the distance outside each band
SCALE = {"forward_speed": 0.02, "net_over_path": 0.20, "hind_swing_load": 0.05,
         "front_stance_load": 0.15, "hind_duty": 0.05, "limb_phase": 0.06}

FAIL_SCORE = 50.0


def unpack(x):
    out = {}
    for (name, lo, hi, _), v in zip(SPACE, x):
        out[name] = float(np.clip(v, lo, hi))
    return out


def rollout(params, seconds, settle, xml):
    """Return the six Gate 2 measurements, or None if the episode is unusable."""
    import mujoco
    from envs.gecko_walk_env import GeckoWalkEnv
    import common.gait_config as gc

    fl = params["fl_touchdown_delay"]
    base_profile = gc.get_gait_profile("lab")
    delays = (0.0, fl, 0.5, (fl + 0.5) % 1.0)
    patched = gc.GaitProfile("lab", base_profile.frequency_hz, delays,
                             base_profile.stance_ratios)
    real_get = gc.get_gait_profile
    gc.get_gait_profile = lambda p="legacy": patched if p == "lab" else real_get(p)
    try:
        lab = {k: params[k] for k in LAB_KEYS}
        env = GeckoWalkEnv(xml_path=xml, control_mode="cpg_residual", max_steps=10**6,
                           gait_profile="lab", hind_stance_compensation="all", seed=0,
                           lab_parameters=lab)
        env.cpg.spine_amp = params["spine_amp"]
        env.cpg.tail_amp = params["tail_amp"]
        env.cpg.tail_phase_lag = params["tail_phase_lag"]
        env.reset(seed=0)
        m, d = env.model, env.data
        # Match the calibration scenario realism_metrics uses: a fixed world
        # heading toward a distant goal. Without it the base has no heading
        # reference and wanders, which changes speed and net/path.
        env.target = d.xpos[env._trunk][:2].copy() + np.array([GOAL_DISTANCE_M, 0.0])
        _, env._prev_dist, _ = env._target_egocentric()
        site = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in SITES]
        sens = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in TOUCH]
        if min(site) < 0 or min(sens) < 0:
            return None
        adr = [m.sensor_adr[s] for s in sens]

        times, loaded, cmd_stance, path, x0, xprev = [], [], [], 0.0, None, None
        zero = np.zeros(m.nu)
        steps = int(seconds / env.dt)
        for _ in range(steps):
            env.step(zero)
            if bool(getattr(env, "fallen", False)):
                return None
            if d.time < settle:
                continue
            tr = d.xpos[env._trunk].copy()
            if x0 is None:
                x0 = tr.copy()
            else:
                path += float(np.linalg.norm(tr[:2] - xprev[:2]))
            xprev = tr.copy()
            times.append(float(d.time))
            loaded.append([float(d.sensordata[a]) > CONTACT_THRESHOLD_N for a in adr])
            cmd_stance.append(env.cpg.commanded_contact_array(d.time) > 0.5)
        env.close()
    finally:
        gc.get_gait_profile = real_get

    if len(times) < 100 or x0 is None or path <= 1e-9:
        return None
    t = np.asarray(times)
    ld = np.asarray(loaded)
    cs = np.asarray(cmd_stance)
    dur = t[-1] - t[0]
    net = float(np.linalg.norm(xprev[:2] - x0[:2]))

    dt = float(np.median(np.diff(t)))
    nmin = max(1, int(round(DEBOUNCE_S / dt)))

    def debounce(col):
        y = col.astype(int).copy()
        i = 0
        while i < len(y):
            j = i
            while j < len(y) and y[j] == y[i]:
                j += 1
            if (j - i) < nmin and i > 0:
                y[i:j] = y[i - 1]
            i = j
        return y.astype(bool)

    db = np.column_stack([debounce(ld[:, i]) for i in range(4)])

    def touchdowns(i):
        return np.array([t[k] for k in range(1, len(db)) if db[k, i] and not db[k - 1, i]])

    duty, td = {}, {}
    for i, f in enumerate(FEET):
        duty[f] = float(db[:, i].mean())
        td[f] = touchdowns(i)
    if any(len(td[f]) < 5 for f in FEET):
        return None

    def circ_mean(v):
        a = (v * FREQ % 1.0) * 2 * math.pi
        return (math.atan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * math.pi)) % 1.0

    phase = [(circ_mean(td["FL"]) - circ_mean(td["HL"])) % 1.0,
             (circ_mean(td["FR"]) - circ_mean(td["HR"])) % 1.0]

    hind_swing, front_stance = [], []
    for i, f in enumerate(FEET):
        st = cs[:, i]
        if f.startswith("H"):
            hind_swing.append(float(ld[~st, i].mean()) if (~st).any() else 1.0)
        else:
            front_stance.append(float(ld[st, i].mean()) if st.any() else 0.0)

    return {"forward_speed": net / dur,
            "net_over_path": net / path,
            "hind_swing_load": max(hind_swing),
            "front_stance_load": min(front_stance),
            "hind_duty": min(duty["HL"], duty["HR"]),
            "limb_phase": float(np.mean(phase))}


def score(meas):
    if meas is None:
        return FAIL_SCORE, None
    total = 0.0
    for name, lo, hi in GATES:
        v = meas[name]
        d = max(lo - v, 0.0) if lo > -math.inf else 0.0
        d = max(d, v - hi) if hi < math.inf else d
        total += max(d, 0.0) / SCALE[name]
    return total, meas


def evaluate(args):
    x, seconds, settle, xml = args
    try:
        return score(rollout(unpack(x), seconds, settle, xml))[0]
    except Exception:
        return FAIL_SCORE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=11.0)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--xml", default="morphology/gecko_body_lab_v2.xml")
    ap.add_argument("--output", default="artifacts/evidence/session3/cma_gate2.json")
    ap.add_argument("--warm-start", default=None,
                    help="JSON from a previous run; its parameters seed this search.")
    ap.add_argument("--scale-override", default=None,
                    help="e.g. hind_duty=0.02,front_stance_load=0.08 to weight a stubborn check harder.")
    a = ap.parse_args()

    import cma
    if a.scale_override:
        for item in a.scale_override.split(","):
            k, v = item.split("="); SCALE[k.strip()] = float(v)
    lo = np.array([s[1] for s in SPACE]); hi = np.array([s[2] for s in SPACE])
    x0 = np.array([s[3] for s in SPACE])
    if a.warm_start:
        prev = json.loads(pathlib.Path(a.warm_start).read_text(encoding="utf8"))["parameters"]
        x0 = np.array([prev[s[0]] for s in SPACE])
        print(f"warm start from {a.warm_start}")
    norm = lambda v: (v - lo) / (hi - lo)
    denorm = lambda u: lo + np.clip(u, 0, 1) * (hi - lo)

    es = cma.CMAEvolutionStrategy(norm(x0).tolist(), a.sigma,
                                  {"bounds": [0, 1], "popsize": a.popsize,
                                   "seed": 1, "verbose": -9})
    base_s, base_m = score(rollout(unpack(x0), a.seconds, a.settle, a.xml))
    print(f"baseline score {base_s:.4f}")
    if base_m:
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in base_m.items()))

    best, best_x, t0 = base_s, x0.copy(), time.time()
    with mp.Pool(a.workers) as pool:
        for gen in range(a.generations):
            us = es.ask()
            xs = [denorm(np.asarray(u)) for u in us]
            fs = pool.map(evaluate, [(x, a.seconds, a.settle, a.xml) for x in xs])
            es.tell(us, fs)
            i = int(np.argmin(fs))
            if fs[i] < best:
                best, best_x = fs[i], xs[i].copy()
            print(f"gen {gen+1:3d}/{a.generations}  best {best:.4f}  "
                  f"gen-best {fs[i]:.4f}  median {np.median(fs):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if best <= 0.0:
                print("all six gates satisfied; stopping early")
                break

    s, meas = score(rollout(unpack(best_x), 17.0, 3.0, a.xml))
    print(f"\nBEST score {s:.4f} (re-scored on a 17 s rollout)")
    params = unpack(best_x)
    if meas:
        for name, lo_, hi_ in GATES:
            ok = (lo_ <= meas[name] <= hi_)
            print(f"  {name:20s} {meas[name]:9.4f}   {'PASS' if ok else 'FAIL'}")
    print("\nparameters:")
    for k, v in params.items():
        print(f"  {k:28s} {v:+.5f}")
    out = pathlib.Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"score": s, "measurements": meas, "parameters": params,
                               "baseline_score": base_s, "baseline_measurements": base_m,
                               "space": [list(s_) for s_ in SPACE],
                               "gates": [[g[0], g[1] if g[1] > -math.inf else None,
                                          g[2] if g[2] < math.inf else None] for g in GATES],
                               "generations": a.generations, "popsize": a.popsize,
                               "eval_seconds": a.seconds, "settle_s": a.settle,
                               "xml": a.xml}, indent=1), encoding="utf8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
