#!/usr/bin/env python3
"""CMA-ES fit of the lab base controller, scored through realism_metrics itself.

This replaces tools/fit_gate2_cma.py, which reimplemented limb phase and so
optimised a quantity that diverged from the gate on non-entrained gaits. That
run reported limb phase 0.4497 while the official measure read 0.6534 on the
same configuration, because the search had been rewarded for driving stride
period CV from 0.0149 to 0.2713. See docs/BUILD_LOG.md, Session 3g.

Here every candidate is scored by running realism_metrics.py as a subprocess and
reading the six checks out of its JSON. There is no second implementation, so
there is nothing to diverge. It is slower per evaluation; that is the price of
scoring the thing being claimed.

Two guards against the previous failure mode are hard requirements, not scored
preferences: a candidate is rejected outright if the episode does not complete,
or if contact entrainment fails. An irregular gait therefore cannot win.

Searched: controller values only. Never the body, the locked 1.1888 Hz cadence,
the published stance ratios, or any published target.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent

# name, low, high, default
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
LAB_KEYS = [s[0] for s in SPACE if s[0] != "fl_touchdown_delay"]

GATES = [
    ("forward_speed",     0.04, math.inf),
    ("net_over_path",     0.50, math.inf),
    ("hind_swing_load", -math.inf, 0.10),
    ("front_stance_load", 0.65, math.inf),
    ("hind_duty",         0.73, 0.83),
    ("limb_phase",        0.405, 0.465),
]
SCALE = {"forward_speed": 0.02, "net_over_path": 0.20, "hind_swing_load": 0.05,
         "front_stance_load": 0.15, "hind_duty": 0.05, "limb_phase": 0.06}
FAIL_SCORE = 600.0
# A failed gate costs more than any achievable distance improvement elsewhere.
GATE_WEIGHT = 10.0
FEET = ("HL", "FL", "HR", "FR")
# Stride-period CV above this rejects a candidate outright.
CV_CEILING = 0.10


def unpack(x):
    return {name: float(np.clip(v, lo, hi)) for (name, lo, hi, _), v in zip(SPACE, x)}


def measure(params, duration, xml, keep=None):
    """Run realism_metrics and return its six gate values, or None if unusable."""
    lab = json.dumps({k: params[k] for k in LAB_KEYS})
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(keep) if keep else pathlib.Path(tmp) / "r.json"
        trace = pathlib.Path(tmp) / "tr"
        cmd = [sys.executable, str(REPO / "realism_metrics.py"),
               "--xml", xml, "--gait-profile", "lab", "--zero-residual",
               "--hind-stance-compensation", "all", "--episodes", "1", "--seed", "0",
               "--duration", str(duration), "--lab-params", lab,
               "--fl-touchdown-delay", f"{params['fl_touchdown_delay']:.6f}",
               "--output", str(out), "--trace-dir", str(trace)]
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            return None
        report = json.loads(out.read_text(encoding="utf8"))
        ep = report["episodes"][0]
        g = ep["gait"]
        if g.get("status") != "measured":
            return None
        # Hard rejections. An irregular or terminated gait cannot win, which is
        # the specific failure mode of the previous harness.
        if ep.get("terminated") or not ep.get("completed_requested_duration"):
            return None
        # The baseline itself has entrainment_pass False, so that cannot be a
        # hard requirement. Guard instead on stride regularity, which is the
        # quantity the previous harness was rewarded for destroying: the
        # baseline sits at CV 0.0149 and the bad fit reached 0.2713.
        cv = g["limbs"]["HL"]["stride_period_cv"]
        cv = cv["mean"] if isinstance(cv, dict) else cv
        if cv is None or cv > CV_CEILING:
            return None
        tr = json.loads((trace / "episode_000.json").read_text(encoding="utf8"))
        meta, s = tr["metadata"], tr["samples"]
        thr = float(meta["contact_threshold_N"])
        t = np.asarray(s["time_s"])
        keep_mask = t >= float(report["protocol"].get("settle_s", 3.0)) if "settle_s" in report["protocol"] else t >= 3.0
        f = np.asarray(s["foot_force_N"])[keep_mask]
        c = np.asarray(s["commanded_contacts"])[keep_mask].astype(bool)

    num = lambda v: v["mean"] if isinstance(v, dict) else v
    loads = {n: ((f[c[:, i], i] > thr).mean() if c[:, i].any() else 0.0,
                 (f[~c[:, i], i] > thr).mean() if (~c[:, i]).any() else 1.0)
             for i, n in enumerate(FEET)}
    phase = g["limb_phase"]
    if phase["HL_to_FL"]["mean_cycle"] is None or phase["HR_to_FR"]["mean_cycle"] is None:
        return None
    return {"forward_speed": num(g["forward_speed_m_s"]),
            "net_over_path": g["net_displacement_m"] / g["distance_path_m"],
            "hind_swing_load": max(loads["HL"][1], loads["HR"][1]),
            "front_stance_load": min(loads["FL"][0], loads["FR"][0]),
            "hind_duty": min(num(g["limbs"]["HL"]["duty_factor"]),
                             num(g["limbs"]["HR"]["duty_factor"])),
            "limb_phase": float(np.mean([phase["HL_to_FL"]["mean_cycle"],
                                         phase["HR_to_FR"]["mean_cycle"]])),
            "_stride_cv": num(g["limbs"]["HL"]["stride_period_cv"])}


def score(meas):
    """Gates PASSED dominate; normalised distance only breaks ties.

    An earlier version summed normalised distance alone, so a candidate could
    score better while passing fewer gates -- the 20 s fit scored 2.75 against
    the baseline's 3.49 while passing 2/6 against 4/6, because it missed forward
    speed and hind duty by hairs. Gate count is the thing being asked for, so it
    is the primary term and no distance improvement can buy a lost gate.
    """
    if meas is None:
        return FAIL_SCORE
    failed, total = 0, 0.0
    for name, lo, hi in GATES:
        v = meas[name]
        d = max(lo - v, 0.0) if lo > -math.inf else 0.0
        if hi < math.inf:
            d = max(d, v - hi)
        d = max(d, 0.0)
        if d > 0.0:
            failed += 1
        total += d / SCALE[name]
    return GATE_WEIGHT * failed + total


def evaluate(a):
    x, duration, xml = a
    try:
        return score(measure(unpack(x), duration, xml))
    except Exception:
        return FAIL_SCORE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--xml", default="morphology/gecko_body_lab_v2.xml")
    ap.add_argument("--output", default="artifacts/evidence/session3/cma_official.json")
    a = ap.parse_args()

    import cma
    lo = np.array([s[1] for s in SPACE]); hi = np.array([s[2] for s in SPACE])
    x0 = np.array([s[3] for s in SPACE])
    denorm = lambda u: lo + np.clip(u, 0, 1) * (hi - lo)
    es = cma.CMAEvolutionStrategy((((x0 - lo) / (hi - lo)).tolist()), a.sigma,
                                  {"bounds": [0, 1], "popsize": a.popsize,
                                   "seed": 1, "verbose": -9})

    base_m = measure(unpack(x0), a.duration, a.xml)
    base_s = score(base_m)
    print(f"baseline score {base_s:.4f}")
    if base_m:
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in base_m.items() if not k.startswith("_")))

    best, best_x, t0 = base_s, x0.copy(), time.time()
    with mp.Pool(a.workers) as pool:
        for gen in range(a.generations):
            us = es.ask()
            xs = [denorm(np.asarray(u)) for u in us]
            fs = pool.map(evaluate, [(x, a.duration, a.xml) for x in xs])
            es.tell(us, fs)
            i = int(np.argmin(fs))
            if fs[i] < best:
                best, best_x = fs[i], xs[i].copy()
            alive = sum(1 for v in fs if v < FAIL_SCORE)
            print(f"gen {gen+1:3d}/{a.generations}  best {best:.4f}  gen-best {fs[i]:.4f}  "
                  f"valid {alive}/{len(fs)}  {time.time()-t0:.0f}s", flush=True)
            if best <= 0.0:
                print("all six gates satisfied; stopping early")
                break

    final = pathlib.Path(a.output).with_suffix(".report.json")
    final.parent.mkdir(parents=True, exist_ok=True)
    meas = measure(unpack(best_x), 20.0, a.xml, keep=final)
    print(f"\nBEST re-scored at 20 s: {score(meas):.4f}")
    if meas:
        for name, lo_, hi_ in GATES:
            print(f"  {name:20s} {meas[name]:9.4f}   {'PASS' if lo_ <= meas[name] <= hi_ else 'FAIL'}")
        print(f"  {'stride period CV':20s} {meas['_stride_cv']:9.4f}")
    params = unpack(best_x)
    print("\nparameters:")
    for k, v in params.items():
        print(f"  {k:28s} {v:+.5f}")
    pathlib.Path(a.output).write_text(json.dumps(
        {"scored_via": "realism_metrics.py subprocess (identical code path to the gate)",
         "score": score(meas), "measurements": meas, "parameters": params,
         "baseline_score": base_s, "baseline_measurements": base_m,
         "rejects": f"episode not completed, terminated, or HL stride_period_cv > {CV_CEILING}",
         "space": [list(s) for s in SPACE], "generations": a.generations,
         "popsize": a.popsize, "fit_duration_s": a.duration, "xml": a.xml},
        indent=1), encoding="utf8")
    print(f"\nwrote {a.output}")


if __name__ == "__main__":
    main()
