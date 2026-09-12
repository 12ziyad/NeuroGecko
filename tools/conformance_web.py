"""Does the browser gecko behave like the Python one? Measure, do not assume.

A reimplementation that is not checked is a SECOND ANIMAL wearing this one's
name, and every number on the public page would then describe a different
thing from the one the ledger measured. This is the check that stops that.

It runs three comparisons on the same inputs:

  1. THE WALKER. `base_ctrl` in Python against `Walker.baseCtrl` in JavaScript,
     over one full gait cycle at 200 sample times, all 30 actuators.
  2. THE BASAL GANGLIA. The extended Prescott network run to convergence from
     the same salience vectors, comparing the released gate vector.
  3. THE CLOCK. `arousal_at` across a whole simulated day.

Tolerance is 1e-9 -- these are the same equations over the same constants, so
anything above float noise means the port has drifted and must be fixed, not
excused.

Usage:  python tools/conformance_web.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TOL = 1e-9
SITE = REPO / "site"


def python_side():
    from envs.gecko_walk_env import GeckoWalkEnv
    from brain.prescott_bg import PrescottParameters, PrescottBasalGanglia
    from brain.gecko_selector import CHANNELS, BASELINE_DOPAMINE
    from brain import arousal as ar

    env = GeckoWalkEnv(xml_path="morphology/gecko_body_lab_v2.xml",
                       gait_profile="lab", control_mode="cpg_residual",
                       hind_stance_compensation=True)
    c = env.cpg
    T = 1.0 / c.freq

    times = [k * T / 200.0 for k in range(200)]
    walk = [[float(x) for x in c.base_ctrl(t, None, 0.0)] for t in times]

    steers = [-0.6, -0.2, 0.0, 0.35, 0.8]
    steer = [[float(x) for x in c.base_ctrl(0.31, None, h)] for h in steers]

    saliences = [
        [0, 0, 0, 0, 0, 0],
        [0, 0.21, 0.105, 0, 0, 0],
        [0.1504, 0.21, 0, 0, 0, 0],
        [0, 0, 0.105, 1.0, 1.0, 0],
        [0.9, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1.0, 0],
        [0.4, 0.05, 0.2, 0.3, 0.15, 0],
    ]
    gates = []
    for s in saliences:
        bg = PrescottBasalGanglia(n_channels=len(CHANNELS), n_primary=len(CHANNELS),
                                  dopamine=BASELINE_DOPAMINE,
                                  parameters=PrescottParameters.extended())
        for _ in range(20):
            bg.converge(np.asarray(s, dtype=float))
        gates.append([float(x) for x in bg.gates()])

    clock = ar.Arousal()
    hours = [h * 0.25 for h in range(96)]
    arous = [float(clock.arousal_at(h * 3600.0)) for h in hours]

    # SEARCH. 4000 control steps -- long enough to cross scan -> turn -> scan
    # -> travel several times over, so every branch of the state machine is
    # exercised, not just the one it starts in. The trunk angle is fed a
    # deterministic ramp that advances only while the pattern says it is
    # turning, which is what the real body does, so the turn-exit test is
    # driven the same way on both sides.
    from brain.search import SearchPattern
    sp = SearchPattern()
    trunk = 0.0
    head, heading, moving, states = [], [], [], []
    for _ in range(4000):
        h, hy, mv = sp.step(trunk)
        head.append(float(hy)); heading.append(float(h))
        moving.append(int(bool(mv))); states.append(sp.state)
        if mv and h != 0.0:
            trunk += 1.7           # deg per step while the body is turning
    search = {"head": head, "heading": heading, "moving": moving,
              "states": states}

    return {"times": times, "walk": walk, "steers": steers, "steer": steer,
            "saliences": saliences, "gates": gates, "hours": hours,
            "arousal": arous, "search": search}


JS = r"""
import fs from 'fs';
import { Walker, BasalGanglia, Clock, SearchPattern } from '../site/gecko.js';
const cfg = JSON.parse(fs.readFileSync(new URL('../site/media/brain.json', import.meta.url), 'utf8'));
const ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));

const w = new Walker(cfg);
const walk = ref.times.map(t => Array.from(w.baseCtrl(t, 0)));
const steer = ref.steers.map(h => Array.from(w.baseCtrl(0.31, h)));

const gates = ref.saliences.map(s => {
  const bg = new BasalGanglia(cfg);
  const v = Float64Array.from(s);
  for (let i = 0; i < 20; i++) bg.converge(v);
  return Array.from(bg.gates());
});

const clock = new Clock(cfg);
const arousal = ref.hours.map(h => clock.arousalAt(h * 3600));

const sp = new SearchPattern(cfg);
let trunk = 0;
const search = { head: [], heading: [], moving: [], states: [] };
for (let i = 0; i < ref.search.head.length; i++) {
  const r = sp.step(trunk);
  search.head.push(r.headYawDeg); search.heading.push(r.headingDeg);
  search.moving.push(r.moving ? 1 : 0); search.states.push(r.state);
  if (r.moving && r.headingDeg !== 0) trunk += 1.7;
}

fs.writeFileSync(process.argv[3], JSON.stringify({ walk, steer, gates, arousal, search }));
"""


def worst(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main():
    print("running Python side ...")
    ref = python_side()
    ref_path = REPO / "artifacts" / "conformance_ref.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps(ref), encoding="utf-8")

    js_path = REPO / "artifacts" / "_conformance.mjs"
    out_path = REPO / "artifacts" / "conformance_js.json"
    js_path.write_text(JS, encoding="utf-8")

    print("running JavaScript side ...")
    r = subprocess.run(["node", str(js_path), str(ref_path), str(out_path)],
                       cwd=str(REPO), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("NODE FAILED:\n" + (r.stderr or r.stdout)[:3000])
        return 1
    got = json.loads(out_path.read_text(encoding="utf-8"))

    checks = [
        ("walker, one gait cycle x 30 actuators", worst(ref["walk"], got["walk"])),
        ("walker under steering", worst(ref["steer"], got["steer"])),
        ("basal ganglia gates", worst(ref["gates"], got["gates"])),
        ("circadian clock over a day", worst(ref["arousal"], got["arousal"])),
        ("search: head yaw, 4000 steps", worst(ref["search"]["head"],
                                               got["search"]["head"])),
        ("search: heading command", worst(ref["search"]["heading"],
                                          got["search"]["heading"])),
        ("search: moving flag", worst(ref["search"]["moving"],
                                      got["search"]["moving"])),
        ("search: state, step for step",
         0.0 if ref["search"]["states"] == got["search"]["states"] else float("inf")),
    ]
    print()
    ok = True
    for name, err in checks:
        flag = "OK " if err <= TOL else "FAIL"
        if err > TOL:
            ok = False
        print(f"  [{flag}] {name:42} max |difference| = {err:.3e}")
    print()
    if ok:
        print(f"CONFORMANT: the browser gecko matches Python to within {TOL:g}.")
        print("It is the same animal, not a lookalike.")
    else:
        print("NOT CONFORMANT. The port has drifted -- fix it rather than widening the tolerance.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
