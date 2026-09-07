"""Does handing the rhythm to the spinal cord change what the walker does?

The accepted walker passes 4 of 6 published gait gates on a closed-form clock.
`brain/spinal_cpg.py` replaces that clock with integrated oscillators so a
descending command has somewhere to land. With its coupling switched off the
two agree to about 1e-13 of a stride in isolation -- but agreeing in isolation
is not the same as producing the same gait, because the phase feeds contact
detection, the stance compensator and the reflex assist.

So this runs the SAME trial twice, once on each, and compares every gate. It is
the acceptance test for attaching the cord, and until it has run nothing is
claimed about the gates.

Usage:  python tools/gate_with_cord.py [--duration 20] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from common.gait_config import get_gait_profile          # noqa: E402
from envs.gecko_walk_env import GeckoWalkEnv             # noqa: E402
from eval.base_diagnostics import diagnose_trace         # noqa: E402
from eval.session2_controller import gate2               # noqa: E402
from realism_metrics import TraceRecorder, analyze_trace  # noqa: E402


#: The accepted walker is the one with the stance compensator on -- that is what
#: took Gate 2 from 2/6 to 4/6. Running parity on the bare base would compare the
#: cord against a configuration nobody accepted, which is the mistake a scouting
#: agent made in session 6d and which this project has recorded twice.
ACCEPTED = "all"


def run(with_cord, duration=20.0, seed=0, compensation=ACCEPTED):
    profile = get_gait_profile("lab")
    xml = REPO / "morphology/gecko_body_lab_v2.xml"
    env = GeckoWalkEnv(xml_path=xml, gait_profile=profile,
                       control_mode="cpg_residual", reset_noise=0,
                       residual_scale=.25, max_steps=round(duration / .02),
                       hind_stance_compensation=compensation)
    try:
        env.reset(seed=seed)
        if with_cord:
            # Attached AFTER reset so the cord starts at t=0 with the episode.
            env.cpg.attach_spinal_cord()
        env.target = env.data.xpos[env._trunk, :2] + 10 * np.array([1.0, 0.0])
        _, env._prev_dist, _ = env._target_egocentric()
        recorder = TraceRecorder(env, {"seed": seed, "gait_profile": "lab"},
                                 physics_substeps=5)
        recorder.record()
        term = trunc = False
        for _ in range(env.max_steps):
            _, _, term, trunc, _ = env.step(np.zeros(env.nu, dtype=np.float32))
            if term or trunc:
                break
        recorder.detach()
        trace = recorder.as_dict()
        if env.data.time <= 3:
            raise RuntimeError("the walker fell before the settling window; "
                               "no gait to compare")
        result = analyze_trace(trace, settle_s=3)
        diagnosis = diagnose_trace(trace, settle_s=3,
                                   entrainment_tolerance_fraction=.10)
        completed = not term and np.isclose(env.data.time, duration)
        return {
            "gate2": gate2(trace, result, diagnosis, completed),
            "duty": {f: result["limbs"][f]["duty_factor"]["mean"]
                     for f in ("HL", "FL", "HR", "FR")},
            "limb_phase": {k: result["limb_phase"][k]["mean_cycle"]
                           for k in ("HL_to_FL", "HR_to_FR")},
            "speed": diagnosis["motion"]["signed_body_forward_speed_m_s"]["mean"],
            "strides": result.get("complete_strides"),
        }
    finally:
        env.close()


def _flatten(gate, prefix=""):
    out = {}
    for key, value in (gate or {}).items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        elif isinstance(value, bool):
            out[f"{prefix}{key}"] = value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out",
                    default="artifacts/evidence/session7/cord_gate_parity.json")
    args = ap.parse_args()

    print("running the accepted walker on the closed-form clock ...")
    clock = run(False, args.duration, args.seed)
    print("running the same trial with the spinal cord driving ...")
    cord = run(True, args.duration, args.seed)

    a, b = _flatten(clock["gate2"]), _flatten(cord["gate2"])
    keys = sorted(set(a) | set(b))
    print(f"\n{'gate':<44} {'clock':>7} {'cord':>7}")
    changed = []
    for k in keys:
        same = a.get(k) == b.get(k)
        if not same:
            changed.append(k)
        print(f"  {k:<42} {str(a.get(k)):>7} {str(b.get(k)):>7}"
              f"{'   <-- CHANGED' if not same else ''}")
    passed_clock = sum(1 for k in keys if a.get(k))
    passed_cord = sum(1 for k in keys if b.get(k))

    print(f"\n{'measurement':<44} {'clock':>10} {'cord':>10} {'delta':>11}")
    for label, ka, kb in (("hind duty HL", clock["duty"]["HL"], cord["duty"]["HL"]),
                          ("fore duty FL", clock["duty"]["FL"], cord["duty"]["FL"]),
                          ("limb phase HL->FL", clock["limb_phase"]["HL_to_FL"],
                           cord["limb_phase"]["HL_to_FL"]),
                          ("forward speed m/s", clock["speed"], cord["speed"])):
        if ka is None or kb is None:
            print(f"  {label:<42} {'-':>10} {'-':>10}")
            continue
        print(f"  {label:<42} {ka:>10.6f} {kb:>10.6f} {kb - ka:>11.2e}")

    # This harness is a PARITY harness, not the official scorer. It records at
    # its own rate, so timing_and_acquisition_valid can be False here and the
    # absolute pass count is NOT the project's Gate 2 result. What it is
    # entitled to conclude is that the two arms are identical, because both
    # arms run through the same harness.
    verdict = ("EQUIVALENT -- every gate agrees, so the cord can drive the "
               "walker without changing the accepted result. NOTE: absolute "
               "pass counts here are this harness's, not the official Gate 2 "
               "run; only the clock-versus-cord comparison is claimed."
               if not changed else
               "NOT EQUIVALENT -- attaching the cord moved a gate. It must not "
               "be attached until this is explained.")
    print(f"\ngates passed: clock {passed_clock}, cord {passed_cord}")
    print(verdict)

    payload = {
        "schema_version": 1,
        "test": "does attaching the integrated spinal cord change the gait gates",
        "generated_by": "tools/gate_with_cord.py",
        "protocol": {"duration_s": args.duration, "seed": args.seed,
                     "residual": "zero -- the open-loop accepted base",
                     "stance_compensation": ACCEPTED,
                     "coupling": "load feedback OFF (sigma = 0)",
                     "caveat": (
                         "PARITY HARNESS, NOT THE OFFICIAL SCORER. It records "
                         "at its own rate, so timing_and_acquisition_valid may "
                         "be False and the absolute pass count is not the "
                         "project's Gate 2 result. Both arms run through the "
                         "same harness, so the comparison is valid; the "
                         "absolute numbers are not a gate claim.")},
        "gates_changed": changed,
        "gates_passed": {"clock": passed_clock, "cord": passed_cord},
        "clock": clock, "cord": cord,
        "verdict": verdict,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float),
                   encoding="utf-8")
    print(f"written: {out}")
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
