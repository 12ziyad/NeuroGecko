#!/usr/bin/env python3
"""Compare measured joint excursions against the published E. macularius table.

Scorecard rows 17-21 (Jagnandan & Higham 2017 Table 1) give peak-to-peak
excursions for six hindlimb and forelimb joints. `realism_metrics.py` already
records an anatomical proxy for each one per trace sample, so the comparison is
a read, not a new measurement.

The proxy convention is recorded in every trace as "Unvalidated against paper
conventions; never compare raw hinge maxima to biological included angles".
That caveat is carried into the output: this tool reports how far the model is
from each published excursion, and that is a direction of travel, not a claim
of anatomical equivalence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# name -> (published excursion deg, published SD deg, source row)
PUBLISHED = {
    "femur_depression":   (52.42, 3.25, "scorecard 18, Jagnandan convention"),
    "femur_retraction":   (82.57, 2.29, "scorecard 17"),
    "knee":               (98.83, 1.39, "scorecard 19"),
    "ankle":              (85.19, 2.38, "scorecard 20"),
    "humerus_depression": (101.38, 15.88, "scorecard 21 (very noisy)"),
    "humerus_retraction": (44.41, 1.80, "scorecard 21"),
    "elbow":              (92.49, 2.27, "scorecard 21"),
    "wrist":              (71.93, 3.09, "scorecard 21; no wrist actuator exists"),
}


def measure(extra_flags, lab_params=None, duration=20.0, settle=3.0, xml=None):
    """Run one rollout and return (excursions_deg, gate_values, report)."""
    from eval.base_diagnostics import diagnose_trace
    from eval.session2_controller import gate2
    from realism_metrics import analyze_trace
    xml = xml or "morphology/gecko_body_lab_v2.xml"
    with tempfile.TemporaryDirectory() as tmp:
        out, traces = Path(tmp) / "r.json", Path(tmp) / "tr"
        command = [sys.executable, str(REPO / "realism_metrics.py"), "--xml", xml,
                   "--gait-profile", "lab", "--zero-residual", "--episodes", "1", "--seed", "0",
                   "--duration", str(duration), "--settle", str(settle),
                   "--output", str(out), "--trace-dir", str(traces), *extra_flags]
        if lab_params:
            command += ["--lab-params", json.dumps(lab_params)]
        finished = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True)
        if finished.returncode != 0 or not out.exists():
            return None, None, (finished.stderr or finished.stdout)[-500:]
        report = json.loads(out.read_text(encoding="utf-8"))
        episode = report["episodes"][0]
        if episode["gait"].get("status") != "measured":
            return None, None, "episode not scorable (fell before settling)"
        trace = json.loads((traces / "episode_000.json").read_text(encoding="utf-8"))

    time = np.asarray(trace["samples"]["time_s"])
    keep = time >= settle
    angles = np.asarray(trace["samples"]["anatomical_proxy_deg"])[keep]
    names = trace["metadata"]["anatomical_proxy_names"]
    excursions = {}
    for base in PUBLISHED:
        sides = [angles[:, names.index(base + "_" + side)] for side in ("L", "R")
                 if base + "_" + side in names]
        if sides:
            excursions[base] = float(np.mean([s.max() - s.min() for s in sides]))
    result = analyze_trace(trace, settle)
    diagnosis = diagnose_trace(trace, settle_s=settle, entrainment_tolerance_fraction=.10)
    gate = gate2(trace, result, diagnosis,
                 episode["completed_requested_duration"] and not episode["terminated"])
    gate["_stride_period_cv"] = result["limbs"]["HL"]["stride_period_cv"]
    return excursions, gate, report


def gate_line(gate):
    checks = gate["checks"]
    worst = lambda k: (min(checks[k]["values"].values()) if "values" in checks[k]
                       else checks[k].get("value", checks[k].get("value_m_s")))
    passed = sum(1 for c in checks.values() if c["pass"])
    marks = "".join("P" if checks[k]["pass"] else "." for k in
                    ("signed_forward", "net_path", "hind_swing_load",
                     "front_stance_load", "hind_duty", "limb_phase"))
    return (f"{passed}/6 {marks}  fwd {worst('signed_forward'):.4f}  "
            f"front {worst('front_stance_load'):.4f}  hduty {worst('hind_duty'):.4f}  "
            f"CV {gate['_stride_period_cv']:.4f}")


def table(excursions, label="measured"):
    lines = [f"{'joint':20} {label:>10} {'published':>10} {'SD':>6} {'ratio':>7}"]
    for name, (target, sd, _) in PUBLISHED.items():
        value = excursions.get(name)
        if value is None:
            continue
        lines.append(f"{name:20} {value:>10.2f} {target:>10.2f} {sd:>6.2f} {value/target:>7.2f}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flags", nargs=argparse.REMAINDER, default=[],
                   help="extra realism_metrics flags, after --flags")
    p.add_argument("--lab-params", default=None, help="JSON lab parameter overrides")
    p.add_argument("--duration", type=float, default=20.0)
    a = p.parse_args(argv)
    lab = json.loads(a.lab_params) if a.lab_params else None
    excursions, gate, report = measure(a.flags, lab, a.duration)
    if excursions is None:
        print("FAILED:", report)
        return 1
    print(table(excursions))
    print()
    print("gates:", gate_line(gate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
