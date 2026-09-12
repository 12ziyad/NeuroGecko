"""What does the efference copy cost, and what does it buy?

THE MEASUREMENT THIS EXISTS TO MAKE. Ledger #213 records the efference copy as
a partial success: "cuts false alarms from 72 % of frames to 3-7 %". That is a
PRECISION measurement, and it was the only one available, because the project
had no column saying whether the prey was on the image at all. So the cost side
was never measured. `tools/hunt_decomposition.py` added that column, and with
it the trade-off becomes visible for the first time.

Measured over 400 steps x 6 seeds of the #220 configuration, on the frames
where the prey IS on the rendered image:

    efference copy ON  (shipped gains)   recall  7.6 %   precision 85.4 %
    efference copy OFF                   recall 65.2 %   precision 67.7 %

The efference copy is spending 58 points of recall to buy 18 points of
precision. Both gains are INVENTED and, by their own docstring, "set so that
the flow a walking animal generates is roughly cancelled" -- i.e. tuned until
the false alarms went away, against a measurement that could only see false
alarms. This sweeps them and reports both sides.

WHAT IS NOT DECIDED HERE. This prints a curve. It does not pick a point. The
operating point is a modelling choice and belongs in the registry with a stated
criterion attached, not in whichever row of a table looks best.

Usage:  python tools/efference_tradeoff.py [--seeds 6] [--steps 400]
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
sys.path.insert(0, str(REPO / "tools"))

from hunt_decomposition import decompose, run  # noqa: E402


def sweep(configs, steps, seeds, seed0=1):
    out = []
    for label, window, efference, gain in configs:
        rows = []
        fovy = prey_r = None
        for k in range(seeds):
            r, fovy, prey_r = run(steps=steps, seed=seed0 + k,
                                  motion_window=window, efference=efference,
                                  flow_gain_scale=gain)
            rows.extend(r)
        s = decompose(rows, fovy, prey_r)
        out.append({
            "label": label,
            "motion_window": window,
            "efference": efference,
            "flow_gain_scale": gain,
            "recall": s["recall"],
            "precision": s["precision"],
            "f1": s["f1"],
            "P_prey_on_image": s["P_prey_on_image"],
            "P_report": s["P_report"],
            "reports": s["reports_total"],
            "rejection_census": s["rejection_census_prey_on_image"],
        })
        r_, p_, f_ = s["recall"], s["precision"], s["f1"]
        print(f"  {label:<34} recall {100*r_:5.1f} %   "
              f"precision {100*p_:5.1f} %   F1 {f_:.3f}   "
              f"({s['reports_total']} reports)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()

    configs = []
    # The shipped baseline, first, so everything below is read against it.
    configs.append(("SHIPPED  w=1 gain=1.00", 1, "scaled", 1.0))
    # The gain curve at the window that measured best.
    for g in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        configs.append((f"w=3      gain={g:.2f}", 3, "unscaled", g))
    # Whether a wider window still helps once the gain is right.
    for w in (1, 6):
        configs.append((f"w={w}      gain=0.20", w, "unscaled", 0.2))

    print(f"\n{args.seeds} seeds x {args.steps} steps, #220 configuration")
    print("recall    = reports / frames where the prey IS on the image")
    print("precision = reports about the prey / all reports\n")
    rows = sweep(configs, args.steps, args.seeds)

    best = max(rows, key=lambda r: (r["f1"] or 0.0))
    print(f"\nhighest F1: {best['label']}  "
          f"recall {100*best['recall']:.1f} %  "
          f"precision {100*best['precision']:.1f} %")
    print("F1 is reported because it is symmetric and assumption-free, NOT "
          "because it is the right objective for a predator.\nChoosing the "
          "operating point needs a stated criterion; this only draws the curve.")

    out = REPO / "artifacts" / "evidence" / "session11" / "efference_tradeoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": 1,
        "test": "what the efference copy costs in recall and buys in precision",
        "run": ("habitat world, accepted walker (lab, no policy), eye on, "
                "scripted creep -- the #220 configuration"),
        "seeds": args.seeds, "steps": args.steps,
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
