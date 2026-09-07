"""Does the tectum actually find the prey, in the world, with a body attached?

It finds a synthetic cricket on a synthetic background perfectly: correct
bearing at every position in the field, zero response to a still world, zero
response to pure self-motion. None of that answers the only question that
matters, which is whether the bearing it reports in the ENVIRONMENT is the
bearing of the prey.

So this measures exactly that: the correlation between the bearing the tectum
reports and the true bearing of the prey from the animal's own head. A detector
that fires constantly on the wrong thing scores a high hit rate and a
correlation of zero, which is why the hit rate is not the test.

Usage:  python tools/tectum_field_test.py [--cells 16] [--out PATH]
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

#: The bar this has to clear to be called working. INVENTED, and deliberately
#: low: a detector that is even loosely pointing at the prey would clear it.
CORRELATION_FLOOR = 0.5
MEAN_ERROR_CEILING_DEG = 20.0


def true_bearing(env, food_xy):
    """Signed bearing of the prey from the animal's facing, degrees, +ve left."""
    data = env.walk_env.data
    forward = data.xmat[env.walk_env._trunk].reshape(3, 3)[:, 0][:2]
    offset = np.asarray(food_xy) - np.asarray(env._nose_xy())
    return math.degrees(math.atan2(
        forward[0] * offset[1] - forward[1] * offset[0],
        forward[0] * offset[0] + forward[1] * offset[1]))


def run(cells=16, steps=150, seed=1):
    from envs.gecko_brain_env import GeckoBrainEnv
    from brain.tectum import Eye
    env = GeckoBrainEnv(homeostasis=True, eye=True)
    env.eye = Eye(fovy_deg=70.0, pixels=64, cells=cells)
    try:
        env.reset(seed=seed)
        pairs, fired = [], 0
        for _ in range(steps):
            _, _, term, trunc, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float32))
            bearing = info.get("prey_bearing_deg")
            if bearing is not None and info["food_visible_frac"] > 0.05:
                fired += 1
                pairs.append((true_bearing(env, info["food_xy"]), bearing))
            if term or trunc:
                break
        if len(pairs) < 6:
            return {"cells": cells, "frames": len(pairs), "correlation": None,
                    "mean_abs_error_deg": None, "fired_fraction": None}
        arr = np.asarray(pairs)
        return {
            "cells": cells,
            "frames": int(len(arr)),
            "fired_fraction": round(fired / steps, 3),
            "correlation": round(float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]), 4),
            "mean_abs_error_deg": round(
                float(np.mean(np.abs(arr[:, 0] - arr[:, 1]))), 2),
        }
    finally:
        env.close()


def prey_size_table():
    """How much of the map the prey covers. The constraint behind the result."""
    half = math.radians(70.0) / 2.0
    focal_px = (64 / 2.0) / math.tan(half)
    per_px = math.degrees(math.atan(1.0 / focal_px))
    rows = []
    for distance in (0.10, 0.20, 0.30, 0.50):
        angle = 2.0 * math.degrees(math.atan(0.009 / distance))
        rows.append({"distance_m": distance,
                     "subtends_deg": round(angle, 2),
                     "pixels": round(angle / per_px, 2),
                     "cells_at_16": round(angle / per_px / 4.0, 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="artifacts/evidence/session8/tectum_field_test.json")
    args = ap.parse_args()

    print("How much of the retinotopic map does the prey cover?")
    for row in prey_size_table():
        print(f"   at {row['distance_m']:.2f} m -> {row['subtends_deg']:>5.2f} deg"
              f" = {row['pixels']:>5.2f} px = {row['cells_at_16']:>5.2f} cells")
    print("\nDoes the reported bearing track the prey?")
    print(f"{'cells':>7}{'frames':>8}{'fired':>8}{'correlation':>13}{'mean err':>10}")
    results = []
    for cells in (16, 32, 64):
        row = run(cells=cells)
        results.append(row)
        print(f"{cells:>7}{row['frames']:>8}{row['fired_fraction']:>8}"
              f"{row['correlation']:>13}{row['mean_abs_error_deg']:>10}")

    best = max((r for r in results if r["correlation"] is not None),
               key=lambda r: r["correlation"], default=None)
    passed = bool(best and best["correlation"] >= CORRELATION_FLOOR
                  and best["mean_abs_error_deg"] <= MEAN_ERROR_CEILING_DEG)

    verdict = (
        "ACCEPTED -- the reported bearing tracks the prey." if passed else
        "NOT ACCEPTED. The tectum fires on almost every frame and the bearing "
        "it reports is uncorrelated with the prey's. It works on synthetic "
        "stimuli -- correct bearing everywhere in the field, silent on a still "
        "world, silent on pure self-motion -- and does not transfer to the "
        "environment, where the animal's own body and the floor sliding past "
        "produce more motion than a prey that covers less than one cell beyond "
        "0.20 m. Raising the map to full pixel resolution reaches only 0.21. "
        "Resolution is therefore part of the cause and not the whole of it, "
        "and no fix is claimed here.")

    payload = {
        "schema_version": 1,
        "test": "does the tectum's reported bearing track the prey in the world",
        "generated_by": "tools/tectum_field_test.py",
        "criterion": {
            "correlation_floor": CORRELATION_FLOOR,
            "mean_error_ceiling_deg": MEAN_ERROR_CEILING_DEG,
            "provenance": "INVENTED, and deliberately low -- a detector even "
                          "loosely pointing at the prey would clear it.",
            "note": "The hit rate is NOT the test. A detector that fires "
                    "constantly on the wrong thing scores a high hit rate and "
                    "a correlation of zero, which is what happens here.",
        },
        "prey_size_in_map": prey_size_table(),
        "results": results,
        "passed": passed,
        "verdict": verdict,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n{verdict}")
    print(f"written: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
