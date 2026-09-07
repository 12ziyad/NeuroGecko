"""The optokinetic drum: the one published behavioural test for this species.

Masseck, Roll & Hoffmann 2008, Vision Research 48:765-772, rotated a striped
drum around *Eublepharis macularius* (n = 4) and measured how well the head
followed. That is the only quantitative sensorimotor measurement that exists
for this animal's vision, and it comes with an asymmetry sharp enough to
falsify a whole class of front ends:

    binocular             0.9 at 20 deg/s, 0.8 at 30, 0.7-0.8 at 40
    monocular temporo-nasal  0.7 / 0.6 / 0.4
    monocular naso-temporal  NO RESPONSE AT ANY VELOCITY

This runs the same experiment on the model. It gates on the three properties
that are PREDICTED by the mechanism, and reports the absolute gains -- which
are fitted -- as context rather than as evidence. A fitted curve passing
through its own fit points proves nothing.

Usage:  python tools/okr_sweep.py [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_retina = _load("_retina_for_okr", "brain/retina.py")
_pretectum = _load("_pretectum_for_okr", "brain/pretectum.py")
Retina = _retina.Retina
Pretectum = _pretectum.Pretectum
PUBLISHED_GAIN = _pretectum.PUBLISHED_GAIN
PUBLISHED_VELOCITIES = _pretectum.PUBLISHED_VELOCITIES

#: The drum. Stripe period in degrees -- INVENTED; the source's spatial
#: frequency was not recovered, and the gain of a flow estimator depends on it,
#: which is one more reason the absolute values here are not evidence.
STRIPE_PERIOD_DEG = 20.0
FRAME_RATE_HZ = 50.0            # the source filmed at 50 Hz
SETTLE_FRAMES = 6
MEASURE_FRAMES = 24


def drum_frame(retina, phase_deg):
    """Render one view of a vertically-striped drum at a given phase.

    Built in ANGLE and then projected to pixels, so the stripes have a constant
    angular period as a real drum does. Painting them at constant pixel spacing
    would put a different spatial frequency in the periphery than at the centre
    and quietly change the measurement.
    """
    columns = np.arange(retina.pixels)
    half = math.radians(retina.fovy_deg) / 2.0
    focal_px = (retina.pixels / 2.0) / math.tan(half)
    azimuth = np.degrees(np.arctan((columns + 0.5 - retina.pixels / 2.0) / focal_px))
    stripes = 0.5 + 0.5 * np.sin(
        2.0 * math.pi * (azimuth - phase_deg) / STRIPE_PERIOD_DEG)
    frame = np.zeros((retina.pixels, retina.pixels, 3), dtype=np.uint8)
    row = np.clip(stripes * 255.0, 0, 255).astype(np.uint8)
    # Grey world with green and blue carrying the pattern; red is dropped by
    # the retina anyway, and leaving it flat makes that explicit.
    frame[:, :, 1] = row[None, :]
    frame[:, :, 2] = row[None, :]
    frame[:, :, 0] = 128
    return frame


def measure_gain(velocity_deg_s, left_eye=True, right_eye=True, direction=+1.0):
    """Head velocity divided by stimulus velocity, as the source measured it."""
    retina = Retina(fovy_deg=70.0, pixels=64, cells=16)
    pretectum = Pretectum(retina, left_eye=left_eye, right_eye=right_eye)
    dt = 1.0 / FRAME_RATE_HZ
    commands = []
    for frame_index in range(SETTLE_FRAMES + MEASURE_FRAMES):
        phase = direction * velocity_deg_s * frame_index * dt
        out = retina.step(drum_frame(retina, phase))
        command = pretectum.step(out, dt)
        if frame_index >= SETTLE_FRAMES:
            commands.append(command)
    mean_command = float(np.mean(commands))
    # Gain is a magnitude ratio: the sign says which way the head went and is
    # checked separately.
    return abs(mean_command) / velocity_deg_s, mean_command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/evidence/session8/okr_sweep.json")
    args = ap.parse_args()

    rows = []
    print(f"{'condition':<22}{'deg/s':>7}{'measured':>10}{'published':>11}")
    # ONE CAMERA. The published experiment covers one of TWO eyes; this body
    # has a single central head_cam, so "binocular" and "monocular" cannot be
    # distinguished on it and the rows below say so rather than pretending.
    # What IS testable is the direction asymmetry, which is the discriminating
    # half of the published result.
    conditions = (
        ("binocular", dict(left_eye=True, right_eye=True), +1.0),
        ("temporo_nasal", dict(left_eye=True, right_eye=False), +1.0),
        ("naso_temporal", dict(left_eye=True, right_eye=False), -1.0),
    )
    for name, eyes, direction in conditions:
        for velocity in PUBLISHED_VELOCITIES:
            gain, signed = measure_gain(velocity, direction=direction, **eyes)
            published = PUBLISHED_GAIN[name][velocity]
            rows.append({"condition": name, "velocity_deg_s": velocity,
                         "measured_gain": round(gain, 4),
                         "signed_command_deg_s": round(signed, 4),
                         "published_gain": published})
            print(f"{name:<22}{velocity:>7.0f}{gain:>10.3f}{published:>11.2f}")

    def gains(name):
        return [r["measured_gain"] for r in rows if r["condition"] == name]

    nt = gains("naso_temporal")
    tn = gains("temporo_nasal")
    binoc = gains("binocular")

    checks = {
        "naso_temporal_is_silent": {
            "predicted": "exactly zero at every velocity",
            "measured": nt,
            "pass": all(g < 1e-9 for g in nt),
        },
        "binocular_beats_monocular": {
            "predicted": "binocular gain exceeds temporo-nasal at every velocity",
            "measured": [round(b - t, 4) for b, t in zip(binoc, tn)],
            "pass": None,
            "untestable": (
                "THE BODY HAS ONE CAMERA. morphology/gecko_world_v1.xml:89 "
                "defines a single central head_cam; there is no second eye to "
                "cover, so binocular and monocular are the same condition here "
                "and the measured difference is identically zero. The "
                "published comparison needs two eyes. Adding one is a body "
                "change and the body is validated 14/14, so it does not happen "
                "as a side effect of a vision module."),
        },
        "gain_falls_with_velocity": {
            "predicted": "gain decreases as stimulus velocity rises",
            "measured": binoc,
            "pass": all(a >= b for a, b in zip(binoc, binoc[1:])),
        },
        "gain_never_reaches_one": {
            "predicted": "a perfectly stabilising loop has over-reproduced the animal",
            "measured": max(binoc) if binoc else None,
            "pass": all(g < 1.0 for g in binoc),
        },
    }
    testable = [c for c in checks.values() if c["pass"] is not None]
    predicted_pass = all(c["pass"] for c in testable)

    print("\n--- PREDICTED properties (the actual test) ---")
    for name, c in checks.items():
        mark = "UNTESTABLE" if c["pass"] is None else ("PASS" if c["pass"] else "FAIL")
        print(f"  {mark:<10} {name}: {c['predicted']}")
        print(f"             measured {c['measured']}")
        if c.get("untestable"):
            print(f"             WHY: {c['untestable'][:72]}...")

    print("\n--- FITTED values (context, NOT evidence) ---")
    print("  peak_gain and half_velocity_deg_s were chosen so the curve passes")
    print("  near the published points, so agreement there is a fit and not a")
    print("  reproduction. The four checks above are what this test gates on.")

    payload = {
        "schema_version": 1,
        "test": "optokinetic drum, Masseck Roll & Hoffmann 2008",
        "generated_by": "tools/okr_sweep.py",
        "published_source": (
            "Masseck, Roll & Hoffmann 2008, Vision Research 48:765-772, "
            "Eublepharis macularius, n = 4, head gain. The only quantitative "
            "sensorimotor measurement published for this species' vision."),
        "protocol": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "stripe_period_deg": STRIPE_PERIOD_DEG,
            "stripe_period_provenance": (
                "INVENTED -- the source's spatial frequency was not recovered, "
                "and a flow estimator's gain depends on it. One more reason "
                "the absolute gains here are not evidence."),
            "velocities_deg_s": list(PUBLISHED_VELOCITIES),
            "extrapolation": "refused -- only the three published velocities",
        },
        "rows": rows,
        "predicted_checks": checks,
        "fitted_not_evidence": ["peak_gain", "half_velocity_deg_s",
                                "stripe_period_deg"],
        "blocked": {
            "binocular_versus_monocular": (
                "needs two eyes; the body has one central camera"),
        },
        "verdict": (
            ("EVERY TESTABLE PREDICTED PROPERTY REPRODUCES, including the "
             "discriminating one: monocular naso-temporal gain is exactly zero "
             "at every velocity, which a symmetric flow estimator cannot "
             "produce. The binocular-versus-monocular comparison is UNTESTABLE "
             "on a one-eyed body and is recorded as blocked, not as passed.")
            if predicted_pass else
            "A PREDICTED PROPERTY FAILED -- the mechanism is wrong, not the fit"),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n{payload['verdict']}")
    print(f"written: {out}")
    return 0 if predicted_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
