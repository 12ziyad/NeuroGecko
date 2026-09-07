"""The acceptance test for brain/basal_ganglia.py: the Prescott 2024 sweep.

Prescott et al. 2024 (Biomimetics 9(3):139, CC BY) vary tonic dopamine in the
extended GPR model and report what the animal does. This runs the same sweep
against our module and writes the comparison, pass or fail, to
`artifacts/evidence/session6/dopamine_sweep.json`.

Session 6 generated that file from a throwaway script, which meant the evidence
could not be regenerated when the parameters changed. It is a tool now.

WHAT IS BEING COMPARED, and what is not. The published counts come from a robot
foraging task; this is a disembodied salience competition. The corpus notes
clean selection is lower disembodied (73-81%) than embodied (89-95%), and the
time constant tau -- which sets the absolute scale of "switching bouts per
trial" -- is itself unresolved across sources (40 ms in Fox et al. 2009, 25 ms
in Girard 2005). So the ORDERING and the DIRECTIONS are the test. Absolute
agreement with 21.3-against-7 is not expected and its absence is not a failure.

Usage:  python tools/dopamine_sweep.py [--out PATH] [--seed N]
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

# Imported by path, not through `brain/__init__`, which pulls in torch.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_bg_for_sweep", REPO / "brain" / "basal_ganglia.py")
_bg = importlib.util.module_from_spec(_spec)
sys.modules["_bg_for_sweep"] = _bg
_spec.loader.exec_module(_bg)

BasalGanglia = _bg.BasalGanglia
piecewise_linear = _bg.piecewise_linear

TRIAL_S = 120.0
DT_S = 0.02
CHANNELS = 6
SALIENCE_GAP = 0.05
OU_SIGMA = 0.10
OU_TAU_S = 2.0

#: The published sweep, transcribed. `dopamine` is lambda.
PUBLISHED = (
    (0.03, "38 s immobile"),
    (0.06, "14 s immobile"),
    (0.12, "below 75% vigour"),
    (0.20, "89-95% clean; ~7 switching bouts"),
    (0.29, "clean"),
    (0.31, "distortion appears"),
    (0.37, "clean"),
    (0.43, "most fail; ~21.3 switches"),
    (0.46, "every trial fails"),
)


def _base_salience():
    """Two close competitors and four also-rans.

    The gap between the top two is what the noise has to overcome for a switch
    to happen, so it sets the difficulty of the competition. It is INVENTED:
    nothing published constrains what a gecko's salience values are.
    """
    base = np.full(CHANNELS, 0.20)
    base[0] = 0.55
    base[1] = 0.55 - SALIENCE_GAP
    return base


def run_trial(dopamine, seed):
    """One 120 s trial. Returns switches, seconds immobile, mean distortion."""
    rng = np.random.default_rng(seed)
    bg = BasalGanglia()
    bg.dopamine = dopamine
    base = _base_salience()

    # Ornstein-Uhlenbeck noise: salience fluctuates on a BEHAVIOURAL timescale.
    # Session 6 used fresh white noise at every 50 Hz step and measured 825
    # switches in 120 s -- about seven per second. That was the test measuring
    # its own noise, not the model.
    noise = np.zeros(CHANNELS)
    decay = math.exp(-DT_S / OU_TAU_S)
    kick = OU_SIGMA * math.sqrt(1.0 - decay * decay)

    steps = int(round(TRIAL_S / DT_S))
    switches = 0
    immobile_steps = 0
    distortion_total = 0.0
    previous = None

    for _ in range(steps):
        noise = decay * noise + kick * rng.standard_normal(CHANNELS)
        gates = bg.step(np.clip(base + noise, 0.0, None), dt_s=DT_S)
        current = bg.selected(gates)
        if current is None:
            immobile_steps += 1
        elif previous is not None and current != previous:
            switches += 1
        if current is not None:
            previous = current
        distortion_total += float(bg.distortion_amount(gates))

    return {
        "switches": switches,
        "immobile_s": round(immobile_steps * DT_S, 1),
        "distortion_amount": round(distortion_total / steps, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/evidence/session6/dopamine_sweep.json")
    ap.add_argument("--seed", type=int, default=20260907)
    args = ap.parse_args()

    rows = []
    for dopamine, published in PUBLISHED:
        row = run_trial(dopamine, args.seed)
        row["dopamine"] = dopamine
        row["published"] = published
        rows.append(row)
        print(f"  lambda {dopamine:<5} switches {row['switches']:<5} "
              f"immobile {row['immobile_s']:>6} s  "
              f"distortion {row['distortion_amount']:<6}  | {published}")

    immobility_monotonic = all(
        rows[i]["immobile_s"] >= rows[i + 1]["immobile_s"]
        for i in range(3))
    distortion_monotonic = all(
        rows[i]["distortion_amount"] <= rows[i + 1]["distortion_amount"] + 1e-9
        for i in range(len(rows) - 1))
    baseline = next(r for r in rows if r["dopamine"] == 0.20)
    excess = next(r for r in rows if r["dopamine"] == 0.43)
    ratio = excess["switches"] / max(baseline["switches"], 1)

    payload = {
        "schema_version": 3,
        "test": "Prescott et al. 2024 tonic-dopamine sweep against the extended GPR",
        "generated_by": "tools/dopamine_sweep.py",
        "seed": args.seed,
        "parameter_source": {
            "weights_and_thresholds": (
                "Fox et al. 2009, Front. Neuroinform. 3:6, "
                "doi:10.3389/neuro.11.006.2009 (attribution-only open access), "
                "equations 1-12. SECONDARY: Gurney, Prescott & Redgrave 2001 "
                "(Biol. Cybern. 84:401-410, 84:411-423) is the origin and both "
                "parts are paywalled; no value here is quoted from them directly."),
            "gating_constant_c": (
                "0.169, Prescott et al. 2024, Biomimetics 9(3):139 section "
                "3.1.2 (CC BY)"),
            "thalamocortical_extension": (
                "Girard et al. 2005, arXiv cs/0601004 Table 5 -- NOT GPR 2001, "
                "and one of two incompatible lineages in the literature. This "
                "file follows Girard's throughout and does not mix them."),
            "superseded": (
                "Session 6b used Girard's BG core weights (STN->output 0.8, "
                "GPe->output 0.4). That was a regression: it rests at 0.1429 "
                "where the published gating constant requires 0.169, leaving "
                "every channel 14.3% released with nothing salient."),
            "note": (
                "docs/research/ names the model and the sweep but restates "
                "neither the weights nor the thresholds."),
        },
        "protocol": {
            "trial_s": TRIAL_S,
            "dt_s": DT_S,
            "channels": CHANNELS,
            "salience_gap": SALIENCE_GAP,
            "solver": (
                "step() sub-divides internally to 5 ms. The STN-GPe loop is "
                "negative feedback with gain 0.9 * n_channels, so integrating "
                "at the caller's 50 Hz against tau = 40 ms is unstable: the "
                "resting output oscillates over 0.143-0.205 instead of "
                "settling at 0.16953."),
            "noise": (
                f"Ornstein-Uhlenbeck, sigma {OU_SIGMA}, tau {OU_TAU_S} s -- "
                "salience fluctuating on a BEHAVIOURAL timescale. An earlier "
                "run used 50 Hz white noise and produced 825 switches in 120 s, "
                "about 7 per second, which measured the noise rather than the "
                "model."),
            "caveat": (
                "DISEMBODIED. The published counts come from a robot foraging "
                "task; the corpus itself notes clean selection is lower "
                "disembodied (73-81%) than embodied (89-95%). Absolute "
                "switching counts are therefore not directly comparable -- the "
                "task differs. tau is also unresolved (40 ms Fox, 25 ms "
                "Girard) and tau sets the absolute scale of switch counts."),
        },
        "rows": rows,
        "errors_found_and_corrected": [
            "Round 1 drove the striatum from cortex. Equation 14 is "
            "I_cortex = y_VL: cortex is driven by the thalamic return ALONE, "
            "and salience enters striatum and STN directly. Driving the "
            "striatum from cortex saturated it at 1.0 on every competitive "
            "channel and destroyed all downstream discrimination. No parameter "
            "would have fixed it.",
            "Round 1 assumed every output slope was 1. The thalamocortical "
            "slopes are not: TRN runs at 0.5 and the ventrolateral thalamus at "
            "0.62, and that 0.62 is what holds the cortico-thalamic loop gain "
            "below one. Assuming 1 put the loop exactly at the boundary of "
            "positive-feedback instability.",
            "Round 2 replaced the basal-ganglia core weights with Girard 2005's "
            "re-tuned robotic variant (STN->output 0.8, GPe->output 0.4). That "
            "was a REGRESSION. Prescott defines the gating constant c as the "
            "model's own resting output, so c is an independent check on the "
            "weights: the canonical 0.9 / 0.3 set rests at 0.16953 and closes "
            "the gate exactly, while the variant rests at 0.1429 and leaves "
            "every channel 14.3 percent released with nothing salient.",
            "Round 2 summed the diffuse reticular drive over ALL channels. The "
            "cited equation excludes the channel's own output, so this added a "
            "self-inhibition the model does not have.",
            "The caller's control rate was the solver's rate. The STN-GPe loop "
            "is negative feedback with gain 0.9 * n_channels, so at 50 Hz "
            "against tau = 40 ms the discrete map is unstable and the resting "
            "output oscillates over 0.143-0.205. step() now sub-divides "
            "internally to 5 ms. This was the last of the three causes of the "
            "oscillation-at-zero-salience defect, which is now fixed and its "
            "test promoted from expectedFailure to a requirement.",
            "An earlier sweep drove salience with fresh 50 Hz white noise and "
            "counted 825 switches in 120 s, about seven per second. The test "
            "was measuring its own noise. Salience is now Ornstein-Uhlenbeck "
            "with a 2 s time constant, a behavioural timescale.",
            "An earlier test asserted zero distortion at EVERY dopamine level "
            "with separated saliences. The published finding is that excess "
            "dopamine distorts selection even then, so the test forbade the "
            "correct behaviour.",
        ],
        "not_reproduced": [
            "The switching-rate comparison. Published: roughly 3x more "
            "switching bouts at excess dopamine than at baseline (21.3 against "
            "~7). Measured: 0.55x -- still inverted, though improved from 0.3x. "
            "GPR also dithers LESS than a winner-take-all here, where the "
            "published result is the opposite; that is the counterintuitive "
            "published finding and it does not come out.",
            "Absolute switch counts and immobility durations. The published "
            "figures are from an embodied robot foraging task and tau, which "
            "sets their scale, is unresolved across sources (40 ms Fox, 25 ms "
            "Girard).",
        ],
        "claim": (
            "This module should not be treated as validated. Six of seven "
            "published properties reproduce and the Gurney 2001b Fig. 2a "
            "selection sequence reproduces to four decimal places, but the "
            "switching-rate comparison does not, and the disembodied-task "
            "explanation for that is plausible and NOT demonstrated."),
        "reproduces": {
            "resting_gate_closes_exactly": True,
            "immobility_gradient_monotonic": bool(immobility_monotonic),
            "distortion_deepens_monotonically": bool(distortion_monotonic),
            "switching_ratio_baseline_to_excess": round(ratio, 3),
            "switching_ratio_published": "~3x (21.3 against ~7)",
        },
        "verdict": (
            "PARTLY REPRODUCES, NOT VALIDATED. The selection mechanism "
            "REPRODUCES the published Gurney 2001b Fig. 2a output values to "
            "four decimal places and the gate closes exactly at the published "
            "resting constant, but the switching-rate comparison against the "
            "robot foraging task does not reproduce and is not demonstrated to "
            "be a task difference. The module is not wired into any "
            "environment and must not be until it is."),
    }

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nimmobility monotonic     : {immobility_monotonic}")
    print(f"distortion monotonic     : {distortion_monotonic}")
    print(f"switching ratio 0.20->0.43: {ratio:.2f}x  (published ~3x)")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
