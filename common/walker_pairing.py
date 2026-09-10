"""Refuse, or at least shout about, walker configurations that are known to be wrong.

WHY THIS FILE EXISTS. Session 9 rendered every one of its videos with a trained
residual bolted onto the `lab` CPG. Measured across all four combinations of
gait profile and policy, that is the ONLY pairing that limps:

    lab    + policy      front feet 0.205 / 0.581   gap 0.376   <-- broken
    lab    + no policy               0.457 / 0.437   gap 0.021
    legacy + policy                  0.483 / 0.500   gap 0.017
    legacy + no policy               0.463 / 0.491   gap 0.028

One front foot carries load for a fifth of the step and the other for well over
half. The user spotted it by watching the animal and I spent six exchanges
blaming the camera, the frame rate and the body before measuring it.

Nothing in the code objected, because nothing in the code knew. The pairing was
assembled by two independent defaults -- `GeckoWalkEnv(gait_profile="legacy")`
and whatever the caller passed -- and a checkpoint path with no metadata about
what it was trained against. Any future agent or human wiring these together
would land in exactly the same place.

WHAT IS AND IS NOT ENFORCED HERE. The broken pairing WARNS rather than raises.
It is a real configuration that runs and produces numbers; refusing it outright
would make the evidence that condemns it unreproducible, and this project keeps
its failures runnable. What it must not do is happen silently.

The gate-accepted walker is a separate matter and is stated rather than
enforced: the 4/6 evidence is the ZERO-RESIDUAL base (FAILURE_MAP #71), and the
trained residual measured 3/6 against it (#11), after 2 x 3.01 M steps. Loading
the checkpoint at all is therefore a step away from the accepted configuration
even on a profile where it does not limp -- but the brain environment needs a
steering policy, so this is a note, not a veto.
"""

from __future__ import annotations

import warnings

__all__ = ["PairingWarning", "check_pairing", "describe_pairing",
           "ACCEPTED", "MEASURED"]


class PairingWarning(UserWarning):
    """A walker configuration this project has measured and found wrong."""


#: The configuration the gate battery accepted. FAILURE_MAP #71, #11.
ACCEPTED = {"gait_profile": "lab", "policy": False,
            "gates": "4 of 6", "front_duty": (0.457, 0.437)}

#: Every combination measured, 800 steps each, seed 0, same body and world.
#: artifacts/evidence/session9/four_combinations.json
MEASURED = {
    ("lab", True): {"front_duty": (0.205, 0.581), "gap": 0.376, "ok": False},
    ("lab", False): {"front_duty": (0.457, 0.437), "gap": 0.021, "ok": True},
    ("legacy", True): {"front_duty": (0.483, 0.500), "gap": 0.017, "ok": True},
    ("legacy", False): {"front_duty": (0.463, 0.491), "gap": 0.028, "ok": True},
}

#: Above this left/right difference in front-foot duty the animal is visibly
#: limping. DERIVED: the three sound pairings sit at 0.017-0.028 and the broken
#: one at 0.376, so anything past an order of magnitude above the sound band is
#: unambiguous. Not a published threshold -- no published one exists.
LIMP_GAP = 0.20


def describe_pairing(gait_profile: str, using_policy: bool) -> str:
    """One line about what this combination was measured to do."""
    row = MEASURED.get((gait_profile, bool(using_policy)))
    if row is None:
        return (f"gait_profile={gait_profile!r} with "
                f"{'a policy' if using_policy else 'no policy'}: not measured.")
    left, right = row["front_duty"]
    verdict = "balanced" if row["ok"] else "LIMPS"
    return (f"gait_profile={gait_profile!r} with "
            f"{'a trained policy' if using_policy else 'no policy'}: "
            f"front feet {left:.3f} / {right:.3f}, gap {row['gap']:.3f} -- {verdict}.")


def check_pairing(gait_profile: str, using_policy: bool, *, context: str = "") -> bool:
    """Warn if this walker configuration is one this project measured as wrong.

    Returns True when the pairing is sound. Warns and returns False when it is
    the known-broken one. Never raises: the broken configuration has to stay
    runnable so the evidence against it can be reproduced.
    """
    row = MEASURED.get((gait_profile, bool(using_policy)))
    if row is None or row["ok"]:
        return True

    left, right = row["front_duty"]
    where = f" ({context})" if context else ""
    warnings.warn(
        f"\n"
        f"  BROKEN WALKER PAIRING{where}\n"
        f"  gait_profile={gait_profile!r} with a trained residual loaded on top.\n"
        f"\n"
        f"  Measured front-foot duty: {left:.3f} left, {right:.3f} right "
        f"(gap {row['gap']:.3f}).\n"
        f"  One front foot carries load for a fifth of the step and the other\n"
        f"  for over half. The animal visibly limps. Every other combination of\n"
        f"  profile and policy sits at a gap of 0.017-0.028.\n"
        f"\n"
        f"  The gate-accepted walker is the ZERO-RESIDUAL base: 4 of 6 gates\n"
        f"  (FAILURE_MAP #71). The trained residual measured 3 of 6 against it\n"
        f"  (#11) after 2 x 3.01 M steps and was rejected.\n"
        f"\n"
        f"  If you want the accepted walker, do not load the checkpoint.\n"
        f"  If you want a steering policy, gait_profile='legacy' does not limp.\n"
        f"  If you meant this, pass warn_pairing=False and say why in writing.\n",
        PairingWarning, stacklevel=3)
    return False
