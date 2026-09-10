"""Brain module 5: smell — the sense this animal actually gates on.

WHY THIS IS NOT AN AFTERTHOUGHT. Everything built before it assumed vision was
the animal's route to prey and to danger. For this species, measured, it is not:

    antipredator reaction to snake SCENT alone      0.21
    antipredator reaction to the SIGHT of a snake   0.07

    chemical      chi2 = 8.098, p < 0.0044
    visual        chi2 < 0.01,  p > 0.9
    mechanosensory chi2 < 0.01, p > 0.9

n = 42, this species, and the authors put it plainly: the initiation of costly
defensive action remains strictly gated by chemoreception. Vision and touch
modulated attention and amplified chemical sampling; neither on its own
produced a defensive response. A flee channel wired to the eye is close to
modelling the wrong sense.

Geckos are also unusual among squamates in being nasal-olfaction specialists --
a large main olfactory bulb relative to the accessory bulb -- and eublepharids
lean on vomerolfaction more than other gekkotans. The vomeronasal organ is
present and well developed in *E. macularius* (µCT, 9 embryos).

WHAT THIS MODULE MAY AND MAY NOT DO, and the line is the whole design.

    MAY: say how strongly the air smells of prey or of predator, and raise
    tongue-flick rate accordingly. Published for this species: 3.0 flicks per
    minute against an odourless control, 14.57 against cricket chemicals, a
    4.9x rise, n = 7.

    MAY NOT: say WHERE. The swab assay that produced those numbers presents the
    stimulus about one centimetre from the snout, in plain view. It shows the
    animal can TELL cricket chemicals from a control. It shows nothing about
    localisation, and NO GECKO HAS EVER BEEN SHOWN TO FIND PREY BY SMELL ALONE.
    Cooper flagged the missing experiment in 1998 -- "experimental selective
    blocking of senses is needed to demonstrate conclusively the separate
    sensory roles" -- and in the decades since nobody has run it.

So this returns a scalar per odour and no bearing. A gradient would be the
easiest thing in the world to write and would be an invention with a published
number's clothes on, which is the defect this project exists to avoid.

WHAT IS INVENTED HERE AND SAYS SO: how concentration falls with distance. No
odour plume has been measured around any prey item for any gecko. A plume in
still air is not a simple function of range, and in moving air it is not a
function of range at all. The inverse-square-with-floor used here is a
placeholder whose only defensible property is that it decreases.
"""

from __future__ import annotations

import math

import numpy as np

from common.provenance import parameter_value

#: How far a scent is detectable at all, metres. INVENTED. No detection
#: distance has been published for any odour for any gecko. Set to the far edge
#: of the prey arena so the sense has a range at all, and deliberately larger
#: than the eye's useful range so the two senses disagree rather than agreeing
#: by construction.
MAX_RANGE_M = 0.60

#: Distance at which concentration is called 1.0. INVENTED. Roughly the swab
#: assay's own presentation distance, which is the only distance anything about
#: this sense has ever been measured at.
REFERENCE_M = 0.01


class Vomeronasal:
    """Concentration in, tongue-flick rate and arousal out. Never a bearing."""

    def __init__(self, max_range_m=MAX_RANGE_M, reference_m=REFERENCE_M):
        self.baseline_rate = float(parameter_value("tongue_flick_baseline_per_min"))
        self.prey_rate = float(parameter_value("tongue_flick_prey_per_min"))
        self.predator_chemical = float(parameter_value("predator_reaction_chemical"))
        self.predator_visual = float(parameter_value("predator_reaction_visual"))
        self.max_range_m = float(max_range_m)
        self.reference_m = float(reference_m)
        if not (self.reference_m > 0 and self.max_range_m > self.reference_m):
            raise ValueError("reference distance must be positive and inside the range")
        self.reset()

    def reset(self):
        self.last = {"prey_odour": 0.0, "predator_odour": 0.0,
                     "tongue_flicks_per_min": self.baseline_rate,
                     "bearing_deg": None}

    # ------------------------------------------------------------- the sense
    def concentration(self, distance_m):
        """Odour strength in [0, 1] at a distance. INVENTED falloff.

        Inverse-square with a floor and a hard cut at max range. Its only
        defensible property is that it decreases: no odour plume has been
        measured around any prey item for any gecko, and a real plume in moving
        air is not a function of range at all.
        """
        if distance_m is None or not np.isfinite(distance_m):
            return 0.0
        d = float(distance_m)
        if d >= self.max_range_m:
            return 0.0
        if d <= self.reference_m:
            return 1.0
        return float(np.clip((self.reference_m / d) ** 2, 0.0, 1.0))

    def flick_rate(self, prey_odour):
        """Tongue-flicks per minute at this prey-odour strength.

        Published endpoints, this species, n=7: 3.0 per minute at an odourless
        control and 14.57 at cricket chemicals. The interpolation BETWEEN them
        is INVENTED -- only the two ends were measured.
        """
        p = float(np.clip(prey_odour, 0.0, 1.0))
        return float(self.baseline_rate + p * (self.prey_rate - self.baseline_rate))

    def step(self, prey_distance_m=None, predator_distance_m=None):
        """One sniff. Returns strengths and a flick rate, and never a direction."""
        prey = self.concentration(prey_distance_m)
        predator = self.concentration(predator_distance_m)
        self.last = {
            "prey_odour": round(prey, 6),
            "predator_odour": round(predator, 6),
            "tongue_flicks_per_min": round(self.flick_rate(prey), 3),
            # Stated explicitly rather than omitted, so that anything reading
            # this output finds an assertion instead of a missing key and
            # cannot quietly invent a gradient later.
            "bearing_deg": None,
            "localisation": "NOT AVAILABLE -- no gecko has been shown to find "
                            "prey by smell alone",
        }
        return dict(self.last)

    # ------------------------------------------------------- the gating rule
    def defensive_probability(self, predator_odour, predator_visible=False):
        """Chance of a defensive reaction, gated on chemistry.

        The published factorial, this species, n=42: scent alone 0.21, sight
        alone 0.07, and only the chemical term is significant. Vision amplifies
        chemical sampling rather than acting on its own, so it contributes here
        only in the presence of odour.
        """
        chemical = float(np.clip(predator_odour, 0.0, 1.0)) * self.predator_chemical
        if predator_visible:
            # Additive, not multiplicative: the authors describe integration as
            # additive, with secondary modalities heightening alertness rather
            # than gating.
            chemical += self.predator_visual * float(predator_odour > 0.0)
        return float(np.clip(chemical, 0.0, 1.0))

    def state(self):
        return dict(
            self.last,
            published={"baseline_flicks_per_min": self.baseline_rate,
                       "prey_flicks_per_min": self.prey_rate,
                       "defensive_p_chemical": self.predator_chemical,
                       "defensive_p_visual": self.predator_visual,
                       "species": "E_macularius"},
            invented=["odour falloff with distance", "MAX_RANGE_M",
                      "REFERENCE_M", "the interpolation between the two "
                      "published flick rates"],
            refuses=["bearing -- smell may say THAT, never WHERE"])
