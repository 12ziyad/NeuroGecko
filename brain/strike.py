"""Brain module 4d: the strike -- the thing that actually catches prey.

WHY THIS EXISTS, AND WHY ITS ABSENCE WAS FATAL. Until now this animal could
only walk. Session 9's literature sweep put three numbers side by side and the
conclusion was arithmetic:

    the walker moves at            0.055 m/s
    the cricket escapes at         0.093 - 0.143 m/s
    a real gecko strikes at        0.851 m/s

A gecko cannot run down a cricket, and the real animal does not try. It creeps
to roughly two centimetres and fires something fifteen times faster than its
walk. Capture success on evasive crickets is 82.9 %. Every session that tried
to make the hunting channel work by improving pursuit or improving the eye was
working on the wrong half of the problem, because there was nothing at the end
of the approach to succeed with.

WHAT IS PUBLISHED AND WHOSE IT IS. Almost none of this is the target species.
No strike distance, speed, acceleration or success rate has ever been measured
in *Eublepharis macularius*. The kinematics come from *Coleonyx variegatus*,
the western banded gecko -- same family, filmed at 500 fps, and mostly from
FIVE JUVENILES. Every number carries that in the registry.

The one thing that IS from the target species is the duration, and it is the
weakest number here: Delheusy, Brillet & Bels (1995) filmed six adult leopard
geckos at 64 fps and could digitise only five lateral capture cycles from two
animals -- not enough to analyse, so capture is described qualitatively there
and excluded from every table. The 80 ms figure is read off a figure. At 64 fps
that cycle is about five frames. It is an order of magnitude, not a
measurement, and the module says so rather than dressing it up.

THE ONE REAL CONTROL LAW. Attack distance and peak strike speed rise together,
r = 0.47, P = 0.009 (n = 9 adults, 29 trials). A longer strike is a faster
strike. That gives a scaling law instead of a constant, and it is the most
useful single relationship in the hunting literature. It is also only a
moderate correlation -- about 22 % of the variance -- so it is a tendency, and
modelling it as deterministic would discard real animal variability. The
residual spread is therefore kept.

TWO STRIKES, NOT ONE. The published *E. macularius* ethogram (Krönke & Xu 2023,
n = 18) names them separately: `snap`, defined with the pointed note that "a
snap can be unsuccessful", and `bag jump`, "jump towards a prey in order to bag
it". *Coleonyx* high-speed video shows the same split by prey type -- against
slow mealworms the animal walks right up and uses head and neck only; against
fast crickets it stops further back and pushes off with the hindlimbs. So the
mode here is chosen by distance, which is the observable the animal has.

WHAT IS DELIBERATELY ABSENT.
  * No tongue. Geckos take prey with the JAWS -- the standard scleroglossan
    pattern -- and the widely circulated "486 m/s^2, 5.8 m/s, 35 cm" figures
    are a chameleon's tongue (Wainwright, Kraklau & Bennett 1991) misattributed
    to geckos. A leopard gecko does not project its tongue 35 cm.
  * No pre-strike tail vibration. It is a real published ethogram category in
    this species and "mostly in context with prey" -- but its FUNCTION has
    never been tested in any eublepharid, and every confident explanation for
    it traces to pet-care sites. Absent and declared beats invented.
  * No head-orientation model. The published rule is that the head is aligned
    TRANSVERSE to the prey's long axis, but our prey is a sphere and has no
    long axis, so there is nothing here to align to.
  * No prey-type discrimination. Only one prey exists in this world.
"""

from __future__ import annotations

import math

import numpy as np

from common.provenance import parameter_value

#: How much of the strike's peak speed the distance scaling may move it by.
#: DERIVED from r = 0.47: the correlation explains r^2 = 22 % of the variance,
#: so the deterministic part of the relationship is given that share of the
#: span and the rest is left to the residual below. This is an interpretation
#: of a correlation coefficient, not a published effect size.
_SCALING_SHARE = 0.47 ** 2

#: Spread left on peak speed after the distance scaling, as a fraction. The
#: published data has real animal-to-animal variability that a deterministic
#: law throws away. INVENTED.
_SPEED_JITTER = 0.12


class Strike:
    """Fire, or do not. And if fired, hit or miss at the published rate."""

    def __init__(self, rng=None, success_probability=None):
        self.trigger_distance_m = float(parameter_value("strike_trigger_distance_m"))
        self.peak_speed_m_s = float(parameter_value("strike_peak_speed_m_s"))
        self.peak_acceleration_m_s2 = float(parameter_value("strike_peak_acceleration_m_s2"))
        self.duration_s = float(parameter_value("strike_duration_s"))
        self.correlation_r = float(parameter_value("strike_distance_speed_correlation_r"))
        self.success_probability = float(
            parameter_value("strike_capture_success") if success_probability is None
            else success_probability)
        if not 0.0 <= self.success_probability <= 1.0:
            raise ValueError("success_probability is a probability.")
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.reset()

    def reset(self):
        self._elapsed = 0.0
        self.active = False
        self.mode = None
        self.speed_m_s = 0.0
        self.will_hit = False
        self.strikes = 0
        self.hits = 0
        self.misses = 0

    # ---------------------------------------------------------------- firing
    def in_range(self, distance_m):
        """Is the prey close enough to strike at?"""
        return bool(np.isfinite(distance_m) and 0.0 <= distance_m <= self.trigger_distance_m)

    def speed_for(self, distance_m):
        """Peak speed for a strike launched from `distance_m`.

        Published in DIRECTION and in strength, not in slope: attack distance
        and peak velocity correlate at r = 0.47. The published means anchor the
        centre; distance moves it within the share of variance the correlation
        actually accounts for.
        """
        if not np.isfinite(distance_m):
            raise ValueError("distance must be finite")
        d = float(np.clip(distance_m, 0.0, self.trigger_distance_m))
        # -1 at contact, +1 at the trigger distance.
        offset = (2.0 * d / self.trigger_distance_m) - 1.0
        scaled = self.peak_speed_m_s * (1.0 + _SCALING_SHARE * offset)
        jitter = float(self._rng.normal(0.0, _SPEED_JITTER * self.peak_speed_m_s))
        return float(max(scaled + jitter, 0.05))

    def mode_for(self, distance_m):
        """`snap` up close, `bag_jump` from further out.

        Both are named categories in the published ethogram for this species,
        and the split by distance is what the *Coleonyx* video shows: head and
        neck only against slow prey approached closely, a hindlimb push-off
        against fast prey attacked from further back. The boundary is INVENTED
        -- no published distance separates them.
        """
        return "snap" if distance_m <= 0.5 * self.trigger_distance_m else "bag_jump"

    def fire(self, distance_m):
        """Commit to a strike. Returns the plan, or None if out of range.

        Whether it lands is decided HERE, at commitment, not on arrival. A real
        strike is ballistic: the animal is already decelerating 9 ms before
        contact and cannot correct. Deciding the outcome at launch is what
        makes the 82.9 % a property of the strike rather than of whatever the
        prey does during it.
        """
        if self.active or not self.in_range(distance_m):
            return None
        self.active = True
        self._elapsed = 0.0
        self.mode = self.mode_for(distance_m)
        self.speed_m_s = self.speed_for(distance_m)
        self.will_hit = bool(self._rng.random() < self.success_probability)
        self.strikes += 1
        return {"mode": self.mode, "peak_speed_m_s": self.speed_m_s,
                "launched_from_m": float(distance_m), "will_hit": self.will_hit}

    # --------------------------------------------------------------- running
    def step(self, dt):
        """Advance an in-flight strike. Returns (fraction, finished, hit)."""
        if not self.active:
            return 0.0, False, False
        self._elapsed += float(dt)
        fraction = min(self._elapsed / self.duration_s, 1.0)
        if fraction < 1.0:
            return fraction, False, False
        hit = self.will_hit
        self.active = False
        self.hits += int(hit)
        self.misses += int(not hit)
        return 1.0, True, hit

    def velocity_profile(self, fraction):
        """Speed at a point through the strike, as a fraction of peak.

        A single smooth rise and fall, peaking BEFORE contact. Published for
        *Coleonyx*: peak acceleration 16.3 ms and peak velocity 9.0 ms before
        contact, out of a strike of this length -- so the peak sits at roughly
        0.89 of the way through and the animal is decelerating on arrival. The
        shape between those points is INVENTED; only the location of the peak
        is published.
        """
        f = float(np.clip(fraction, 0.0, 1.0))
        peak_at = 1.0 - 9.03e-3 / self.duration_s
        if f <= peak_at:
            return float(math.sin(0.5 * math.pi * f / max(peak_at, 1e-6)))
        tail = (f - peak_at) / max(1.0 - peak_at, 1e-6)
        return float(math.cos(0.5 * math.pi * tail))

    def state(self):
        return {
            "active": self.active, "mode": self.mode,
            "strikes": self.strikes, "hits": self.hits, "misses": self.misses,
            "success_rate": (self.hits / self.strikes) if self.strikes else None,
            "published": {
                "trigger_distance_m": self.trigger_distance_m,
                "peak_speed_m_s": self.peak_speed_m_s,
                "capture_success": self.success_probability,
                "species": "Coleonyx variegatus for everything except duration",
            },
            "invented": ["mode boundary at half the trigger distance",
                         "speed jitter", "velocity profile shape"],
        }
