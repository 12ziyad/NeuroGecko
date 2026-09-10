"""Brain module 6: the day/night clock — when the animal is awake at all.

WHAT THIS IS AND, MORE IMPORTANTLY, WHAT IT IS NOT.

It is a circadian gate: a model of WHEN this animal is active, driving an
arousal level that everything else can be gated on. A gecko asleep in a crevice
at noon should not be hunting, and until now nothing in this project knew what
time it was.

It is NOT a model of sleep architecture. That distinction is the whole reason
this module is small. *E. macularius* does appear in the reptile sleep
literature -- Bergel et al. 2026 recorded seven lizard species and found
sleep-dependent infraslow rhythms conserved between reptiles and mammals, with
n = 2 leopard geckos. That paper establishes that this animal sleeps, and that
its eye movements can be measured at all (electrodes under each eyelid; it
closes them, unlike a tokay, which has none).

What it does not give is a cycle period this project can defend. A well-known
figure exists for the bearded dragon and borrowing it would take one line --
and would put an agamid's number in a eublepharid's mouth, which is the exact
substitution `config/proxies.yaml` exists to make visible. So
`sleep_cycle_period_s` is recorded as **null**, and any future module that
wants a sleep cycle has to declare that it is inventing one.

WHAT IS PUBLISHED, AND HOW THIN IT IS.

    onset after dark   81 min   n = 1     SD 89, range 11-247
    evening peak       18:00-23:00        n = 18, CAPTIVE

The first is the only actograph-style dataset that exists for this species and
its standard deviation exceeds its mean. The second comes from a welfare study
that scored behaviour during the window it expected activity in, so it is
partly a description of the experimenters. There is NO field activity budget
for a wild leopard gecko: no emergence times, no retreat times, no
activity-versus-light-level curve.

Both are used because they are what exists, and both are labelled `uncertain`.

CREPUSCULAR, NOT SIMPLY NOCTURNAL. The literature is genuinely split -- the same
species is called nocturnal in one paper and crepuscular in another -- and the
mechanistic result favours the second: preferred body temperature rises through
the light phase and peaks toward its end, which Angilletta et al. propose is
what initiates evening emergence. So arousal here rises before dark rather than
switching on at it.
"""

from __future__ import annotations

import math

import numpy as np

from common.provenance import parameter_value

#: Length of a day, seconds. Not a measurement.
DAY_S = 24 * 3600.0

#: Fraction of the day that is dark. INVENTED as an engineering default -- the
#: captive studies used 12:12 and 10h45:13h15 light cycles, and no wild
#: photoperiod has been recorded for this animal at its own latitude.
DARK_FRACTION = 0.5

#: How sharply arousal rises around dusk, in hours. INVENTED. The animal is
#: crepuscular rather than switched, and the one mechanistic hint -- preferred
#: temperature rising through the afternoon -- says the rise begins BEFORE
#: dark, but nothing measures its slope.
DUSK_RAMP_H = 1.5


class Arousal:
    """Time of day in, arousal in [0, 1] out."""

    def __init__(self, dark_fraction=DARK_FRACTION, dusk_ramp_h=DUSK_RAMP_H):
        self.dark_fraction = float(dark_fraction)
        self.dusk_ramp_h = float(dusk_ramp_h)
        if not 0.0 < self.dark_fraction < 1.0:
            raise ValueError("dark_fraction is a fraction of the day")
        self.onset_after_dark_s = float(
            parameter_value("activity_onset_after_dark_s"))
        self.peak_window_h = float(parameter_value("activity_peak_window_h"))
        #: null in the registry, on purpose. Read so that anything depending on
        #: it fails loudly rather than silently substituting another species.
        self.sleep_cycle_period_s = parameter_value("sleep_cycle_period_s")
        self.reset()

    def reset(self, time_of_day_s=0.0):
        self.time_of_day_s = float(time_of_day_s) % DAY_S
        self.last = {}

    # ------------------------------------------------------------ the clock
    @property
    def dusk_s(self):
        """When darkness begins, seconds from midnight."""
        return DAY_S * (1.0 - self.dark_fraction)

    def is_dark(self, time_of_day_s=None):
        t = self.time_of_day_s if time_of_day_s is None else float(time_of_day_s) % DAY_S
        return bool(t >= self.dusk_s or t < DAY_S * (1.0 - self.dark_fraction) - DAY_S * (1.0 - self.dark_fraction))

    def arousal_at(self, time_of_day_s):
        """Arousal in [0, 1] at a time of day.

        Zero through the light phase, rising across dusk, peaking through the
        published evening window, and decaying toward dawn. The SHAPE is
        INVENTED; the peak window and the onset lag are not, and both are thin.
        """
        t = float(time_of_day_s) % DAY_S
        dusk = self.dusk_s
        ramp = self.dusk_ramp_h * 3600.0
        # Hours since dusk, wrapping through midnight.
        since = t - dusk
        if since < 0:
            since += DAY_S
        night_length = DAY_S * self.dark_fraction
        if since > night_length:
            # Daytime. The animal shelters; arousal is the crepuscular rise
            # only in the last stretch before dusk.
            until_dusk = DAY_S - since
            if until_dusk <= ramp:
                return float(np.clip(1.0 - until_dusk / ramp, 0.0, 1.0)) * 0.6
            return 0.0
        # Night. Rise over the onset lag, hold through the peak window, decay.
        peak = self.peak_window_h * 3600.0
        if since < self.onset_after_dark_s:
            return float(0.6 + 0.4 * (since / max(self.onset_after_dark_s, 1e-9)))
        if since <= peak:
            return 1.0
        remaining = night_length - peak
        return float(np.clip(1.0 - (since - peak) / max(remaining, 1e-9), 0.0, 1.0))

    def step(self, dt_s):
        """Advance the clock. Returns arousal and whether it is dark."""
        self.time_of_day_s = (self.time_of_day_s + float(dt_s)) % DAY_S
        arousal = self.arousal_at(self.time_of_day_s)
        dark = self.time_of_day_s >= self.dusk_s
        self.last = {
            "time_of_day_h": round(self.time_of_day_s / 3600.0, 4),
            "arousal": round(arousal, 6),
            "dark": bool(dark),
            "asleep": bool(arousal < 0.1),
        }
        return dict(self.last)

    def state(self):
        return dict(
            self.last,
            published={"onset_after_dark_min": round(self.onset_after_dark_s / 60, 1),
                       "onset_n": 1,
                       "peak_window_h": self.peak_window_h,
                       "peak_window_n": 18,
                       "species": "E_macularius"},
            invented=["the shape of the arousal curve", "DARK_FRACTION",
                      "DUSK_RAMP_H"],
            not_in_corpus=["sleep_cycle_period_s -- this animal is in the "
                           "reptile sleep literature but no cycle period has "
                           "been established for it, and borrowing the bearded "
                           "dragon's would put an agamid's number in a "
                           "eublepharid's mouth"],
            caveats=["the onset lag is n=1 with SD larger than its mean",
                     "the peak window is captive and partly describes the "
                     "experimenters' schedule",
                     "no field activity budget exists for this species"])
