"""How a hunt is scored, and the shadow signal that makes a failure attributable.

TWO THINGS, BOTH INSTRUMENTATION, BOTH BEFORE THE ORACLE COMES OUT.

1. PATH EFFICIENCY, NOT TIME TO CAPTURE.

For this species the question is settled and it does not go the way a
simulation would naturally go. Across 42 animals, path length is significant
across training phase (F = 24.157, df = 538, p < 0.0001) and latency is NOT
(F = 0.326, df = 208, p = 0.568). Time is the wrong readout.

Worse than wrong -- actively misleading. Geckos found a hidden goal FASTER in
complete darkness than at the end of normal training, by moving roughly twice
as fast along paths about twice as long. A fast undirected search solves a
latency-scored task with no spatial knowledge whatsoever. Score a hunt by time
to capture and that is the strategy it rewards.

So the score here is path efficiency: straight-line distance to the prey
divided by the distance actually travelled getting there. An animal that walks
straight at its food scores near 1. One that thrashes about and stumbles into
it scores near 0, however fast it arrives.

2. THE ORACLE AS A SHADOW.

When the privileged prey position is disconnected from the policy, hunting will
get worse for two reasons at once: the eye may not see the prey, and the body
may not be able to reach it. Capture rate cannot tell those apart, and they
need opposite fixes.

So the oracle keeps running -- into the LOGGER ONLY, never into the animal.
Every step records what the eye said and what the truth was, side by side. The
difference between "never detected it" and "detected it and could not close" is
then a column rather than an argument.

WHAT THIS MODULE MUST NOT BECOME. The shadow must never reach the policy. That
is the whole point, and it is exactly the defect this project has shipped three
times: a privileged channel that was supposed to be off and was not (#26, #33,
#165). `shadow_only` is asserted on construction and the values are namespaced
under `shadow_` so a stray read is visible in a diff.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["HuntMetrics"]


class HuntMetrics:
    """Scores a hunt by how directly it was done, and logs the truth alongside."""

    #: Absolute bearing inside which the animal counts as approaching, degrees.
    #: PUBLISHED as a behavioural criterion in the mouse prey-capture work,
    #: where an approach is co-occurring decreasing range, |azimuth| < 90 deg
    #: and speed above a floor -- a combination that predicted 100 % of
    #: interceptions. Transferred, not measured in any lizard.
    APPROACH_BEARING_DEG = 90.0
    #: Speed floor for the same criterion, m/s. The mouse figure is 0.05 m/s
    #: against a 0.164 m/s locomotor speed -- roughly a third. Scaled to this
    #: animal's 0.055 m/s walk. DERIVED, and the scaling is an assumption.
    APPROACH_SPEED_M_S = 0.018

    def __init__(self, shadow_only=True):
        if not shadow_only:
            raise ValueError(
                "The oracle shadow exists to be logged, never to be consumed. "
                "Three privileged channels in this project were supposed to be "
                "off and were not; this one refuses to be switchable.")
        self.shadow_only = True
        self.reset()

    def reset(self):
        self._path_m = 0.0
        self._previous_xy = None
        self._start_range_m = None
        self.episodes = []
        self.last = {}

    # ------------------------------------------------------------- the score
    def begin_pursuit(self, mouth_xy, prey_xy):
        """Mark the start of an approach, so efficiency has a baseline."""
        self._path_m = 0.0
        self._previous_xy = np.asarray(mouth_xy, dtype=float)[:2].copy()
        self._start_range_m = float(np.linalg.norm(
            np.asarray(prey_xy, dtype=float)[:2] - self._previous_xy))

    def path_efficiency(self, mouth_xy, prey_xy):
        """Straight-line progress divided by distance actually walked.

        1.0 is a direct approach. Near 0 is a thrash that happened to end up in
        the right place. Undefined until the animal has moved.
        """
        if self._start_range_m is None or self._path_m <= 1e-9:
            return None
        here = np.asarray(mouth_xy, dtype=float)[:2]
        remaining = float(np.linalg.norm(np.asarray(prey_xy, dtype=float)[:2] - here))
        closed = self._start_range_m - remaining
        return float(np.clip(closed / self._path_m, -1.0, 1.0))

    def is_approaching(self, bearing_deg, speed_m_s, range_closing):
        """The published three-part criterion, all three required."""
        return bool(abs(bearing_deg) < self.APPROACH_BEARING_DEG
                    and speed_m_s > self.APPROACH_SPEED_M_S
                    and range_closing)

    # ------------------------------------------------------------ the shadow
    def step(self, *, mouth_xy, prey_xy, true_bearing_deg,
             eye_bearing_deg=None, eye_salience=0.0, speed_m_s=0.0):
        """One step of scoring. Returns a row; consumes nothing.

        `true_bearing_deg` and `prey_xy` are the ORACLE. They are recorded and
        must not reach the animal.
        """
        here = np.asarray(mouth_xy, dtype=float)[:2]
        if self._previous_xy is None:
            self.begin_pursuit(here, prey_xy)
        self._path_m += float(np.linalg.norm(here - self._previous_xy))
        self._previous_xy = here.copy()

        prey = np.asarray(prey_xy, dtype=float)[:2]
        true_range = float(np.linalg.norm(prey - here))
        closing = (self._start_range_m is not None
                   and true_range < self._start_range_m)

        detected = bool(eye_salience > 0.0 and eye_bearing_deg is not None)
        error = (abs(float(eye_bearing_deg) - float(true_bearing_deg))
                 if detected else None)

        self.last = {
            "path_efficiency": self.path_efficiency(here, prey),
            "path_travelled_m": round(self._path_m, 4),
            "approaching": self.is_approaching(true_bearing_deg, speed_m_s, closing),
            # Everything below is the shadow. Namespaced so a stray read shows
            # up in a diff, and never returned to the policy.
            "shadow_true_range_m": round(true_range, 4),
            "shadow_true_bearing_deg": round(float(true_bearing_deg), 2),
            "shadow_eye_detected": detected,
            "shadow_eye_bearing_deg": (round(float(eye_bearing_deg), 2)
                                       if detected else None),
            "shadow_bearing_error_deg": (round(error, 2) if error is not None else None),
            "shadow_eye_salience": round(float(eye_salience), 4),
            # The column that separates the two failures from each other.
            "shadow_failure_mode": (
                "not detected" if not detected else
                "detected, bearing wrong" if error is not None and error > 30.0 else
                "detected, bearing usable"),
        }
        return dict(self.last)

    def summarise(self, rows):
        """Turn a run of rows into the numbers worth arguing about."""
        rows = [r for r in rows if r]
        if not rows:
            return {}
        eff = [r["path_efficiency"] for r in rows if r["path_efficiency"] is not None]
        det = [r for r in rows if r["shadow_eye_detected"]]
        err = [r["shadow_bearing_error_deg"] for r in det
               if r["shadow_bearing_error_deg"] is not None]
        modes = {}
        for r in rows:
            modes[r["shadow_failure_mode"]] = modes.get(r["shadow_failure_mode"], 0) + 1
        return {
            "steps": len(rows),
            "mean_path_efficiency": round(float(np.mean(eff)), 4) if eff else None,
            "approaching_fraction": round(
                float(np.mean([r["approaching"] for r in rows])), 4),
            "eye_detected_fraction": round(len(det) / len(rows), 4),
            "eye_mean_bearing_error_deg": round(float(np.mean(err)), 2) if err else None,
            "failure_modes": modes,
        }
