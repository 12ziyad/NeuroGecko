"""Brain module 3a: the spinal central pattern generator -- the leg rhythm.

WHAT WAS THERE BEFORE, and why it had to change. The walking controller
computes each foot's place in its stride as

    phase = (time * frequency + touchdown_delay) % 1.0

That is a closed-form clock. It has no state, the legs are not coupled to each
other, nothing on the ground can influence it, and -- the reason this module
exists -- THERE IS NO INPUT FOR A DESCENDING COMMAND. A brainstem's whole job
is to say "go faster" or "turn", and there was nowhere for that to land: of the
four output channels a brainstem is specified to produce, exactly one had
anything to connect to. So the cord has to become a real oscillator before the
brainstem is worth building. That ordering was the plan's, reversed.

WHAT THIS IS. Four per-leg phase oscillators, integrated rather than evaluated:

    dphi_i/dt = omega  -  sigma * N_i * cos(phi_i)

`omega` is the stride frequency, now a settable input rather than a constant.
`N_i` is the ground reaction force under leg i and `sigma` is the strength of
that feedback -- the decentralised-control law named in the research corpus
(Owaki & Ishiguro). With `sigma = 0` the law reduces to dphi/dt = omega, which
is the old clock, and that is the default.

SIGMA IS ZERO BY DEFAULT AND TAGGED INVENTED. The corpus NAMES this law but
gives no value for sigma, for omega, or for how N is scaled, and neither source
paper has been read. Shipping it switched on with a number chosen to look right
is precisely the failure this project exists to avoid, so the mechanism is here,
the default is off, and the constant is declared.

ON "BIT-IDENTICAL", WHICH THE PLAN ASKED FOR AND IS NOT ACHIEVABLE. The
intention was right: at sigma = 0 the new cord must reproduce the accepted
walker exactly, so that any later change in the gait gates is attributable to
the coupling and not to the rewrite. But an integrated phase and a closed-form
one cannot agree bit for bit:

  * the clock computes `time * frequency` as one product; the oscillator
    accumulates `omega * dt` a step at a time, and floating-point addition is
    not associative;
  * the closed form lands exactly on cycle boundaries, and the lab profile has
    explicit hygiene snapping values within 1e-12 to 0.0 or to the stance
    ratio. An accumulator does not land on those boundaries at all, so that
    hygiene is a property of the closed form rather than of the gait.

What IS achievable, and what the test asserts, is agreement to about 1e-12 over
a full minute of walking -- far below any quantity the gait gates measure.
Claiming bit-identity here would have been an overclaim, and the plan's
requirement is recorded as unachievable rather than quietly downgraded.
"""

from __future__ import annotations

import math

import numpy as np

#: Foot order is POSITIONAL and is taken from the repository rather than
#: restated here. Writing it out again is how a silent index mismatch gets in:
#: the canonical order is hind-left, fore-left, hind-right, fore-right, which is
#: not the order anyone guesses.
try:
    from common.gait_config import FOOT_ORDER as FEET
except Exception:                                    # pragma: no cover
    FEET = ("HL", "FL", "HR", "FR")


class SpinalCPG:
    """Four coupled leg oscillators with a settable descending drive.

    The cord owns the rhythm. The brainstem will own `omega` and the left/right
    asymmetry; nothing else about a decision reaches the legs.
    """

    #: Largest internal integration step, seconds. The phase derivative is
    #: bounded, so this is about resolution rather than stability -- but the
    #: caller's control rate should not be the solver's rate. That mistake cost
    #: this project two sessions in the basal ganglia, where a 50 Hz step
    #: against a 40 ms time constant made a stable model oscillate.
    MAX_INTERNAL_DT_S = 0.002

    def __init__(self, frequency_hz, touchdown_delays, load_feedback=0.0,
                 feet=FEET):
        if not math.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive.")
        self.feet = tuple(feet)
        delays = np.asarray([float(touchdown_delays[f]) for f in self.feet])
        if not np.all(np.isfinite(delays)):
            raise ValueError("touchdown delays must be finite.")
        self.base_frequency_hz = float(frequency_hz)
        self.touchdown_delays = delays
        #: sigma. INVENTED, and zero unless someone deliberately turns it on.
        self.load_feedback = float(load_feedback)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Start at t = 0, which puts each foot at its own touchdown delay."""
        self._t = 0.0
        # The clock's phase at t=0 is (0*f + delay) % 1, so the oscillator
        # starts where the clock starts. Anything else would make the two
        # disagree from the first step for a reason that is not the coupling.
        self._phase = np.mod(-self.touchdown_delays, 1.0)
        self._frequency = np.full(len(self.feet), self.base_frequency_hz)
        return self

    @property
    def elapsed_s(self):
        return self._t

    # ------------------------------------------------ the descending command
    def set_drive(self, frequency_hz=None, left_right_bias=0.0):
        """The brainstem's seam. Sets stride frequency and a turning asymmetry.

        `left_right_bias` in [-1, 1] speeds one side and slows the other, which
        is how a symmetric pattern generator produces a turn. The MAPPING from
        a descending command to a frequency is NOT set here and is NOT invented
        here -- this takes a frequency directly. Whatever converts "drive" into
        hertz belongs in the brainstem, with its own provenance.
        """
        f = self.base_frequency_hz if frequency_hz is None else float(frequency_hz)
        if not math.isfinite(f) or f <= 0:
            raise ValueError("frequency_hz must be finite and positive.")
        bias = float(left_right_bias)
        if not math.isfinite(bias) or not -1.0 <= bias <= 1.0:
            raise ValueError("left_right_bias must lie in [-1, 1].")
        left = np.array([foot.endswith("L") for foot in self.feet])
        self._frequency = np.where(left, f * (1.0 + bias), f * (1.0 - bias))
        return self._frequency.copy()

    # ------------------------------------------------------------------- step
    def step(self, dt_s, loads=None):
        """Advance the rhythm. `loads` is the per-foot ground reaction force.

        Returns the phase vector. Loads are ignored entirely when the feedback
        gain is zero, which is the default, so a caller that has no force
        readings loses nothing.
        """
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive.")
        if loads is None:
            load = np.zeros(len(self.feet))
        else:
            load = np.asarray([float(loads[f]) for f in self.feet]) \
                if isinstance(loads, dict) else np.asarray(loads, dtype=float)
            if load.shape != (len(self.feet),):
                raise ValueError("loads must give one value per foot.")
            if not np.all(np.isfinite(load)):
                raise ValueError("loads must be finite.")

        substeps = int(math.ceil(dt_s / self.MAX_INTERNAL_DT_S))
        inner = dt_s / substeps
        for _ in range(substeps):
            dphi = self._frequency
            if self.load_feedback != 0.0:
                # Owaki & Ishiguro's decentralised law: a loaded leg is held in
                # stance, an unloaded one is hurried through swing, and the legs
                # coordinate through the ground rather than through each other.
                dphi = dphi - self.load_feedback * load * np.cos(
                    2.0 * math.pi * self._phase)
            self._phase = np.mod(self._phase + dphi * inner, 1.0)
            self._t += inner
        return self.phases()

    def advance_to(self, time_s, loads=None):
        """Integrate forward to an absolute time and return the phase vector.

        The walking controller is written against absolute time -- callers ask
        "where is this foot at t?" rather than stepping a clock -- so the cord
        has to answer that question without changing every caller.

        Asking for a time in the PAST replays from zero rather than raising.
        The gait scorer probes times out of order, and with the coupling off
        the trajectory is a pure function of elapsed time, so replaying is
        exact. With the coupling ON it is not: phase would then depend on the
        load history, and a replay with different loads would silently give a
        different answer. That case is refused rather than approximated.
        """
        t = float(time_s)
        if not math.isfinite(t) or t < 0.0:
            raise ValueError("time_s must be finite and non-negative.")
        if t < self._t:
            if self.load_feedback != 0.0:
                raise ValueError(
                    "cannot replay to an earlier time with load feedback on: "
                    "phase depends on the load history, so the answer would "
                    "depend on loads this call does not have.")
            saved = self._frequency.copy()
            self.reset()
            self._frequency = saved
        remaining = t - self._t
        if remaining > 0:
            self.step(remaining, loads=loads)
        return self.phases()

    # ---------------------------------------------------------------- readout
    def phases(self):
        return self._phase.copy()

    def phase_fraction(self, foot):
        """This foot's place in its own stride, in [0, 1)."""
        return float(self._phase[self.feet.index(foot)])

    def clock_phase_fraction(self, foot, time_s=None):
        """What the OLD closed-form clock would say, for comparison only.

        Kept so the regression test can be written against the thing it is
        replacing rather than against a remembered version of it.
        """
        t = self._t if time_s is None else float(time_s)
        i = self.feet.index(foot)
        cycle = (t * self.base_frequency_hz) % 1.0
        return float((cycle - self.touchdown_delays[i]) % 1.0)

    def max_clock_deviation(self, seconds=60.0, dt_s=0.004):
        """Largest disagreement with the old clock over a run, as a fraction of
        a stride. Zero coupling only -- with feedback on, disagreeing is the
        entire point."""
        probe = SpinalCPG(self.base_frequency_hz,
                          dict(zip(self.feet, self.touchdown_delays)),
                          load_feedback=0.0, feet=self.feet)
        worst = 0.0
        steps = int(round(seconds / dt_s))
        for _ in range(steps):
            probe.step(dt_s)
            for foot in probe.feet:
                a = probe.phase_fraction(foot)
                b = probe.clock_phase_fraction(foot)
                d = abs(a - b)
                worst = max(worst, min(d, 1.0 - d))   # phase is circular
        return worst

    def state(self):
        return {
            "feet": list(self.feet),
            "phase": [round(float(x), 9) for x in self._phase],
            "frequency_hz": [round(float(x), 6) for x in self._frequency],
            "elapsed_s": round(self._t, 6),
            "load_feedback": self.load_feedback,
        }
