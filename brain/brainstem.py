"""Brain module 3b: the brainstem -- turning a decision into a command.

The basal ganglia releases ONE behaviour. The spinal cord produces the leg
rhythm. The brainstem is the piece between them: it takes "the animal has
chosen to hunt, and the prey is over there" and produces the two things the
cord can actually accept -- how fast to step, and how much to favour one side.

WHAT IS PUBLISHED HERE AND WHAT IS NOT. This is the thinnest-evidence module in
the project so far and the docstring says so up front.

  PUBLISHED, direction only:
    * Stimulating the midbrain locomotor region drives locomotion, and stronger
      stimulation gives a faster rhythm. Measured in salamander, where the
      preparation steps at 0.08-0.16 Hz -- an order of magnitude below a
      gecko's stride -- so the DIRECTION transfers and the numbers do not.
    * Turning comes from left/right asymmetry in the descending drive, not
      from a separate turning command.
    * Midbrain stimulation in *Gekko gecko* itself elicits walking and both
      left and right turns. That is the only work in the target family, its
      locus is published, and its currents and frequencies are not: the
      chapter is paywalled and only the abstract was read.

  NOT PUBLISHED, and therefore INVENTED, declared, and kept in one place:
    * every number in the drive-to-frequency map. There is no published
      relationship between descending drive and stride frequency for this
      species, or for any reptile.

  DELIBERATELY ABSENT:
    * a gait-selection channel. This species does NOT change footfall pattern
      with speed -- relative phase is not significantly different across
      0.2-1.1 m/s, and limb phase is 44 +/- 1.1 % walking against 43 +/- 1.8 %
      running. A channel that switches gaits would be modelling something the
      animal does not do, so there is none, and that absence is a finding
      rather than an omission.

THE FREQUENCY BAND IS THE HONEST PROBLEM. The cord is locked at 1.1888 Hz, a
number tagged INVENTED that exists to keep an old checkpoint loadable. The only
published stride frequency for this species is 2.03 +/- 0.18 Hz at 0.18 m/s.
The band here is anchored on the LOCK, not on the published figure, because the
walker that passes 4 of 6 gates runs at the lock and changing it invalidates
that walker. So the drive maps onto a band around a number that is not the
animal's. That is recorded here, in the registry, and in the failure map, and
it is the first thing to fix if the lock is ever broken.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_scpg = _load("_spinal_cpg_for_brainstem", "spinal_cpg.py")
SpinalCPG = _scpg.SpinalCPG

#: Behaviours that mean "move". Everything else stops the legs. Which
#: behaviours are locomotor is a fact about the ethogram, not about the
#: brainstem, so it is stated here once rather than assumed at each use.
LOCOMOTOR = {"hunt": 1.0, "flee": 1.0, "explore": 0.55, "bask": 0.35}

#: INVENTED. The drive-to-frequency map. Nothing published relates descending
#: drive to stride frequency in this species or any reptile.
#:
#: `DRIVE_THRESHOLD` is the drive below which the animal does not step at all:
#: a real locomotor region has a threshold, below which stimulation produces
#: nothing, and that much is published in direction. Its VALUE is invented.
DRIVE_THRESHOLD = 0.15
#: INVENTED. Half-width of the frequency band, as a fraction of the locked
#: frequency. Anchored on the LOCK and not on the published 2.03 Hz -- see the
#: module docstring; this is the compromise the frequency lock forces.
FREQUENCY_SPAN = 0.10
#: INVENTED. How hard a bearing error turns the animal.
TURN_GAIN = 0.6
#: Largest asymmetry the cord will be asked for. Beyond this the two sides are
#: doing different gaits rather than turning.
MAX_TURN = 0.35


class Brainstem:
    """Chosen behaviour plus a bearing, in. Stride frequency and turn, out."""

    def __init__(self, cord, locked_frequency_hz=None, locomotor=None):
        # Duck-typed on purpose. `brain/__init__` pulls in torch, so every
        # module here is loaded by path, and the same file loaded under two
        # names yields two distinct classes -- an isinstance check against one
        # of them rejects a perfectly good cord from the other. Check for what
        # is actually needed instead.
        for required in ("set_drive", "base_frequency_hz", "phases"):
            if not hasattr(cord, required):
                raise TypeError(
                    f"the brainstem drives a spinal cord; this object has no "
                    f"{required!r}")
        self.cord = cord
        self.base_hz = float(locked_frequency_hz
                             if locked_frequency_hz is not None
                             else cord.base_frequency_hz)
        if not math.isfinite(self.base_hz) or self.base_hz <= 0:
            raise ValueError("locked_frequency_hz must be finite and positive.")
        self.locomotor = dict(LOCOMOTOR if locomotor is None else locomotor)
        self.last = {"behaviour": None, "drive": 0.0, "frequency_hz": 0.0,
                     "turn": 0.0, "stepping": False}

    # ------------------------------------------------------------------ drive
    def drive_for(self, behaviour, urgency=1.0):
        """How hard to run, in [0, 1], for a chosen behaviour.

        `urgency` is the gate value the basal ganglia released the behaviour
        with, so a half-released behaviour drives the legs half as hard. That
        is the whole reason for using a disinhibition model rather than a
        maximum: the strength of the decision survives into the movement.
        """
        if behaviour is None:
            return 0.0
        if not math.isfinite(urgency):
            raise ValueError("urgency must be finite.")
        return float(self.locomotor.get(behaviour, 0.0)
                     * min(max(urgency, 0.0), 1.0))

    def frequency_for(self, drive):
        """Stride frequency for a drive. Below threshold the animal stands.

        Returns 0.0 when not stepping, which is a real state and not an error:
        a resting or grooming gecko has no stride frequency.
        """
        if drive <= DRIVE_THRESHOLD:
            return 0.0
        span = (drive - DRIVE_THRESHOLD) / (1.0 - DRIVE_THRESHOLD)
        low = self.base_hz * (1.0 - FREQUENCY_SPAN)
        high = self.base_hz * (1.0 + FREQUENCY_SPAN)
        return float(low + span * (high - low))

    def turn_for(self, bearing_error_rad):
        """Left/right asymmetry from a bearing error, clipped.

        Positive bearing error means the target is to the LEFT, so the right
        side must step faster -- which is a NEGATIVE bias, because the cord's
        bias speeds the left side. Getting that sign wrong produces an animal
        that turns away from everything it wants, so it is asserted by test.
        """
        if not math.isfinite(bearing_error_rad):
            raise ValueError("bearing_error_rad must be finite.")
        turn = -TURN_GAIN * math.sin(float(bearing_error_rad))
        return float(max(-MAX_TURN, min(MAX_TURN, turn)))

    # ------------------------------------------------------------------- step
    def step(self, behaviour, urgency=1.0, bearing_error_rad=0.0):
        """Command the cord. Returns what was commanded.

        Does NOT advance the cord: the caller owns the clock, because the cord
        is stepped by the walking controller at its own rate. This sets the
        command and leaves the rhythm to run.
        """
        drive = self.drive_for(behaviour, urgency)
        frequency = self.frequency_for(drive)
        turn = self.turn_for(bearing_error_rad) if frequency > 0.0 else 0.0
        if frequency > 0.0:
            self.cord.set_drive(frequency_hz=frequency, left_right_bias=turn)
        self.last = {"behaviour": behaviour, "drive": round(drive, 6),
                     "frequency_hz": round(frequency, 6),
                     "turn": round(turn, 6), "stepping": frequency > 0.0}
        return dict(self.last)

    def step_from_selector(self, selector, bearing_error_rad=0.0):
        """Drive straight from the basal ganglia's own output.

        The seam between brain 2 and brain 3: the chosen behaviour AND the gate
        value it was chosen with, so a hesitant decision produces a hesitant
        walk.
        """
        behaviour = selector.selected()
        if behaviour is None:
            return self.step(None)
        gates = selector.gates()
        urgency = float(gates[list(selector.channels).index(behaviour)])
        return self.step(behaviour, urgency, bearing_error_rad)

    def state(self):
        return dict(self.last, base_frequency_hz=self.base_hz,
                    invented=["DRIVE_THRESHOLD", "FREQUENCY_SPAN", "TURN_GAIN"])
