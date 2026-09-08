"""Brain module 4c: the optic tectum -- "there is prey, and it is over there".

This is the piece that finally retires the colour matcher. The old prey signal
counted pixels matching a colour, which is not how eyes work, which went
silently blind when the prey stopped being green, and which cannot work anyway:
the prey is a brown cricket on sand and the nearest chromatic confuser is the
gecko's own spots, 0.145 away in chromaticity.

WHAT MAKES SOMETHING PREY, AND WHY IT IS NOT A COLOUR.

The tectum's job is to separate a small moving object from a moving world. That
distinction is the mechanism, and it is the same one that separates this module
from the pretectum next door:

    FULL-FIELD motion is SELF-MOTION. The whole scene sliding together is the
    animal walking or turning. It goes to the pretectum and stabilises gaze.

    LOCAL motion that DISAGREES with the field is an OBJECT. Something moving
    differently from its own surroundings is, to a first approximation,
    something alive.

So the tectum asks two questions in this order, and the ORDER was wrong the
first time: first, does the motion FILL THE FIELD, judged on the raw map --
because that is the world going past. Only then does it subtract the field and
look for what is left.

Both are needed and neither is sufficient, which is a measurement rather than
an opinion. Against pure self-motion with no prey present:

    field subtraction alone, no size test    salience 0.266   (1.000 without)
    size test applied AFTER subtraction      salience 0.066   leaks
    size test applied BEFORE, as shipped     salience 0.000

Subtracting first leaves a ragged residual whose surviving peaks then look
small and local and slip through. An earlier version of this docstring claimed
the subtraction alone stops the animal hunting its own optic flow; measurement
says it reduces it fourfold and does not stop it, and the claim is corrected
rather than quietly kept.

SMALL. Prey subtends about 3.4 degrees at 30 cm -- roughly three pixels at this
field of view. So the detector is centre-surround in SPACE as well: a target
that fills the field is a wall, not a cricket. Real tectal prey-recognition is
far more selective than this (the toad literature's worm-versus-anti-worm
configural selectivity is the classic result), and nothing of that kind has been
measured in a gecko, so this stops at small-and-moving and says so.

WHAT IS PUBLISHED HERE AND WHAT IS NOT.

  PUBLISHED, qualitative: the retina projects topographically to tectal layers
  8-14 in *Gekko gecko*, n=12, silver degeneration. That is the warrant for a
  retinotopic map and a bearing readout, and it is a structure, not a number.

  PUBLISHED, quantitative, and used as the reach test: prey capture distance
  4.07 cm, strike speed 0.851 m/s, capture success 82.9 %. The tectum must
  still see the prey at the distance the animal is measured to strike from.

  NOT PUBLISHED, and therefore INVENTED and declared: every threshold here.
  No tectal recording exists for this species, and no visual feature has been
  shown to trigger prey-directed behaviour in it.
"""

from __future__ import annotations

import math

import numpy as np

#: INVENTED. Motion, after the field is subtracted, below which a cell is
#: called empty. Set from the noise floor of a still scene, not from what makes
#: a demo look good: a perfectly still world must return exactly zero.
MOTION_FLOOR = 1e-4
#: INVENTED. Largest fraction of the visual field a target may fill before it
#: is treated as background rather than as an object.
MAX_TARGET_FRACTION = 0.25


class Tectum:
    """Retinotopic map in; bearing and salience out."""

    def __init__(self, retina, motion_floor=MOTION_FLOOR,
                 max_target_fraction=MAX_TARGET_FRACTION,
                 subtract_field=True, surround_cells=3):
        self.retina = retina
        self.motion_floor = float(motion_floor)
        self.max_target_fraction = float(max_target_fraction)
        #: Leave this on. Off only in the test that measures what it is worth,
        #: which is a fourfold reduction in the response to self-motion.
        self.subtract_field = bool(subtract_field)
        #: Width of the surround each cell is compared against, in cells.
        #: INVENTED.
        self.surround_cells = int(surround_cells)
        self.last = {"salience": 0.0, "bearing_deg": 0.0, "elevation_cell": None}

    def step(self, retina_output):
        """Returns salience in [0, 1] and the bearing of the best target.

        Bearing is signed degrees from the optical axis, negative to the LEFT,
        which is the convention the brainstem's turn channel expects.
        """
        motion = np.asarray(retina_output["motion"], dtype=float)
        raw_peak = float(motion.max()) if motion.size else 0.0

        # ORDER MATTERS, and it was wrong the first time. "Is this a
        # whole-field event?" is asked of the RAW motion, before anything is
        # subtracted. Asking it afterwards lets self-motion through: removing
        # the median leaves a ragged residual with a few surviving peaks, which
        # then looks small and local and passes the size test. Measured, pure
        # self-motion with no prey present:
        #
        #     size test after subtraction   salience 0.066   <- leaks
        #     size test before subtraction  salience 0.000
        #
        # A target that fills the field is a wall, or the world going past,
        # and either way it is not a cricket.
        if raw_peak > self.motion_floor:
            filling = float(np.mean(motion > 0.5 * raw_peak))
            if filling > self.max_target_fraction:
                self.last = {"salience": 0.0, "bearing_deg": 0.0,
                             "elevation_cell": None, "rejected": "fills the field"}
                return 0.0, 0.0

        if self.subtract_field:
            # SUBTRACTING A SCALAR WAS A NO-OP AND THIS IS THE CORRECTION.
            #
            # The previous version did `motion - float(np.median(motion))`.
            # clip(x - c, 0, None) is monotonic in x for any SCALAR c, so it
            # cannot move the argmax -- and the argmax is where the reported
            # bearing comes from. Checked on 20000 random maps: the argmax was
            # identical every single time. The stage had never changed a
            # reported bearing, and an earlier "measured worth: 1.000 -> 0.266"
            # in this file was measuring the salience VALUE, not the bearing,
            # which is the quantity that matters.
            #
            # A local-versus-global comparison has to be SPATIALLY VARYING to
            # do anything. Each cell is now measured against its own
            # surroundings, which is what object-motion-sensitive retinal cells
            # actually do: they respond when local motion differs from the
            # background and fall silent when the whole field moves together.
            local = motion - _surround(motion, self.surround_cells)
        else:
            local = motion.copy()
        local = np.clip(local, 0.0, None)

        peak = float(local.max()) if local.size else 0.0
        if peak <= self.motion_floor:
            self.last = {"salience": 0.0, "bearing_deg": 0.0,
                         "elevation_cell": None}
            return 0.0, 0.0

        row, column = np.unravel_index(int(np.argmax(local)), local.shape)
        bearing = self.retina.azimuth_of(int(column))
        # Salience is the target's strength relative to the strongest thing the
        # retina can report, so it arrives in [0, 1] as the selector expects.
        salience = float(min(peak / max(raw_peak, 1e-9), 1.0))
        self.last = {"salience": round(salience, 6),
                     "bearing_deg": round(bearing, 4),
                     "elevation_cell": int(row),
                     "field_subtracted": self.subtract_field}
        return salience, bearing

    def state(self):
        return dict(self.last,
                    invented=["MOTION_FLOOR", "MAX_TARGET_FRACTION"])


def _surround(plane, width):
    """Mean of the neighbourhood around each cell, edges extended.

    This has to be SPATIALLY VARYING. A scalar reference cannot change which
    cell is the largest, which is the whole reason the previous version did
    nothing.
    """
    if width < 1:
        return np.zeros_like(plane)
    padded = np.pad(plane, width, mode="edge")
    out = np.zeros_like(plane)
    n = 0
    for dy in range(-width, width + 1):
        for dx in range(-width, width + 1):
            if dx == 0 and dy == 0:
                continue
            out += padded[width + dy:width + dy + plane.shape[0],
                          width + dx:width + dx + plane.shape[1]]
            n += 1
    return out / max(n, 1)


class Eye:
    """Retina, pretectum and tectum as one object, because they share a frame.

    The three run off the same image and it would be easy to step them out of
    order or to give two of them different frames. This makes that impossible.
    """

    def __init__(self, fovy_deg=70.0, pixels=64, cells=16,
                 render_pixels=None):
        from brain.retina import Retina
        from brain.pretectum import Pretectum
        self.retina = Retina(fovy_deg=fovy_deg, pixels=pixels, cells=cells,
                             render_pixels=render_pixels)
        self.pretectum = Pretectum(self.retina)
        self.tectum = Tectum(self.retina)
        self.last = {}

    def reset(self):
        self.retina.reset()
        return self

    def step(self, image, dt_s):
        """One frame through the whole eye."""
        retina_output = self.retina.step(image)
        salience, bearing = self.tectum.step(retina_output)
        gaze = self.pretectum.step(retina_output, dt_s)
        self.last = {
            "prey_salience": salience,
            "prey_bearing_deg": bearing,
            "gaze_command_deg_s": gaze,
        }
        return dict(self.last)
