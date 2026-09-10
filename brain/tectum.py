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


def _published(name):
    from common.provenance import parameter_value
    return float(parameter_value(name))


class Tectum:
    """Retinotopic map in; bearing and salience out."""

    def __init__(self, retina, motion_floor=MOTION_FLOOR,
                 max_target_fraction=MAX_TARGET_FRACTION,
                 subtract_field=True, surround_cells=3,
                 velocity_band=True, elevation_switch=True):
        self.retina = retina
        # TWO PUBLISHED RULES THE DETECTOR DID NOT HAVE.
        #
        # It asked only "small?" and "moving differently from its background?".
        # Two of the six filters the literature describes, and the four missing
        # ones are why it fires on anything that moves. These add the two with
        # the strongest evidence behind them. The other two are deliberately
        # NOT added, and the reasons are in the module docstring.
        #
        # VELOCITY BAND. Four independent lineages converge on the same order:
        # toad prey-catching peaks 30-60 deg/s and dies above ~200; pit viper
        # tectal units are near-silent at 5 deg/s and peak near 50; mouse
        # looming escape is released only at 35 deg/s. The pit viper is a
        # squamate -- the same tectum this module is a model of -- and it is
        # the closest species anyone has recorded. No lizard has ever had its
        # visual tectal velocity tuning measured, so this is a transfer, and it
        # says so.
        #
        # ELEVATION SWITCH. The same stimulus means opposite things above and
        # below the horizon. Identical looming disc: 75 % escape overhead
        # against 53 % and much slower to the side. Identical small sweeping
        # disc: freeze overhead, and 80 % APPROACH to the side. Floor-mounted
        # looms produced zero escapes. It is anatomically grounded -- medial
        # colliculus carries the upper field to the escape pathways, lateral
        # carries the lower field to the hunting pathway -- and the ecology
        # (raptors above, insects below) applies to this animal at least as
        # strongly. No reptile has been tested.
        self.velocity_band = bool(velocity_band)
        self.elevation_switch = bool(elevation_switch)
        self.v_peak = _published("prey_velocity_peak_deg_s")
        self.v_low = _published("prey_velocity_low_cut_deg_s")
        self.v_high = _published("prey_velocity_high_cut_deg_s")
        self.motion_floor = float(motion_floor)
        self.max_target_fraction = float(max_target_fraction)
        #: Leave this on. Off only in the test that measures what it is worth,
        #: which is a fourfold reduction in the response to self-motion.
        self.subtract_field = bool(subtract_field)
        #: Width of the surround each cell is compared against, in cells.
        #: INVENTED.
        self.surround_cells = int(surround_cells)
        self.last = {"salience": 0.0, "bearing_deg": 0.0, "elevation_cell": None}
        #: Frames of peak-position history the speed estimate is fitted over.
        #: INVENTED. 25 frames is half a second at 50 Hz, which is roughly what
        #: prey at 0.30 m needs to cross one cell.
        self.speed_window = 25
        #: Displacement below which the estimate is refused rather than
        #: guessed. ONE FULL CELL, and the value was measured rather than
        #: picked. At half a cell the estimator returned answers it could not
        #: support: a 9 deg/s target read 52 (480 % error) and a 27 deg/s one
        #: read 58 (114 %). At one cell the surviving estimates are accurate to
        #: about 3 % from 60 deg/s upward, and slower targets are refused
        #: instead of guessed. DERIVED from the map geometry.
        self.min_resolvable_deg = 1.0 * (retina.fovy_deg / retina.cells)
        self._history = []

    def velocity_gain(self, deg_per_s):
        """How much a target moving at this angular speed counts.

        1.0 at the peak, falling to zero at both published cuts. The SHAPE
        between them is INVENTED -- no tuning curve has been published for any
        lizard -- but the peak and the two cuts are not.

        WHAT THIS RULE CANNOT DO AT THE CURRENT MAP RESOLUTION, measured over
        120 frames of sustained motion:

            true 9 deg/s    measured 52    484 % error
            true 27         measured 58    114 %
            true 60         measured 62      3 %
            true 150        measured 155     3 %
            true 400        measured 414     3 %

        Accurate from roughly 60 deg/s upward and wrong below it. A cricket at
        the registry's ambient speed subtends 5-27 deg/s across the working
        range of 0.10-0.50 m, which is entirely inside the unreliable band, so
        THIS RULE DOES NOT CURRENTLY DISCRIMINATE REAL PREY. It rejects things
        that are genuinely far too fast or too slow and passes everything in
        between.

        It does no harm -- the over-estimate lands near the peak, so prey is
        accepted rather than wrongly rejected -- but it is not doing the job
        the literature describes. Fixing that needs a finer retinotopic map or
        a longer window, not a different threshold: raising the resolvability
        floor from half a cell to a full cell changed the 9 deg/s error from
        480 % to 484 %.
        """
        v = abs(float(deg_per_s))
        if not (self.v_low < v < self.v_high):
            return 0.0
        if v <= self.v_peak:
            return float((v - self.v_low) / max(self.v_peak - self.v_low, 1e-9))
        return float((self.v_high - v) / max(self.v_high - self.v_peak, 1e-9))

    def elevation_gain(self, row, rows):
        """Below the horizon is food; above it is not.

        Returns 1.0 in the lower half of the visual field and 0.0 in the upper.
        A hard switch rather than a soft weight, because that is what was
        measured: the same stimulus produced escape overhead and approach to
        the side, not a graded blend.
        """
        if rows < 2:
            return 1.0
        return 1.0 if row >= rows / 2.0 else 0.0

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

        rejected = None
        gain = 1.0
        # How fast the peak moved across the retina since the last frame, in
        # degrees per second. Measured between successive peak azimuths rather
        # than from the motion magnitude, because the magnitude is contrast and
        # this rule is about angular speed.
        speed = None
        dt = float(retina_output.get("dt_s") or 0.0)
        # SUB-CELL, and the first version was not. Taking the peak CELL and
        # differencing its azimuth quantises speed to one cell width per frame:
        # on a 16-cell map across 70 degrees at 50 Hz that is 219 deg/s per
        # step, so the estimate could only ever read 0, 219, 438... and the
        # whole velocity band -- 5 to 200 -- fell between two of its values.
        # The rule would have looked present and been arithmetic noise.
        #
        # A brightness-weighted centroid of the cells around the peak gives a
        # position between cells, which is what makes a 9 deg/s cricket
        # distinguishable from a stationary one at all.
        centre = _centroid_column(local, int(column))
        fine_bearing = self.retina.azimuth_of(centre)
        # A WINDOW, BECAUSE ONE FRAME CANNOT POSSIBLY MEASURE THIS.
        #
        # Real prey moves a small fraction of one cell per frame. At the
        # registry's 0.047 m/s ambient speed, on a 16-cell map across 70
        # degrees at 50 Hz:
        #
        #     0.10 m   26.9 deg/s   0.123 cells/frame    8 frames per cell
        #     0.30 m    9.0 deg/s   0.041 cells/frame   24 frames per cell
        #     0.50 m    5.4 deg/s   0.025 cells/frame   41 frames per cell
        #
        # Differencing successive frames therefore reads whatever the centroid
        # noise happens to be: measured, the same 0.3 px/frame stimulus gave
        # 110 deg/s and a 0.8 px/frame one gave 0.0. The band it feeds -- 5 to
        # 200 deg/s -- is finer than the instrument, which is precisely the
        # kind of rule that looks present and is arithmetic noise. This module
        # has shipped one of those already.
        #
        # So displacement is measured across the longest window held, and the
        # estimate is REFUSED rather than guessed when the target has not yet
        # moved far enough to be resolvable.
        self._history.append((fine_bearing, dt))
        if len(self._history) > self.speed_window:
            self._history.pop(0)
        if len(self._history) >= 2:
            span = sum(d for _, d in self._history[1:])
            travelled = abs(self._history[-1][0] - self._history[0][0])
            if span > 0 and travelled >= self.min_resolvable_deg:
                speed = travelled / span

        # A refused estimate does not veto the target. The rule can only
        # reject what it has actually measured; treating "not yet resolvable"
        # as "too slow" would blind the animal for the first half-second of
        # every encounter.
        if self.velocity_band and speed is not None:
            g = self.velocity_gain(speed)
            gain *= g
            if g <= 0.0:
                rejected = ("too slow" if speed <= self.v_low else "too fast")

        if self.elevation_switch:
            g = self.elevation_gain(int(row), int(local.shape[0]))
            gain *= g
            if g <= 0.0:
                rejected = "above the horizon"

        if gain <= 0.0:
            self.last = {"salience": 0.0, "bearing_deg": 0.0,
                         "elevation_cell": int(row), "rejected": rejected,
                         "peak_speed_deg_s": speed}
            return 0.0, 0.0
        # Salience is the target's strength relative to the strongest thing the
        # retina can report, so it arrives in [0, 1] as the selector expects.
        salience = float(min(peak / max(raw_peak, 1e-9), 1.0)) * gain
        self.last = {"salience": round(salience, 6),
                     "bearing_deg": round(bearing, 4),
                     "elevation_cell": int(row),
                     "field_subtracted": self.subtract_field,
                     "velocity_gain": round(gain, 4),
                     "peak_speed_deg_s": (round(speed, 2) if speed is not None
                                          else None)}
        return salience, bearing

    def state(self):
        return dict(self.last,
                    invented=["MOTION_FLOOR", "MAX_TARGET_FRACTION",
                              "the shape of the velocity curve between its "
                              "published peak and cuts"],
                    published=["prey_velocity_peak_deg_s",
                               "prey_velocity_low_cut_deg_s",
                               "prey_velocity_high_cut_deg_s",
                               "prey_elevation_switch"],
                    rules={"small": True, "moves differently": True,
                           "velocity band": self.velocity_band,
                           "below the horizon": self.elevation_switch,
                           "shape vs direction": False,
                           "smell confirms": False})


def _centroid_column(plane, peak_column, width=1):
    """Brightness-weighted column of the peak and its immediate neighbours.

    Returns a FRACTIONAL column index. Without this the reported position moves
    in whole-cell jumps and any speed derived from it is quantised far coarser
    than the band it is being compared against.
    """
    lo = max(peak_column - width, 0)
    hi = min(peak_column + width + 1, plane.shape[1])
    weights = plane[:, lo:hi].sum(axis=0)
    total = float(weights.sum())
    if total <= 0:
        return float(peak_column)
    columns = np.arange(lo, hi, dtype=float)
    return float((weights * columns).sum() / total)


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
        # The tectum's velocity rule needs to know how long a frame was, and
        # the retina does not carry a clock. Without this the rule would find
        # no dt, silently skip itself, and the whole filter would look present
        # while doing nothing -- which is the exact shape of the no-op this
        # module already shipped once (the scalar field subtraction).
        retina_output = dict(retina_output, dt_s=float(dt_s))
        salience, bearing = self.tectum.step(retina_output)
        gaze = self.pretectum.step(retina_output, dt_s)
        self.last = {
            "prey_salience": salience,
            "prey_bearing_deg": bearing,
            "gaze_command_deg_s": gaze,
        }
        return dict(self.last)
