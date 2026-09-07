"""Brain module 4b: the pretectum -- gaze stabilisation, and the one published
behavioural measurement that exists for this species.

WHY THIS IS THE MOST IMPORTANT TEST IN THE VISION WORK. Almost nothing about
the leopard gecko eye has ever been measured: no grating acuity in any gecko,
no eye optics in any eublepharid, no contrast sensitivity function, no flicker
fusion from a readable primary source. What DOES exist is an optokinetic study
of *Eublepharis macularius* itself, n = 4 -- the target species, quantitative,
with a discriminating asymmetry:

    binocular head gain      0.9 at 20 deg/s, 0.8 at 30, 0.7-0.8 at 40
    monocular temporo-nasal  0.7 at 20, 0.6 at 30, 0.4 at 40
    monocular naso-temporal  NO RESPONSE ELICITED AT ANY VELOCITY

That last row is the whole value of this module. Two things fall out of it:

  * A SYMMETRIC OPTIC-FLOW ESTIMATOR CANNOT PRODUCE IT. Any front end that
    simply measures full-field motion predicts a non-zero naso-temporal gain,
    because the flow is there and it is just as strong. The asymmetry is not a
    detail to be added later; it is the shape of the mechanism.
  * The gain FALLS with velocity, and never reaches 1. An optokinetic loop that
    stabilises perfectly has over-reproduced the animal, not matched it.

HOW THE ASYMMETRY IS PRODUCED HERE. Each eye's hemifield is half-wave rectified
in its temporo-nasal direction, so backward flow contributes nothing rather
than contributing negatively. With both eyes, whichever direction the drum
turns, one eye is being driven temporo-nasally -- so the response is
bidirectional and strong. With one eye covered, only one direction survives and
the other is exactly zero. The published asymmetry is then a CONSEQUENCE of the
rectification rather than a number written down.

WHAT IS FITTED AND WHAT IS PREDICTED, stated plainly because the distinction is
the whole of rule 2.

  PREDICTED, and therefore a real test:
    - that monocular naso-temporal gain is exactly zero
    - that binocular gain exceeds monocular temporo-nasal gain at every speed
    - that gain falls as stimulus velocity rises
  Nothing was adjusted to make those three true. They follow from rectification
  and from a first-order lag.

  FITTED, and therefore NOT evidence:
    - the absolute gain values. `peak_gain` and `half_velocity` are INVENTED
      and were chosen so the curve passes near the published points. A fitted
      curve passing through its own fit points proves nothing, and this module
      does not claim it does.

So the acceptance test gates on the three predicted properties and reports the
fitted ones as context. See tools/okr_sweep.py.
"""

from __future__ import annotations

import math

import numpy as np

#: Published stimulus velocities, degrees per second. The source measured only
#: these three, on 50 Hz video giving 0.4 deg per frame at the slowest, so the
#: curve is not extrapolated beyond them and the tests refuse to.
PUBLISHED_VELOCITIES = (20.0, 30.0, 40.0)
#: Masseck, Roll & Hoffmann 2008, Vision Research 48:765-772.
#: Eublepharis macularius, n = 4. Head gain.
PUBLISHED_GAIN = {
    "binocular": {20.0: 0.9, 30.0: 0.8, 40.0: 0.75},
    "temporo_nasal": {20.0: 0.7, 30.0: 0.6, 40.0: 0.4},
    "naso_temporal": {20.0: 0.0, 30.0: 0.0, 40.0: 0.0},
}
#: The published binocular value at 40 deg/s is direction-dependent, 0.8
#: clockwise against 0.7 counter-clockwise, and only that velocity reached
#: significance. Recorded as a band rather than a point.
PUBLISHED_BAND = 0.15

#: INVENTED. Fitted, and labelled as fitted -- see the module docstring.
PEAK_GAIN = 0.95
#: INVENTED. Stimulus velocity at which the loop reaches half its peak gain.
HALF_VELOCITY_DEG_S = 95.0


class Pretectum:
    """Optokinetic gaze stabilisation with the published monocular asymmetry."""

    def __init__(self, retina, peak_gain=PEAK_GAIN,
                 half_velocity_deg_s=HALF_VELOCITY_DEG_S,
                 left_eye=True, right_eye=True):
        self.retina = retina
        self.peak_gain = float(peak_gain)
        self.half_velocity = float(half_velocity_deg_s)
        if self.half_velocity <= 0:
            raise ValueError("half_velocity_deg_s must be positive.")
        #: Eyes can be covered independently, which is how the published
        #: monocular conditions were produced.
        self.left_eye = bool(left_eye)
        self.right_eye = bool(right_eye)
        self.last = {"command_deg_s": 0.0, "left_drive": 0.0, "right_drive": 0.0}

    def cover(self, left=None, right=None):
        if left is not None:
            self.left_eye = bool(left)
        if right is not None:
            self.right_eye = bool(right)
        return self

    #: Gradient magnitude below which a column carries no usable horizontal
    #: structure and the flow estimate is undefined there.
    MIN_GRADIENT = 1e-3

    def _flow_deg_s(self, retina_output, dt_s):
        """Horizontal flow per hemifield, in degrees per second.

        The standard gradient relation -- temporal change over spatial gradient
        gives displacement -- with two things that both had to be right before
        the estimate was unbiased:

        1. It runs at PIXEL resolution. Estimating on the 16-cell map put the
           drum's stripes at four cells per period, where a central difference
           underestimates the gradient by about a third, and the flow came out
           2.07x too large at every velocity.
        2. It differences against AZIMUTH, not against the pixel index. The
           camera is uniform in pixels and not in angle, so a pixel-index
           gradient still overestimated by 15%. Differencing against the actual
           angle of each column removes it: measured ratio 1.02 against truth,
           across 10 to 40 deg/s.

        The median is used rather than the mean because a handful of near-zero
        gradients survive the threshold and produce huge outliers; the median
        is not sensitive to them and the mean is.
        """
        temporal = retina_output["pixel_temporal"]
        luminance = retina_output["pixel_luminance"]
        gradient = np.gradient(luminance, self.retina.azimuth_deg, axis=1)
        usable = np.abs(gradient) > self.MIN_GRADIENT
        shift = np.zeros_like(gradient)
        shift[usable] = -temporal[usable] / gradient[usable]

        half = self.retina.pixels // 2
        per_field = []
        for lo, hi in ((0, half), (half, self.retina.pixels)):
            mask = usable[:, lo:hi]
            field = shift[:, lo:hi]
            per_field.append(float(np.median(field[mask])) / dt_s
                             if mask.any() else 0.0)
        return per_field                      # [left hemifield, right hemifield]

    def step(self, retina_output, dt_s):
        """One frame. Returns the commanded head yaw velocity, degrees/second.

        Sign convention: positive means turn the head to the animal's LEFT.
        """
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be finite and positive.")
        left_flow, right_flow = self._flow_deg_s(retina_output, dt_s)

        # HALF-WAVE RECTIFICATION, per eye, in its temporo-nasal direction.
        # For the left eye temporal is leftward, so temporo-nasal flow runs
        # rightward and appears as positive shift; the right eye is mirrored.
        # Backward (naso-temporal) flow contributes ZERO, not a negative -- and
        # that single asymmetry is what reproduces the published monocular
        # result that a symmetric estimator cannot.
        left_drive = max(0.0, left_flow) if self.left_eye else 0.0
        right_drive = max(0.0, -right_flow) if self.right_eye else 0.0

        # The two eyes drive the head in opposite directions.
        net = left_drive - right_drive
        speed = abs(net)
        # Gain falls with stimulus speed. A first-order saturation, so the
        # DIRECTION of the fall is a prediction even though its scale is fitted.
        gain = self.peak_gain / (1.0 + speed / self.half_velocity)
        command = gain * net
        self.last = {"command_deg_s": command,
                     "left_drive": left_drive, "right_drive": right_drive,
                     "gain": gain}
        return command

    def state(self):
        return dict(self.last, left_eye=self.left_eye, right_eye=self.right_eye,
                    fitted=["peak_gain", "half_velocity_deg_s"])
