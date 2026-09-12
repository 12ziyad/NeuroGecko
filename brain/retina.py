"""Brain module 4a: the retina -- what the eye sends, rather than what it sees.

WHAT WAS THERE BEFORE. The camera rendered a 64x64x3 image and two things
happened to it, both wrong:

  * the encoder average-pooled it to 128 numbers over the WHOLE field, so the
    policy received "how much" and never "where". There was no retinotopic map
    anywhere in the animal.
  * prey were found by matching a colour, which is not how eyes work and which
    had silently gone blind when the prey stopped being green.

WHAT THIS IS. A retina keeps position and throws away almost everything else.
This produces, on a spatial grid:

  * TEMPORAL CONTRAST -- what changed since the last frame. This is the channel
    that finds a camouflaged cricket, because a brown cricket on sand is
    separable by motion and, as measured, barely separable by colour: the
    nearest chromatic confuser is the gecko's own spots at a distance of 0.145.
  * SPATIAL CONTRAST -- centre-surround, the classic retinal operation.

FOUR THINGS ARE DELIBERATE AND EACH HAS A REASON THAT IS NOT AESTHETIC.

1. SAMPLING IS UNIFORM. Strictly nocturnal geckos completely lack foveae and
   run about 1-2 photoreceptors per ganglion cell, uniformly. So no fovea, no
   centre weighting, no log-polar transform. Note the camera currently violates
   this in the OPPOSITE direction -- a rectilinear render is uniform in PIXELS,
   not in ANGLE, so at fovy 120 the corners sample 4.0x finer than the centre.
   That is an INVERSE fovea, worst exactly where the animal aims.
   `angular_sampling()` measures it so the defect is visible rather than
   assumed away.

2. THE RED CHANNEL IS DROPPED. This animal's gene complement is three visual
   opsins with RH1 and SWS2 absent, and the longest-wavelength pigment sits at
   about 521 nm -- spectrally green despite being called LWS. Keeping red hands
   the policy an axis with no receptor behind it, and any prey-detection
   strategy that separates prey from floor on red-versus-green is solving a
   problem this animal cannot solve. The two kept channels are a MAPPING, not
   a match: MuJoCo's sRGB primaries are not gecko cone fundamentals, and no
   behavioural colour result exists for this species. Tagged accordingly.

3. UV IS NOT MODELLED. The third opsin peaks near 364 nm. The renderer has no
   ultraviolet, so it is absent, and absent-and-declared beats invented.

4. THERE IS NO TAPETUM AND NO LOW-LIGHT GAIN. No primary source confirms a
   gecko tapetum -- every confident claim traced to a pet-care site -- and the
   published optical sensitivity advantage is explained by pupil, focal length
   and outer-segment size without invoking one.
"""

from __future__ import annotations

import math

import numpy as np

#: Green and blue survive; red is dropped. See point 2 above.
KEPT_CHANNELS = (1, 2)

#: The animal's own resolving power, cycles per degree. DERIVED, from published
#: cone outer-segment spacing (10-15 um) and focal length (3.5 mm) in
#: *Tarentola chazaliae* -- not measured in any eublepharid, and a SAMPLING
#: limit rather than an optical one. The eye is multifocal with roughly 15 D
#: between zones and positive spherical aberration, so the true optical cutoff
#: is LOWER than this and has never been measured. The conservative end of the
#: range is used, which errs toward the animal seeing less rather than more.
ACUITY_CYC_DEG = 2.0
#: What those channels are being read AS. A mapping, not a match.
CHANNEL_NM = (521.0, 467.0)


def angular_sampling(fovy_deg, pixels):
    """Degrees per pixel on the optical axis, and at the corner.

    A rectilinear projection is uniform in pixels and NOT in angle. Returned as
    a pair so the non-uniformity is a number someone can look at rather than an
    assumption: the animal's retina is uniform, and this camera is not.
    """
    if not 0.0 < fovy_deg < 180.0:
        raise ValueError("fovy_deg must lie strictly between 0 and 180.")
    if pixels < 2:
        raise ValueError("pixels must be at least 2.")
    half = math.radians(fovy_deg) / 2.0
    focal_px = (pixels / 2.0) / math.tan(half)
    on_axis = math.degrees(math.atan(1.0 / focal_px))
    # At the corner the same pixel spans a smaller angle, because the image
    # plane is further from the nodal point there.
    corner_px = math.hypot(pixels / 2.0, pixels / 2.0)
    corner = math.degrees(
        math.atan((corner_px + 1.0) / focal_px) - math.atan(corner_px / focal_px))
    return on_axis, corner


def resolvable(fovy_deg, pixels, feature_deg):
    """How many pixels a feature of a given angular size spans, on axis."""
    on_axis, _ = angular_sampling(fovy_deg, pixels)
    return float(feature_deg) / on_axis


class Retina:
    """Turns frames into a retinotopic map of motion and contrast."""

    #: INVENTED, but bounded by a measurement. Number of control frames the
    #: prey-motion map is differenced across. 1 reproduces the single-frame
    #: behaviour exactly and is the default, so nothing changes unless asked.
    #:
    #: WHY THIS EXISTS. Measured over 3200 steps of a real hunt
    #: (artifacts/evidence/session11/hunt_decomposition.json): on the frames
    #: where the prey IS on the rendered image and the eye stays silent,
    #: 92.3 % are rejected as "below motion floor". The prey's image moves a
    #: median of 1.01 px per control step, which is a QUARTER of a 4x4-pixel
    #: cell, and a quarter-cell displacement of a ~5 px object survives neither
    #: the mean-pool nor the surround subtraction. The band is finer than the
    #: instrument -- which is ledger #188, restated at the cell level.
    #:
    #: WHAT BOUNDS IT. The control loop runs at 50 Hz. Scotopic flicker fusion
    #: in two nocturnal geckos is "up to 18 flashes per sec" (Dodt & Jessen
    #: 1961, J Gen Physiol 44(6):1143-1158, Hemidactylus turcicus and Tarentola
    #: mauritanica -- neither is an eublepharid). 50/18 = 2.8, so an animal-
    #: appropriate integration window is about three control frames. That is a
    #: bound on the ORDER, not a measurement of this species: no eublepharid
    #: temporal resolution has ever been published.
    def __init__(self, fovy_deg=70.0, pixels=64, cells=16, motion_window=3,
                 surround_ratio=3.0, render_pixels=None,
                 acuity_cyc_deg=ACUITY_CYC_DEG):
        """
        `render_pixels` is the resolution the WORLD is rendered at, which may be
        far higher than the resolution the animal sees. That is not a cheat and
        it is not a superpower -- it is how an eye actually works, and skipping
        it is the error.

        A real eye receives a sharp image, its OPTICS blur it, and only then do
        the receptors sample it. Rendering coarse and sampling coarse omits the
        blur entirely, so fine detail that the eye could never resolve does not
        get removed -- it gets ALIASED, and turns into false structure that
        moves when the animal moves. Floor texture shimmering like a moving
        object is exactly that error, and a motion detector cannot tell the
        difference.

        So: render high, low-pass to `acuity_cyc_deg`, then sample. What
        reaches the brain is limited to what the animal's receptors can carry,
        whatever the render resolution was. Raising the render resolution makes
        the model MORE faithful and never sharper than the animal.
        """
        if cells < 2 or pixels % cells:
            raise ValueError("pixels must be a whole multiple of cells.")
        self.render_pixels = int(render_pixels) if render_pixels else int(pixels)
        if self.render_pixels < pixels or self.render_pixels % pixels:
            raise ValueError("render_pixels must be a whole multiple of pixels.")
        self.acuity_cyc_deg = float(acuity_cyc_deg)
        if self.acuity_cyc_deg <= 0:
            raise ValueError("acuity_cyc_deg must be positive.")
        self.fovy_deg = float(fovy_deg)
        self.pixels = int(pixels)
        self.cells = int(cells)
        self.surround_ratio = float(surround_ratio)
        self._previous = None
        self.motion_window = max(1, int(motion_window))
        #: Luminance frames, oldest first, at most motion_window + 1 of them.
        self._frames = []
        self.on_axis_deg, self.corner_deg = angular_sampling(fovy_deg, pixels)
        # Azimuth of every pixel column, in degrees. Flow must be differenced
        # against THIS and not against the pixel index: the camera is uniform
        # in pixels and not in angle, so a pixel-index gradient overestimates
        # flow by about 15% at this field of view -- the inverse-fovea defect
        # showing up as a measurement error rather than as a picture.
        half = math.radians(self.fovy_deg) / 2.0
        focal_px = (self.pixels / 2.0) / math.tan(half)
        self.azimuth_deg = np.degrees(np.arctan(
            (np.arange(self.pixels) + 0.5 - self.pixels / 2.0) / focal_px))

    # ----------------------------------------------------------------- state
    def reset(self):
        """Forget the previous frame. Motion is undefined on the first frame
        after a reset, and this returns zeros there rather than a spurious
        transient -- an episode should not begin with the world appearing to
        explode.

        THE WINDOW RING IS CLEARED HERE TOO, and the first version of this
        change forgot to. Frames then leaked across episode boundaries and the
        windowed difference compared the new episode's first frame against the
        old episode's last one, so the peak landed on where the target used to
        be. `test_it_reports_the_side_the_target_is_on` caught it by reporting
        a left bearing for a right-hand target."""
        self._previous = None
        self._frames = []
        return self

    # ------------------------------------------------------------ the signal
    def optical_limit_px(self):
        """Blur width, in RENDER pixels, that limits the image to the animal's
        own resolving power. A feature finer than one cycle at
        `acuity_cyc_deg` cannot be transmitted by the eye and must not survive
        into the retinal image."""
        per_px = self.fovy_deg / self.render_pixels
        cycle_deg = 1.0 / self.acuity_cyc_deg
        return max(1.0, (cycle_deg / 2.0) / per_px)

    def _receptors(self, image):
        """uint8 HxWx3 -> float, red dropped, blurred to the animal's acuity,
        then sampled at the receptor grid."""
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("expected an HxWx3 image")
        if frame.shape[0] != frame.shape[1]:
            raise ValueError("expected a square image")
        if frame.shape[0] not in (self.pixels, self.render_pixels):
            raise ValueError(
                f"expected {self.pixels} or {self.render_pixels} px, "
                f"got {frame.shape[0]}")
        planes = frame[:, :, KEPT_CHANNELS].astype(np.float64) / 255.0

        if frame.shape[0] == self.render_pixels and                 self.render_pixels != self.pixels:
            # THE OPTICS. Low-pass to what the eye can transmit, THEN sample.
            # Doing it in this order is the whole point: sampling first would
            # alias detail the animal cannot resolve into false structure.
            width = self.optical_limit_px()
            planes = np.stack([_blur(planes[:, :, i], width)
                               for i in range(planes.shape[2])], axis=-1)
            k = self.render_pixels // self.pixels
            planes = planes.reshape(self.pixels, k, self.pixels, k,
                                    planes.shape[2]).mean(axis=(1, 3))
        return planes

    def _bin(self, plane):
        """Pool to the cell grid. Uniform: every cell gets the same count."""
        k = self.pixels // self.cells
        return plane.reshape(self.cells, k, self.cells, k).mean(axis=(1, 3))

    def step(self, image):
        """One frame. Returns a dict of retinotopic maps, each cells x cells.

        `motion` is |temporal contrast|, `signed_motion` keeps its sign per
        channel-mean, and `contrast` is centre-surround. All three are maps,
        not summaries: position is the thing a retina is for.
        """
        receptors = self._receptors(image)
        luminance = receptors.mean(axis=2)

        if self._previous is None:
            temporal = np.zeros_like(luminance)
        else:
            temporal = luminance - self._previous
        self._previous = luminance

        # The windowed map is SEPARATE from `temporal` on purpose. The
        # pretectum's optokinetic reflex reads `pixel_temporal`, and that
        # result is validated (#123 ratio 1.02 against truth, #124 the
        # published naso-temporal asymmetry, #126 the published gain). Widening
        # the window under it would change a reproduced published number, so it
        # is left exactly alone and prey motion gets its own baseline.
        self._frames.append(luminance)
        if len(self._frames) > self.motion_window + 1:
            self._frames.pop(0)
        if len(self._frames) < 2:
            windowed = np.zeros_like(luminance)
            span = 0
        else:
            windowed = luminance - self._frames[0]
            span = len(self._frames) - 1

        centre = self._bin(luminance)
        surround = _blur(centre, self.surround_ratio)
        return {
            # Pixel-resolution planes, for anything that needs to differentiate
            # in space. Binning first destroys the spatial sampling: at 16
            # cells the drum's stripes span four cells and a central difference
            # underestimates the gradient by a third, which came out as a 2x
            # error in the flow estimate.
            "pixel_temporal": temporal,
            "pixel_luminance": luminance,
            "motion": self._bin(np.abs(temporal)),
            # Differenced across `span` frames rather than one. Identical to
            # `motion` when motion_window == 1.
            "motion_windowed": self._bin(np.abs(windowed)),
            #: How many frames the windowed map actually spans. Fewer than
            #: motion_window while the ring fills, so a consumer scaling
            #: anything by the window uses what was measured, not what was
            #: asked for.
            "motion_window_span": int(span),
            "signed_motion": self._bin(temporal),
            "contrast": centre - surround,
            "luminance": centre,
            "channels": np.stack([self._bin(receptors[:, :, i])
                                  for i in range(receptors.shape[2])], axis=-1),
        }

    # ------------------------------------------------------------- geometry
    def azimuth_of(self, column):
        """Signed horizontal angle of a cell column, degrees.

        Negative is left of the optical axis. Computed through the actual
        projection rather than by linear interpolation across the field: the
        linear version is what produced the refuted "0.85 px" and "1.46 px"
        figures in the world file's own header note.
        """
        half = math.radians(self.fovy_deg) / 2.0
        focal_px = (self.pixels / 2.0) / math.tan(half)
        centre_px = (column + 0.5) * (self.pixels / self.cells) - self.pixels / 2.0
        return math.degrees(math.atan(centre_px / focal_px))

    def elevation_of(self, row):
        """Signed vertical angle of a cell row, degrees. Negative is BELOW.

        The same projection as `azimuth_of`, with the image's row order --
        row 0 is the top of the frame -- so the sign comes out negative for
        things on the ground, which is where a cricket is.

        WHY THIS EXISTS. Measured over 2400 steps of a real hunt
        (artifacts/evidence/session11/hunt_decomposition_rows.jsonl), the prey's
        elevation in camera coordinates falls as the animal closes:

            0.30-0.50 m   -4.2 deg    on image  54 %
            0.08-0.15 m  -10.8 deg    on image  85 %
            0.04-0.08 m  -20.2 deg    on image 100 %
            0.00-0.04 m  -35.1 deg    on image  45 %   <- falls out of frame

        At fovy 70 the frame stops at -35 deg, so the animal goes blind exactly
        where the strike has to fire. The tectum has always known the target's
        row; nothing ever converted it to an angle.
        """
        half = math.radians(self.fovy_deg) / 2.0
        focal_px = (self.pixels / 2.0) / math.tan(half)
        centre_px = (row + 0.5) * (self.pixels / self.cells) - self.pixels / 2.0
        return -math.degrees(math.atan(centre_px / focal_px))

    def hemifields(self, plane):
        """Split a map into (left, right) halves about the optical axis.

        The optokinetic reflex is computed per hemifield, because the published
        asymmetry in this species is monocular and direction-specific.
        """
        half = self.cells // 2
        return plane[:, :half], plane[:, half:]


def _blur(plane, ratio):
    """Cheap separable box surround. Ratio is the surround width in cells."""
    width = max(1, int(round(ratio)))
    if width <= 1:
        return plane.copy()
    padded = np.pad(plane, width, mode="edge")
    kernel = np.ones(2 * width + 1) / (2 * width + 1)
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, padded)
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), 0, smoothed)
    return smoothed[width:-width, width:-width]
