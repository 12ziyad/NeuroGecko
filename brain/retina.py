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

    def __init__(self, fovy_deg=70.0, pixels=64, cells=16,
                 surround_ratio=3.0):
        if cells < 2 or pixels % cells:
            raise ValueError("pixels must be a whole multiple of cells.")
        self.fovy_deg = float(fovy_deg)
        self.pixels = int(pixels)
        self.cells = int(cells)
        self.surround_ratio = float(surround_ratio)
        self._previous = None
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
        explode."""
        self._previous = None
        return self

    # ------------------------------------------------------------ the signal
    def _receptors(self, image):
        """uint8 HxWx3 -> float HxWx2, red dropped, normalised to [0, 1]."""
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("expected an HxWx3 image")
        if frame.shape[0] != self.pixels or frame.shape[1] != self.pixels:
            raise ValueError(
                f"expected {self.pixels}x{self.pixels}, got "
                f"{frame.shape[0]}x{frame.shape[1]}")
        return frame[:, :, KEPT_CHANNELS].astype(np.float64) / 255.0

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
