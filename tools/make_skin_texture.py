"""Generate the leopard gecko's spots, so the animal stops being flat yellow paint.

WHAT THIS IS AND IS NOT. It is APPEARANCE ONLY. A texture changes no mass, no
joint, no contact and no gate: `assert_body_unchanged` compares bodies, geoms,
sites and actuators by name, and a material's texture is none of those. Nothing
the brain reads changes either -- the animal's own `head_cam` sees the world,
not itself.

WHY IT MATTERS ANYWAY. The species is called the LEOPARD gecko because of the
spots, and this body has been solid `rgba="0.83 0.73 0.42 1"` since it was
built. Shape is what a mesh fixes; pattern is what this fixes, and they are
independent.

WHAT IS PUBLISHED HERE. Almost nothing, and that is stated rather than dressed
up. The base colour, spot size, spot density and banding are read off
photographs of adult *E. macularius*, not from a paper: no quantitative
colour-pattern morphometrics for this species were found by the session-12
literature sweep. This file's output is INVENTED and is tagged as such in the
XML comment it is referenced from. Wild-type animals are yellow-to-buff with
dark brown/black spotting, a pale unspotted venter, and a banded tail -- that
much is uncontroversial natural history, and it is all this reproduces.

Usage:  python tools/make_skin_texture.py
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "morphology" / "textures"

#: Wild-type dorsal ground colour, and the spot colour. INVENTED (photographs).
BASE = np.array([214, 186, 108], dtype=float)
SPOT = np.array([56, 42, 28], dtype=float)
PALE = np.array([238, 228, 200], dtype=float)
SIZE = 512


def _spots(rng, w, h, n, r_lo, r_hi, softness=1.6, wrap_y=True):
    """A field in [0,1] with n soft dark blobs of random radius.

    A spot is zero outside its own radius, so it is drawn into a WINDOW around
    its centre rather than evaluated over the whole image nine times. The first
    version did the latter: at 512 x 2048 with 1120 spots that is about ten
    billion element operations and the call did not return in two minutes.
    Wrapping is done by drawing the window at each wrapped offset it overlaps,
    which is at most a couple of small windows instead of nine full images.

    `wrap_y` is False for the whole-body image, whose v axis runs nose to tail
    once and does not repeat -- wrapping it would put the tail's spots on the
    snout.
    """
    field = np.zeros((h, w), dtype=float)
    for _ in range(n):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(r_lo, r_hi)
        # slight ellipticity so the spots are not perfect circles
        ax, ay = r * rng.uniform(0.75, 1.3), r * rng.uniform(0.75, 1.3)
        rot = rng.uniform(0, np.pi)
        reach = int(math.ceil(max(ax, ay))) + 1
        for dx in ((-w, 0, w)):
            for dy in ((-h, 0, h) if wrap_y else (0,)):
                ox, oy = cx + dx, cy + dy
                x0, x1 = max(0, int(ox - reach)), min(w, int(ox + reach) + 1)
                y0, y1 = max(0, int(oy - reach)), min(h, int(oy + reach) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
                px, py = xx - ox, yy - oy
                qx = px * np.cos(rot) + py * np.sin(rot)
                qy = -px * np.sin(rot) + py * np.cos(rot)
                d = (qx / ax) ** 2 + (qy / ay) ** 2
                blob = np.clip(1.0 - d ** softness, 0.0, 1.0)
                np.maximum(field[y0:y1, x0:x1], blob, out=field[y0:y1, x0:x1])
    return field


def skin(seed=7):
    rng = np.random.default_rng(seed)
    h = w = SIZE
    img = np.tile(BASE, (h, w, 1))

    # fine tubercle speckle: the skin is granular, not smooth
    grain = rng.normal(0.0, 6.0, (h, w, 1))
    img += grain

    # two spot scales, as on the animal: large dorsal blotches with smaller
    # ones filling between them
    big = _spots(rng, w, h, 26, 16, 34)
    small = _spots(rng, w, h, 90, 5, 12)
    field = np.clip(big + 0.75 * small, 0.0, 1.0)[..., None]
    img = img * (1.0 - field) + SPOT * field

    # NO pale venter strip. A cube texture with `texuniform` maps by geom
    # coordinates, so a pale band in the image lands on WHOLE GEOMS rather than
    # on undersides: the jowls and shoulders came out solid cream. The belly is
    # a separate material's job, not this texture's.
    return np.clip(img, 0, 255).astype(np.uint8)


def tail(seed=11):
    """Paler, spotted, with the banding still faintly showing.

    BAND DIRECTION. The mesh generator writes u AROUND the body and v ALONG it,
    and an image's u is its COLUMN axis. The first version varied the bands
    across columns, which wrapped them around the long axis instead of around
    the girth: the tail came out as a barber's pole. The bands here vary down
    the ROWS, so they land as rings.

    STRENGTH. Bold black-and-white rings are the JUVENILE pattern. Adults keep
    the rings only as a faint interruption in a spotted, paler tail, which is
    what this reproduces -- the animal being modelled is an adult, since every
    published measurement the body is gated on is from adults. INVENTED
    (photographs), like the rest of the pattern.
    """
    rng = np.random.default_rng(seed)
    h = w = SIZE
    img = np.tile(PALE * 0.22 + BASE * 0.78, (h, w, 1))
    img += rng.normal(0.0, 6.0, (h, w, 1))
    rings = (np.sin(np.linspace(0, 2 * np.pi * 3.0, h))[:, None, None] + 1.0) * 0.5
    band_field = np.clip((rings - 0.62) / 0.34, 0.0, 1.0) * 0.45
    img = img * (1.0 - band_field) + SPOT * band_field
    blot = np.clip(_spots(rng, w, h, 30, 9, 20)
                   + 0.7 * _spots(rng, w, h, 70, 4, 9), 0.0, 1.0)[..., None]
    img = img * (1.0 - 0.88 * blot) + SPOT * 0.88 * blot
    return np.clip(img, 0, 255).astype(np.uint8)


def _tubercles(rng, w, h, spacing, radius, jitter=0.35, light=(-0.6, -0.8)):
    """A signed shading field for raised bumps: highlight one side, shadow the other.

    WHY THIS IS PAINTED RATHER THAN MODELLED. A gecko's skin is covered in
    raised tubercles, and a perfectly smooth surface is the single thing that
    most makes a model read as plastic. The obvious fix is a normal map, and
    MuJoCo 3.9 ACCEPTS one -- `<material><layer role="normal"/></material>`
    compiles without complaint. It then ignores it: a flat slab rendered with
    and without a strong normal map differs in 0 of 57600 pixels, maximum
    channel difference 0. Measured before relying on it (#329). Real geometry
    at tubercle scale would need roughly ten times the vertices, so the bumps
    are baked into the colour instead, the way they were before normal maps
    existed: a light side and a dark side per bump.

    Arrangement is a jittered grid, INVENTED from photographs. Returns a field
    in roughly [-1, 1] to be added to brightness.
    """
    field = np.zeros((h, w), dtype=float)
    lx, ly = light
    norm = math.hypot(lx, ly)
    lx, ly = lx / norm, ly / norm
    reach = int(math.ceil(radius)) + 1
    cols = max(1, int(w / spacing))
    rows = max(1, int(h / spacing))
    for r_i in range(rows):
        for c_i in range(cols):
            cx = (c_i + 0.5 + rng.uniform(-jitter, jitter)) * (w / cols)
            cy = (r_i + 0.5 + rng.uniform(-jitter, jitter)) * (h / rows)
            rad = radius * rng.uniform(0.7, 1.25)
            for dx in (-w, 0, w):
                ox, oy = cx + dx, cy
                x0, x1 = max(0, int(ox - reach)), min(w, int(ox + reach) + 1)
                y0, y1 = max(0, int(oy - reach)), min(h, int(oy + reach) + 1)
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
                px, py = xx - ox, yy - oy
                rr = np.sqrt(px * px + py * py) / rad
                inside = rr < 1.0
                if not inside.any():
                    continue
                # a dome's surface normal tilts away from its centre, so the
                # shading is the sideways offset projected on the light
                shade = np.where(inside,
                                 -(px * lx + py * ly) / rad
                                 * np.sqrt(np.clip(1.0 - rr * rr, 0.0, 1.0)),
                                 0.0)
                block = field[y0:y1, x0:x1]
                field[y0:y1, x0:x1] = np.where(np.abs(shade) > np.abs(block),
                                               shade, block)
    return field


def _paint_face(img, landmarks, w, h):
    """Mouth line, nostrils and ear openings, painted where the skin says they are.

    The skin's u runs round the body with 0 at the left flank, 0.25 the spine,
    0.5 the right flank, 0.75 the belly; v runs nose to tail. The three head
    landmarks arrive as v from the skin generator, read off the actual rings,
    so nothing here guesses where the eye or the jaw angle is. Offsets round
    the body are INVENTED (photographs): the mouth line sits a little below
    the flank midline, the ear a little behind and below the jaw angle, the
    nostril just above the midline at the very tip of the snout.
    """
    if not landmarks or landmarks.get("jaw_v") is None:
        return img
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    uu, vv = xx / w, yy / h
    dark = SPOT * 0.75

    def paint(mask, colour, strength=1.0):
        m = np.clip(mask, 0.0, 1.0)[..., None] * strength
        return img * (1.0 - m) + colour * m

    jaw_v = landmarks["jaw_v"]
    nose_v = landmarks.get("nostril_v", 0.01)
    # ---- mouth: a thin line along each flank from the snout to the jaw angle
    for u0 in (1.0 - 18.0 / 360.0, 0.5 + 18.0 / 360.0):
        du = np.minimum(np.abs(uu - u0), 1.0 - np.abs(uu - u0))
        along = np.clip((vv - nose_v) / max(1e-6, jaw_v - nose_v), 0.0, 1.0)
        # the line curves up very slightly toward the jaw angle: the "smile"
        du = du - 0.006 * along
        line = np.exp(-(du / 0.0045) ** 2) * ((vv >= nose_v - 0.004) & (vv <= jaw_v + 0.006))
        img = paint(line, dark, 0.85)
    # ---- ears: an open oval just behind and below the jaw angle
    ear_v = jaw_v + 0.010
    for u0 in (1.0 - 9.0 / 360.0, 0.5 + 9.0 / 360.0):
        du = np.minimum(np.abs(uu - u0), 1.0 - np.abs(uu - u0))
        d = (du / 0.020) ** 2 + ((vv - ear_v) / 0.0075) ** 2
        img = paint(np.clip(1.4 - 1.4 * d, 0.0, 1.0), SPOT * 0.55, 0.95)
    # ---- nostrils: a small dark pit each side, just above the midline
    for u0 in (24.0 / 360.0, 0.5 - 24.0 / 360.0):
        du = np.minimum(np.abs(uu - u0), 1.0 - np.abs(uu - u0))
        d = (du / 0.011) ** 2 + ((vv - nose_v) / 0.0035) ** 2
        img = paint(np.clip(1.5 - 1.5 * d, 0.0, 1.0), SPOT * 0.6, 0.9)
    return img


def whole_body(tail_start_v, width=512, height=2048, seed=23, landmarks=None):
    """One image for the whole animal, for the single continuous skin.

    THE COORDINATES ARE THE ANIMAL'S, WHICH IS WHY THIS CAN DO WHAT THE TILED
    TEXTURES COULD NOT. The skin's u runs around the girth and its v runs nose
    (0) to tail tip (1) without repeating, so a row of this image is a ring
    around the body at a known position along it and a column is a line down its
    length. That makes three things expressible that a tiling texture cannot:
    the tail can be banded while the trunk is spotted, the spots can thin out
    toward the belly, and the belly can be pale.

    The aspect ratio is not arbitrary. The girth is about 60 mm and the animal
    about 220 mm long, so 512 x 2048 puts ~8500 pixels per metre around against
    ~9300 along: a circle drawn here comes out a circle on the animal.

    WHICH WAY IS UP. `make_gecko_mesh.ring_at` places a ring's vertices at
    y = w*cos(angle), z = h*sin(angle), so u = 0.25 is the DORSUM and u = 0.75
    is the VENTER. Everything below keys off that.

    Everything here is INVENTED, read off photographs of adult
    *E. macularius*: no quantitative colour-pattern morphometrics exist for this
    species. What is uncontroversial natural history and all this reproduces:
    yellow-to-buff ground, dark brown-black dorsal spotting, a pale unspotted
    venter, and a paler banded tail.
    """
    rng = np.random.default_rng(seed)
    h, w = height, width
    vv = np.broadcast_to((np.arange(h) / float(h))[:, None], (h, w))
    uu = np.broadcast_to((np.arange(w) / float(w))[None, :], (h, w))

    # How far round from the dorsal midline, as a fraction of the way round.
    # 0 is the spine, 0.5 is the belly. The pattern covers the back and the
    # FLANKS -- a leopard gecko is spotted right down to the level of the limbs
    # and only the underside is clear -- so the fade starts late and is quick.
    off = np.abs(((uu - 0.25 + 0.5) % 1.0) - 0.5)
    belly = np.clip((off - 0.30) / 0.13, 0.0, 1.0)

    # COLOUR, from the reference animal (#343): a bright yellow back, orange
    # down the flanks, a WHITE belly, and a tail that is white with orange
    # patches near its base. Spots are near-black, not brown. The old buff was
    # read, correctly, as mud. INVENTED (photographs).
    YELLOW = np.array([242, 204, 58], dtype=float)
    ORANGE = np.array([236, 128, 44], dtype=float)
    WHITE = np.array([246, 243, 236], dtype=float)
    INK = np.array([26, 20, 17], dtype=float)
    tailness = np.clip((vv - tail_start_v) / 0.06, 0.0, 1.0)
    # flank: how far round from the spine, 0 on the spine, 1 at the belly line
    flank = np.clip((off - 0.10) / 0.22, 0.0, 1.0)
    ground = YELLOW * (1.0 - flank[..., None]) + ORANGE * flank[..., None]
    # tail: white ground, orange only in a broad patch just behind the vent
    tail_orange = np.clip(1.0 - (vv - tail_start_v) / 0.16, 0.0, 1.0) * tailness
    tail_ground = WHITE * (1.0 - 0.75 * tail_orange[..., None]) + ORANGE * (0.75 * tail_orange[..., None])
    ground = ground * (1.0 - tailness[..., None]) + tail_ground * tailness[..., None]
    img = ground * (1.0 - 0.92 * belly[..., None]) + WHITE * (0.92 * belly[..., None])
    img = img + rng.normal(0.0, 4.0, (h, w, 1))

    # tail rings: constant in u, varying down v, so they land as rings around
    # the tail rather than stripes along it
    rings = (np.sin(vv * (h / 9300.0) * 2 * np.pi * 10.0) + 1.0) * 0.5
    band = np.clip((rings - 0.55) / 0.28, 0.0, 1.0) * tailness * 0.55
    band = band * (1.0 - 0.85 * belly)
    img = img * (1.0 - band[..., None]) + INK * band[..., None]

    # spots, at the same physical size everywhere; radii in px at ~9000 px/m,
    # so 9-22 px is a 2-5 mm blotch and 3-8 px is the fine speckle between them
    # Harder-edged than before (softness 2.6 against 1.6): the reference
    # animal's spots are crisp blots, and soft ones read as mud (#341).
    big = _spots(rng, w, h, 300, 9, 22, softness=2.6, wrap_y=False)
    small = _spots(rng, w, h, 820, 3, 8, softness=2.2, wrap_y=False)
    field = np.clip(big + 0.8 * small, 0.0, 1.0) * (1.0 - belly)
    img = img * (1.0 - field[..., None]) + INK[None, None, :] * field[..., None]

    # the face: mouth line, nostrils, ear openings, at the positions the skin
    # generator measured (#341)
    img = _paint_face(img, landmarks, w, h)

    # THE INSIDE OF THE MOUTH. The mandible skin's upper face is hidden while
    # the mouth is shut and is all you see when it opens, so it is painted the
    # dark pink of a mouth floor. The jaw tube maps to v 0.02-0.06 with its
    # top at u ~ 0.25 (#343).
    # The jaw skin maps to v >= 0.995 with its TOP at u = 0.20 and its
    # underside at u = 0.30 (make_gecko_skin.build_surface), so only the top
    # is painted and the chin stays skin-coloured. Nothing else on the animal
    # maps there except the last dome of the tail tip, which is a point.
    PINK = np.array([176, 78, 92], dtype=float)
    mouth = (np.clip(1.0 - np.abs(uu - 0.20) / 0.045, 0.0, 1.0)
             * (vv >= 0.994))
    img = img * (1.0 - mouth[..., None]) + PINK * mouth[..., None]

    # TUBERCLES, on top of the colour so they read on spots and ground alike.
    # Two scales: the large raised bumps that sit in rough rows down the back,
    # and the fine granular scaling between them. Both fade out on the belly,
    # which is smooth. Spacing and radius in pixels at about 9300 px per metre,
    # so 17 px is a bump roughly every 1.8 mm. INVENTED (photographs).
    bumps = _tubercles(rng, w, h, spacing=17.0, radius=5.2)
    grain = _tubercles(rng, w, h, spacing=6.0, radius=2.1, jitter=0.45)
    relief = (bumps * 26.0 + grain * 11.0) * (1.0 - 0.88 * belly)
    img = img + relief[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, arr in (("gecko_skin.png", skin()), ("gecko_tail.png", tail())):
        Image.fromarray(arr).save(OUT / name)
        print(f"wrote {OUT / name}  {arr.shape[1]}x{arr.shape[0]}")


if __name__ == "__main__":
    main()
