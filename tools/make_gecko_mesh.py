"""Generate the gecko's skin as a mesh, by wrapping its own gated primitives.

WHY GENERATE RATHER THAN DOWNLOAD. A bought or AI-generated mesh arrives as one
solid sculpt in somebody else's pose, and every step after that is loss: cut it
into the pieces this skeleton has, re-pose each piece, hope the seams line up,
and carry whatever licence it came with. Generating it from the body inverts all
of that -- the mesh is born already split per body, already in the neutral pose,
already seamless at the joints because adjacent pieces share a cross-section,
already UV-mapped so the spot texture lands correctly, and owned outright.

WHAT IT IS, AND WHY THIS VERSION IS DIFFERENT. A swept surface -- but the sweep
is taken from the body's OWN GEOMS, not from a guess. At each station along the
animal's long axis the generator intersects every visual primitive with that
plane, takes the union of the cross-sections, and fits one ellipse to it. The
skin is therefore the shrink-wrap of the primitives that passed the 14
morphology gates, station by station, and cannot disagree with them.

  The first version of this file did NOT do that. It placed one ring at each
  body's ORIGIN and interpolated between origins. A body's origin is its joint,
  not its middle, so the head -- whose geoms run 32.6 mm forward of its joint --
  got a mesh 20 mm long that had already tapered to a tenth of its width by the
  eyes. The eyes, the lip lines and the jaw all ended up outside the skin.
  Recorded as #313.

WHAT IS PUBLISHED IN THE SHAPE, AND WHAT IS NOT. Every dimension is now read
from the body rather than invented: the ellipse at each station comes from the
primitives, so head width, head height, trunk width, the per-segment tail radii
and the limb bone lengths are reproduced by construction rather than by a
profile table. What remains INVENTED is the styling multiplier in PROFILE --
how much the fat tail bulges beyond its collision capsule and how much the neck
is drawn in -- read off photographs of adult *Eublepharis macularius*, because
no published body-outline morphometrics exist for this species.

EVERY INVENTED NUMBER IN THIS FILE, IN ONE PLACE. Rule 1 says each number
declares which it is; none of these is in `config/proxies.yaml` because that
registry only admits verified / likely / uncertain published values, so they are
declared here, at the place they are used, as the rest of the project's invented
constants are.

  PROFILE            styling multiplier per body, 1.00 = exactly the measured
                     primitives. Only the fat tail and the neck depart from 1.
  LIMB_SCALE  1.12   how far the skin stands off a limb bone.
  JAW_WIDTH_OF_HEAD  0.88   mandible width as a fraction of the skull's.
  VENT_PINCH  0.82   depth of the constriction at the vent.
  k           6      rolling-maximum half-window, in stations. Chosen as half a
                     tail segment: the artefact it removes has that period.
  win         13     longitudinal smoothing window, in stations.
  U_TILES     2      texture tiles around the girth. Must be a whole number.
  TILE_M      0.030  metres of skin per texture tile along the animal.
  RING, STATIONS     mesh resolution; affects nothing but triangle count.

The mesh is APPEARANCE: mass 0, no collision, so the primitives remain the
physics and the mesh cannot change a single gate. None of the numbers above can
move a gate, and that is why they are allowed to be invented at all.

Usage:  python tools/make_gecko_mesh.py [--body morphology/gecko_body_lab_v2.xml]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "morphology" / "meshes"

#: Vertices around each ring, and stations along the animal. 28 x 200 puts a
#: vertex about every 2 mm around the girth and every 1.1 mm along the body,
#: which is fine enough that the silhouette has no visible facets at the sizes
#: this animal is rendered at, and still about a two-hundredth of the
#: AI-generated sculpt's triangle count.
RING = 28
STATIONS = 200

#: Nose-to-tail ordering. Anything absent from the model is skipped.
CHAIN = ["head", "neck", "trunk_anterior", "trunk_middle",
         "tail1", "tail2", "tail3", "tail4", "tail5"]

#: STYLING MULTIPLIER, INVENTED (photographs). Applied to the ellipse MEASURED
#: from that body's primitives, so a value of 1.0 is "exactly the primitives"
#: and the gated dimensions survive untouched. Only the departures are invented.
#:   width scale, height scale
PROFILE = {
    "head":           (1.00, 1.00),
    "neck":           (0.94, 0.94),
    "trunk_anterior": (1.02, 1.00),
    "trunk_middle":   (1.04, 1.00),
    # The fat tail: pinched at the vent, bulging through the first third, then
    # tapering to a point. After the spots this is the most recognisable thing
    # about the animal's silhouette, and the collision capsules do not carry it.
    "tail1":          (1.14, 1.12),
    "tail2":          (1.20, 1.18),
    "tail3":          (1.04, 1.02),
    "tail4":          (0.92, 0.90),
    "tail5":          (0.80, 0.78),
}
LIMB_SCALE = 1.12

#: One texture tile per 30 mm of skin ALONG the animal, so the spots come out
#: the same size on the neck as on the tail instead of being stretched by each
#: piece's length.
TILE_M = 0.030
#: Tiles AROUND the girth. This has to be a WHOLE NUMBER and the same for every
#: ring: u ran 0..2pi*w/TILE_M, which is neither. A fractional wrap leaves the
#: texture discontinuous at the seam, and a wrap that changes from ring to ring
#: shears that discontinuity along the body -- which is why the tail came out
#: as a barber's pole. Two tiles round a ~60 mm girth keeps the spots roughly
#: as wide as they are long.
U_TILES = 2


# --------------------------------------------------------------------------
# cross-sections: what does a primitive look like on the plane x = X
# --------------------------------------------------------------------------

def _spheres_of(model, data, g):
    """Represent a geom as spheres, for the types where that is exact enough."""
    import mujoco
    t = model.geom_type[g]
    c = np.array(data.geom_xpos[g], float)
    R = np.array(data.geom_xmat[g], float).reshape(3, 3)
    s = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [(c, float(s[0]))]
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        # MuJoCo's capsule/cylinder axis is the geom's local Z; size = (r, half)
        axis = R[:, 2]
        r, half = float(s[0]), float(s[1])
        n = 14
        return [(c + axis * (half * (2.0 * k / n - 1.0)), r) for k in range(n + 1)]
    return []


def section(model, data, geoms, X):
    """Union cross-section of `geoms` on the plane x = X.

    Returns (half_width, z_low, z_high) or None if the plane misses them all.
    Width is measured in |y| because the animal is symmetric about y = 0.
    """
    import mujoco
    half_w, z_lo, z_hi = 0.0, None, None

    def note(w, zl, zh):
        nonlocal half_w, z_lo, z_hi
        half_w = max(half_w, w)
        z_lo = zl if z_lo is None else min(z_lo, zl)
        z_hi = zh if z_hi is None else max(z_hi, zh)

    for g in geoms:
        t = model.geom_type[g]
        c = np.array(data.geom_xpos[g], float)
        s = model.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            a = float(s[0])
            if abs(X - c[0]) >= a:
                continue
            k = math.sqrt(max(0.0, 1.0 - ((X - c[0]) / a) ** 2))
            note(abs(c[1]) + float(s[1]) * k, c[2] - float(s[2]) * k,
                 c[2] + float(s[2]) * k)
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            if abs(X - c[0]) >= float(s[0]):
                continue
            note(abs(c[1]) + float(s[1]), c[2] - float(s[2]), c[2] + float(s[2]))
        else:
            for p, r in _spheres_of(model, data, g):
                if abs(X - p[0]) >= r:
                    continue
                k = math.sqrt(max(0.0, r * r - (X - p[0]) ** 2))
                note(abs(p[1]) + k, p[2] - k, p[2] + k)
    if z_lo is None or half_w <= 0.0:
        return None
    return half_w, z_lo, z_hi


def skin_geoms(model, bid):
    """The visual primitives of a body that describe its OUTLINE.

    Eyes, teeth, tongue and lid are features sitting on or in the skin, not the
    skin, so wrapping them would put a bulge where the eye is. They are excluded
    by material name; everything else in the visual group counts.
    """
    import mujoco
    skip = set()
    for name in ("eye", "teeth", "tongue", "lid"):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if i >= 0:
            skip.add(i)
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        if model.geom_group[g] != 1:
            continue
        if int(model.geom_matid[g]) in skip:
            continue
        out.append(g)
    return out


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------

def axial_bodies(model):
    import mujoco
    out = []
    for n in CHAIN:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if b >= 0:
            out.append((n, b))
    return out


def build_axis(model, data):
    """Rings along the animal's long axis, each tagged with the body it belongs to.

    The ellipse at every station is MEASURED from the primitives crossing that
    station; PROFILE only styles it afterwards.
    """
    bodies = axial_bodies(model)
    geoms = {bid: skin_geoms(model, bid) for _, bid in bodies}

    xs = []
    for bid, gs in geoms.items():
        for g in gs:
            c = data.geom_xpos[g]
            xs += [c[0] - 0.04, c[0] + 0.04]
    lo, hi = min(xs), max(xs)
    # tighten to where something actually exists
    grid = np.linspace(lo, hi, 900)
    live = [X for X in grid
            if any(section(model, data, gs, X) for gs in geoms.values())]
    lo, hi = min(live), max(live)

    # ---- pass 1: measure the envelope at every station -------------------
    raw, prev_rank = [], -1
    gap = {}
    order = {bid: i for i, (_, bid) in enumerate(bodies)}
    # Sweep NOSE TO TAIL -- the same direction CHAIN is written in. The animal
    # faces +x, so that means descending x. Sweeping the other way makes the
    # monotonic-ownership rule below award every ring to the tail tip (#313b).
    for k in range(STATIONS + 1):
        X = hi - (hi - lo) * k / STATIONS
        best, best_bid = None, None
        for name, bid in bodies:
            sec = section(model, data, geoms[bid], X)
            if sec is None:
                continue
            sw, sh = PROFILE.get(name, (1.0, 1.0))
            w = sec[0] * sw
            h = (sec[2] - sec[1]) * 0.5 * sh
            zc = (sec[1] + sec[2]) * 0.5
            area = w * h
            if best is None or area > best[3]:
                best, best_bid = (w, h, zc, area), bid
        if best is None:
            # A station with no primitive at all. It must still produce a ring,
            # or the station spacing stops being uniform: the vent has a 6 mm
            # stretch with no visual geom, and SKIPPING it made the one-ring
            # overlap between neighbouring pieces 16 mm wide instead of 1.6 mm,
            # so the trunk and the tail each poked out through the other. Marked
            # here and interpolated below (#313d).
            raw.append((X, None, None, None, None))
            continue
        # Ownership must advance nose-to-tail and never go back, or a piece of
        # skin would be split across two non-adjacent bodies.
        rank = max(prev_rank, order[best_bid])
        prev_rank = rank
        raw.append((X, best[0], best[1], best[2], bodies[rank][1]))

    # drop leading/trailing gaps, then bridge the interior ones
    while raw and raw[0][1] is None:
        raw.pop(0)
    while raw and raw[-1][1] is None:
        raw.pop()
    for i, r in enumerate(raw):
        if r[1] is not None:
            continue
        a = next(j for j in range(i, -1, -1) if raw[j][1] is not None)
        b = next(j for j in range(i, len(raw)) if raw[j][1] is not None)
        # ANCHOR ON SOLID MATERIAL, NOT ON THE TAPERING TIP. Both primitives
        # either side of the vent are ellipsoids, and an ellipsoid's last
        # station before it ends is almost a point: interpolating tip-to-tip
        # gave the animal a 9.8 mm waist between a 28 mm trunk and a 32 mm tail,
        # so the tail looked stalked. The anchors are taken a few stations back
        # into solid flesh instead. The waist that remains is the one the
        # trunk's and tail's own widths imply -- still a vent constriction,
        # which the species has, but not a stalk.
        A = max(raw[j][1] for j in range(max(0, a - 4), a + 1)
                if raw[j][1] is not None)
        Ah = max(raw[j][2] for j in range(max(0, a - 4), a + 1)
                 if raw[j][2] is not None)
        B = max(raw[j][1] for j in range(b, min(len(raw), b + 5))
                if raw[j][1] is not None)
        Bh = max(raw[j][2] for j in range(b, min(len(raw), b + 5))
                 if raw[j][2] is not None)
        t = (i - a) / float(b - a)
        # a shallow cosine dip between the two anchors: the constriction is
        # real, its depth is INVENTED (photographs) and declared here
        # 1.0 = NO PINCH. It was 0.82, and the user was right that it looked
        # like the animal had been pinched behind the legs: a real leopard
        # gecko goes smoothly from hip into tail (#341). The sculpt in
        # gecko_sculpt.py now shapes this stretch; this value only feeds zc.
        VENT_PINCH = 1.0
        dip = 1.0 - (1.0 - VENT_PINCH) * math.sin(math.pi * t)
        raw[i] = (r[0],
                  (A + (B - A) * t) * dip,
                  (Ah + (Bh - Ah) * t) * dip,
                  raw[a][3] + (raw[b][3] - raw[a][3]) * t,
                  raw[a][4] if t < 0.5 else raw[b][4])
        gap[i] = True

    # ---- pass 2: smooth ALONG the animal ---------------------------------
    # Each body's primitives are a separate blob, so the measured envelope has
    # a bulge at every body centre and a pinch at every joint: a caterpillar,
    # not a gecko. Smoothing the width, height and centreline longitudinally
    # before the rings are built restores one continuous silhouette. The mean
    # is preserved, so the gated dimensions are not shrunk -- only the
    # segmentation between them is removed. Window INVENTED (#313c).
    arr = np.array([[r[1], r[2], r[3]] for r in raw], float)

    # Each axial primitive is an ellipsoid or capsule that tapers to nothing at
    # its own two ends, so where two of them meet tip-to-tip the union has a
    # WAIST -- the tail measured 10.6 mm, 6.3 mm, 11.7 mm at 10 mm intervals.
    # Smoothing alone cannot remove a real valley, it only rounds it, which is
    # why the tail came out as a stack of cones. A rolling maximum first fills
    # the valleys between the peaks while leaving every peak at its measured
    # value; the smoothing then rounds the result. Half-width 3 stations is
    # about half a tail segment: wide enough to bridge the joins, narrow enough
    # to leave the neck its real taper. INVENTED (#313e).
    #
    # THE WINDOW IS HALF A SEGMENT, AND THE VENT IS EXEMPT. The tail beads sit
    # ~20 mm apart, so filling the valley between two of them needs a window of
    # ~10 mm; at 1.25 mm per station that is 6 either side. But the animal also
    # has a REAL waist -- the vent, where the trunk ends and the fat tail has
    # not started, and which shows up as the one stretch with no primitive at
    # all. A blind maximum erases it along with the artefacts. The stations that
    # were interpolated across that gap, and a window either side of them, are
    # therefore left at their measured value: the artificial waists are filled,
    # the anatomical one is kept.
    k = 6
    wid = np.stack([np.max(np.stack([np.roll(arr[:, c], t)
                                     for t in range(-k, k + 1)]), axis=0)
                    for c in (0, 1)], axis=1)
    wid[:k] = arr[:k, :2]
    wid[-k:] = arr[-k:, :2]
    keep = np.zeros(len(arr), bool)
    for i in gap:
        keep[max(0, i - k):min(len(arr), i + k + 1)] = True
    wid[keep] = arr[keep, :2]
    arr = np.concatenate([wid, arr[:, 2:3]], axis=1)

    win = 13
    pad = np.vstack([np.repeat(arr[:1], win, axis=0), arr,
                     np.repeat(arr[-1:], win, axis=0)])
    ker = np.hanning(win + 2)[1:-1]
    ker = ker / ker.sum()
    sm = np.stack([np.convolve(pad[:, c], ker, mode="same")
                   for c in range(3)], axis=1)[win:win + len(arr)]

    # ---- pass 3: rings ---------------------------------------------------
    rings, uvs, owner = [], [], []
    v_arc, prev_c = 0.0, None
    n = len(raw)
    for i, (X, _, _, _, bid) in enumerate(raw):
        w, h, zc = sm[i]
        # DO NOT TAPER THE ENDS HERE. The measured envelope ALREADY closes: the
        # snout's own rounded tip primitive and the tail's last segment both
        # shrink to nothing on their own. Multiplying that by a second taper
        # closed the tube twice, and the rings piled up almost on top of each
        # other at the nose -- which is the crease that was visible running back
        # from the snout. The two ends get a proper dome below instead (#328).
        c = np.array([X, 0.0, zc])
        if prev_c is not None:
            v_arc += float(np.linalg.norm(c - prev_c))
        prev_c = c
        pts, uv = ring_at(c, w, h, v_arc / TILE_M)
        rings.append(pts)
        uvs.append(uv)
        owner.append(bid)

    # ---- pass 3b: the head is sculpted, not swept (#340) ------------------
    rings, uvs = sculpt_head(model, data, rings, uvs, owner)
    # ---- pass 3c: and so is everything behind it (#341) -------------------
    # neck, belly, hips, fat tail with ridges, rounded tip. Anchored to the
    # measured trunk and tail maxima; see tools/gecko_sculpt.py.
    from gecko_sculpt import sculpt_body
    rings = sculpt_body(model, data, rings, uvs, owner, RING, CHAIN)

    # ---- pass 4: a real dome on each end ---------------------------------
    # The snout and the tail tip are now closed by a quarter-ellipse of rings
    # that bulges OUTWARD from the last measured cross-section and ends on a
    # single point, rather than by squashing the last few rings toward the axis.
    # A squashed ring is still a ring: twenty vertices a fraction of a
    # millimetre apart, which shades as a crease no matter how fine the mesh is.
    head_dome = dome(rings[0], uvs[0], +1.0, v_arc)
    tail_dome = dome(rings[-1], uvs[-1], -1.0, v_arc)
    rings = head_dome[0][::-1] + rings + tail_dome[0]
    uvs = head_dome[1][::-1] + uvs + tail_dome[1]
    owner = [owner[0]] * len(head_dome[0]) + owner + [owner[-1]] * len(tail_dome[0])
    return rings, uvs, owner


#: Rings in each end dome. INVENTED; it only sets how round the nose looks.
DOME_RINGS = 5


def dome(ring, uv, direction, v_total):
    """Close one end of the tube with a quarter-ellipse, ending on a point.

    `direction` is +1 to build forward off the nose ring and -1 to build back
    off the tail ring. The dome's length is the mean half-width of the ring it
    grows from, so a blunt snout gets a blunt dome and a thin tail tip a thin
    one -- the shape is taken from the body rather than chosen.
    """
    centre = ring.mean(axis=0)
    half_w = float(np.abs(ring[:, 1] - centre[1]).max())
    half_h = float(np.abs(ring[:, 2] - centre[2]).max())
    reach = 0.5 * (half_w + half_h)
    out_rings, out_uvs = [], []
    for k in range(1, DOME_RINGS + 1):
        t = k / float(DOME_RINGS)
        scale = math.sqrt(max(0.0, 1.0 - t * t))
        x = centre[0] + direction * reach * t
        pts = np.stack([np.full(len(ring), x),
                        centre[1] + (ring[:, 1] - centre[1]) * scale,
                        centre[2] + (ring[:, 2] - centre[2]) * scale], axis=1)
        out_rings.append(pts)
        out_uvs.append(np.stack([uv[:, 0],
                                 uv[:, 1] + direction * reach * t / TILE_M],
                                axis=1))
    return out_rings, out_uvs


def ring_at(centre, half_w, half_h, v):
    """One elliptical ring. RING+1 points: the seam is duplicated so the texture
    can run 0 -> 1 around the body without the last quad wrapping backwards."""
    ang = np.linspace(0.0, 2.0 * math.pi, RING + 1)
    pts = np.stack([np.full(RING + 1, centre[0]),
                    centre[1] + half_w * np.cos(ang),
                    centre[2] + half_h * np.sin(ang)], axis=1)
    uv = np.stack([ang / (2.0 * math.pi) * U_TILES, np.full(RING + 1, v)], axis=1)
    return pts, uv


def limb_tubes(model, data):
    """A tapered tube along each limb bone, as {bid: (rings, uvs)}."""
    import mujoco
    out = {}
    for side in ("L", "R"):
        for chain in (("humerus", "forearm", "manus"), ("femur", "tibia", "pes")):
            for seg in chain:
                name = f"{seg}_{side}"
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    continue
                kids = [c for c in range(model.nbody)
                        if model.body_parentid[c] == bid]
                start = np.array(data.xpos[bid], float)
                end = (np.array(data.xpos[kids[0]], float) if kids
                       else start + np.array([0.006, 0.0, 0.0]))
                r = 0.004
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == bid and model.geom_group[g] == 1:
                        r = max(r, float(model.geom_size[g][0]))
                r *= LIMB_SCALE
                axis = end - start
                L = float(np.linalg.norm(axis))
                if L < 1e-6:
                    continue
                axis = axis / L
                tmp = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(tmp, axis))) > 0.95:
                    tmp = np.array([0.0, 1.0, 0.0])
                u1 = np.cross(axis, tmp); u1 /= np.linalg.norm(u1)
                u2 = np.cross(axis, u1)
                rings, uvs = [], []
                N = 6
                ang = np.linspace(0.0, 2 * math.pi, RING + 1)
                u_rep = 1
                for k in range(N + 1):
                    t = k / N
                    # taper 0.25, not 0.45: a limb that halves in radius over
                    # one bone leaves a step at every joint for the knuckle to
                    # have to hide (#341)
                    rr = r * (1.0 - 0.25 * t)
                    # (the socket used to be shrunk to 0.72 r and tucked in; with a
                    # joint sphere covering each articulation that only made a
                    # flared collar at the ankle, #341)
                    c = start + axis * (L * t)
                    pts = np.stack([c + rr * (math.cos(a) * u1 + math.sin(a) * u2)
                                    for a in ang])
                    rings.append(pts)
                    uvs.append(np.stack([ang / (2 * math.pi) * u_rep,
                                         np.full(RING + 1, t * L / TILE_M)], axis=1))
                out[bid] = (rings, uvs)
    return out


def teeth_rim(jaw_rings):
    """A pale rim along the mandible's upper edge.

    A box cannot follow a tapering jaw: sized to show at the hinge it juts out
    past the chin, and sized to fit the chin it vanishes inside the jaw. The rim
    is therefore built from the JAW'S OWN rings -- a thin, flattened tube a few
    per cent wider than the mandible and sitting at its upper edge, so only its
    two side edges emerge, as a tooth line that tapers with the jaw.
    """
    ang = np.linspace(0.0, 2.0 * math.pi, RING + 1)
    rings, uvs = [], []
    for i, r in enumerate(jaw_rings):
        w = float(np.abs(r[:, 1]).max())
        zc = float(r[:, 2].mean())
        ztop = float(r[:, 2].max())
        h = (ztop - zc)
        if w < 1e-5:
            continue
        rings.append(np.stack([np.full(RING + 1, r[0, 0]),
                               (w * 1.035) * np.cos(ang),
                               (zc + h * 0.42) + 0.0006 * np.sin(ang)], axis=1))
        uvs.append(np.stack([ang / (2 * math.pi), np.full(RING + 1, i / 8.0)],
                            axis=1))
    return rings, uvs


def jaw_tube(model, data):
    """The lower jaw, as wide as the head it closes against.

    A gecko's mandible is nearly the width of its skull; the collision capsule
    is a 2.9 mm rod because that is all the physics needs. Sweeping the jaw with
    the HEAD's measured half-width at each station -- narrowed by the ratio
    below -- makes the closed mouth a seam instead of a gap, and the open mouth
    a mouth. The narrowing is INVENTED (photographs).
    """
    import mujoco
    jb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw")
    hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    if jb < 0 or hb < 0:
        return None
    head_g = skin_geoms(model, hb)
    jaw_g = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == jb and model.geom_type[g] ==
             mujoco.mjtGeom.mjGEOM_CAPSULE]
    if not jaw_g:
        return None
    xs = []
    for g in jaw_g:
        for p, r in _spheres_of(model, data, g):
            xs += [p[0] - r, p[0] + r]
    lo, hi = min(xs), max(xs)
    JAW_WIDTH_OF_HEAD = 0.88
    rings, uvs = [], []
    N = 16
    ang = np.linspace(0.0, 2 * math.pi, RING + 1)
    for k in range(N + 1):
        t = k / N
        X = lo + (hi - lo) * t
        sec = section(model, data, jaw_g, X)
        if sec is None:
            continue
        hs = section(model, data, head_g, X)
        w = (hs[0] * JAW_WIDTH_OF_HEAD) if hs else sec[0]
        w = max(w, sec[0])
        # taper to the chin
        w *= (1.0 - 0.42 * t ** 2)
        h = 0.0034 * (1.0 - 0.45 * t)
        zc = (sec[1] + sec[2]) * 0.5
        if k == N:
            w, h = w * 0.25, h * 0.25
        c = np.array([X, 0.0, zc])
        u_rep = U_TILES
        rings.append(np.stack([np.full(RING + 1, c[0]),
                               w * np.cos(ang), c[2] + h * np.sin(ang)], axis=1))
        uvs.append(np.stack([ang / (2 * math.pi) * u_rep,
                             np.full(RING + 1, t * (hi - lo) / TILE_M)], axis=1))
    return jb, rings, uvs


# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------

def stitch(rings, uvs):
    """Triangulate consecutive rings into a tube, with smooth vertex normals."""
    per = RING + 1
    verts = np.concatenate(rings, axis=0)
    uv = np.concatenate(uvs, axis=0)
    faces = []
    for i in range(len(rings) - 1):
        a0, b0 = i * per, (i + 1) * per
        for j in range(RING):
            faces.append((a0 + j, b0 + j, b0 + j + 1))
            faces.append((a0 + j, b0 + j + 1, a0 + j + 1))

    # area-weighted vertex normals, so the tube shades as a curve rather than
    # as a stack of faceted bands
    nrm = np.zeros_like(verts)
    for a, b, c in faces:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        nrm[a] += n; nrm[b] += n; nrm[c] += n
    # the duplicated seam column is one edge of the surface, not two
    for i in range(len(rings)):
        a, b = i * per, i * per + RING
        s = nrm[a] + nrm[b]
        nrm[a] = s; nrm[b] = s
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln < 1e-12, 1.0, ln)

    # Winding check, rather than a winding assumption. A tube swept in -x and
    # one swept in +x have opposite face orientation, and a mesh whose normals
    # point inward renders as a dark silhouette with no shading.
    ctr = np.repeat(np.array([r.mean(axis=0) for r in rings]), per, axis=0)
    if float((nrm * (verts - ctr)).sum()) < 0.0:
        faces = [(a, c, b) for a, b, c in faces]
        nrm = -nrm
    return verts, uv, nrm, faces


def write_obj(path, verts, uvs, nrms, faces):
    lines = ["# generated by tools/make_gecko_mesh.py -- do not hand-edit"]
    lines += [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in verts]
    lines += [f"vt {u[0]:.5f} {u[1]:.5f}" for u in uvs]
    lines += [f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}" for n in nrms]
    for a, b, c in faces:
        lines.append(f"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(verts), len(faces)


def emit(model, data, bid, rings, uvs, written):
    import mujoco
    v, uv, nrm, f = stitch(rings, uvs)
    R = np.array(data.xmat[bid], float).reshape(3, 3)
    v = (v - np.array(data.xpos[bid], float)) @ R
    nrm = nrm @ R
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    nv, nf = write_obj(OUT / f"{name}.obj", v, uv, nrm, f)
    written.append((name, nv, nf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="morphology/gecko_body_lab_v2.xml")
    args = ap.parse_args()
    import mujoco

    # Compile the body WITHOUT its mesh geoms. The skin is generated from the
    # primitives, so reading a body that already references the meshes would
    # both be circular and make the generator unable to run after a clean --
    # the very first thing it does is delete the files that body points at.
    src = REPO / args.body
    text = src.read_text(encoding="utf-8")
    # Strip mesh geoms as ELEMENTS, and the skin too. Filtering by LINE drops
    # whatever else happens to share that line, which is how the eye spheres
    # disappeared once two geoms ended up written on one line (#330).
    text = re.sub(r"[ \t]*<deformable>.*?</deformable>\n?", "", text, flags=re.S)
    text = re.sub(r'<geom[^>]*type="mesh"[^>]*/>', '', text)
    kept = [ln for ln in text.splitlines() if not re.match(r'\s*<mesh ', ln)]
    tmp = src.with_name(src.stem + "_nomesh_tmp.xml")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral").id)
    mujoco.mj_forward(model, data)

    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    rings, uvs, owner = build_axis(model, data)
    for bid in sorted(set(owner)):
        idx = [i for i, o in enumerate(owner) if o == bid]
        lo, hi = min(idx), max(idx)
        # one ring of overlap each side, so neighbouring pieces share a
        # cross-section and the seam does not open when the joint bends
        lo = max(0, lo - 1); hi = min(len(rings) - 1, hi + 1)
        if hi - lo < 1:
            continue
        emit(model, data, bid, rings[lo:hi + 1], uvs[lo:hi + 1], written)

    for bid, (lr, lu) in limb_tubes(model, data).items():
        emit(model, data, bid, lr, lu, written)

    jaw = jaw_tube(model, data)
    if jaw is not None:
        emit(model, data, jaw[0], jaw[1], jaw[2], written)
        tr, tu = teeth_rim(jaw[1])
        v, uv, nrm, f = stitch(tr, tu)
        R = np.array(data.xmat[jaw[0]], float).reshape(3, 3)
        v = (v - np.array(data.xpos[jaw[0]], float)) @ R
        nv, nf = write_obj(OUT / "teeth.obj", v, uv, nrm @ R, f)
        written.append(("teeth", nv, nf))

    total_f = sum(f for _, _, f in written)
    print(f"{len(written)} meshes, {total_f} triangles total "
          f"(the AI-generated sculpt was 1,958,254)")
    for n, nv, nf in sorted(written):
        print(f"   {n:16} {nv:5d} verts  {nf:5d} tris")
    print(f"\nwritten to {OUT}")



# --------------------------------------------------------------------------
# THE HEAD IS SCULPTED, NOT SWEPT (#340)
#
# Everything above sweeps one oval along the animal's axis. That is fine for a
# trunk and a tail, which really are tubes. It is not fine for a head, and the
# result was fair comment: a missile. An oval swept along a line can be fat here
# and thin there and that is the whole of its vocabulary -- it has no way to
# express a flat crown, a brow, a jaw angle, a cheek, or a snout that narrows
# faster than it drops. So the head is built from its own cross-sections here.
#
# TWO THINGS MAKE IT A HEAD RATHER THAN A CONE.
#
#   1. The cross-section is a SUPERELLIPSE with a different exponent above and
#      below the midline. A gecko's skull is flat on top and rounded underneath;
#      an ellipse is equally round in both directions, which is most of why the
#      old head read as a nose cone.
#   2. The width and the height run on SEPARATE curves. The snout narrows much
#      faster than it flattens, and the widest point of the head is at the jaw
#      angle, well behind the eyes. A single radius cannot say either of those.
#
# The profile is INVENTED, read off photographs of adult E. macularius, and is
# ANCHORED to the body: it is scaled so its widest station equals the widest
# half-width actually measured from the head's own primitives, so the dimension
# the gates check is reproduced and only the shape between stations is styled.
# --------------------------------------------------------------------------

#: (fraction from occiput to snout tip, width, height, centreline drop in m).
#: Width and height are fractions of the measured maximum. INVENTED.
HEAD_PROFILE = (
    (0.00, 0.72, 0.70,  0.0000),   # occiput, meets the neck
    (0.10, 0.93, 0.88,  0.0000),
    (0.22, 1.00, 0.95,  0.0000),   # jaw angle: the widest part of the head
    (0.38, 0.99, 1.00,  0.0000),   # brow ridge is the tallest part
    (0.52, 0.93, 0.97, -0.0004),   # eyes
    (0.64, 0.72, 0.80, -0.0010),   # the snout starts here and narrows fast
    (0.76, 0.53, 0.60, -0.0016),
    (0.88, 0.40, 0.45, -0.0020),
    (0.96, 0.30, 0.34, -0.0022),
    (1.00, 0.17, 0.21, -0.0024),   # blunt, not pointed
)
#: Superellipse exponents above and below the midline. 2 is an ellipse; higher
#: is flatter-topped. INVENTED.
HEAD_N_TOP = 2.7
HEAD_N_BOTTOM = 2.1


def _interp_profile(t):
    ts = [p[0] for p in HEAD_PROFILE]
    if t <= ts[0]:
        return HEAD_PROFILE[0][1:]
    if t >= ts[-1]:
        return HEAD_PROFILE[-1][1:]
    for i in range(len(HEAD_PROFILE) - 1):
        a, b = HEAD_PROFILE[i], HEAD_PROFILE[i + 1]
        if a[0] <= t <= b[0]:
            u = (t - a[0]) / (b[0] - a[0])
            u = u * u * (3.0 - 2.0 * u)          # smooth, no corners
            return tuple(a[k] + (b[k] - a[k]) * u for k in (1, 2, 3))
    return HEAD_PROFILE[-1][1:]


def head_ring(centre, half_w, half_h, v):
    """One superellipse cross-section: flat on top, rounded underneath."""
    ang = np.linspace(0.0, 2.0 * math.pi, RING + 1)
    c, s = np.cos(ang), np.sin(ang)
    n = np.where(s >= 0.0, HEAD_N_TOP, HEAD_N_BOTTOM)
    y = np.sign(c) * np.abs(c) ** (2.0 / n) * half_w
    z = np.sign(s) * np.abs(s) ** (2.0 / n) * half_h
    pts = np.stack([np.full(RING + 1, centre[0]),
                    centre[1] + y, centre[2] + z], axis=1)
    uv = np.stack([ang / (2.0 * math.pi) * U_TILES, np.full(RING + 1, v)], axis=1)
    return pts, uv


def sculpt_head(model, data, rings, uvs, owner):
    """Replace the head's swept ovals with the sculpted profile, in place.

    Runs after the sweep so the head keeps the same stations, the same ring
    size and the same texture coordinates -- only the CROSS-SECTION changes, so
    the join to the neck is automatic and nothing downstream has to know.

    The profile is anchored to the body: its widest and tallest stations are set
    to the widest and tallest the sweep actually measured from the head's own
    primitives. The shape between them is what is invented.
    """
    import mujoco
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    idx = [i for i, o in enumerate(owner) if o == head]
    if len(idx) < 4:
        return rings, uvs
    xs = np.array([rings[i][0, 0] for i in idx])
    tip, occiput = float(xs.max()), float(xs.min())
    if tip - occiput < 1e-6:
        return rings, uvs
    W = max(float(np.abs(rings[i][:, 1]).max()) for i in idx)
    H = max(0.5 * float(rings[i][:, 2].max() - rings[i][:, 2].min()) for i in idx)
    Z = float(np.mean([rings[i][:, 2].mean() for i in idx]))
    # BLEND INTO THE NECK RATHER THAN STEP INTO IT. The sculpt's occiput ring
    # is 0.72 of the head's widest, and the neck ring next to it is whatever the
    # sweep measured; butting the two together left a visible notch at the back
    # of the skull. The first fifth of the head fades from the measured ring to
    # the sculpted one.
    BLEND_T = 0.20
    for i in idx:
        x = float(rings[i][0, 0])
        t = (x - occiput) / (tip - occiput)
        wf, hf, dz = _interp_profile(t)
        centre = np.array([x, 0.0, Z + dz])
        pts, _ = head_ring(centre, W * wf, H * hf, 0.0)
        if t < BLEND_T:
            u = t / BLEND_T
            u = u * u * (3.0 - 2.0 * u)
            pts = rings[i] * (1.0 - u) + pts * u
        rings[i] = pts
        # keep the texture coordinates the sweep already assigned
        uvs[i] = np.stack([uvs[i][:, 0], uvs[i][:, 1]], axis=1)
    return rings, uvs

if __name__ == "__main__":
    main()
