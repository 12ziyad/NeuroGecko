"""Sculpt the body: neck, belly, hips, tail, toes. Everything the sweep could not.

WHY THIS FILE EXISTS. `make_gecko_mesh.build_axis` sweeps one oval along the
animal, sized station by station from the primitives. That is a faithful
shrink-wrap of the physics and it looks like a missile, because a shrink-wrap of
capsules IS a missile. It has no neck (the neck capsule is nearly as wide as the
trunk), no belly (a capsule is the same width all along), no hips, and its tail
ends in a needle. The head was fixed by sculpting it from a profile instead
(#340). This does the same for everything behind the head.

HOW IT STAYS HONEST. The profile is INVENTED, read off photographs of adult
*E. macularius*, and is ANCHORED to the body at three stations: the widest
measured half-width of the trunk, of the tail, and the head's occiput ring that
`sculpt_head` already fixed. Between anchors the shape is styled; at the
anchors it is the measured body. The primitives are still the physics and this
cannot move a gate.

TOES. Five per foot, and they are not placed by this file: the body already
defines five digit capsules per foot with explicit endpoints, so the toes are
built along those, exactly where the body says the digits are. This species has
NO adhesive toe pads (verified, #337), so the toes are slender tapered tubes with
a rounded tip -- not the round-ended paddles of the climbing geckos.

INVENTED HERE: every number in the profile tables, the tail ridge amplitude and
period, the toe skin-standoff and taper. None can move a gate.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------
# cross-section: superellipse, flatter on top than below
# --------------------------------------------------------------------------

def superellipse_ring(centre, half_w, half_h, n_top, n_bot, ring_n):
    """One cross-section. n = 2 is an ellipse; higher is flatter-topped."""
    ang = np.linspace(0.0, 2.0 * math.pi, ring_n + 1)
    c, s = np.cos(ang), np.sin(ang)
    n = np.where(s >= 0.0, n_top, n_bot)
    y = np.sign(c) * np.abs(c) ** (2.0 / n) * half_w
    z = np.sign(s) * np.abs(s) ** (2.0 / n) * half_h
    return np.stack([np.full(ring_n + 1, centre[0]), centre[1] + y, centre[2] + z],
                    axis=1)


def _smooth_curve(points, f):
    """Piecewise smoothstep through (fraction, value) points."""
    if f <= points[0][0]:
        return points[0][1]
    if f >= points[-1][0]:
        return points[-1][1]
    for (fa, va), (fb, vb) in zip(points[:-1], points[1:]):
        if fa <= f <= fb:
            u = (f - fa) / max(1e-12, fb - fa)
            u = u * u * (3.0 - 2.0 * u)
            return va + (vb - va) * u
    return points[-1][1]


# --------------------------------------------------------------------------
# the body profile, region by region
# --------------------------------------------------------------------------

#: NECK, as fractions of the value at its own two ends: starts at the head's
#: occiput ring, ends at the shoulder. Dips between. INVENTED.
NECK_DIP = 0.84
NECK_DIP_AT = 0.40

#: TRUNK, fractions of the trunk's measured maximum. Widest at 0.45 of the
#: way back -- a leopard gecko carries its belly behind the middle. INVENTED.
TRUNK_W = ((0.00, 0.86), (0.25, 0.97), (0.45, 1.00), (0.70, 0.96), (1.00, 0.80))
TRUNK_H = ((0.00, 0.86), (0.25, 0.96), (0.45, 1.00), (0.70, 0.94), (1.00, 0.78))

#: TAIL, fractions of the tail's measured maximum. Fat through the first
#: third, then a long even taper to a ROUNDED tip. No pinch at the vent: the
#: real animal goes smoothly from hip into tail. INVENTED.
TAIL_W = ((0.00, 0.78), (0.10, 0.94), (0.24, 1.00), (0.42, 0.90), (0.60, 0.66),
          (0.78, 0.42), (0.92, 0.24), (1.00, 0.16))
TAIL_H = ((0.00, 0.76), (0.10, 0.92), (0.24, 1.00), (0.42, 0.90), (0.60, 0.66),
          (0.78, 0.42), (0.92, 0.24), (1.00, 0.16))

#: Tail ridges: the fat tail is banded by transverse rows of tubercles that
#: give it a visibly wavy outline. Amplitude as a fraction of radius, period
#: in metres. INVENTED (photographs).
TAIL_RIDGE_AMP = 0.035
TAIL_RIDGE_PERIOD_M = 0.0046

#: Superellipse exponents: a slightly flat back and a round belly. INVENTED.
BODY_N_TOP = 2.35
BODY_N_BOT = 2.0
TAIL_N_TOP = 2.15
TAIL_N_BOT = 2.0


def sculpt_body(model, data, rings, uvs, owner, ring_n, chain_names):
    """Override every station behind the head with the sculpted profile.

    `rings` are nose-to-tail, each (ring_n+1, 3) in world space; `owner` is the
    body id per ring. The head's rings are left exactly as `sculpt_head` made
    them; this starts at the neck.
    """
    import mujoco

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    head = bid("head")
    neck = bid("neck")
    trunk_ids = {bid(n) for n in ("trunk_anterior", "trunk_middle") if bid(n) >= 0}
    tail_ids = {bid(n) for n in chain_names if n.startswith("tail") and bid(n) >= 0}

    idx_head = [i for i, o in enumerate(owner) if o == head]
    idx_neck = [i for i, o in enumerate(owner) if o == neck]
    idx_trunk = [i for i, o in enumerate(owner) if o in trunk_ids]
    idx_tail = [i for i, o in enumerate(owner) if o in tail_ids]
    if not (idx_head and idx_neck and idx_trunk and idx_tail):
        return rings

    def half_w(i):
        return float(np.abs(rings[i][:, 1]).max())

    def half_h(i):
        return 0.5 * float(rings[i][:, 2].max() - rings[i][:, 2].min())

    def zc(i):
        return 0.5 * float(rings[i][:, 2].max() + rings[i][:, 2].min())

    # ---- anchors, measured from the body -----------------------------------
    W_trunk = max(half_w(i) for i in idx_trunk)
    H_trunk = max(half_h(i) for i in idx_trunk)
    W_tail = max(half_w(i) for i in idx_tail[: max(3, len(idx_tail) // 2)])
    H_tail = max(half_h(i) for i in idx_tail[: max(3, len(idx_tail) // 2)])
    occ = idx_head[-1]                       # last head ring = occiput
    w_occ, h_occ = half_w(occ), half_h(occ)

    x_neck0, x_neck1 = rings[idx_neck[0]][0, 0], rings[idx_neck[-1]][0, 0]
    x_trunk0, x_trunk1 = rings[idx_trunk[0]][0, 0], rings[idx_trunk[-1]][0, 0]
    x_tail0, x_tail1 = rings[idx_tail[0]][0, 0], rings[idx_tail[-1]][0, 0]

    def frac(x, a, b):
        return float((a - x) / max(1e-9, a - b))     # x descends nose -> tail

    # ---- neck: from the occiput ring to the first trunk station -----------
    w_sh, h_sh = W_trunk * TRUNK_W[0][1], H_trunk * TRUNK_H[0][1]
    for i in idx_neck:
        x = rings[i][0, 0]
        f = frac(x, x_neck0, x_trunk0)
        dip = 1.0 - (1.0 - NECK_DIP) * math.exp(-((f - NECK_DIP_AT) / 0.28) ** 2)
        w = (w_occ + (w_sh - w_occ) * _smooth_curve(((0, 0), (1, 1)), f)) * dip
        h = (h_occ + (h_sh - h_occ) * _smooth_curve(((0, 0), (1, 1)), f)) * dip
        rings[i] = superellipse_ring((x, 0.0, zc(i)), w, h,
                                     BODY_N_TOP, BODY_N_BOT, ring_n)

    # ---- trunk ------------------------------------------------------------
    for i in idx_trunk:
        x = rings[i][0, 0]
        f = frac(x, x_trunk0, x_trunk1)
        w = W_trunk * _smooth_curve(TRUNK_W, f)
        h = H_trunk * _smooth_curve(TRUNK_H, f)
        rings[i] = superellipse_ring((x, 0.0, zc(i)), w, h,
                                     BODY_N_TOP, BODY_N_BOT, ring_n)

    # ---- tail: continuous with the hip, no pinch --------------------------
    w_hip, h_hip = W_trunk * TRUNK_W[-1][1], H_trunk * TRUNK_H[-1][1]
    for i in idx_tail:
        x = rings[i][0, 0]
        f = frac(x, x_tail0, x_tail1)
        w = W_tail * _smooth_curve(TAIL_W, f)
        h = H_tail * _smooth_curve(TAIL_H, f)
        # the first tenth blends from the hip so the join is invisible
        if f < 0.10:
            u = f / 0.10
            u = u * u * (3.0 - 2.0 * u)
            w = w_hip + (w - w_hip) * u
            h = h_hip + (h - h_hip) * u
        # transverse ridges over the fat part, fading out toward the tip
        gain = math.sin(math.pi * min(1.0, f / 0.85)) if f < 0.85 else 0.0
        ridge = 1.0 + TAIL_RIDGE_AMP * gain * math.sin(
            2.0 * math.pi * (x_tail0 - x) / TAIL_RIDGE_PERIOD_M)
        rings[i] = superellipse_ring((x, 0.0, zc(i)), w * ridge, h * ridge,
                                     TAIL_N_TOP, TAIL_N_BOT, ring_n)
    return rings


# --------------------------------------------------------------------------
# toes
# --------------------------------------------------------------------------

#: Skin stands this far off the digit capsule, and the toe tapers to this
#: fraction of its base radius before the rounded tip. INVENTED.
TOE_SKIN = 1.45
TOE_TAPER = 0.62
TOE_RINGS = 6


def toe_tubes(model, data, ring_n):
    """Five toes per foot, along the digit capsules the body defines.

    Returns a list of (foot body id, rings, uvs). The digits are found as the
    thin capsules in each foot body, so nothing here decides where a toe goes.
    """
    import mujoco
    out = []
    ang = np.linspace(0.0, 2.0 * math.pi, ring_n + 1)
    for side in ("L", "R"):
        for foot in ("manus", "pes"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{foot}_{side}")
            if bid < 0:
                continue
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != bid:
                    continue
                if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                    continue
                r0, half = float(model.geom_size[g][0]), float(model.geom_size[g][1])
                if r0 > 0.0015:
                    continue                      # not a digit
                c = np.array(data.geom_xpos[g], float)
                axis = np.array(data.geom_xmat[g], float).reshape(3, 3)[:, 2]
                start, end = c - axis * half, c + axis * half
                # digits point away from the body; make sure the tube does too
                foot_c = np.array(data.xpos[bid], float)
                if np.linalg.norm(start - foot_c) > np.linalg.norm(end - foot_c):
                    start, end, axis = end, start, -axis
                L = float(np.linalg.norm(end - start))
                # Angled up about 14 degrees and a tenth shorter: the flat
                # digit capsules put the toe tips below the foot's collision
                # box, and the skin drawn on them went into the floor (#343).
                axis = axis + np.array([0.0, 0.0, 0.25])
                axis = axis / np.linalg.norm(axis)
                L *= 0.90
                end = start + axis * L
                r = r0 * TOE_SKIN
                tmp = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(tmp, axis))) > 0.95:
                    tmp = np.array([0.0, 1.0, 0.0])
                u1 = np.cross(axis, tmp); u1 /= np.linalg.norm(u1)
                u2 = np.cross(axis, u1)
                rings, uvs = [], []
                # socket ring, sunk into the foot so the toe grows out of it
                for k in range(TOE_RINGS + 1):
                    t = k / TOE_RINGS
                    rr = r * (1.0 + (TOE_TAPER - 1.0) * t)
                    cc = start + axis * (L * t - (0.0008 if k == 0 else 0.0))
                    rings.append(np.stack([cc + rr * (math.cos(a) * u1 + math.sin(a) * u2)
                                           for a in ang]))
                    uvs.append(np.stack([0.15 + 0.20 * ang / (2 * math.pi), np.full(ring_n + 1, 0.30 + 0.02 * t)], axis=1))
                # rounded tip: three shrinking rings past the end
                rtip = r * TOE_TAPER
                for k, (sc, adv) in enumerate(((0.80, 0.35), (0.48, 0.70), (0.06, 0.95))):
                    cc = end + axis * (rtip * adv)
                    rings.append(np.stack([cc + rtip * sc * (math.cos(a) * u1 + math.sin(a) * u2)
                                           for a in ang]))
                    uvs.append(np.stack([0.15 + 0.20 * ang / (2 * math.pi), np.full(ring_n + 1, 0.32 + 0.004 * k)], axis=1))
                out.append((bid, rings, uvs))
    return out


# --------------------------------------------------------------------------
# joints
# --------------------------------------------------------------------------

#: A sphere at every limb articulation, a little larger than the tubes that
#: meet there. Two tapered tubes meeting at an angle leave a wedge open on the
#: outside of the bend and a twisted collar where their end rings cross; the
#: sphere hides both, which is what a knuckle is for. INVENTED standoff.
JOINT_SCALE = 0.72   # was 1.06: sat between tube ENDS that had tapered to 0.55 r, so it looked like a ping-pong ball
JOINT_BODIES = ("humerus", "forearm", "manus", "femur", "tibia", "pes")


def joint_spheres(model, data, ring_n, limb_scale):
    """UV spheres at the origin of each limb segment. Returns (bid, rings, uvs)."""
    import mujoco
    out = []
    ang = np.linspace(0.0, 2.0 * math.pi, ring_n + 1)
    for side in ("L", "R"):
        for seg in JOINT_BODIES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{seg}_{side}")
            if bid < 0:
                continue
            r = 0.004
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid and model.geom_group[g] == 1:
                    r = max(r, float(model.geom_size[g][0]))
            r = r * limb_scale * JOINT_SCALE
            if seg in ("manus", "pes"):
                r *= 0.85
            c = np.array(data.xpos[bid], float)
            rings, uvs = [], []
            n_lat = 8
            for i in range(n_lat + 1):
                polar = math.pi * i / n_lat
                rr = r * math.sin(polar)
                z = r * math.cos(polar)
                pts = np.stack([c + np.array([rr * math.cos(a), rr * math.sin(a), z])
                                for a in ang])
                rings.append(pts)
                # u confined to the spotted dorsal band: a full 0..1 wrap paints
                # half of every knuckle with the pale belly
                uvs.append(np.stack([0.15 + 0.20 * ang / (2 * math.pi),
                                     np.full(ring_n + 1, 0.22 + 0.03 * i / n_lat)], axis=1))
            out.append((bid, rings, uvs))
    return out


# --------------------------------------------------------------------------
# ONE SKIN PER LIMB, shoulder to foot, through the joints (#343)
#
# Six separate tubes with a ball at each articulation was the wrong answer to
# "why does the leg come apart when it bends": it was the same mistake the
# body made before it got one continuous skin. The limb is now a single tube
# along a smoothed path through the joint positions, with a PARALLEL-TRANSPORT
# frame so the rings never twist against each other where the path bends. The
# root is sunk into the trunk so no open end shows at the shoulder or hip, and
# the foot end is closed with a dome that the toes grow out of.
# --------------------------------------------------------------------------

LIMB_STATIONS = 34

#: How much the limb is flattened top-to-bottom, as a fraction of its width.
#: INVENTED. A sprawling lizard's limb is not a circular rod -- it is wider
#: than it is deep -- but NO limb cross-section has ever been published for
#: this species or any eublepharid, and the 42-agent surface pass (#337) killed
#: 26 of 36 appearance claims. So this is a shape choice that says it is one.
LIMB_FLATTEN = 0.74

#: The radius profile along one limb, as multipliers on the segment radius at
#: five stations: shoulder, mid-upper-arm, ELBOW, mid-forearm, WRIST.
#: INVENTED, and the whole point of it is the two pinches. The old profile ran
#: 1.08, 1.05, 0.96, 0.80, 0.62 -- monotonically down -- so nothing narrowed at
#: an articulation and a smooth curve through it reads as a banana (#342, #341,
#: #317). Muscle belly proximally, waist at the joint, smaller belly distally
#: is the shape of a real limb; the NUMBERS are chosen, not measured.
LIMB_BELLY = 1.22
LIMB_JOINT_WAIST = 0.66
LIMB_ROOT_SINK_M = 0.004

_UP = np.array([0.0, 0.0, 1.0])


def _chaikin(pts, passes=2):
    pts = [np.asarray(p, float) for p in pts]
    for _ in range(passes):
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(pts[-1])
        pts = out
    return np.array(pts)


def limb_skins(model, data, ring_n, limb_scale, v_band=(0.16, 0.34)):
    """One continuous tube per limb. Returns [(root body id, rings, uvs)]."""
    import mujoco
    out = []
    ang = np.linspace(0.0, 2.0 * math.pi, ring_n + 1)
    for side in ("L", "R"):
        for chain in (("humerus", "forearm", "manus"), ("femur", "tibia", "pes")):
            bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_{side}")
                    for s in chain]
            if any(b < 0 for b in bids):
                continue
            joints = [np.array(data.xpos[b], float) for b in bids]
            # the foot's far end: along the mean direction of its own digits
            d = None
            for g in range(model.ngeom):
                if (model.geom_bodyid[g] == bids[-1]
                        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                        and model.geom_size[g][0] < 0.0015):
                    ax = np.array(data.geom_xmat[g], float).reshape(3, 3)[:, 2]
                    d = ax if d is None else d + ax
            if d is None or np.linalg.norm(d) < 1e-9:
                d = joints[-1] - joints[-2]
            d = d / np.linalg.norm(d)
            if float(np.dot(d, joints[-1] - joints[-2])) < 0.0:
                d = -d
            end = joints[-1] + d * 0.005

            radii = []
            for b in bids:
                r = 0.004
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == b and model.geom_group[g] == 1:
                        r = max(r, float(model.geom_size[g][0]))
                radii.append(r * limb_scale)
            root_dir = joints[0] - joints[1]
            root_dir /= np.linalg.norm(root_dir)
            # Extra control points at the MIDDLE of each long segment, so the
            # profile can carry a muscle belly between two joint waists rather
            # than sliding monotonically from shoulder to toe.
            mid_upper = (joints[0] + joints[1]) * 0.5
            mid_lower = (joints[1] + joints[2]) * 0.5
            ctrl = [joints[0] + root_dir * LIMB_ROOT_SINK_M,
                    joints[0], mid_upper, joints[1], mid_lower, joints[2], end]
            r_ctrl = [radii[0] * 1.14,                      # where it meets the body
                      radii[0] * 1.06,                      # shoulder / hip
                      radii[0] * LIMB_BELLY,                # upper-arm belly
                      radii[1] * LIMB_JOINT_WAIST,          # ELBOW / KNEE, pinched
                      radii[1] * (LIMB_BELLY * 0.74),       # forearm / shank belly
                      radii[2] * LIMB_JOINT_WAIST * 0.80,   # WRIST / ANKLE, pinched
                      radii[2] * 0.78]                      # foot, a paddle not a stump

            # ONE pass, not two. Chaikin cuts corners, and the elbow and the
            # knee ARE corners -- the physics puts them there and two passes
            # rounded them into a single arc. One pass still kills the hard
            # kink without erasing the joint (#364).
            pts = _chaikin(ctrl, 1)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(arc[-1])
            cseg = np.linalg.norm(np.diff(np.array(ctrl), axis=0), axis=1)
            cf = np.concatenate([[0.0], np.cumsum(cseg)])
            cf = cf / cf[-1]

            rings, uvs = [], []
            n_prev = None
            frame = None
            for k in range(LIMB_STATIONS + 1):
                s = total * k / LIMB_STATIONS
                i = int(min(len(arc) - 2, max(0, np.searchsorted(arc, s, side="right") - 1)))
                u = (s - arc[i]) / max(1e-9, arc[i + 1] - arc[i])
                c = pts[i] + (pts[i + 1] - pts[i]) * u
                t = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
                t = t / max(1e-12, np.linalg.norm(t))
                if n_prev is None:
                    tmp = np.array([0.0, 0.0, 1.0])
                    if abs(float(np.dot(tmp, t))) > 0.95:
                        tmp = np.array([0.0, 1.0, 0.0])
                    n = np.cross(t, tmp)
                else:
                    n = n_prev - t * float(np.dot(n_prev, t))
                    if np.linalg.norm(n) < 1e-9:
                        n = np.cross(t, np.array([0.0, 0.0, 1.0]))
                n = n / np.linalg.norm(n)
                bn = np.cross(t, n)
                n_prev = n
                frame = (c, t, n, bn)
                r = float(np.interp(s / total, cf, r_ctrl))
                # Flattened cross-section: squash the world-vertical component
                # of each offset, so the limb is wider than it is deep however
                # the frame happens to be rolled. INVENTED -- see LIMB_FLATTEN.
                # The foot flattens harder than the rest of the limb: a gecko
                # stands on a flat sole, not on the end of a rod. INVENTED, and
                # the ramp is a shape choice like the rest of the profile.
                u_arc = s / total
                flat = LIMB_FLATTEN - 0.26 * max(0.0, (u_arc - 0.70) / 0.30)
                ring = []
                for a in ang:
                    off = r * (math.cos(a) * n + math.sin(a) * bn)
                    off = off - _UP * ((1.0 - flat) * float(np.dot(off, _UP)))
                    ring.append(c + off)
                rings.append(np.stack(ring))
                uvs.append(np.stack([0.15 + 0.20 * ang / (2.0 * math.pi),
                                     np.full(ring_n + 1, v_band[0]
                                             + (v_band[1] - v_band[0]) * s / total)],
                                    axis=1))
            # dome at the foot end
            c, t, n, bn = frame
            r_end = r_ctrl[-1]
            for sc, adv in ((0.92, 0.30), (0.70, 0.66), (0.28, 0.90), (0.05, 1.02)):
                cc = c + t * r_end * adv
                dome = []
                for a in ang:
                    off = r_end * sc * (math.cos(a) * n + math.sin(a) * bn)
                    off = off - _UP * ((1.0 - LIMB_FLATTEN * 0.62) * float(np.dot(off, _UP)))
                    dome.append(cc + off)
                rings.append(np.stack(dome))
                uvs.append(np.stack([0.15 + 0.20 * ang / (2.0 * math.pi),
                                     np.full(ring_n + 1, v_band[1])], axis=1))
            out.append((bids[0], rings, uvs))
    return out
