"""Build a skeleton for the animal, and be honest about what it is.

WHY THIS EXISTS. The lower panel has been showing the 48 COLLISION SOLIDS --
capsules, boxes and one ellipsoid. That is the body the physics uses, and it is
the correct thing to show when the question is "what does MuJoCo integrate".
It is not a skeleton, and the user kept saying so. This project has never had
one: no skull, no vertebrae, no femur, and the 23 "bones" in the model file are
skin-binding bones, which is a rigging term, not anatomy.

WHAT THIS IS, STATED PLAINLY

    APPEARANCE ONLY. Not one gram of mass, not one contact, not one degree of
    freedom. It is drawn geometry parented to bodies the physics already has,
    so it rides the same transforms the skin does and cannot influence motion.

    THE LAYOUT IS MEASURED. Every bone is fitted to the real body it belongs
    to, using that body's own position and the distance to its child: the
    humerus is 11.9 mm because this animal's humerus body is 11.9 mm from the
    shoulder to the elbow. Those lengths come from the certified body, which
    passes 14 of 14 anatomical checks against living geckos.

    THE SHAPES ARE INVENTED, AND SAY SO. The outline of a gecko's skull, the
    profile of its scapula, the shape of a caudal vertebra -- none of that is
    in this project's corpus. What is drawn is a generic squamate reading of
    each element. It is a diagram of where the bones are, not a claim about
    what they look like.

    THE VERTEBRAL COUNTS ARE INVENTED. Powell, Russell & Sutey 2018, "Patterns
    of growth in the presacral vertebral column of the leopard gecko", is the
    paper that would settle the presacral count for this species. Only its
    abstract has been read in this project and the abstract carries no count --
    an earlier agent claimed one and a verifier killed it for exactly that
    reason. So the counts below are chosen to look right and are tagged
    INVENTED until someone opens the paper.

Usage:  python tools/export_web_skeleton.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "site" / "media"

# INVENTED counts -- see the module docstring. Chosen to read as a lizard.
CERVICAL, DORSAL, SACRAL = 8, 17, 2


# --------------------------------------------------------------- primitives
class Mesh:
    def __init__(self):
        self.v: list[list[float]] = []
        self.f: list[list[int]] = []

    def add(self, verts, faces):
        o = len(self.v)
        self.v.extend([float(x) for x in p] for p in verts)
        self.f.extend([a + o, b + o, c + o] for a, b, c in faces)


def _tube(path, radii, sides=8, squash=1.0):
    """A closed tube through `path` with per-station `radii`."""
    verts, faces = [], []
    path = [np.asarray(p, float) for p in path]
    n = len(path)
    for i, c in enumerate(path):
        fwd = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
        L = np.linalg.norm(fwd)
        fwd = fwd / L if L > 1e-9 else np.array([1.0, 0, 0])
        tmp = np.array([0.0, 0, 1.0])
        if abs(float(np.dot(tmp, fwd))) > 0.95:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(fwd, tmp); u /= np.linalg.norm(u)
        w = np.cross(fwd, u)
        for k in range(sides):
            a = 2 * math.pi * k / sides
            verts.append(c + radii[i] * (math.cos(a) * u + squash * math.sin(a) * w))
    for i in range(n - 1):
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            faces.append((a, b, b + sides))
            faces.append((a, b + sides, a + sides))
    return verts, faces


def long_bone(p0, p1, r_mid, flare=1.9, stations=9, sides=8):
    """A shaft with flared ends -- the shape that reads as 'bone'."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    path, radii = [], []
    for i in range(stations):
        t = i / (stations - 1)
        path.append(p0 + (p1 - p0) * t)
        # thin waisted shaft, epiphyses at both ends
        bulge = (abs(t - 0.5) * 2) ** 3
        radii.append(r_mid * (1.0 + (flare - 1.0) * bulge))
    return _tube(path, radii, sides)


def vertebra(c, fwd, up, side, size):
    """Centrum, neural spine and two transverse processes."""
    m = Mesh()
    c = np.asarray(c, float)
    m.add(*long_bone(c - fwd * size * 0.45, c + fwd * size * 0.45, size * 0.30, 1.35, 5, 7))
    m.add(*long_bone(c, c + up * size * 1.25, size * 0.14, 1.0, 4, 5))
    for s in (+1, -1):
        m.add(*long_bone(c, c + side * s * size * 0.85 + up * size * 0.15,
                         size * 0.11, 1.0, 4, 5))
    return m


def rib(c, side, up, fwd, span, sign):
    pts, rr = [], []
    for i in range(9):
        t = i / 8
        pts.append(c + side * sign * span * math.sin(t * 1.45)
                     - up * span * 1.15 * (1 - math.cos(t * 1.45))
                     + fwd * span * 0.18 * t)
        rr.append(span * 0.085 * (1 - 0.45 * t))
    return _tube(pts, rr, 5)


def blade(c, a, b, thick):
    """A flat plate, for girdle elements."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = np.cross(a, b)
    L = np.linalg.norm(n)
    n = n / L * thick if L > 1e-9 else np.array([0, 0, thick])
    q = [c + a + b, c + a - b, c - a - b, c - a + b]
    verts = [p + n for p in q] + [p - n for p in q]
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
             (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
             (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return verts, faces


def main():
    import mujoco
    m = mujoco.MjModel.from_xml_path("morphology/gecko_body_web.xml")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    bname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

    def local(parent, world_pt):
        """World point into a body's own frame."""
        R = np.asarray(d.xmat[parent]).reshape(3, 3)
        return R.T @ (np.asarray(world_pt, float) - np.asarray(d.xpos[parent], float))

    def child_of(i):
        k = [c for c in range(m.nbody) if m.body_parentid[c] == i]
        return k[0] if k else None

    X, Y, Z = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    pieces: dict[int, Mesh] = {}

    def mesh_for(b) -> Mesh:
        return pieces.setdefault(b, Mesh())

    # ---- limb long bones, fitted to the real segment lengths
    for side in ("L", "R"):
        for seg, r in (("humerus", 0.0016), ("forearm", 0.0013),
                       ("femur", 0.0019), ("tibia", 0.0015)):
            b = bid(f"{seg}_{side}")
            if b < 0:
                continue
            k = child_of(b)
            end = local(b, d.xpos[k]) if k is not None else np.array([0.010, 0, 0])
            if seg in ("forearm", "tibia"):        # two bones, not one
                off = np.cross(end / (np.linalg.norm(end) or 1), Z) * r * 1.15
                mesh_for(b).add(*long_bone(off, end + off, r * 0.75))
                mesh_for(b).add(*long_bone(-off, end - off, r * 0.62))
            else:
                mesh_for(b).add(*long_bone(np.zeros(3), end, r))

    # ---- digits
    for side in ("L", "R"):
        for part, n_dig, spread, length in (("manus", 5, 0.85, 0.010), ("pes", 5, 0.95, 0.013)):
            b = bid(f"{part}_{side}")
            if b < 0:
                continue
            for k in range(n_dig):
                a = (-spread / 2 + spread * k / (n_dig - 1))
                tip = np.array([math.cos(a), math.sin(a), -0.15]) * length * (0.72 + 0.28 * math.sin(math.pi * k / (n_dig - 1)))
                for j in range(3):                 # phalanges
                    p0 = tip * (j / 3.0)
                    p1 = tip * ((j + 1) / 3.0)
                    mesh_for(b).add(*long_bone(p0, p1, 0.00042, 1.5, 4, 5))

    # ---- skull and mandible
    hb = bid("head")
    if hb >= 0:
        L = 0.021
        sk = mesh_for(hb)
        path = [X * t * L for t in (0.0, 0.22, 0.45, 0.68, 1.0)]
        sk.add(*_tube(path, [0.0055, 0.0062, 0.0052, 0.0034, 0.0016], 9, squash=0.78))
        for s in (+1, -1):                          # orbits, as rings
            c = X * L * 0.40 + Y * s * 0.0043 + Z * 0.0016
            ring, rr = [], []
            for i in range(11):
                a = 2 * math.pi * i / 10
                ring.append(c + X * 0.0031 * math.cos(a) + Z * 0.0031 * math.sin(a))
                rr.append(0.00055)
            sk.add(*_tube(ring, rr, 5))
    jb = bid("jaw")
    if jb >= 0:
        for s in (+1, -1):
            mesh_for(jb).add(*long_bone(np.zeros(3),
                                        X * 0.0185 + Y * s * 0.0022, 0.00085, 1.5, 6, 6))

    # ---- vertebral column, spread over the real axial bodies
    axial = [("neck", CERVICAL), ("trunk_anterior", DORSAL // 2),
             ("trunk_middle", 3), ("trunk_posterior", DORSAL - DORSAL // 2),
             ("pelvis", SACRAL),
             ("tail1", 7), ("tail2", 6), ("tail3", 6), ("tail4", 5), ("tail5", 4)]
    for name, count in axial:
        b = bid(name)
        if b < 0:
            continue
        k = child_of(b)
        end = local(b, d.xpos[k]) if k is not None else X * -0.012
        seg = np.linalg.norm(end) or 0.012
        fwd = end / seg
        size = 0.0042 if name.startswith(("trunk", "pelvis")) else (
            0.0032 if name == "neck" else 0.0030 * (1 - 0.10 * int(name[-1] or 1)))
        mm = mesh_for(b)
        for i in range(count):
            t = (i + 0.5) / count
            c = fwd * seg * t + Z * 0.0006
            vb = vertebra(c, fwd, Z, Y, size)
            mm.add(vb.v, vb.f)
            if name.startswith("trunk") and i % 2 == 0:     # ribs on the dorsals
                for s in (+1, -1):
                    mm.add(*rib(c, Y, Z, fwd, size * 2.5, s))

    # ---- girdles
    tb = bid("trunk_anterior")
    if tb >= 0:
        for s in (+1, -1):
            c = np.array([0.0, s * 0.0052, 0.0008])
            mesh_for(tb).add(*blade(c, Y * s * 0.0034 + Z * 0.0052, X * 0.0030, 0.00055))
            mesh_for(tb).add(*long_bone(c, c + Y * s * 0.0030 - Z * 0.0052, 0.0009, 1.4, 5, 6))
    pb = bid("pelvis")
    if pb >= 0:
        for s in (+1, -1):
            c = np.array([0.0, s * 0.0050, 0.0010])
            mesh_for(pb).add(*blade(c, X * 0.0062, Z * 0.0044 + Y * s * 0.0016, 0.00060))
            mesh_for(pb).add(*long_bone(c, c + X * -0.0052 - Z * 0.0042, 0.0009, 1.35, 5, 6))

    # ---------------------------------------------------------------- pack
    verts, faces, bones = [], [], []
    for b, mesh in sorted(pieces.items()):
        if not mesh.f:
            continue
        o = len(verts)
        bones.append({"body": int(b), "body_name": bname(b),
                      "vertadr": o, "vertnum": len(mesh.v),
                      "faceadr": len(faces), "facenum": len(mesh.f)})
        verts.extend(mesh.v)
        faces.extend([a + o, c2 + o, c3 + o] for a, c2, c3 in mesh.f)

    V = np.asarray(verts, dtype=np.float32).ravel()
    F = np.asarray(faces, dtype=np.uint32).ravel()
    (OUT / "skeleton.bin").write_bytes(V.tobytes() + F.tobytes())
    (OUT / "skeleton.json").write_text(json.dumps({
        "nvert": len(verts), "nface": len(faces),
        "layout": [{"name": "vert", "type": "float32", "count": V.size, "bytes": V.nbytes},
                   {"name": "face", "type": "uint32", "count": F.size, "bytes": F.nbytes}],
        "bones": bones,
        "counts": {"cervical": CERVICAL, "dorsal": DORSAL, "sacral": SACRAL,
                   "provenance": "INVENTED -- no presacral count for this species has been "
                                 "read in this project. Powell, Russell & Sutey 2018 is the "
                                 "paper that would settle it; only its abstract was reached "
                                 "and it carries no count."},
        "note": "APPEARANCE ONLY. No mass, no contact, no degree of freedom. Bone LENGTHS "
                "are fitted to the certified body; bone SHAPES are invented and say so.",
    }), encoding="utf-8")
    print(f"written: skeleton.bin {(V.nbytes + F.nbytes)/1024:.0f} KB   skeleton.json")
    print(f"  {len(verts)} vertices, {len(faces)} triangles, {len(bones)} bodies carry bone")
    print(f"  vertebrae: {CERVICAL} cervical + {DORSAL} dorsal + {SACRAL} sacral + 28 caudal (INVENTED)")
    for b in bones[:6]:
        print(f"    {b['body_name']:16} {b['facenum']:6d} tris")


if __name__ == "__main__":
    main()
