"""Generate the eyeball as a textured mesh: an iris and a vertical slit pupil.

WHY THE EYE NEEDED ITS OWN TOOL. The eye has been a plain black sphere since the
body was built, and #310 is the record of how much trouble that one geom has
already caused -- it spent most of the project drawing CREAM because a class
default beat its material, and the "two pale blobs on the head" that took three
wrong guesses to identify were these. Black was the fix. Black is still wrong: a
face is read almost entirely from its eyes, and a uniform dark sphere is the
single strongest signal that a model is a toy.

WHY IT IS A MESH AND NOT A TEXTURED SPHERE. #309 established that a `type="2d"`
texture does not map onto a curved PRIMITIVE, and a `cube` texture is addressed
by position rather than by texture coordinate, so neither can put a pupil in a
specific place on a sphere geom. A mesh carries its own UVs, so it can.

THE MAPPING IS A DISC, NOT A CYLINDER. The usual sphere UV layout runs the
texture's rows from pole to pole, which would smear the pupil across the whole
top edge of the image. Instead the vertices are laid out azimuthal-equidistant
about the eye's OUTWARD axis: the centre of the image lands on the point of the
eye that faces away from the skull, and the image is simply a picture of an eye,
pupil in the middle. The outward axis is DERIVED, not chosen -- it is the
direction from the head's own main ellipsoid to the eye geom's centre.

INVENTED HERE, all appearance and none of it able to move a gate. The eye geom
itself is untouched and still carries the radius the morphology audit measures;
this mesh is drawn at the same radius and the primitive is simply hidden.
  PUPIL_HALF_WIDTH 0.085  slit half-width as a fraction of the eye radius
  PUPIL_HALF_HEIGHT 0.62  slit half-height, same units
  IRIS_OUTER 0.86         where the iris ends and the dark rim begins
  colours                 read off photographs of adult E. macularius

Usage:  python tools/make_gecko_eye.py
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
MESH_OUT = REPO / "morphology" / "meshes"
TEX_OUT = REPO / "morphology" / "textures"
SRC = REPO / "morphology" / "gecko_body_r.xml"

#: Sphere resolution. An eye is 3.6 mm across and fills the frame in a close-up,
#: so it gets a finer mesh than the body does.
RINGS = 28
SEGMENTS = 40

#: The eye as the reference photograph shows it: a large, dark, glossy dome
#: with the pupil dilated -- the animal is nocturnal and the picture was not
#: taken in bright sun. The slit is still there, wide open. The ring at the
#: edge of the iris is now LIGHTER than the iris, not darker: a dark ring on a
#: dark eye read as a black hole with a black border (#343). INVENTED.
PUPIL_HALF_WIDTH = 0.24
PUPIL_HALF_HEIGHT = 0.66
IRIS_OUTER = 0.86

IRIS_PALE = np.array([74, 56, 44], dtype=float)
IRIS_DARK = np.array([26, 19, 16], dtype=float)
RIM = np.array([128, 112, 94], dtype=float)
PUPIL = np.array([8, 6, 8], dtype=float)


def eye_texture(size=512, seed=5):
    """A picture of an eye: slit pupil, veined iris, dark rim."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    u = (xx - size / 2.0) / (size / 2.0)
    v = (yy - size / 2.0) / (size / 2.0)
    r = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)

    # iris: radial veining, so the streaks run out from the pupil
    veins = np.zeros_like(r)
    for k in range(46):
        phase = rng.uniform(0, 2 * math.pi)
        width = rng.uniform(0.035, 0.11)
        strength = rng.uniform(0.45, 1.0)
        d = np.abs(((theta - phase + math.pi) % (2 * math.pi)) - math.pi)
        veins = np.maximum(veins, strength * np.exp(-(d / width) ** 2))
    veins = veins * np.clip((r - 0.12) / 0.30, 0.0, 1.0)
    veins = veins * np.clip((IRIS_OUTER - r) / 0.22, 0.0, 1.0)
    mottle = rng.normal(0.0, 0.08, (size, size))
    iris = np.clip(veins + 0.35 + mottle, 0.0, 1.0)[..., None]
    img = IRIS_PALE * (1.0 - iris) + IRIS_DARK * iris

    # darken toward the rim, then the rim itself
    shade = np.clip((r - 0.55) / 0.30, 0.0, 1.0)[..., None]
    img = img * (1.0 - 0.55 * shade) + RIM * (0.55 * shade)
    rim = np.clip((r - IRIS_OUTER) / 0.06, 0.0, 1.0)[..., None]
    img = img * (1.0 - rim) + RIM * rim

    # THE SLIT. Vertical, because this animal is nocturnal and its pupil closes
    # to a slit in light, the same optical solution a cat uses. The edges are
    # slightly wavy rather than straight.
    wobble = 1.0 + 0.22 * np.sin(v * 11.0) + 0.10 * np.sin(v * 23.0 + 1.2)
    ellipse = ((u / (PUPIL_HALF_WIDTH * wobble)) ** 2
               + (v / PUPIL_HALF_HEIGHT) ** 2)
    pupil = np.clip(1.6 - 1.6 * ellipse, 0.0, 1.0)[..., None]
    img = img * (1.0 - pupil) + PUPIL * pupil

    # outside the eyeball's visible disc, match the rim so any wrap is invisible
    outside = np.clip((r - 0.97) / 0.03, 0.0, 1.0)[..., None]
    img = img * (1.0 - outside) + RIM * outside
    return np.clip(img, 0, 255).astype(np.uint8)


def sphere(centre, radius, outward, flip_u):
    """A UV sphere whose texture disc is centred on `outward`.

    Returns (vertices, uvs, normals, faces) in the frame `centre` is given in.
    """
    outward = np.asarray(outward, float)
    outward = outward / np.linalg.norm(outward)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, outward))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(outward, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(outward, e1)

    verts, uvs = [], []
    for i in range(RINGS + 1):
        polar = math.pi * i / RINGS          # 0 at the outward pole
        for j in range(SEGMENTS + 1):
            az = 2.0 * math.pi * j / SEGMENTS
            d = (outward * math.cos(polar)
                 + (e1 * math.cos(az) + e2 * math.sin(az)) * math.sin(polar))
            verts.append(centre + d * radius)
            # azimuthal-equidistant: image centre on the pole
            rr = polar / math.pi
            su = -1.0 if flip_u else 1.0
            uvs.append([0.5 + su * rr * math.cos(az), 0.5 + rr * math.sin(az)])
    per = SEGMENTS + 1
    faces = []
    for i in range(RINGS):
        for j in range(SEGMENTS):
            a, b = i * per + j, (i + 1) * per + j
            faces.append((a, b, b + 1))
            faces.append((a, b + 1, a + 1))
    verts = np.array(verts, float)
    normals = verts - centre
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return verts, np.array(uvs, float), normals, faces


#: How far round from its axis the eyelid hood reaches, and how far it stands
#: off the eyeball. INVENTED; both are appearance only. 64 degrees was the first
#: try and it swallowed the eye whole: the eye's own outward axis is only about
#: 68 degrees from the hood's, so a 64 degree cap reaches the pupil. 33 leaves
#: a brow that overhangs the top of the eye, which is what it should be.
LID_ARC_DEG = 48.0
LID_GAP = 0.0008


#: The pale rim of skin around the eye opening: standoff from the eyeball and
#: tube radius, in metres. INVENTED (photographs -- the rim is the single most
#: recognisable thing about this animal's face after the spots).
RIM_STANDOFF = 0.00025
RIM_TUBE = 0.00048


def rim_ring(centre, radius, outward):
    """A torus around the eye at the skin surface: the pale eyelid margin."""
    outward = np.asarray(outward, float)
    outward /= np.linalg.norm(outward)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, outward))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(outward, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(outward, e1)
    big = radius + RIM_STANDOFF
    # the ring sits where the eyeball meets the skin: a little inboard of the
    # eye's equator, so it hugs the socket rather than floating at the equator
    plane = centre - outward * radius * 0.28
    segs, tube = 40, 12
    verts, uvs = [], []
    for i in range(segs + 1):
        a = 2.0 * math.pi * i / segs
        radial = e1 * math.cos(a) + e2 * math.sin(a)
        ring_c = plane + radial * big * 0.96
        for j in range(tube + 1):
            b = 2.0 * math.pi * j / tube
            p = ring_c + (radial * math.cos(b) + outward * math.sin(b)) * RIM_TUBE
            verts.append(p)
            uvs.append([i / segs, j / tube])
    per = tube + 1
    faces = []
    for i in range(segs):
        for j in range(tube):
            a0, b0 = i * per + j, (i + 1) * per + j
            faces.append((a0, b0, b0 + 1))
            faces.append((a0, b0 + 1, a0 + 1))
    verts = np.array(verts, float)
    nrm = np.zeros_like(verts)
    for a, b, c in faces:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        nrm[a] += n; nrm[b] += n; nrm[c] += n
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln < 1e-12, 1.0, ln)
    return verts, np.array(uvs, float), nrm, faces


def lid_cap(centre, radius, outward):
    """A spherical hood over the top of the eyeball.

    THE EYELID IS THE POINT OF THIS ANIMAL'S FAMILY. Eublepharidae means "true
    eyelid": unlike the geckos most people picture, which have a fixed
    transparent spectacle and lick it clean, this one has a movable lid and
    blinks. The body already has the joint for it. What it did not have was a
    lid that looked like one -- two flat discs sitting above the eyes, sized by
    hand twice and still reading as a cap rather than a lid.

    Generated instead as a shell concentric with the eyeball, so it fits by
    construction. Its axis leans from straight up toward the eye's own outward
    direction, because a lid closes over the eye rather than over the skull.
    """
    # The hood sits on the UPPER-OUTER face of the eyeball, not on its crown:
    # the sculpted skull's brow is taller than the old swept one, and a hood
    # on the crown was buried inside the head skin and never rendered (#343).
    # Leaning the axis 0.9 toward outward puts it where the eye stands proud
    # of the skin, and the roll hinge then sweeps it down over the pupil.
    axis = np.array([0.0, 0.0, 1.0]) + 0.9 * np.asarray(outward, float)
    axis /= np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(tmp, axis))) > 0.95:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    # A THICK HOOD, NOT A FILM (#343). Rendered as a single surface the lid was
    # a paper-thin shell that showed its unlit back face as a dark sliver and
    # vanished against the eye. The verified anatomy (#337) says the lids are
    # THICK fleshy folds. So: an outer cap, an inner cap LID_THICK inside it,
    # and a rim joining them -- a closed solid with volume. Textured from a
    # yellow stretch of the body image (mid-neck, dorsal band).
    LID_THICK = 0.0010
    r_in = radius + LID_GAP
    r_out = r_in + LID_THICK
    arc = math.radians(LID_ARC_DEG)
    rings, segs = 8, 32
    per = segs + 1
    verts, uvs = [], []

    def shell(r_shell):
        base = len(verts)
        for i in range(rings + 1):
            polar = arc * i / rings
            for j in range(segs + 1):
                az = 2.0 * math.pi * j / segs
                dvec = (axis * math.cos(polar)
                        + (e1 * math.cos(az) + e2 * math.sin(az)) * math.sin(polar))
                verts.append(centre + dvec * r_shell)
                # MESH texture rows run the OTHER way from the skin's (#343):
                # measured -- the image is yellow at (0.22, 0.10) and the lid
                # rendered white, the colour at (0.22, 0.90). A mesh geom's v
                # counts from the bottom of the image; the skin's counts from
                # the top. So the lid asks for 1 - v.
                uvs.append([0.22 + 0.05 * (i / rings) * math.cos(az),
                            0.90 - 0.012 * (i / rings) * math.sin(az)])
        return base

    outer = shell(r_out)
    inner = shell(r_in)
    faces = []
    for i in range(rings):
        for j in range(segs):
            a, b = outer + i * per + j, outer + (i + 1) * per + j
            faces.append((a, b + 1, b)); faces.append((a, a + 1, b + 1))
            a, b = inner + i * per + j, inner + (i + 1) * per + j
            faces.append((a, b, b + 1)); faces.append((a, b + 1, a + 1))
    # the rim: join the outer and inner last rings
    ro, ri = outer + rings * per, inner + rings * per
    for j in range(segs):
        faces.append((ro + j, ro + j + 1, ri + j + 1))
        faces.append((ro + j, ri + j + 1, ri + j))
    verts = np.array(verts, float)
    nrm = np.zeros_like(verts)
    for a, b, c in faces:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        nrm[a] += n; nrm[b] += n; nrm[c] += n
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(ln < 1e-12, 1.0, ln)
    return verts, np.array(uvs, float), nrm, faces


def write_obj(path, verts, uvs, nrms, faces, note):
    lines = [f"# generated by tools/make_gecko_eye.py -- {note}"]
    lines += [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in verts]
    lines += [f"vt {a:.5f} {b:.5f}" for a, b in uvs]
    lines += [f"vn {a:.5f} {b:.5f} {c:.5f}" for a, b, c in nrms]
    for a, b, c in faces:
        lines.append(f"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import mujoco
    from PIL import Image

    # compile without meshes or the skin, the same way the other generators do
    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"[ \t]*<deformable>.*?</deformable>\n?", "", text, flags=re.S)
    # Strip mesh geoms as ELEMENTS. Filtering by line drops whatever else
    # happens to share the line, which is exactly how the eye spheres
    # disappeared and this generator reported "found 0" (#330).
    text = re.sub(r'<geom[^>]*type="mesh"[^>]*/>', '', text)
    kept = [ln for ln in text.splitlines() if not re.match(r"\s*<mesh ", ln)]
    tmp = SRC.with_name(SRC.stem + "_eyesrc_tmp.xml")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    eye_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "eye")
    eyes = [g for g in range(model.ngeom)
            if model.geom_bodyid[g] == head and int(model.geom_matid[g]) == eye_mat]
    if len(eyes) != 2:
        raise SystemExit(f"expected 2 eye geoms, found {len(eyes)}")

    # the head's largest ellipsoid is the cranium; the outward axis of an eye is
    # the direction from that ellipsoid's centre to the eye. DERIVED.
    crania = [(float(np.prod(model.geom_size[g])), g) for g in range(model.ngeom)
              if model.geom_bodyid[g] == head
              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_ELLIPSOID]
    cranium = np.array(model.geom_pos[max(crania)[1]], float)

    MESH_OUT.mkdir(parents=True, exist_ok=True)
    TEX_OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(eye_texture()).save(TEX_OUT / "gecko_eye.png")
    print(f"texture: gecko_eye.png 512x512, slit pupil")

    for g in eyes:
        pos = np.array(model.geom_pos[g], float)
        radius = float(model.geom_size[g][0])
        # WHERE THE PUPIL POINTS. The first version used the direction from the
        # cranium's centre to the eye, which is (0.295, 0.793, 0.533): 53
        # degrees above horizontal, because the eye geom sits high on a skull
        # that is flatter than it is wide. The pupil then faced the sky and from
        # any normal viewing angle only the pale outer rim of the iris was
        # visible. The surface NORMAL of the cranium ellipsoid at that point is
        # worse still, at (0.165, 0.703, 0.692), for the same reason: an
        # ellipsoid's normal tilts toward its short axis.
        #
        # Neither is what the animal does. A gecko's eye sits high on the skull
        # and still looks SIDEWAYS -- the pupil is on the lateral face of the
        # eyeball, not on its top. So the lateral and forward components are
        # taken from the ellipsoid normal, which is geometry, and the vertical
        # one is set to a small fixed rise. INVENTED, and declared: 0.18.
        rel = pos - cranium
        size = np.array(model.geom_size[max(crania)[1]], float)
        normal = rel / (size ** 2)
        outward = np.array([normal[0], normal[1], 0.0], float)
        outward /= np.linalg.norm(outward)
        outward[2] = 0.18
        outward /= np.linalg.norm(outward)
        # SEAT IT PROUD OF THE SKIN. The eye geom's centre sits far enough inside
        # the head that the skin cuts across the eyeball, and the skin's rings
        # are 2 mm apart against a 7.2 mm eye, so that cut is a visible polygon
        # rather than a curve -- the eye read as a crater with a stepped rim. The
        # VISUAL sphere is pushed out along its own axis so most of it is clear
        # of the skin, which is also what a real eye does. The eye GEOM does not
        # move: it still carries the position and radius the inner-eye-gap gate
        # measures. Offset INVENTED, 1.1 mm.
        pos = pos + outward * 0.0003
        side = "L" if pos[1] > 0 else "R"
        # mirror the image on the right eye so the veining is not a copy
        v, uv, nrm, faces = sphere(pos, radius, outward, flip_u=(side == "R"))
        write_obj(MESH_OUT / f"eye_{side}.obj", v, uv, nrm, faces, "eyeball")
        print(f"  eye_{side}.obj  {len(v)} verts  {len(faces)} tris  "
              f"radius {radius * 1000:.2f} mm  outward {outward.round(3)}")

        # the lid, in the EYELIDS body's own frame so the blink joint moves it
        # one lid body per side now (#343): each hinges about the head's long
        # axis so it closes DOWN over its own eye. Falls back to the old single
        # `eyelids` body if the source has not been split yet.
        lids = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"eyelid_{side}")
        if lids < 0:
            lids = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "eyelids")
        if lids >= 0:
            origin = np.array(model.body_pos[lids], float)
            lv, luv, lnrm, lfaces = lid_cap(pos - origin, radius, outward)
            write_obj(MESH_OUT / f"lid_{side}.obj", lv, luv, lnrm, lfaces, "eyelid")
            print(f"  lid_{side}.obj  {len(lv)} verts  {len(lfaces)} tris")

        # the pale rim around the eye opening, fixed to the head
        rv, ruv, rnrm, rfaces = rim_ring(pos, radius, outward)
        write_obj(MESH_OUT / f"rim_{side}.obj", rv, ruv, rnrm, rfaces, "eye rim")
        print(f"  rim_{side}.obj  {len(rv)} verts  {len(rfaces)} tris")


if __name__ == "__main__":
    main()
