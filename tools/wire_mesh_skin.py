"""Wire the generated skin meshes into the source body, idempotently.

This is a patch script rather than hand-editing because the same mistakes kept
recurring by hand, and each one cost a full rebuild to notice:

  1. A mesh geom that inherits `class="visual"` also inherits that class's
     rgba="0.78 0.72 0.45 1", and MuJoCo MULTIPLIES a geom's rgba into its
     material's texture. Skin texture (0.84, 0.73, 0.42) times that rgba is
     (0.65, 0.52, 0.19) -- dark olive. Every mesh geom therefore needs an
     explicit rgba="1 1 1 1" to show the texture at full strength. Same root
     cause as #310, different symptom.
  2. A mesh's texture coordinates are read ONLY from a type="2d" texture. #309
     established that type="2d" does not map onto curved PRIMITIVES, so every
     texture here had been declared type="cube" -- and a cube texture is
     addressed by position, so MuJoCo ignored the UVs the generator writes and
     drew the skin plain grey. Both kinds are now declared from the same PNG.
  3. The primitive face details -- lip lines, brows, snout bridge, snout tip --
     were positioned against a FLAT-SHADED head. Once the head is a mesh they
     poke through it as cream boxes. They stay in the file because the
     morphology audit measures them; they are drawn transparent.
  4. The tail has its own banded texture, generated but never referenced by any
     material, so the tail was drawn with the spotted body texture.
  5. The lower jaw was a 2.9 mm collision rod drawn as itself. It now gets the
     generated jaw mesh, which is swept at the head's own width.

Nothing here changes a body, joint, actuator, mass or contact: it edits colours,
materials and the size of four visual-only geoms. Run it after
tools/make_gecko_mesh.py, then rebuild the lab bodies.

Usage:  python tools/wire_mesh_skin.py
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "morphology" / "gecko_body_r.xml"

#: Face primitives that the head mesh now supersedes. Kept in the file (the
#: morphology audit measures them) and drawn transparent.
SUPERSEDED = ("L brow", "R brow", "snout bridge", "rounded snout tip",
              "L lip line", "R lip line")

#: The 23 rigid per-body mesh geoms are superseded by the continuous skin from
#: tools/make_gecko_skin.py. Set False to draw them again and hide the skin --
#: that is the fallback if linear blend skinning ever misbehaves.
RIGID_MESHES_SUPERSEDED = True

CUBE_TAIL = '    <texture name="gecko_tail" type="cube" file="gecko_tail.png"/>'
UV_TEXTURES = (
    CUBE_TAIL + "\n"
    '    <texture name="gecko_skin_uv" type="2d" file="gecko_skin.png"/>'
    "  <!-- the same image, addressed by the mesh's own UVs: a cube texture is\n"
    "         addressed by POSITION and ignores texcoord entirely -->\n"
    '    <texture name="gecko_tail_uv" type="2d" file="gecko_tail.png"/>'
)


def patch(text: str) -> str:
    # ---- 1. every mesh geom shows its texture at full strength -----------
    def full_bright(m):
        line = m.group(0)
        if "skin_teeth" in line:
            # THE TOOTH RIM CANNOT SURVIVE THE SKIN. It is bolted rigidly to the
            # jaw body, while the skin around it is blended between the jaw and
            # the head, so the two surfaces no longer coincide and the rim juts
            # out of the cheek as a pale wing. Colour-coded to identify it after
            # two wrong guesses. Drawn transparent; the mouth reads from the
            # skin's own jaw seam instead.
            return re.sub(r'rgba="[^"]*"', 'rgba="0 0 0 0"', line)
        if "skin_rim_" in line:
            # the eyelid margin is PALE, not white: untextured, so rgba is its
            # colour, and 1 1 1 1 made it glow
            return re.sub(r'rgba="[^"]*"', 'rgba="0.90 0.86 0.78 1"', line)
        if any(k in line for k in ("skin_eye_", "skin_lid_")):
            # The eyeball and the eyelid are NOT superseded by the skin: the
            # skin deliberately excludes them (an eye is not skin, and wrapping
            # it would put a bulge where the pupil is). They must stay lit, and
            # the blanket rule below would hide them -- which it did once, and
            # the animal came out with no eyes at all.
            return re.sub(r'rgba="[^"]*"', 'rgba="1 1 1 1"', line)
        if RIGID_MESHES_SUPERSEDED:
            # the continuous skin draws all of this now; the rigid pieces stay
            # in the file as a one-constant fallback and are drawn transparent
            if "rgba=" in line:
                return re.sub(r'rgba="[^"]*"', 'rgba="0 0 0 0"', line)
            return line.replace('type="mesh"', 'rgba="0 0 0 0" type="mesh"')
        if "rgba=" in line:
            return re.sub(r'rgba="[^"]*"', 'rgba="1 1 1 1"', line)
        return line.replace('type="mesh"', 'rgba="1 1 1 1" type="mesh"')

    text = re.sub(r'<geom[^>]*type="mesh"[^>]*/>', full_bright, text)

    # ---- 2. the tail is banded, not spotted ------------------------------
    if 'name="tailskin"' not in text:
        text = text.replace(
            '    <material name="spot"',
            '    <material name="tailskin" texture="gecko_tail" texuniform="false"'
            ' rgba="1 1 1 1" specular="0.25" shininess="0.12"/>'
            '  <!-- the tail is RINGED, not spotted: gecko_tail.png -->\n'
            '    <material name="spot"', 1)
    text = re.sub(r'(mesh="skin_tail\d")\s+material="skin"',
                  lambda m: m.group(1) + ' material="tailskin"', text)

    # ---- 3. face primitives the head mesh supersedes ---------------------
    for tag in SUPERSEDED:
        pat = re.compile(r'(<geom(?:(?!/>).)*?/>)(\s*<!--\s*'
                         + re.escape(tag) + r'\s*-->)')

        def hide(m):
            g = m.group(1)
            g = (re.sub(r'rgba="[^"]*"', 'rgba="0 0 0 0"', g) if "rgba=" in g
                 else g.replace("<geom", '<geom rgba="0 0 0 0"', 1))
            return g + m.group(2)
        text = pat.sub(hide, text)

    # ---- 4. the jaw ------------------------------------------------------
    if 'name="skin_jaw"' not in text:
        text = text.replace('    <mesh name="skin_manus_L"',
                            '    <mesh name="skin_jaw" file="jaw.obj"/>\n'
                            '    <mesh name="skin_manus_L"', 1)
    text = text.replace(
        '              <geom class="visual" type="capsule"'
        ' fromto="0.0005 0 0 0.0245 0 0" size="0.0029" material="jaw"/>',
        '              <geom rgba="1 1 1 1" class="visual" type="mesh"'
        ' mesh="skin_jaw" material="jaw"/>'
        "  <!-- generated mandible, swept at the head's own width -->")

    # ---- 5. mesh UVs need type="2d" textures -----------------------------
    if 'name="gecko_skin_uv"' not in text:
        text = text.replace(CUBE_TAIL, UV_TEXTURES, 1)
    for mat, tex in (("skin", "gecko_skin_uv"), ("jaw", "gecko_skin_uv"),
                     ("tailskin", "gecko_tail_uv")):
        text = re.sub(r'(<material name="' + mat + r'"\s+)texture="[^"]*"',
                      lambda m, t=tex: m.group(1) + 'texture="' + t + '"', text)

    # ---- 6. the tooth line follows the mandible, so it is a mesh too -----
    if 'name="skin_teeth"' not in text:
        text = text.replace('    <mesh name="skin_tibia_L"',
                            '    <mesh name="skin_teeth" file="teeth.obj"/>\n'
                            '    <mesh name="skin_tibia_L"', 1)
    text = re.sub(
        r'<geom class="visual" type="box" pos="0\.0\d+ 0 0?\.?0*\d*"'
        r' size="0\.0\d+ 0\.00\d+ 0\.000\d+"',
        '<geom class="visual" type="mesh" mesh="skin_teeth"',
        text)

    # ---- 6a. the continuous skin's material, and retiring the rigid pieces -
    # `tools/make_gecko_skin.py` binds ONE surface to all the bodies, so the 23
    # rigid per-body mesh geoms are superseded: they are kept (they cost no mass
    # and make a one-line fallback if the skin ever misbehaves) and drawn
    # transparent. The skin's image is not tiled -- v runs nose to tail once --
    # so it gets its own texture rather than sharing the tiling one.
    if 'name="gecko_body_uv"' not in text:
        text = text.replace(
            '    <texture name="gecko_tail_uv" type="2d" file="gecko_tail.png"/>',
            '    <texture name="gecko_tail_uv" type="2d" file="gecko_tail.png"/>\n'
            '    <texture name="gecko_body_uv" type="2d" file="gecko_body.png"/>'
            '  <!-- whole animal, nose to tail, NOT tiled -->\n'
            '    <material name="body" texture="gecko_body_uv" texuniform="false"'
            ' rgba="1 1 1 1" specular="0.25" shininess="0.12"/>', 1)
    # ---- 6a2. the cream digit fans are superseded by real toes (#341) ------
    # The skin now grows five tapered toes per foot along these very capsules,
    # so the capsules themselves are drawn transparent. They stay: the toes are
    # built FROM them, and the audit measures them.
    text = re.sub(r'(<geom class="visual" type="capsule"[^>]*?)rgba="[^"]*"'
                  r'([^>]*/>\s*<!--\s*digit \d \(visual\)\s*-->)',
                  lambda m: m.group(1) + 'rgba="0 0 0 0"' + m.group(2), text)

    # ---- 6b. measurement markers are for measuring, not for looking at ---
    # Two translucent red slabs under the belly and two red pips at the vent and
    # tail tip. They exist so the audit can find those landmarks; group 4 is the
    # group this body already uses for landmarks that are not meant to be seen.
    for site in ("belly_mid", "belly_post", "vent", "tail_tip"):
        text = re.sub(r'(<site name="' + site + r'"(?![^>]*group=)[^>]*)/>',
                      lambda m: m.group(1) + ' group="4"/>', text)

    # ---- 7. the throat is now a BONE of the skin, not a lump under it ----
    # `make_gecko_skin.py` binds the gular body into the skin, so the breathing
    # movement shows as the skin itself swelling. The separate pale ellipsoid
    # that used to stand in for it is drawn transparent: it was poking out
    # through the skin at the mouth corner.
    text = re.sub(r'(<geom class="visual" type="ellipsoid" pos="0\.00[0-9]+ 0 [-0-9.]+"'
                  r' size="0\.009[0-9]+ 0\.005[0-9]+ 0\.001[0-9]+")'
                  r' rgba="[^"]*"',
                  lambda m: m.group(1) + ' rgba="0 0 0 0"', text)

    # ---- 7b. the eyeball gets an iris and a slit pupil --------------------
    # A plain dark sphere is the strongest single signal that a model is a toy,
    # and a sphere GEOM cannot carry a pupil: #309 found `type="2d"` does not
    # map onto a curved primitive, and a `cube` texture is addressed by position
    # rather than by texture coordinate, so neither can put a mark in a chosen
    # place. `tools/make_gecko_eye.py` generates the eyeball as a mesh with its
    # own UVs instead. The sphere geoms STAY, drawn transparent: the morphology
    # audit measures the inner-eye gap from them, and the eye generator reads
    # its position and radius from them.
    if 'name="gecko_eye"' not in text:
        text = text.replace(
            '    <texture name="gecko_body_uv" type="2d" file="gecko_body.png"/>',
            '    <texture name="gecko_eye" type="2d" file="gecko_eye.png"/>\n'
            '    <material name="eyeball" texture="gecko_eye" texuniform="false"'
            ' rgba="1 1 1 1" specular="0.55" shininess="0.62" reflectance="0.05"/>'
            '  <!-- wet, unlike the skin: an eye is the one part that should'
            ' catch a highlight -->\n'
            '    <texture name="gecko_body_uv" type="2d" file="gecko_body.png"/>', 1)
    if 'name="skin_eye_L"' not in text:
        text = text.replace('    <mesh name="skin_femur_L"',
                            '    <mesh name="skin_eye_L" file="eye_L.obj"/>\n'
                            '    <mesh name="skin_eye_R" file="eye_R.obj"/>\n'
                            '    <mesh name="skin_lid_L" file="lid_L.obj"/>\n'
                            '    <mesh name="skin_lid_R" file="lid_R.obj"/>\n'
                            '    <mesh name="skin_femur_L"', 1)
    for side, sign in (("L", ""), ("R", "-")):
        old = ('<geom class="visual" type="sphere" pos="0.01682 %s0.01012 0.0068"'
               ' size="0.0036" rgba="0.05 0.05 0.06 1" material="eye"/>' % sign)
        new = ('<geom class="visual" type="sphere" pos="0.01682 %s0.01012 0.0068"'
               ' size="0.0036" rgba="0 0 0 0" material="eye"/>\n'
               '            <geom class="visual" type="mesh" mesh="skin_eye_%s"'
               ' rgba="1 1 1 1" material="eyeball"/>' % (sign, side))
        text = text.replace(old, new)

    # ---- 7c. the pale rim around each eye opening (#341) -------------------
    # The most recognisable thing about this animal's face after the spots is
    # the pale margin of skin round the eye. Generated as a torus at the skin
    # surface by make_gecko_eye.rim_ring, fixed to the head.
    if 'name="skin_rim_L"' not in text:
        text = text.replace('    <mesh name="skin_lid_L" file="lid_L.obj"/>',
                            '    <mesh name="skin_rim_L" file="rim_L.obj"/>\n'
                            '    <mesh name="skin_rim_R" file="rim_R.obj"/>\n'
                            '    <mesh name="skin_lid_L" file="lid_L.obj"/>', 1)
    for side in ("L", "R"):
        if 'mesh="skin_rim_%s"' % side in text:
            continue
        text = re.sub(
            r'(<geom class="visual" type="mesh" mesh="skin_eye_%s"[^>]*/>)' % side,
            lambda m: m.group(1) + '\n            <geom class="visual" type="mesh"'
                      ' mesh="skin_rim_%s" rgba="0.94 0.91 0.86 1" material="pale"/>' % side,
            text, count=1)

    # ---- 8. the lid is a generated hood, concentric with the eyeball ------
    # Sized by hand twice and wrong both times: too small and it was a flat
    # disc floating above the eye, too large and it covered the eyeball almost
    # completely. `make_gecko_eye.lid_cap` builds it as a shell around the eye
    # instead, so it fits by construction. The ellipsoids stay, transparent.
    for side, sign in (("L", ""), ("R", "-")):
        if 'mesh="skin_lid_%s"' % side in text:
            continue          # already wired; re-running must not stack copies
        pat = re.compile(
            r'<geom class="visual" type="ellipsoid" pos="0 ' + sign +
            r'0\.01012 0\.00\d+" size="0\.00\d+ 0\.00\d+ 0\.00\d+"'
            r'(?: rgba="[^"]*")? material="lid"/>')
        text = pat.sub(
            '<geom class="visual" type="ellipsoid" pos="0 %s0.01012 0.0026"'
            ' size="0.0042 0.0042 0.0014" rgba="0 0 0 0" material="lid"/>\n'
            '              <geom class="visual" type="mesh" mesh="skin_lid_%s"'
            ' rgba="1 1 1 1" material="body"/>' % (sign, side), text)
    return text


def main():
    before = SRC.read_text(encoding="utf-8")
    after = patch(before)
    SRC.write_text(after, encoding="utf-8")
    import mujoco
    mujoco.MjModel.from_xml_path(str(SRC))
    n_mesh = len(re.findall(r'type="mesh"', after))
    print("ok: %d mesh geoms at full texture brightness, UV-addressed; "
          "%d superseded face primitives hidden; tail banded"
          % (n_mesh, len(SUPERSEDED)))
    if before == after:
        print("   (already wired -- no change)")


if __name__ == "__main__":
    main()
