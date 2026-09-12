"""One continuous skin over the whole animal, bound to the bodies it already has.

WHY THIS EXISTS. `tools/make_gecko_mesh.py` cuts the generated surface into one
rigid mesh per body. That is the standard MuJoCo pattern and it is what every
Menagerie robot does -- but a robot is made of separate hard shells, and this
animal is not. When the head yaws 25 degrees the head piece and the neck piece
rotate about different frames and a wedge opens between them: a visible notch in
the neck, in exactly the movement this project has spent the most effort on.

WHAT A SKIN IS. A MuJoCo `<skin>` is one triangle surface whose vertices are
bound to several bodies with weights, and re-posed by linear blend skinning
every frame. The bones are the bodies that already exist. From the XML
reference: skins "are only used for visualization and do not affect the physics
in any way" -- so this changes no mass, no contact, no joint and no gate, the
same guarantee the per-body meshes carry.

  Verified in this MuJoCo (3.9.0) before any of it was written: `<deformable>`
  `<skin>` compiles with inline `vertex`, `texcoord`, `face` and `<bone>`, and
  accepts `material` and `group`. Written inline rather than as a `.skn` binary
  so the whole animal stays inside the one file whose SHA-256 the provenance
  chain hashes.

HOW THE WEIGHTS ARE COMPUTED. Each candidate body's primitives are sampled into
a point cloud; a vertex's weight on a body falls off as exp(-(d/SIGMA)^2) in the
distance to that cloud. Candidates are restricted by region -- a limb's vertices
can only be driven by that limb's own chain plus the trunk it hangs from, and
the jaw's only by the jaw and the head -- because a purely distance-based bind
gives a foot resting against the belly a share of the trunk, and the foot then
swims when the animal breathes.

  MuJoCo normalises the weights itself, and refuses to compile if any vertex has
  zero total weight ("vertex %d must have positive total weight in skin"), so
  every vertex is given its nearest body outright if the falloff leaves it bare.

THE TEXTURE. With one skin there is one material, so the tail cannot carry a
separate banded image the way it did when it was five rigid pieces. Instead the
v coordinate runs 0 at the nose to 1 at the tail tip -- no tiling -- and ONE
image carries the whole animal: spotted over the head and trunk, paler and
banded behind the vent. The image is 512 x 2048 because that is the aspect ratio
that makes a circle drawn in it come out circular on the animal (about 8500
pixels per metre around the girth against 9300 along the body).

INVENTED IN THIS FILE, all appearance-only and none able to move a gate:
  SIGMA   0.009 m  weight falloff distance. Roughly one body radius: smaller and
                   the skin creases at the joints, larger and the tail smears
                   into the trunk.
  MAX_BONES   4    bones kept per vertex, largest first.
  MIN_WEIGHT  0.02 below this a bone is dropped rather than kept at noise level.
  LIMB_V           the stretch of the texture limbs are mapped into, so they get
                   body spots rather than whatever happens to be at their own
                   position along the animal.

Usage:  python tools/make_gecko_skin.py
"""

from __future__ import annotations

import math
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "morphology" / "gecko_body_r.xml"

SIGMA = 0.009
MAX_BONES = 4
MIN_WEIGHT = 0.02
LIMB_V = (0.16, 0.34)

AXIAL = ["head", "neck", "trunk_anterior", "trunk_middle",
         "tail1", "tail2", "tail3", "tail4", "tail5"]
LIMB_CHAINS = [("humerus", "forearm", "manus"), ("femur", "tibia", "pes")]


# --------------------------------------------------------------------------

def compile_without_skin(path):
    """Compile the body with its mesh geoms and any existing skin removed."""
    import mujoco
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"[ \t]*<deformable>.*?</deformable>\n?", "", text, flags=re.S)
    # Strip mesh geoms as ELEMENTS. Filtering by line drops whatever else
    # happens to share the line, which is exactly how the eye spheres
    # disappeared and this generator reported "found 0" (#330).
    text = re.sub(r'<geom[^>]*type="mesh"[^>]*/>', '', text)
    kept = [ln for ln in text.splitlines() if not re.match(r"\s*<mesh ", ln)]
    tmp = path.with_name(path.stem + "_skinsrc_tmp.xml")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def bone_cloud(model, data, bid, n=400):
    """A point cloud on a body's primitives, for distance queries."""
    import mujoco
    pts = []
    skip = set()
    for name in ("eye", "teeth", "tongue"):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if i >= 0:
            skip.add(i)
    rng = np.random.default_rng(0)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or int(model.geom_matid[g]) in skip:
            continue
        t = model.geom_type[g]
        c = np.array(data.geom_xpos[g], float)
        R = np.array(data.geom_xmat[g], float).reshape(3, 3)
        s = model.geom_size[g]
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts.append(c + u * float(s[0]))
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            pts.append(c + (u * np.array([s[0], s[1], s[2]], float)) @ R.T)
        elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            r, half = float(s[0]), float(s[1])
            z = rng.uniform(-half, half, n)
            a = rng.uniform(0, 2 * math.pi, n)
            local = np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)
            pts.append(c + local @ R.T)
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            local = rng.uniform(-1, 1, (n, 3)) * np.array([s[0], s[1], s[2]], float)
            pts.append(c + local @ R.T)
    if not pts:
        return np.array(data.xpos[bid], float)[None, :]
    return np.concatenate(pts, axis=0)


def nearest(cloud, verts, chunk=512):
    """Min distance from each vertex to a point cloud, without scipy."""
    out = np.empty(len(verts))
    for i in range(0, len(verts), chunk):
        block = verts[i:i + chunk]
        d = np.linalg.norm(block[:, None, :] - cloud[None, :, :], axis=2)
        out[i:i + chunk] = d.min(axis=1)
    return out


# --------------------------------------------------------------------------

def build_surface(model, data):
    """One vertex/uv/face set for the whole animal, in the bind pose.

    Returns (verts, uvs, faces, region) where `region` names, per vertex, which
    group of bones is allowed to drive it.
    """
    import sys
    sys.path.insert(0, str(REPO / "tools"))
    import make_gecko_mesh as G
    import mujoco

    verts, uvs, faces, region = [], [], [], []

    def add_tube(rings, ring_uvs, tag):
        base = sum(len(v) for v in verts)
        per = len(rings[0])
        for r, u in zip(rings, ring_uvs):
            verts.append(np.asarray(r, float))
            uvs.append(np.asarray(u, float))
            region.extend([tag] * per)
        for i in range(len(rings) - 1):
            a0, b0 = base + i * per, base + (i + 1) * per
            for j in range(per - 1):
                faces.append((a0 + j, b0 + j, b0 + j + 1))
                faces.append((a0 + j, b0 + j + 1, a0 + j + 1))

    # ---- the axial chain, NOT cut into pieces --------------------------
    rings, _, owner = G.build_axis(model, data)
    centres = np.array([r.mean(axis=0) for r in rings])
    seg = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    ang = np.linspace(0.0, 2.0 * math.pi, len(rings[0]))
    axial_uv = [np.stack([ang / (2 * math.pi), np.full(len(ang), a / total)],
                         axis=1) for a in arc]
    add_tube(rings, axial_uv, "axial")

    # where the tail starts, as a fraction along the animal: the texture needs
    # it to know where to stop spotting and start banding
    tail_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in AXIAL if n.startswith("tail")}
    first_tail = next((i for i, o in enumerate(owner) if o in tail_ids), len(owner) - 1)
    tail_start_v = float(arc[first_tail] / total)

    # ---- limbs, each mapped into a spotted stretch of the texture -------
    # ONE continuous skin per limb through the joints, not six tubes and six
    # balls (#343). See gecko_sculpt.limb_skins.
    from gecko_sculpt import limb_skins
    for bid, lr, lu in limb_skins(model, data, G.RING, G.LIMB_SCALE, LIMB_V):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        add_tube(lr, lu, "limb:" + name[-1])

    # (knuckle spheres retired in #343: the limb is one continuous skin now)

    # ---- toes: five per foot, along the digits the body defines (#341) ---
    from gecko_sculpt import toe_tubes
    for bid, tr, tu in toe_tubes(model, data, G.RING):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        add_tube(tr, tu, "limb:" + name[-1])

    # ---- the jaw --------------------------------------------------------
    jaw = G.jaw_tube(model, data)
    if jaw is not None:
        _, jr, ju = jaw
        n = len(jr)
        # THE JAW GETS ITS OWN STRIP OF TEXTURE. It used v 0.02-0.06, which is
        # also the snout's v on the same image, so painting the mouth floor
        # pink put a pink smear on the tip of the nose (#343). The last half a
        # percent of v is the tail tip's final dome, a point in practice, so
        # the jaw lives there: u kept to the dorsal band so its top and its
        # underside land on different columns and only the top is painted.
        ang = np.linspace(0.0, 2.0 * math.pi, len(ju[0]))
        uv = [np.stack([0.15 + 0.20 * ang / (2.0 * math.pi),
                        np.full(len(u), 0.995 + 0.004 * k / max(1, n - 1))], axis=1)
              for k, u in enumerate(ju)]
        add_tube(jr, uv, "jaw")

    # ---- where the face's features land on the texture (#341) -------------
    # v is arc length nose to tail, so a head landmark's v is read off the
    # ring nearest its x. u is the angle round the body, 0 at the left flank.
    ring_x = np.array([r[0, 0] for r in rings])

    def v_at(x):
        return float(arc[int(np.argmin(np.abs(ring_x - x)))] / total)

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    eye_mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "eye")
    eye_x = next((float(data.geom_xpos[g][0]) for g in range(model.ngeom)
                  if model.geom_bodyid[g] == head
                  and int(model.geom_matid[g]) == eye_mat), None)
    jaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw")
    tip_x = float(ring_x.max())
    landmarks = {
        "tail_start_v": tail_start_v,
        "eye_v": v_at(eye_x) if eye_x is not None else None,
        "jaw_v": v_at(float(data.xpos[jaw][0])) if jaw >= 0 else None,
        "nostril_v": v_at(tip_x - 0.0030),
    }
    return (np.concatenate(verts), np.concatenate(uvs),
            np.array(faces, int), region, landmarks)


def candidate_bones(model, region_tag):
    """Which bodies are allowed to drive a vertex in this region."""
    import mujoco

    def ids(names):
        out = []
        for n in names:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b >= 0:
                out.append(b)
        return out

    if region_tag == "axial":
        # The GULAR is a bone here even though it is not part of the axial
        # chain. It is the throat, and the visible pulsing under a gecko's chin
        # is what it exists to produce -- so the skin itself has to carry that
        # movement. Left out, the throat's own pale ellipsoid had to be drawn
        # separately and poked through the skin at the mouth corner.
        return ids(AXIAL + ["gular"])
    if region_tag == "jaw":
        return ids(["jaw", "head"])
    side = region_tag.split(":")[1]
    names = [f"{seg}_{side}" for chain in LIMB_CHAINS for seg in chain]
    return ids(names + ["trunk_anterior", "trunk_middle"])


def main():
    import mujoco
    model = compile_without_skin(SRC)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral").id)
    mujoco.mj_forward(model, data)

    verts, uvs, faces, region, landmarks = build_surface(model, data)
    tail_start_v = landmarks["tail_start_v"]
    print(f"surface: {len(verts)} vertices, {len(faces)} triangles, "
          f"tail starts at v={tail_start_v:.3f}")

    # ---- weights --------------------------------------------------------
    all_bids = sorted({b for tag in set(region)
                       for b in candidate_bones(model, tag)})
    clouds = {b: bone_cloud(model, data, b) for b in all_bids}
    dist = {b: nearest(clouds[b], verts) for b in all_bids}

    allowed = {tag: set(candidate_bones(model, tag)) for tag in set(region)}
    weights = {b: {} for b in all_bids}
    bare = 0
    # THE THROAT PULLS HARDER. The gular is a small body under a large head, so
    # by plain distance the head owns nearly all the throat skin and the
    # breathing that the gular exists to produce was invisible. Its weight is
    # multiplied so the throat floor follows it. INVENTED gain (#343).
    gain = {}
    _g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gular")
    if _g >= 0:
        gain[_g] = 3.0
    for i in range(len(verts)):
        ok = allowed[region[i]]
        cand = [(math.exp(-(dist[b][i] / SIGMA) ** 2) * gain.get(b, 1.0), b)
                for b in ok]
        cand.sort(reverse=True)
        cand = [(w, b) for w, b in cand[:MAX_BONES] if w >= MIN_WEIGHT]
        if not cand:
            # never leave a vertex unbound: MuJoCo refuses to compile it, and
            # a vertex bound to nothing is a hole in the animal
            b = min(ok, key=lambda b: dist[b][i])
            cand = [(1.0, b)]
            bare += 1
        for w, b in cand:
            weights[b][i] = w
    print(f"weights: {sum(len(v) for v in weights.values())} bone-vertex links, "
          f"{bare} vertices fell back to their nearest body")

    # ---- texture --------------------------------------------------------
    import make_skin_texture as T
    from PIL import Image
    img = T.whole_body(tail_start_v, landmarks=landmarks)
    out = REPO / "morphology" / "textures" / "gecko_body.png"
    Image.fromarray(img).save(out)
    print(f"texture: {out.name} {img.shape[1]}x{img.shape[0]}, "
          f"banded below v={tail_start_v:.3f}")

    # ---- emit -----------------------------------------------------------
    def nums(a, fmt="%.6f"):
        return " ".join(fmt % v for v in np.asarray(a).ravel())

    bones = []
    for b in all_bids:
        w = weights[b]
        if not w:
            continue
        idx = sorted(w)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        bones.append(
            '      <bone body="%s" bindpos="%s" bindquat="%s"\n'
            '            vertid="%s"\n'
            '            vertweight="%s"/>'
            % (name, nums(data.xpos[b]), nums(data.xquat[b]),
               " ".join(str(i) for i in idx),
               " ".join("%.4f" % w[i] for i in idx)))

    block = (
        "  <deformable>\n"
        "    <!-- ONE CONTINUOUS SKIN, generated by tools/make_gecko_skin.py.\n"
        "         Visualization only: a MuJoCo skin changes no mass, no contact,\n"
        "         no joint and no gate. It replaces 23 rigid per-body meshes\n"
        "         whose seams opened at every joint, most visibly as a wedge in\n"
        "         the neck whenever the head yawed. -->\n"
        '    <skin name="gecko_skin" material="body" group="1" inflate="0"\n'
        '          vertex="%s"\n'
        '          texcoord="%s"\n'
        '          face="%s">\n%s\n    </skin>\n  </deformable>\n'
        % (nums(verts), nums(uvs, "%.5f"),
           " ".join(str(i) for i in faces.ravel()), "\n".join(bones)))

    text = SRC.read_text(encoding="utf-8")
    text = re.sub(r"[ \t]*<deformable>.*?</deformable>\n", "", text, flags=re.S)
    text = text.replace("</mujoco>", block + "</mujoco>")

    # TWO PARSERS HAVE TO ACCEPT THIS, NOT ONE, AND THE SECOND ONE FAILS
    # SILENTLY. MuJoCo's reader tolerates a bare "--" inside a comment;
    # ElementTree, which `build_lab_morphology` parses the source with, rejects
    # it outright. Writing one produced a source that COMPILED and could not be
    # built FROM: the generator threw, the previous lab body stayed on disk, the
    # world was rebuilt from that, and the skin was quietly absent from the
    # world while every check that looked at the source said it was present.
    # Both parsers now run before the file is kept (#322).
    import xml.etree.ElementTree as ET
    ET.fromstring(text)
    SRC.write_text(text, encoding="utf-8")
    mujoco.MjModel.from_xml_path(str(SRC))
    print(f"wrote the skin into {SRC.name} ({len(block) / 1024:.0f} KB), "
          f"{len(bones)} bones; model compiles")


if __name__ == "__main__":
    main()
