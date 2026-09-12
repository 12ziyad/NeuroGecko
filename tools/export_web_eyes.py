#!/usr/bin/env python3
"""Export the animal's eyes for the browser.

WHY THIS EXISTS. The eyes were in the model the whole time and had never
reached the web render. `tools/export_web_skin.py` exports MuJoCo's *skin*,
which is the deformable body surface; the eyes are two separate rigid meshes
(`eye_L.obj`, `eye_R.obj`) carrying their own texture (`gecko_eye.png`), parented
to the head body. Nothing in the browser knew about them, so the animal had
blank sockets -- and eyelid geckos are named for the one feature the render was
missing.

READ THE COMPILED MESH, NOT THE OBJ. This is the whole correctness point of the
file and the first version got it wrong. MuJoCo's compiler RE-CENTRES a mesh
asset: it subtracts the mesh's own centroid from every vertex and records that
shift in `mesh_pos`, and `geom_pos` is then expressed relative to the re-centred
frame. For this eye the two are the same number -- centroid (16.94, 10.40, 6.85)
mm, geom_pos (16.89, 10.41, 6.85) mm -- because the OBJ was authored in head
coordinates. So taking raw OBJ vertices AND adding geom_pos applies the offset
TWICE, and the eyes float about 17 mm in front of the face and 10 mm out to the
side. Reading `model.mesh_vert` gives the vertices MuJoCo itself draws, already
re-centred, and then `geom_pos`/`geom_quat` put them exactly where the physics
engine puts them.

WHAT IS PUBLISHED AND WHAT IS NOT. The geometry, its placement on the head and
the texture are the project's own assets and are exported verbatim -- not
redrawn here. Eublepharidae are the EYELID geckos: alone among Gekkota they have
movable eyelids instead of a fused spectacle, and that is why the eyelid joints
exist at all. What has never been published for this species is a blink rate, a
blink duration or a lid excursion, which is why the eyelid angle is commanded
from a declared-invented constant and says so where it is used.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import struct
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "media"
XML = REPO / "morphology" / "gecko_body_lab_v2.xml"


def compiled_mesh(model, mid):
    """Per-corner position / uv / normal buffers from the COMPILED mesh.

    MuJoCo keeps positions, normals and texture coordinates in three separate
    arrays with their own per-face index triples, which is the OBJ convention.
    A GPU wants one vertex per corner, so the three are flattened together.
    """
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    vert = np.asarray(model.mesh_vert[va:va + vn]).reshape(-1, 3)
    face = np.asarray(model.mesh_face[fa:fa + fn]).reshape(-1, 3)

    na = int(model.mesh_normaladr[mid])
    nn = int(model.mesh_normalnum[mid])
    normal = (np.asarray(model.mesh_normal[na:na + nn]).reshape(-1, 3)
              if nn > 0 else None)
    fnorm = (np.asarray(model.mesh_facenormal[fa:fa + fn]).reshape(-1, 3)
             if nn > 0 else None)

    ta = int(model.mesh_texcoordadr[mid])
    tn = int(model.mesh_texcoordnum[mid]) if ta >= 0 else 0
    texc = (np.asarray(model.mesh_texcoord[ta:ta + tn]).reshape(-1, 2)
            if tn > 0 else None)
    ftex = (np.asarray(model.mesh_facetexcoord[fa:fa + fn]).reshape(-1, 3)
            if tn > 0 else None)

    pos, uv, nrm, tri = [], [], [], []
    for f in range(face.shape[0]):
        for k in range(3):
            pos.extend(float(x) for x in vert[face[f, k]])
            if texc is not None:
                t = texc[ftex[f, k]]
                # MuJoCo stores v with the image's own row order; the loader is
                # told not to flip, so this passes straight through.
                uv.extend((float(t[0]), float(t[1])))
            else:
                uv.extend((0.0, 0.0))
            if normal is not None:
                nrm.extend(float(x) for x in normal[fnorm[f, k]])
            else:
                nrm.extend((0.0, 0.0, 1.0))
            tri.append(len(tri))
    return pos, uv, nrm, tri


def main():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    if head < 0:
        raise SystemExit("no head body in the morphology")

    eyes = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != head:
            continue
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        if "eye" not in mesh_name.lower():
            continue
        pos, uv, nrm, tri = compiled_mesh(model, mid)
        r = float(np.abs(np.asarray(pos).reshape(-1, 3)).max())
        eyes.append({
            "name": mesh_name,
            "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
            "pos": [float(x) for x in model.geom_pos[g]],
            "quat": [float(x) for x in model.geom_quat[g]],
            "radius": r,
            "counts": {"verts": len(pos) // 3, "tris": len(tri) // 3},
            "_buf": (pos, uv, nrm, tri),
        })
        print(f"  {mesh_name}: {len(pos)//3} corners, {len(tri)//3} tris, "
              f"radius {r*1000:.2f} mm, at "
              f"({', '.join(f'{x*1000:.2f}' for x in model.geom_pos[g])}) mm on the head")

    if not eyes:
        raise SystemExit("no eye meshes found on the head")

    blob, header = bytearray(), []
    for e in eyes:
        pos, uv, nrm, tri = e.pop("_buf")
        rec = dict(e)
        rec["offsets"] = {}
        for key, arr, fmt in (("pos", pos, "f"), ("uv", uv, "f"),
                              ("nrm", nrm, "f"), ("tri", tri, "I")):
            rec["offsets"][key] = [len(blob), len(arr)]
            blob.extend(struct.pack(f"<{len(arr)}{fmt}", *arr))
        header.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eyes.bin").write_bytes(bytes(blob))
    (OUT / "eyes.json").write_text(json.dumps({
        "generated_by": "tools/export_web_eyes.py",
        "source": "morphology/gecko_body_lab_v2.xml",
        "note": "vertices are model.mesh_vert -- the COMPILED, re-centred mesh. "
                "Using the raw OBJ here double-counts mesh_pos and floats the "
                "eyes off the face.",
        "head_body": "head",
        "texture": "eye.png",
        "meshes": header,
    }), encoding="utf-8")

    tex = REPO / "morphology" / "textures" / "gecko_eye.png"
    if tex.exists():
        shutil.copyfile(tex, OUT / "eye.png")
        print(f"  texture: {tex.name} -> eye.png ({tex.stat().st_size/1024:.0f} KB)")
    else:
        print("  ! gecko_eye.png missing; the browser will fall back to flat colour")

    print(f"written: {OUT/'eyes.json'} + eyes.bin ({len(blob)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
