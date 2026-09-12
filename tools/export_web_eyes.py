#!/usr/bin/env python3
"""Export the animal's eyes for the browser.

WHY THIS EXISTS. The eyes were in the model the whole time and had never
reached the web render. `tools/export_web_skin.py` exports MuJoCo's *skin*,
which is the deformable body surface; the eyes are two separate rigid meshes
(`eye_L.obj`, `eye_R.obj`) carrying their own texture (`gecko_eye.png`), parented
to the head body. Nothing in the browser knew about them, so the animal had
blank sockets -- and eyelid geckos are named for the one feature the render was
missing.

WHAT IS PUBLISHED AND WHAT IS NOT. The mesh geometry, its placement on the head
and the texture are the project's own assets and are exported verbatim -- not
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

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "media"
XML = REPO / "morphology" / "gecko_body_lab_v2.xml"


def read_obj(path):
    """Positions, normals, texcoords and triangles from a Wavefront OBJ."""
    vs, vts, vns, faces = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vts.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("vn "):
            vns.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                parts = (tok.split("/") + ["", ""])[:3]
                v = int(parts[0]) - 1
                t = int(parts[1]) - 1 if parts[1] else -1
                n = int(parts[2]) - 1 if parts[2] else -1
                idx.append((v, t, n))
            for k in range(1, len(idx) - 1):       # fan-triangulate
                faces.append((idx[0], idx[k], idx[k + 1]))
    return vs, vts, vns, faces


def flatten(vs, vts, vns, faces):
    """One vertex per (v, t, n) corner, which is what a GPU buffer wants."""
    order, seen = [], {}
    pos, uv, nrm, tri = [], [], [], []
    for f in faces:
        for corner in f:
            if corner not in seen:
                seen[corner] = len(order)
                order.append(corner)
                v, t, n = corner
                pos.extend(vs[v])
                uv.extend(vts[t] if 0 <= t < len(vts) else (0.0, 0.0))
                nrm.extend(vns[n] if 0 <= n < len(vns) else (0.0, 0.0, 1.0))
            tri.append(seen[corner])
    return pos, uv, nrm, tri


def main():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(XML))

    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    if head < 0:
        raise SystemExit("no head body in the morphology")

    # Every eye mesh geom on the head, with its own local placement. Read off
    # the compiled model rather than the XML text, so what ships is what the
    # physics engine itself resolved.
    eyes = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if int(model.geom_bodyid[g]) != head:
            continue
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mid = int(model.geom_dataid[g])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        if "eye" not in mesh_name.lower():
            continue
        obj = REPO / "morphology" / "meshes" / f"{mesh_name.replace('skin_', '')}.obj"
        if not obj.exists():
            print(f"  ! {mesh_name}: {obj.name} not found, skipped")
            continue
        vs, vts, vns, faces = read_obj(obj)
        pos, uv, nrm, tri = flatten(vs, vts, vns, faces)
        eyes.append({
            "name": mesh_name,
            "geom": name,
            "pos": [float(x) for x in model.geom_pos[g]],
            "quat": [float(x) for x in model.geom_quat[g]],
            "counts": {"verts": len(pos) // 3, "tris": len(tri) // 3},
            "_buf": (pos, uv, nrm, tri),
        })
        print(f"  {mesh_name}: {len(pos)//3} verts, {len(tri)//3} tris, "
              f"local pos {tuple(round(float(x), 5) for x in model.geom_pos[g])}")

    if not eyes:
        raise SystemExit("no eye meshes found on the head")

    # One binary, laid out mesh after mesh.
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
