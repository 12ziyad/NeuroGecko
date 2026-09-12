"""Export the animal's SKIN, so the browser draws the gecko and not its physics proxy.

WHAT WENT WRONG WITHOUT THIS. The first browser build rendered the 48 collision
solids -- capsules, boxes and one ellipsoid. That is the body the PHYSICS uses
and it is the right thing to draw in the nervous-system view, but it is not
what this animal looks like. The leopard gecko is the SKIN: one continuous
deformable surface of 21,663 vertices over 23 bones, carrying the spot texture,
built across sessions 10-13 (#307 to #345). Shipping the proxy and calling it
the animal is the appearance equivalent of a plausible number in published
clothes.

WHY THE SKIN IS EXPORTED SEPARATELY RATHER THAN LOADED BY MuJoCo IN THE BROWSER.
The physics file the browser runs is the stripped body -- 54 KB, verified
physics-neutral against the full one at exactly 0 difference over 400 steps.
Keeping it stripped keeps the physics honest and small. The skin is pure
appearance (#318 measured the trajectory as bit-identical with and without it),
so it travels as its own asset and is posed in the browser by linear blend
skinning from the same body transforms the physics produces.

WHAT COMES OUT

    skin.bin    float32 rest vertices, float32 texcoords, uint32 faces,
                per-bone bind poses and vertex weights -- one packed buffer
    skin.json   the header describing that buffer, plus bone -> body names
    skin.png    the texture the material actually points at

Usage:  python tools/export_web_skin.py
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "site" / "media"


def main():
    import mujoco

    m = mujoco.MjModel.from_xml_path("morphology/gecko_body_lab_v2.xml")
    if m.nskin < 1:
        raise SystemExit("this body has no skin")
    s = 0
    bname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body{i}"

    va, vn = int(m.skin_vertadr[s]), int(m.skin_vertnum[s])
    fa, fn = int(m.skin_faceadr[s]), int(m.skin_facenum[s])
    ba, bn = int(m.skin_boneadr[s]), int(m.skin_bonenum[s])

    vert = np.asarray(m.skin_vert[va * 3:(va + vn) * 3], dtype=np.float32)
    face = np.asarray(m.skin_face[fa * 3:(fa + fn) * 3], dtype=np.uint32)
    txc = np.asarray(m.skin_texcoord[va * 2:(va + vn) * 2], dtype=np.float32) \
        if m.nskintexvert else np.zeros(vn * 2, dtype=np.float32)

    bones = []
    bverts, bweights, boffsets = [], [], []
    for b in range(ba, ba + bn):
        bid = int(m.skin_bonebodyid[b])
        adr, num = int(m.skin_bonevertadr[b]), int(m.skin_bonevertnum[b])
        ids = np.asarray(m.skin_bonevertid[adr:adr + num], dtype=np.uint32)
        wts = np.asarray(m.skin_bonevertweight[adr:adr + num], dtype=np.float32)
        boffsets.append([len(bverts), num])
        bverts.append(ids)
        bweights.append(wts)
        bones.append({
            "body": bid,
            "body_name": bname(bid),
            "bindpos": [float(x) for x in m.skin_bonebindpos[b]],
            "bindquat": [float(x) for x in m.skin_bonebindquat[b]],
            "vertadr": int(boffsets[-1][0]),
            "vertnum": num,
        })
        boffsets[-1][0] = int(sum(len(a) for a in bverts[:-1]))
    for i, b in enumerate(bones):
        b["vertadr"] = int(sum(len(a) for a in bverts[:i]))

    bverts = np.concatenate(bverts).astype(np.uint32) if bverts else np.zeros(0, np.uint32)
    bweights = np.concatenate(bweights).astype(np.float32) if bweights else np.zeros(0, np.float32)

    buf = b"".join([vert.tobytes(), txc.tobytes(), face.tobytes(),
                    bverts.tobytes(), bweights.tobytes()])
    (OUT / "skin.bin").write_bytes(buf)

    header = {
        "nvert": vn, "nface": fn, "nbone": bn,
        "layout": [
            {"name": "vert", "type": "float32", "count": vn * 3, "bytes": vert.nbytes},
            {"name": "texcoord", "type": "float32", "count": vn * 2, "bytes": txc.nbytes},
            {"name": "face", "type": "uint32", "count": fn * 3, "bytes": face.nbytes},
            {"name": "bonevertid", "type": "uint32", "count": int(bverts.size), "bytes": bverts.nbytes},
            {"name": "bonevertweight", "type": "float32", "count": int(bweights.size), "bytes": bweights.nbytes},
        ],
        "bones": bones,
        "body_names": [bname(i) for i in range(m.nbody)],
    }

    # ---- the texture this skin's material actually points at
    mat = int(m.skin_matid[s])
    texid = -1
    if mat >= 0:
        tex = np.asarray(m.mat_texid[mat]).ravel()
        cand = [int(t) for t in tex if int(t) >= 0]
        texid = cand[0] if cand else -1
    if texid >= 0:
        w, h = int(m.tex_width[texid]), int(m.tex_height[texid])
        nch = int(m.tex_nchannel[texid])
        adr = int(m.tex_adr[texid])
        raw = np.asarray(m.tex_data[adr:adr + w * h * nch], dtype=np.uint8).reshape(h, w, nch)
        try:
            from PIL import Image
            Image.fromarray(raw[:, :, :3] if nch >= 3 else
                            np.repeat(raw, 3, axis=2)).save(OUT / "skin.png")
            header["texture"] = {"file": "skin.png", "w": w, "h": h,
                                 "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, texid)}
            print(f"  texture: {header['texture']['name']} {w}x{h} -> skin.png")
        except Exception as exc:                                   # pragma: no cover
            print(f"  texture export skipped: {exc}")
    if mat >= 0:
        header["material_rgba"] = [float(x) for x in m.mat_rgba[mat]]
    header["skin_rgba"] = [float(x) for x in m.skin_rgba[s]]

    (OUT / "skin.json").write_text(json.dumps(header), encoding="utf-8")
    print(f"written: skin.bin {len(buf)/1048576:.2f} MB   skin.json")
    print(f"  {vn} vertices, {fn} faces, {bn} bones, {bverts.size} bone-vertex weights")
    print(f"  bones -> bodies: {', '.join(b['body_name'] for b in bones[:8])} ...")


if __name__ == "__main__":
    main()
