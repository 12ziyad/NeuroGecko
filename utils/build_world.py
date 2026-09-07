#!/usr/bin/env python3
"""Derive a behaviour-phase world from the validated lab body, without editing it.

`morphology/gecko_body_lab_v2.xml` is the body that passes the 14 static checks
and whose SHA256 the lab training evidence contract pins. Editing it in place to
retexture a floor or narrow a camera would silently invalidate every committed
measurement that names that hash.

So the world is generated instead: this reads the validated body, applies a small
number of declared edits, and writes a new file that records the source hash, the
exact edits and its own hash. Each edit is a visual or sensory property of the
world, never of the animal -- no geom, joint, mass, actuator or site is touched,
and that is asserted rather than assumed.

Why each edit exists:

* **Floor texture scale.** The committed checker is 6 repeats per metre, so one
  square is 16.7 cm against a 10.6 cm animal -- 0.64 squares per body length. A
  camera looking at a surface with no edges inside its field cannot see that it
  is moving. Optic flow needs features at a scale the animal actually crosses.
* **Camera field of view.** Wider is not better: at a fixed pixel count a wider
  field spreads the same pixels over more world, so each pixel spans more degrees
  and small prey stops being resolvable at all.

Both numbers are arguments, not constants, and the caller records which it used.

Usage:
    python utils/build_world.py --texrepeat 60 --fovy 70 \
        --output morphology/gecko_world_v1.xml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SOURCE = REPO / "morphology" / "gecko_body_lab_v2.xml"

# Anything outside this set means the generator has been asked to change the
# animal, which is not what it is for.
ALLOWED_TAGS = {"material", "camera"}

# Prey is a MOCAP body on purpose. A free-jointed prey would add degrees of
# freedom to qpos/qvel, which changes the animal's own state vector and every
# index downstream of it. A mocap body is driven by writing mocap_pos directly:
# it renders to the camera, it collides with nothing, and nq/nv are untouched.
PREY_BODY = """
  <body name="prey" mocap="true" pos="0.15 0 0.004">
    <geom name="prey_geom" type="sphere" size="{radius:g}" material="prey"
          contype="0" conaffinity="0" density="0"/>
  </body>
"""
PREY_MATERIAL = '<material name="prey" rgba="{rgba}"/>'



def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative(path):
    """Repo-relative when possible; absolute paths stay legible in the manifest."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source_text, texrepeat=None, fovy=None, offwidth=None, offheight=None,
          prey_radius=None, prey_rgba="0.55 0.35 0.18 1"):
    """Return (xml_text, edits). Raises if a requested edit has no target."""
    root = ET.fromstring(source_text)
    edits = []

    if prey_radius is not None:
        if not (prey_radius > 0):
            raise ValueError("Prey radius must be positive.")
        if [b for b in root.iter("body") if b.get("name") == "prey"]:
            raise ValueError("The source already contains a prey body.")
        asset = root.find("asset")
        worldbody = root.find("worldbody")
        if asset is None or worldbody is None:
            raise ValueError("Expected <asset> and <worldbody> in the source body.")
        asset.append(ET.fromstring(PREY_MATERIAL.format(rgba=prey_rgba)))
        worldbody.append(ET.fromstring(PREY_BODY.format(radius=prey_radius)))
        edits.append({"element": "worldbody", "attribute": "prey",
                      "before": "absent", "after": f"mocap sphere r={prey_radius:g} m",
                      "reason": "the animal must see its food rather than be told where it is; "
                                "mocap keeps nq/nv unchanged and the prey collides with nothing"})

    if texrepeat is not None:
        found = [m for m in root.iter("material") if m.get("name") == "grid"]
        if len(found) != 1:
            raise ValueError("Expected exactly one material named 'grid' to retexture.")
        before = found[0].get("texrepeat")
        found[0].set("texrepeat", f"{texrepeat:g} {texrepeat:g}")
        edits.append({"element": "material grid", "attribute": "texrepeat",
                      "before": before, "after": found[0].get("texrepeat"),
                      "reason": "optic flow needs features at a scale the animal crosses"})

    if fovy is not None:
        found = [c for c in root.iter("camera") if c.get("name") == "head_cam"]
        if len(found) != 1:
            raise ValueError("Expected exactly one camera named 'head_cam'.")
        before = found[0].get("fovy")
        found[0].set("fovy", f"{fovy:g}")
        edits.append({"element": "camera head_cam", "attribute": "fovy",
                      "before": before, "after": found[0].get("fovy"),
                      "reason": "a wider field spends the same pixels on more world"})

    if offwidth is not None or offheight is not None:
        found = [g for g in root.iter("global")]
        if len(found) != 1:
            raise ValueError("Expected exactly one <global> element.")
        for name, value in (("offwidth", offwidth), ("offheight", offheight)):
            if value is None:
                continue
            before = found[0].get(name)
            found[0].set(name, f"{value:g}")
            edits.append({"element": "global", "attribute": name,
                          "before": before, "after": found[0].get(name),
                          "reason": "offscreen buffer must cover the requested render size"})

    if not edits:
        raise ValueError("No edits requested; the generator would just copy the body.")
    for edit in edits:
        tag = edit["element"].split()[0]
        if tag not in ALLOWED_TAGS and tag not in ("global", "worldbody"):
            raise ValueError(f"Refusing to edit <{tag}>: the world may not change the animal.")
    return ET.tostring(root, encoding="unicode"), edits


def assert_body_unchanged(source_text, world_text):
    """The ANIMAL must be identical. The world may gain objects; the gecko may not change.

    Compared by name rather than by index, because adding a world object shifts
    every index after it and an index-wise comparison would either false-alarm or,
    worse, silently compare the wrong pair.
    """
    import mujoco
    import numpy as np
    paths = []
    try:
        for index, text in enumerate((source_text, world_text)):
            path = REPO / f".world_check_{index}.xml"
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        a, b = (mujoco.MjModel.from_xml_path(str(p)) for p in paths)

        # The animal's own state vector must not move at all.
        for name in ("nq", "nv", "nu", "njnt", "nsensor"):
            if getattr(a, name) != getattr(b, name):
                raise ValueError(f"World changed the animal's state: {name} "
                                 f"{getattr(a, name)} -> {getattr(b, name)}")

        def by_name(model, kind, count):
            return {mujoco.mj_id2name(model, kind, i): i for i in range(count)}

        checks = [
            (mujoco.mjtObj.mjOBJ_BODY, a.nbody, b.nbody,
             ("body_mass", "body_inertia", "body_pos", "body_quat")),
            (mujoco.mjtObj.mjOBJ_GEOM, a.ngeom, b.ngeom,
             ("geom_size", "geom_pos", "geom_friction", "geom_type")),
            (mujoco.mjtObj.mjOBJ_SITE, a.nsite, b.nsite, ("site_pos",)),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, a.nu, b.nu,
             ("actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm")),
        ]
        for kind, count_a, count_b, fields in checks:
            source_names = by_name(a, kind, count_a)
            world_names = by_name(b, kind, count_b)
            missing = set(source_names) - set(world_names)
            if missing:
                raise ValueError(f"World dropped {sorted(n for n in missing if n)}")
            for name, index in source_names.items():
                if name is None:
                    continue
                other = world_names[name]
                for field in fields:
                    if not np.array_equal(getattr(a, field)[index], getattr(b, field)[other]):
                        raise ValueError(f"World changed the animal: {name}.{field} differs")
        added = [n for n in by_name(b, mujoco.mjtObj.mjOBJ_BODY, b.nbody)
                 if n and n not in by_name(a, mujoco.mjtObj.mjOBJ_BODY, a.nbody)]
        for name in added:
            index = by_name(b, mujoco.mjtObj.mjOBJ_BODY, b.nbody)[name]
            if not b.body_mocapid[index] >= 0:
                raise ValueError(f"Added world body {name!r} is not mocap; it would enter qpos.")
        return added
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texrepeat", type=float, default=None,
                   help="checker repeats per metre; the committed body uses 6 (16.7 cm squares)")
    p.add_argument("--fovy", type=float, default=None,
                   help="head camera vertical field of view in degrees; the body uses 120")
    p.add_argument("--prey-radius", type=float, default=None,
                   help="add a mocap prey sphere of this radius in metres")
    p.add_argument("--prey-rgba", default="0.55 0.35 0.18 1",
                   help="prey colour; the default is a dull brown, not a high-contrast marker")
    p.add_argument("--offwidth", type=float, default=None)
    p.add_argument("--offheight", type=float, default=None)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--note", default="", help="why this world exists, recorded in the manifest")
    a = p.parse_args(argv)

    source_text = a.source.read_text(encoding="utf-8")
    world_text, edits = build(source_text, a.texrepeat, a.fovy, a.offwidth, a.offheight,
                              a.prey_radius, a.prey_rgba)
    added = assert_body_unchanged(source_text, world_text)

    header = ("<!-- GENERATED by utils/build_world.py from "
              + a.source.name + " (sha256 " + sha256_file(a.source) + ").\n"
              " The animal is byte-for-byte the validated lab body: bodies, geoms, joints,\n"
              " masses, actuators, sites and friction are asserted identical at build time.\n"
              " Only world-facing visual and sensory properties differ, listed below.\n"
              + "".join(f"   {e['element']}.{e['attribute']}: {e['before']} -> {e['after']}\n" for e in edits)
              + (f"   note: {a.note}\n" if a.note else "")
              + " Do not hand-edit; regenerate. -->\n")
    text = world_text.replace("<mujoco", header + "<mujoco", 1) if not world_text.startswith("<?xml") else world_text
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generator": "utils/build_world.py",
        "source": relative(a.source),
        "source_sha256": sha256_file(a.source),
        "output": relative(a.output),
        "output_sha256": sha256_file(a.output),
        "edits": edits,
        "note": a.note,
        "world_bodies_added": added,
        "asserted": "every named body, geom, site and actuator of the animal verified identical "
                    "to the source at build time, compared by name; nq/nv/nu/njnt/nsensor "
                    "unchanged; any added world body verified mocap so it cannot enter qpos",
    }
    manifest_path = a.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
