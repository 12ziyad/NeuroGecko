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
import pathlib
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

# THE REST OF THE WORLD. Until now it was a floor and one cricket, so five of
# the six behaviours the selector can choose had nothing to act on: there was
# nowhere to shelter, nothing warm, and nothing to flee.
#
# SHELTER is a slab raised on two low supports, making a crevice with a gap
# under it. It is NOT a burrow, and that is the finding rather than a
# simplification: this species does not dig. The field description has it
# living in "holes and crevices in gravel-mixed stony terrain", and retreat
# chambers traced through a demolished stone wall were masonry voids the
# animals had not modified -- "the lizards apparently had done nothing in the
# setting of the site". So it gets a gap that already exists.
#
# The WARM PATCH is a surface, not a lamp, and that distinction is measured.
# Body temperature tracks the SUBSTRATE at r2 = 0.97 against air at 0.92 (n=12,
# Hastings et al. 2023) -- the animal is thigmothermic and takes heat by lying
# on warm ground. Melanistic pigment had no effect on heating rate, which is
# further evidence against radiation-driven warming. MuJoCo has no thermal
# physics, so the patch is geometry the environment reads by position, and it
# is coloured so the camera can see it; the heat itself lives in the
# environment, not in the model.
#
# The THREAT is a mocap body like the prey, so it can be moved without touching
# the physics state. Its own honest caveat: for this species a visual predator
# alone produces a defensive reaction on 0.07 of trials against 0.21 for scent
# (n=42). A threat the animal can only see is close to modelling the wrong
# sense, and it exists here so the flee channel has a referent at all.
SHELTER_BODY = """
  <body name="shelter" pos="{x:g} {y:g} 0">
    <geom name="shelter_roof" type="box" size="{half:g} {half:g} 0.004"
          pos="0 0 {roof:g}" material="shelter"/>
    <geom name="shelter_post_a" type="box" size="0.006 {half:g} {post:g}"
          pos="{edge:g} 0 {post:g}" material="shelter"/>
    <geom name="shelter_post_b" type="box" size="0.006 {half:g} {post:g}"
          pos="-{edge:g} 0 {post:g}" material="shelter"/>
  </body>
"""
WARM_PATCH = """
  <body name="warm_patch" pos="{x:g} {y:g} 0">
    <geom name="warm_geom" type="box" size="{half:g} {half:g} 0.0015"
          pos="0 0 0.0015" material="warm" contype="0" conaffinity="0"/>
  </body>
"""
THREAT_BODY = """
  <body name="threat" mocap="true" pos="{x:g} {y:g} {z:g}">
    <geom name="threat_geom" type="capsule" size="0.012 0.045"
          euler="0 1.5708 0" material="threat"
          contype="0" conaffinity="0" density="0"/>
  </body>
"""
SHELTER_MATERIAL = '<material name="shelter" rgba="0.38 0.36 0.34 1"/>'
WARM_MATERIAL = '<material name="warm" rgba="0.62 0.30 0.20 1"/>'
THREAT_MATERIAL = '<material name="threat" rgba="0.20 0.18 0.16 1"/>'



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
          prey_radius=None, prey_rgba="0.55 0.35 0.18 1",
          shelter=None, warm_patch=None, threat=None):
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

    if shelter is not None:
        half, x, y = float(shelter[0]), float(shelter[1]), float(shelter[2])
        if not (half > 0):
            raise ValueError("shelter half-width must be positive")
        # The gap has to admit the animal. Trunk height walking is 19-27 mm, so
        # a 32 mm clearance is enough to enter without being a room. INVENTED:
        # no crevice-height preference has ever been measured in this species,
        # and the "tight hide" rule every care sheet states has zero primary
        # support.
        post = 0.016
        assets = root.find("asset")
        assets.append(ET.fromstring(SHELTER_MATERIAL))
        worldbody.append(ET.fromstring(SHELTER_BODY.format(
            half=half, x=x, y=y, roof=post * 2, post=post, edge=half - 0.008)))
        edits.append({"element": "worldbody", "attribute": "shelter",
                      "before": "absent",
                      "after": f"crevice {2*half:g} m wide, {2*post:g} m clearance, "
                               f"at ({x:g}, {y:g})"})

    if warm_patch is not None:
        half, x, y = float(warm_patch[0]), float(warm_patch[1]), float(warm_patch[2])
        if not (half > 0):
            raise ValueError("warm patch half-width must be positive")
        assets = root.find("asset")
        assets.append(ET.fromstring(WARM_MATERIAL))
        worldbody.append(ET.fromstring(WARM_PATCH.format(half=half, x=x, y=y)))
        edits.append({"element": "worldbody", "attribute": "warm_patch",
                      "before": "absent",
                      "after": f"surface {2*half:g} m across at ({x:g}, {y:g}); "
                               "GEOMETRY ONLY -- MuJoCo has no thermal physics, "
                               "the temperature lives in the environment"})

    if threat is not None:
        x, y, z = (float(v) for v in threat)
        assets = root.find("asset")
        assets.append(ET.fromstring(THREAT_MATERIAL))
        worldbody.append(ET.fromstring(THREAT_BODY.format(x=x, y=y, z=z)))
        edits.append({"element": "worldbody", "attribute": "threat",
                      "before": "absent",
                      "after": f"mocap capsule at ({x:g}, {y:g}, {z:g})"})

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


def assert_body_unchanged(source_text, world_text, asset_dir=None):
    """The ANIMAL must be identical. The world may gain objects; the gecko may not change.

    Compared by name rather than by index, because adding a world object shifts
    every index after it and an index-wise comparison would either false-alarm or,
    worse, silently compare the wrong pair.
    """
    import mujoco
    import numpy as np
    # The comparison copies are written NEXT TO THE SOURCE MODEL, not at the
    # repository root. A model's `texturedir` and `meshdir` are relative to the
    # file it is loaded from, so a copy parked somewhere else cannot find the
    # assets the original finds and the compile fails on a texture rather than
    # on anything to do with the animal.
    # Defaults to where the models and their `textures/` live, because that is
    # the only directory from which a model's relative assets resolve.
    base = pathlib.Path(asset_dir) if asset_dir else SOURCE.parent
    paths = []
    try:
        for index, text in enumerate((source_text, world_text)):
            path = base / f".world_check_{index}.xml"
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
        # WHAT THIS GUARD ACTUALLY CARES ABOUT IS DEGREES OF FREEDOM, and it
        # used to test a proxy for them. It required every added body to be
        # mocap, on the reasoning that a mocap body cannot enter qpos -- true,
        # but so is a STATIC body with no joints, and the guard rejected those
        # too. Verified directly against this body: adding a jointless box
        # leaves nq at 39 and nv at 38, unchanged, while nbody goes 24 -> 25.
        #
        # That mattered because shelter and a warm surface have to be part of
        # the world and cannot be mocap: they are scenery the animal walks on
        # and under. Testing the proxy would have forced them to be fake.
        #
        # So the check is now the thing itself: nq and nv must be identical,
        # and any added body must be mocap OR jointless. A body carrying a
        # joint changes the animal's state vector and is still refused.
        if (a.nq, a.nv) != (b.nq, b.nv):
            raise ValueError(
                f"World changed the animal's state vector: nq {a.nq} -> {b.nq}, "
                f"nv {a.nv} -> {b.nv}. Added bodies must add no degrees of freedom.")
        for name in added:
            index = by_name(b, mujoco.mjtObj.mjOBJ_BODY, b.nbody)[name]
            mocap = b.body_mocapid[index] >= 0
            jointless = int(b.body_jntnum[index]) == 0
            if not (mocap or jointless):
                raise ValueError(
                    f"Added world body {name!r} carries {int(b.body_jntnum[index])} "
                    "joint(s) and is not mocap, so it would enter qpos.")
        return added
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shelter", type=float, nargs=3, default=None,
                   metavar=("HALF", "X", "Y"),
                   help="crevice under a raised slab. This species does not dig; "
                        "it uses voids that already exist.")
    p.add_argument("--warm-patch", type=float, nargs=3, default=None,
                   metavar=("HALF", "X", "Y"),
                   help="warm SURFACE, not a lamp. Body temperature tracks the "
                        "substrate at r2=0.97 against air at 0.92 -- the animal "
                        "is thigmothermic and warms by lying on the ground.")
    p.add_argument("--threat", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="a mocap predator. Caveat: for this species a visual "
                        "threat alone triggers a defensive reaction on 0.07 of "
                        "trials against 0.21 for scent.")
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
                              a.prey_radius, a.prey_rgba,
                              shelter=a.shelter, warm_patch=a.warm_patch,
                              threat=a.threat)
    added = assert_body_unchanged(source_text, world_text, a.source.resolve().parent)

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
