"""Rebuild the animal, end to end, in the one order that works.

WHY THIS EXISTS. The body is produced by six steps that must run in sequence,
and running a subset leaves the repository in a state that COMPILES, passes its
gates, and is still wrong:

  * `build_lab_morphology.py --fit-com` writes the v2 body ONLY. The v1 body is
    written by the same script WITHOUT that flag. Rebuilding v2 and forgetting
    v1 leaves v1 carrying a stale source hash, and three reproducibility tests
    fail with a message about a missing marker that says nothing about the real
    cause. This has now happened twice.
  * The mesh, eye and skin generators all read the SOURCE body, so they must run
    before it is rebuilt, and the skin must run after the mesh because it reuses
    the mesh generator's cross-sections.
  * `wire_mesh_skin.py` is what actually points the body at the generated files,
    so it runs after all three generators and before any build.

Each step prints what it did. Nothing here is new logic; it is the order.

Usage:  python tools/rebuild_body.py [--skip-skin]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable

WORLD_ARGS = [
    "--prey-radius", "0.009", "--texrepeat", "50", "--fovy", "70",
    "--offwidth", "1280", "--offheight", "960",
]


def run(label, args, grep=None):
    print(f"\n=== {label} ===", flush=True)
    # Running a script BY PATH puts that script's own directory on sys.path, not
    # the repository root -- so `utils/build_lab_morphology.py` died on
    # `from common.morphology_audit import ...` and the pipeline never reached
    # the steps after it. The one place this script is supposed to be
    # authoritative is order, and it could not get there. PYTHONPATH fixes every
    # step at once rather than rewriting each invocation.
    env = {**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([PY] + args, cwd=REPO, capture_output=True, text=True, env=env)
    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        print(out[-2500:])
        raise SystemExit(f"FAILED: {label}")
    for line in out.splitlines():
        if grep is None or any(k in line for k in grep):
            print("  " + line.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-skin", action="store_true",
                    help="skip the slow skin bind; use when only colour changed")
    args = ap.parse_args()

    run("surface meshes", ["tools/make_gecko_mesh.py"], grep=["meshes,"])
    run("eyes and eyelids", ["tools/make_gecko_eye.py"], grep=["eye_", "lid_", "texture:"])
    if not args.skip_skin:
        run("continuous skin", ["tools/make_gecko_skin.py"],
            grep=["surface:", "weights:", "texture:", "wrote the skin"])
    run("wire into the source body", ["tools/wire_mesh_skin.py"])

    # BOTH lab bodies. v1 is not a leftover: the reproducibility tests compare
    # against it, and it goes stale the moment the source changes.
    run("lab body v1", ["utils/build_lab_morphology.py"], grep=["gates passed"])
    run("lab body v2 (mass-calibrated)",
        ["utils/build_lab_morphology.py", "--fit-com"],
        grep=["gates passed", "FAIL"])
    run("world", ["utils/build_world.py",
                  "--source", "morphology/gecko_body_lab_v2.xml",
                  "--output", "morphology/gecko_world_v1.xml"] + WORLD_ARGS,
        grep=["__never__"])
    print("  world written")

    furnished = REPO / "morphology" / "gecko_world_furnished_v1.xml"
    if furnished.exists():
        run("furnished world (shelter, warm patch, threat)",
            ["utils/build_world.py",
             "--source", "morphology/gecko_body_lab_v2.xml",
             "--output", str(furnished.relative_to(REPO)),
             "--shelter", "0.055", "0.28", "0.16",
             "--warm-patch", "0.060", "-0.22", "0.26",
             "--threat", "0.45", "-0.30", "0.05"] + WORLD_ARGS,
            grep=["__never__"])
        print("  furnished world written")

    # The habitat was NOT in this list and it drifted for it: the committed copy
    # was generated from source db650f6b while the committed body records
    # 9fd73bad, i.e. a body two generations back, before the animal had textures
    # at all. It is not decorative -- `tools/export_web_brain.py` lifts the
    # shelter and warm-patch geometry out of it for the public page, and
    # `tools/everything_video.py` films in it. Its flags differ from the others
    # (its own manifest records no offwidth/offheight), so they are spelled out.
    habitat = REPO / "morphology" / "gecko_habitat_v1.xml"
    if habitat.exists():
        run("habitat (the world the page and the films use)",
            ["utils/build_world.py",
             "--source", "morphology/gecko_body_lab_v2.xml",
             "--output", str(habitat.relative_to(REPO)),
             "--shelter", "0.055", "-0.3", "0.16",
             "--warm-patch", "0.070", "0.28", "-0.2",
             "--threat", "0", "0.45", "0.3",
             "--prey-radius", "0.009", "--texrepeat", "50", "--fovy", "70"],
            grep=["__never__"])
        print("  habitat written")

    import mujoco
    for name in ("gecko_world_v1.xml", "gecko_world_furnished_v1.xml"):
        path = REPO / "morphology" / name
        if not path.exists():
            continue
        model = mujoco.MjModel.from_xml_path(str(path))
        print(f"\n{name}: nskin {model.nskin}, skin vertices {model.nskinvert}, "
              f"bones {model.nskinbone}, geoms {model.ngeom}")


if __name__ == "__main__":
    main()
