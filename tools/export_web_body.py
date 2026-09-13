"""Strip the certified body down to the one the browser runs.

WHY THIS FILE EXISTS AT ALL. `morphology/gecko_body_web.xml` has been in the
repository, and shipped to the public page, with no generator. It was produced
by hand once and then never rebuilt, so it drifted: the committed web body
records source `a9d423ed...` while the committed `gecko_body_lab_v2.xml` records
`9fd73bad...`. Two different source bodies, while `site/live.js` tells every
visitor the browser runs "the certified body". An artifact nobody can regenerate
cannot be kept honest, and this one wasn't.

WHAT THE STRIP REMOVES, and why each is safe:
  * every <texture> and <material>            -- appearance only
  * every <mesh> asset and the <skin> element -- the linear-blend skin; visual
  * every geom with type="mesh"               -- all are class="visual", which
                                                 the body's own default sets to
                                                 contype=0 conaffinity=0 mass=0,
                                                 so they carry no contact and no
                                                 inertia
  * every material="..." attribute left behind on a surviving geom

WHAT IT MUST NOT CHANGE. Not one body, joint, actuator, collision geom, mass,
inertia or keyframe. `--verify` asserts that directly: it compiles both bodies,
checks nq/nv/nu/nbody/njnt and every mass and inertia, then steps both 400 times
under identical controls and reports the largest joint difference. The original
hand-made strip claimed "maximum joint difference exactly 0.0" and that claim is
now something anyone can re-run instead of having to believe.

Usage:  python tools/export_web_body.py [--verify]
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCE = REPO / "morphology" / "gecko_body_lab_v2.xml"
OUTPUT = REPO / "morphology" / "gecko_body_web.xml"

#: A line is dropped whole if it opens one of these. The source writes each on
#: its own line, which is checked below rather than assumed.
DROP = re.compile(r'^\s*<(texture|material|mesh)\s')
#: <skin> lives inside a <deformable> wrapper, so dropping the skin line alone
#: would leave an orphaned open tag and an uncompilable body. The whole block
#: goes, <bone> children and all -- linear-blend skinning is drawn, never
#: simulated.
DEFORMABLE_OPEN = re.compile(r'^\s*<deformable>\s*$')
DEFORMABLE_CLOSE = re.compile(r'^\s*</deformable>\s*$')
#: Visual mesh geoms. The body's `visual` default is contype=0 conaffinity=0
#: mass=0, so removing them cannot change contact or inertia -- `--verify`
#: proves it rather than trusting the class name.
DROP_GEOM = re.compile(r'^\s*<geom\b[^>]*\btype="mesh"')
#: Left-over references to materials that no longer exist.
MATERIAL_ATTR = re.compile(r'\s+material="[^"]*"')


def strip(text: str) -> str:
    out, dropped, in_deformable = [], 0, False
    for line in text.split("\n"):
        if DEFORMABLE_OPEN.match(line):
            in_deformable = True
        if in_deformable:
            dropped += 1
            if DEFORMABLE_CLOSE.match(line):
                in_deformable = False
            continue
        if DROP.match(line) or DROP_GEOM.match(line):
            # A multi-line element would leave its tail behind, so refuse
            # rather than silently emit a broken file.
            if not line.rstrip().endswith(("/>", ">")):
                raise ValueError(f"Expected a single-line element, got: {line[:80]}")
            dropped += 1
            continue
        out.append(MATERIAL_ATTR.sub("", line))
    if in_deformable:
        raise ValueError("Unclosed <deformable>; refusing to emit a broken body.")
    if not dropped:
        raise ValueError("Nothing was stripped; the source is not the skinned body.")
    return "\n".join(out)


def verify():
    import numpy as np
    import mujoco

    full = mujoco.MjModel.from_xml_path(str(SOURCE))
    web = mujoco.MjModel.from_xml_path(str(OUTPUT))
    for field in ("nq", "nv", "nu", "nbody", "njnt", "nkey"):
        a, b = getattr(full, field), getattr(web, field)
        if a != b:
            raise SystemExit(f"{field} differs: certified {a}, web {b}")
    if not np.allclose(full.body_mass, web.body_mass, atol=0, rtol=0):
        raise SystemExit("body masses differ")
    if not np.allclose(full.body_inertia, web.body_inertia, atol=0, rtol=0):
        raise SystemExit("body inertias differ")
    if not np.allclose(full.key_qpos, web.key_qpos, atol=0, rtol=0):
        raise SystemExit("keyframes differ")

    # 400 steps under identical controls, the original claim, re-measurable.
    da, db = mujoco.MjData(full), mujoco.MjData(web)
    for data, model in ((da, full), (db, web)):
        mujoco.mj_resetDataKeyframe(model, data,
                                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    rng = np.random.default_rng(20260913)
    worst = 0.
    for _ in range(400):
        ctrl = rng.uniform(-.15, .15, full.nu)
        da.ctrl[:] = ctrl
        db.ctrl[:] = ctrl
        mujoco.mj_step(full, da)
        mujoco.mj_step(web, db)
        worst = max(worst, float(np.max(np.abs(da.qpos - db.qpos))))
    print(f"  nq/nv/nu/nbody/njnt/nkey identical; masses, inertias and keyframes identical")
    print(f"  400 steps under identical controls: maximum joint difference {worst:.3e}")
    if worst != 0.:
        raise SystemExit("THE STRIP IS NOT PHYSICS-NEUTRAL; do not ship this body.")
    print("  physics-neutral, exactly.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Compile both bodies and prove the strip changed no physics")
    args = parser.parse_args(argv)
    raw = SOURCE.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    stripped = strip(raw.replace("\r\n", "\n"))
    OUTPUT.write_bytes(stripped.replace("\n", newline).encode("utf-8"))
    print(f"wrote {OUTPUT.relative_to(REPO)} from {SOURCE.relative_to(REPO)}")
    if args.verify:
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
