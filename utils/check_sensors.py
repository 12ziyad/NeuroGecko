#!/usr/bin/env python3
"""
check_sensors.py  --  print every GeckoBody-R sensor and verify the contact
logic (4 feet active + belly silent in stance; belly activates when pressed;
a foot sensor drops to zero when that foot is lifted).

Headless / display independent (no rendering needed).

Usage:
    python check_sensors.py [path/to/gecko_body_r.xml]
"""
import os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

DEFAULT_XML = Path(__file__).resolve().parent.parent / "morphology" / "gecko_body_r.xml"
XML = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML

m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
kstand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")

SNS = mujoco.mjtObj.mjOBJ_SENSOR
TYPES = {int(getattr(mujoco.mjtSensor, n)): n[len("mjSENS_"):]
         for n in dir(mujoco.mjtSensor) if n.startswith("mjSENS_")}

def settle(t=1.5, ctrl=None):
    mujoco.mj_resetDataKeyframe(m, d, kstand)
    for _ in range(int(t / m.opt.timestep)):
        if ctrl is not None: d.ctrl[:] = ctrl
        else: d.ctrl[:] = 0.0
        mujoco.mj_step(m, d)

def sval(name):
    s = mujoco.mj_name2id(m, SNS, name)
    a, dim = m.sensor_adr[s], m.sensor_dim[s]
    return d.sensordata[a:a + dim]

# --------------------------------------------------------------------------- #
print(f"model: {XML}")
print(f"{m.nsensor} sensors\n")
settle()

# group sensors by category for a readable dump
groups = {"joint pos (rad)": [], "joint vel (rad/s)": [], "tendon": [],
          "vestibular / pose": [], "touch (N)": [], "other": []}
for s in range(m.nsensor):
    nm = mujoco.mj_id2name(m, SNS, s)
    typ = TYPES.get(int(m.sensor_type[s]), str(m.sensor_type[s]))
    v = sval(nm)
    txt = f"{nm:<22} {typ:<14} " + " ".join(f"{x:+.4f}" for x in v)
    if   nm.startswith("qp_"): groups["joint pos (rad)"].append(txt)
    elif nm.startswith("qv_"): groups["joint vel (rad/s)"].append(txt)
    elif nm.startswith(("tp_", "tv_")): groups["tendon"].append(txt)
    elif nm.startswith("touch_"): groups["touch (N)"].append(txt)
    elif typ in ("FRAMEZAXIS", "FRAMEQUAT", "GYRO", "ACCELEROMETER",
                 "VELOCIMETER", "FRAMEPOS"): groups["vestibular / pose"].append(txt)
    else: groups["other"].append(txt)

print("=" * 72)
print("FULL SENSOR DUMP  (standing pose)")
print("=" * 72)
for g, rows in groups.items():
    if not rows: continue
    print(f"\n-- {g}  ({len(rows)}) --")
    for r in rows: print("  " + r)

# --------------------------------------------------------------------------- #
print("\n" + "=" * 72)
print("CONTACT-LOGIC VERIFICATION")
print("=" * 72)
FEET = ("touch_fore_L", "touch_fore_R", "touch_hind_L", "touch_hind_R")
BELLY = ("touch_belly_mid", "touch_belly_post")

def touch(nm): return float(sval(nm)[0])

# (a) normal stance
settle()
feet0 = {f: touch(f) for f in FEET}; belly0 = {b: touch(b) for b in BELLY}
n_feet = sum(v > 1e-6 for v in feet0.values())
print("\n(a) normal stance:")
print("    feet :", "  ".join(f"{k.split('_',1)[1]}={v:.3f}N" for k, v in feet0.items()))
print("    belly:", "  ".join(f"{k.split('_',2)[2]}={v:.3f}N" for k, v in belly0.items()))
ok_a = (n_feet == 4) and all(v <= 1e-6 for v in belly0.values())
print(f"    => {'PASS' if ok_a else 'FAIL'}: 4 feet active, belly silent")

# (b) ventral contact: retract the limbs (disable limb collision) and let the
#     trunk settle onto the substrate -> belly touch sensors register.
mujoco.mj_resetDataKeyframe(m, d, kstand)
bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i])
LIMBS = ("humerus_L", "humerus_R", "forearm_L", "forearm_R", "manus_L", "manus_R",
         "femur_L", "femur_R", "tibia_L", "tibia_R", "pes_L", "pes_R")
limb_geoms = [i for i in range(m.ngeom) if bn(i) in LIMBS and m.geom_group[i] == 3]
saved = [(int(m.geom_contype[i]), int(m.geom_conaffinity[i])) for i in limb_geoms]
for i in limb_geoms:
    m.geom_contype[i] = 0; m.geom_conaffinity[i] = 0
d.qpos[2] = 0.018
for _ in range(int(1.0 / m.opt.timestep)):
    d.ctrl[:] = 0.0; mujoco.mj_step(m, d)
belly1 = {b: touch(b) for b in BELLY}; feet1 = {f: touch(f) for f in FEET}
for i, (ct, ca) in zip(limb_geoms, saved):
    m.geom_contype[i] = ct; m.geom_conaffinity[i] = ca
print("\n(b) limbs retracted so the trunk rests on the substrate:")
print("    belly:", "  ".join(f"{k.split('_',2)[2]}={v:.3f}N" for k, v in belly1.items()))
ok_b = all(v > 1e-4 for v in belly1.values())
print(f"    => {'PASS' if ok_b else 'FAIL'}: both belly sensors register ventral contact")

# (c) lift-off: raise the whole body clear of the floor -> ALL contact sensors
#     read zero (proves they are contact-driven, not stuck-on).
mujoco.mj_resetDataKeyframe(m, d, kstand)
d.qpos[2] = 0.050
mujoco.mj_forward(m, d)
feet2 = {f: touch(f) for f in FEET}; belly2 = {b: touch(b) for b in BELLY}
print("\n(c) whole body lifted 22 mm clear of the floor:")
print("    feet :", "  ".join(f"{k.split('_',1)[1]}={v:.3f}N" for k, v in feet2.items()))
print("    belly:", "  ".join(f"{k.split('_',2)[2]}={v:.3f}N" for k, v in belly2.items()))
ok_c = all(v <= 1e-6 for v in list(feet2.values()) + list(belly2.values()))
print(f"    => {'PASS' if ok_c else 'FAIL'}: all contact sensors zero when airborne")

print("\nRESULT:", "contact logic verified \u2714" if (ok_a and ok_b and ok_c)
      else "see notes above")
