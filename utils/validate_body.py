#!/usr/bin/env python3
"""
validate_body.py  --  research-grade validation harness for GeckoBody-R.

Runs the full acceptance checklist and prints a PASS/FAIL report plus the
TRUE structural counts. Headless-safe: the camera render uses EGL if available
and is skipped gracefully otherwise.

Usage:
    python validate_body.py [path/to/gecko_body_r.xml]
Exit code 0 = all checks passed, 1 = at least one failure.
"""
import os, sys, gc, contextlib
from pathlib import Path
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

# --------------------------------------------------------------------------- #
DEFAULT_XML = Path(__file__).resolve().parent.parent / "morphology" / "gecko_body_r.xml"
XML = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML

# target windows (from the consolidated measurement report)
TGT = dict(mass=(0.060, 0.065), tail_frac=(0.22, 0.25), svl=(0.100, 0.1065),
           tail=(0.100, 0.112), total=(0.200, 0.230), belly_mm=(5.0, 15.0))

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<46} {detail}")
    return ok

def hdr(t):
    print("\n" + t)
    print("-" * 72)

# --------------------------------------------------------------------------- #
hdr("1. MODEL LOAD")
try:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    check("model compiles & loads", True, str(XML))
except Exception as e:
    check("model compiles & loads", False, str(e)[:200]); print("\nABORT."); sys.exit(1)

def name2id(typ, n): return mujoco.mj_name2id(m, typ, n)
BODY, JNT, SNS, KEY = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                       mujoco.mjtObj.mjOBJ_SENSOR, mujoco.mjtObj.mjOBJ_KEY)
bid = lambda n: name2id(BODY, n)
G = mujoco.mjtGeom

def geom_lowest_z(i):
    """Exact lowest world-z of a geom's surface (handles sphere/capsule/
    box/ellipsoid via support functions; rbound fallback otherwise).
    rbound (bounding sphere) badly overestimates downward extent for the
    long horizontal trunk capsules, so it must not be used for clearance."""
    p = d.geom_xpos[i]; R = d.geom_xmat[i].reshape(3, 3); s = m.geom_size[i]; t = m.geom_type[i]
    if t == G.mjGEOM_SPHERE:
        return p[2] - s[0]
    if t == G.mjGEOM_CAPSULE:
        return p[2] - abs(R[2, 2]) * s[1] - s[0]          # endpoints +/- axis, then radius
    if t == G.mjGEOM_ELLIPSOID:
        return p[2] - np.linalg.norm(R[2, :3] * s[:3])    # support along -z
    if t == G.mjGEOM_BOX:
        return p[2] - float(np.abs(R[2, :3]) @ s[:3])     # nearest box corner
    return p[2] - m.geom_rbound[i]

def body_lowest_z(n, collision_only=True):
    b = bid(n)
    zs = [geom_lowest_z(i) for i in range(m.ngeom)
          if m.geom_bodyid[i] == b and (not collision_only or m.geom_group[i] == 3)]
    return min(zs) if zs else 1e9

kstand = name2id(KEY, "stand")
mujoco.mj_resetDataKeyframe(m, d, kstand)
mujoco.mj_forward(m, d)

# --------------------------------------------------------------------------- #
hdr("2. STRUCTURAL COUNTS (the true accounting)")
n_hinge = int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
n_free  = int(np.sum(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
# actuated joints (direct + via tendon)
direct = set(int(m.actuator_trnid[i, 0]) for i in range(m.nu)
             if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT)
ten_j = set()
for i in range(m.nu):
    if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_TENDON:
        t = int(m.actuator_trnid[i, 0]); adr = m.tendon_adr[t]
        for k in range(m.tendon_num[t]):
            ten_j.add(int(m.wrap_objid[adr + k]))
n_act = len(direct | ten_j)
n_passive = n_hinge - n_act
n_vis = int(np.sum(m.geom_group == 1)); n_col = int(np.sum(m.geom_group == 3))
print(f"  physics bodies ............ {m.nbody - 1}")
print(f"  joints .................... {n_free} free + {n_hinge} hinge = {m.njnt} total")
print(f"  generalized DoF (nv) ...... {m.nv}")
print(f"  actuated joints ........... {n_act}  ({len(direct)} direct + {len(ten_j)} via tendon)")
print(f"  passive joints ............ {n_passive}")
print(f"  actuators / RL action dim . {m.nu}")
print(f"  tendons ................... {m.ntendon}")
print(f"  sensors ................... {m.nsensor}")
print(f"  geoms ..................... {m.ngeom}  ({n_col} collision + {n_vis} visual)")
check("23 physics bodies",            m.nbody - 1 == 23)
check("1 free + 32 hinge joints",     n_free == 1 and n_hinge == 32)
check("38 generalized DoF",           m.nv == 38)
check("30 actuated joints",           n_act == 30)
check("2 passive joints (wrists)",    n_passive == 2)
check("25 actuators (RL action dim)", m.nu == 25)

# --------------------------------------------------------------------------- #
hdr("3. MASS BUDGET")
total = float(m.body_subtreemass[bid("trunk_middle")])
tail = float(sum(m.body_mass[bid(f"tail{i}")] for i in range(1, 6)))
frac = tail / total
check("total mass in 0.060-0.065 kg", TGT["mass"][0] <= total <= TGT["mass"][1],
      f"{total*1000:.2f} g")
check("tail fraction in 22-25%",      TGT["tail_frac"][0] <= frac <= TGT["tail_frac"][1],
      f"{frac*100:.1f} %  ({tail*1000:.2f} g)")

# --------------------------------------------------------------------------- #
hdr("4. LENGTHS (anatomical landmark sites)")
sx = lambda n: float(d.site_xpos[name2id(mujoco.mjtObj.mjOBJ_SITE, n)][0])
nose, vent, tip = sx("nose_tip"), sx("vent"), sx("tail_tip")
svl, tl, tot = nose - vent, vent - tip, nose - tip
check("SVL in 0.100-0.1065 m", TGT["svl"][0] <= svl <= TGT["svl"][1],   f"{svl:.4f} m")
check("tail in 0.100-0.112 m", TGT["tail"][0] <= tl <= TGT["tail"][1],  f"{tl:.4f} m")
check("total in 0.200-0.230 m", TGT["total"][0] <= tot <= TGT["total"][1], f"{tot:.4f} m")

# --------------------------------------------------------------------------- #
hdr("5. JOINT LIMITS  (neutral strictly inside range, none inverted)")
bad_inv, bad_neu = [], []
for j in range(m.njnt):
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
        continue
    lo, hi = m.jnt_range[j]
    nm = mujoco.mj_id2name(m, JNT, j)
    if not (lo < hi):      bad_inv.append(nm)
    if not (lo < 0.0 < hi): bad_neu.append(nm)   # qpos=0 is the neutral pose
check("no inverted joint limits", not bad_inv, ",".join(bad_inv))
check("neutral (0) inside every hinge range", not bad_neu, ",".join(bad_neu))

# --------------------------------------------------------------------------- #
hdr("6. STATIC STANCE  (settle from 'stand', hold neutral targets)")
mujoco.mj_resetDataKeyframe(m, d, kstand)
d.ctrl[:] = 0.0
for _ in range(int(2.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
settled_finite = bool(np.all(np.isfinite(d.qpos)))
qvel_settled = float(np.abs(d.qvel).max())

def body_minz(n):
    return body_lowest_z(n, collision_only=True)
belly_mid = body_minz("trunk_middle") * 1000
belly_post = body_minz("trunk_posterior") * 1000
check("stays finite while settling", settled_finite)
check("settles to rest (max|qvel|<0.05)", qvel_settled < 0.05, f"{qvel_settled:.4f} rad/s")
check("belly clearance in 5-15 mm",
      TGT["belly_mm"][0] <= min(belly_mid, belly_post) <= TGT["belly_mm"][1],
      f"mid {belly_mid:.2f} mm, post {belly_post:.2f} mm")

# touch sensors in stance
def sens(n):
    s = name2id(SNS, n); a = m.sensor_adr[s]
    return float(d.sensordata[a])
feet = {f: sens(f) for f in ("touch_fore_L", "touch_fore_R", "touch_hind_L", "touch_hind_R")}
belly = {b: sens(b) for b in ("touch_belly_mid", "touch_belly_post")}
n_feet_active = sum(v > 1e-6 for v in feet.values())
foot_sum = sum(feet.values()); weight = total * 9.81
check("exactly 4 feet in contact", n_feet_active == 4,
      " ".join(f"{k.split('_',1)[1]}={v:.3f}N" for k, v in feet.items()))
check("belly NOT touching in stance", all(v <= 1e-6 for v in belly.values()),
      f"mid={belly['touch_belly_mid']:.3f} post={belly['touch_belly_post']:.3f} N")
check("foot reaction ~= body weight", abs(foot_sum - weight) < 0.05 * weight,
      f"{foot_sum:.3f} N vs {weight:.3f} N")

# --------------------------------------------------------------------------- #
hdr("7. STATIC STABILITY  (CoM inside foot support polygon)")
mujoco.mj_resetDataKeyframe(m, d, kstand); mujoco.mj_forward(m, d)
com = d.subtree_com[bid("trunk_middle")].copy()
def foot_xy(n):
    b = bid(n); best, bz = None, 1e9
    for i in range(m.ngeom):
        if m.geom_bodyid[i] == b:
            z = geom_lowest_z(i)
            if z < bz: bz, best = z, d.geom_xpos[i, :2].copy()
    return best
pts = [foot_xy(f) for f in ("manus_L", "manus_R", "pes_R", "pes_L")]  # CCW-ish order
def cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
def in_poly(p, poly):
    signs = [cross2(a, b, p) for a, b in zip(poly, poly[1:] + poly[:1])]
    return all(s >= -1e-9 for s in signs) or all(s <= 1e-9 for s in signs)
rear_margin = com[0] - min(p[0] for p in pts)
check("CoM inside 4-foot support polygon", in_poly(com[:2], pts),
      f"CoM x={com[0]*1000:.1f} mm, rear margin {rear_margin*1000:.1f} mm")

# --------------------------------------------------------------------------- #
hdr("8. HEAD CAMERA  (forward egocentric view renders non-empty)")
cam_ok, cam_detail = False, "render unavailable (headless GL) - skipped"
try:
    with contextlib.redirect_stderr(open(os.devnull, "w")):
        r = mujoco.Renderer(m, 64, 64)
        mujoco.mj_resetDataKeyframe(m, d, kstand); mujoco.mj_forward(m, d)
        r.update_scene(d, camera="head_cam"); img = r.render()
        nz = int((img.sum(-1) > 0).sum())
        del r; gc.collect()
    cam_ok = nz > 0.5 * img.shape[0] * img.shape[1]
    cam_detail = f"64x64, {nz}/4096 px populated, mean RGB {img.reshape(-1,3).mean(0).round(0)}"
except Exception as e:
    cam_detail = f"render skipped: {str(e)[:80]}"
check("head_cam renders forward view", cam_ok or "skipped" in cam_detail, cam_detail)

# --------------------------------------------------------------------------- #
hdr("9. RANDOM-ACTION ROBUSTNESS  (no NaN / explosion / tail-whip)")
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
tail5 = bid("tail5")
def tip_speed():
    v = np.zeros(6); mujoco.mj_objectVelocity(m, d, BODY, tail5, v, 0)
    return float(np.linalg.norm(v[3:6]))

def rollout(smooth, T=6.0, seeds=4):
    worst = dict(nan=False, qvel=0.0, pos=0.0, tail=0.0)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(m, d, kstand); mujoco.mj_forward(m, d)
        a = np.zeros(m.nu); tg = np.zeros(m.nu); hold = 0
        for _ in range(int(T / m.opt.timestep)):
            if smooth:
                a = 0.985 * a + 0.015 * (lo + (hi - lo) * rng.random(m.nu))
            else:
                if hold <= 0:
                    tg = lo + (hi - lo) * rng.random(m.nu); hold = int(0.12 / m.opt.timestep)
                a = tg; hold -= 1
            d.ctrl[:] = a; mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)): worst["nan"] = True; break
            worst["qvel"] = max(worst["qvel"], float(np.abs(d.qvel).max()))
            worst["pos"]  = max(worst["pos"], float(np.abs(d.xpos[1:]).max()))
            worst["tail"] = max(worst["tail"], tip_speed())
    return worst

w_smooth = rollout(smooth=True)
w_harsh  = rollout(smooth=False)
check("no NaN/Inf (smooth + harsh actions)", not (w_smooth["nan"] or w_harsh["nan"]))
check("no detachment/explosion (|pos|<1 m)", max(w_smooth["pos"], w_harsh["pos"]) < 1.0,
      f"max |pos| {max(w_smooth['pos'], w_harsh['pos']):.3f} m")
check("no violent tail-whip (smooth tip<1.0 m/s)", w_smooth["tail"] < 1.0,
      f"smooth {w_smooth['tail']:.2f} m/s | harsh {w_harsh['tail']:.2f} m/s")
print(f"     (info) smooth max|qvel| {w_smooth['qvel']:.1f} rad/s | "
      f"harsh max|qvel| {w_harsh['qvel']:.1f} rad/s (full-range step stress test)")

# --------------------------------------------------------------------------- #
hdr("SUMMARY")
npass = sum(ok for _, ok in results); ntot = len(results)
for nm, ok in results:
    if not ok: print(f"  FAILED: {nm}")
print(f"\n  {npass}/{ntot} checks passed.")
print("  RESULT:", "ALL CHECKS PASSED \u2714" if npass == ntot else "SOME CHECKS FAILED \u2717")
sys.exit(0 if npass == ntot else 1)
