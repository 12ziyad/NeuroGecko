"""Measured single-speed gait diagnostics, not a certificate of biological validity.

Protocol references: Jagnandan & Higham (2017), doi:10.1038/s41598-017-11484-7;
McElroy et al. (2008), doi:10.1242/jeb.015503. The former reports stance-phase
joint kinematics and some speed-residualized means, not raw MuJoCo hinge angles.
All anatomical angles below remain explicitly labelled landmark proxies.

Run a trusted local checkpoint (SB3 archives/normalizers can execute code):
    python realism_metrics.py --model models/v4_5b_speed_polish_1m/final.zip \
      --vecnormalize models/v4_5b_speed_polish_1m/vecnormalize.pkl \
      --episodes 20 --output runs/baseline/report.json --trace-dir runs/baseline/traces

The policy is frozen. We normalize observations with VecNormalize but step the
base environment directly: vector-env auto-reset would destroy terminal samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import os

import numpy as np

FEET = ("HL", "FL", "HR", "FR")
FOOT_SENSORS = ("touch_hind_L", "touch_fore_L", "touch_hind_R", "touch_fore_R")
FOOT_SITES = ("footzone_hind_L", "footzone_fore_L", "footzone_hind_R", "footzone_fore_R")
TOUCH_SENSORS = FOOT_SENSORS + ("touch_belly_mid", "touch_belly_post")
REPO = Path(__file__).resolve().parent


def numeric_summary(values):
    """Finite-value sample summary; missing is null, never an invented zero."""
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "mean": None, "sd": None, "sem": None, "min": None, "max": None}
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else None
    return {"n": int(len(a)), "mean": float(np.mean(a)), "sd": sd,
            "sem": None if sd is None else sd / np.sqrt(len(a)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def circular_summary(phases):
    a = np.asarray(phases, dtype=float)
    a = a[np.isfinite(a)] % 1.0
    if not len(a):
        return {"n": 0, "mean_cycle": None, "resultant_length": None}
    z = np.mean(np.exp(2j * np.pi * a))
    return {"n": int(len(a)), "mean_cycle": None if abs(z) < 1e-12 else float(np.angle(z) / (2*np.pi) % 1),
            "resultant_length": float(abs(z))}


def validate_times(time_s):
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 3 or not np.all(np.isfinite(t)):
        raise ValueError("Need at least three finite timestamps.")
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-9):
        raise ValueError("Timestamps must be strictly increasing and uniformly sampled.")
    return t, dt


def _runs(values):
    x = np.asarray(values, dtype=bool)
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    ends = np.r_[starts[1:], len(x)]
    return list(zip(starts.tolist(), ends.tolist()))


def debounce_contacts(contact, dt, minimum_s):
    """Replace short bounded runs by their neighbours, leaving censored edges.

    Both false and true runs are treated symmetrically. Each pass removes the
    shortest bounded run first; ties go to the earliest. This deterministic
    offline convention is reported because the paper does not specify one.
    Exactly minimum_s is retained. No initial contact is invented as touchdown.
    """
    if dt <= 0 or minimum_s < 0:
        raise ValueError("dt must be positive and minimum_s nonnegative.")
    x = np.asarray(contact, dtype=bool).copy()
    if x.ndim != 1:
        raise ValueError("Debounce accepts a one-dimensional contact signal.")
    while len(x):
        runs = _runs(x)
        short = [(b-a, a, b) for a, b in runs[1:-1]
                 if (b-a)*dt < minimum_s-1e-10]
        if not short:
            break
        _, a, b = min(short)
        x[a:b] = x[a-1]
    return x


def complete_strides(time_s, contact, settle_s=0):
    """Return observed touchdown-to-touchdown cycles with one intervening liftoff."""
    t, _ = validate_times(time_s)
    x = np.asarray(contact, dtype=bool)
    if x.shape != t.shape:
        raise ValueError("Contact and timestamps must have the same length.")
    touchdowns = np.flatnonzero(x[1:] & ~x[:-1]) + 1
    strides = []
    for start, end in zip(touchdowns[:-1], touchdowns[1:]):
        if t[start] < settle_s-1e-10:
            continue
        liftoffs = np.flatnonzero(~x[start+1:end+1] & x[start:end]) + start + 1
        if len(liftoffs) == 1:
            off = int(liftoffs[0])
            period = float(t[end]-t[start])
            strides.append({"start": int(start), "end": int(end), "off": off,
                            "period_s": period, "stance_s": float(t[off]-t[start]),
                            "duty_factor": float((t[off]-t[start])/period)})
    return strides


def _angle(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/den, -1, 1)))) if den > 1e-12 else np.nan


def _window_angles(array, windows):
    return {
        "minimum_deg": numeric_summary([np.min(array[a:b+1]) for a, b in windows]),
        "maximum_deg": numeric_summary([np.max(array[a:b+1]) for a, b in windows]),
        "excursion_deg": numeric_summary([np.ptp(array[a:b+1]) for a, b in windows]),
    }


def analyze_trace(trace, settle_s=None, debounce_s=None):
    """Analyze a JSON-compatible raw trace with explicit acquisition metadata.

    Required samples: time_s, trunk_position_m [N,3], foot_force_N [N,4].
    Optional qpos, sites, quaternion, proxy angles and work retain missing status.
    End samples describe the preceding step; finite-difference stance slip is
    included only when contact exists at both endpoints.
    """
    from common.provenance import parameter_value
    settle_s = parameter_value("evaluation_settle_s") if settle_s is None else settle_s
    debounce_s = parameter_value("contact_debounce_s") if debounce_s is None else debounce_s
    meta, s = trace["metadata"], trace["samples"]
    t, dt = validate_times(s["time_s"])
    if dt > .02000001:
        raise ValueError("Gait protocol requires at least 50 Hz acquisition; do not upsample coarse traces.")
    svl = float(meta["svl_m"])
    threshold = float(meta["contact_threshold_N"])
    if svl <= 0 or threshold < 0 or not np.isfinite([svl, threshold]).all():
        raise ValueError("SVL must be positive; contact threshold finite and nonnegative.")
    if settle_s < t[0] or settle_s >= t[-1]:
        raise ValueError("Settling cutoff must leave an observation interval.")
    xyz = np.asarray(s["trunk_position_m"], dtype=float)
    forces = np.asarray(s["foot_force_N"], dtype=float)
    if xyz.shape != (len(t), 3) or forces.shape != (len(t), 4):
        raise ValueError("Expected trunk[N,3] and foot forces[N,4].")
    if not np.isfinite(xyz).all() or not np.isfinite(forces).all():
        raise ValueError("Nonfinite positions or contacts cannot be scored.")
    keep = t >= settle_s-1e-10
    interval_keep = keep[:-1] & keep[1:]
    contacts = np.column_stack([debounce_contacts(forces[:, j] > threshold, dt, debounce_s)
                               for j in range(4)])
    strides = {f: complete_strides(t, contacts[:, j], settle_s) for j, f in enumerate(FEET)}
    planar_delta = np.diff(xyz[:, :2], axis=0)
    distance = float(np.sum(np.linalg.norm(planar_delta[interval_keep], axis=1)))
    first = int(np.flatnonzero(keep)[0])
    elapsed = float(t[-1]-t[first])
    result = {"status": "measured", "observed_duration_s": elapsed, "sample_rate_hz": 1/dt,
              "settle_s": settle_s, "contact_debounce_s": debounce_s,
              "foot_order": list(FEET), "distance_path_m": distance,
              "stride_length_definition": "Planar net displacement of trunk_middle between observed same-foot touchdowns; not a speed-residualized laboratory mean.",
              "event_time_resolution_s": dt,
              "mean_path_speed_m_s": distance/elapsed,
              "mean_path_speed_svl_s": distance/elapsed/svl,
              "net_displacement_m": float(np.linalg.norm(xyz[-1, :2]-xyz[first, :2])),
              "limbs": {}, "joint_angles": {}, "anatomical_proxies": {},
              "limitations": [
                  "Single-speed calibration, not statistical or kinematic validation.",
                  "Touch threshold is an engineering load detector, not tactile receptor sensitivity.",
                  "Foot-site velocity includes pad rotation/rolling; it is not exact contact-point slip.",
                  "Joint-angle proxies are not yet validated against the Jagnandan landmark convention.",
                  "Biological SEM and raw/residualized multi-lab values are not universal tolerances.",
                  "Temperatures label the literature validity envelope; this model has no thermal dynamics.",
              ],
              "deferred": {name: "requires descending drive and a matched multi-speed protocol"
                           for name in ("frequency_speed_regression", "stride_length_speed_regression",
                                        "tail_amplitude_speed_regression", "tail_height_speed_regression")}}
    foot_xyz = np.asarray(s["foot_position_m"], dtype=float) if "foot_position_m" in s else None
    for j, foot in enumerate(FEET):
        cyc = strides[foot]
        periods = [c["period_s"] for c in cyc]
        length = [float(np.linalg.norm(xyz[c["end"], :2]-xyz[c["start"], :2]))/svl for c in cyc]
        limb = {"complete_strides": len(cyc), "duty_factor": numeric_summary([c["duty_factor"] for c in cyc]),
                "stance_duration_s": numeric_summary([c["stance_s"] for c in cyc]),
                "stride_period_s": numeric_summary(periods),
                "stride_frequency_hz": numeric_summary([1/p for p in periods]),
                "stride_length_svl": numeric_summary(length),
                "contact_time_fraction_including_censored": float(np.mean(contacts[keep, j])),
                "stride_period_cv": float(np.std(periods, ddof=1)/np.mean(periods)) if len(periods)>1 else None,
                "stride_windows": cyc,
                "boundary_policy": "Only observed touchdown-to-touchdown cycles wholly after settling."}
        if foot_xyz is not None:
            if foot_xyz.shape != (len(t), 4, 3):
                raise ValueError("Expected foot_position_m[N,4,3].")
            speed = np.linalg.norm(np.diff(foot_xyz[:, j, :2], axis=0), axis=1)/dt
            stance = contacts[:-1, j] & contacts[1:, j] & interval_keep
            limb["stance_foot_site_speed_m_s"] = numeric_summary(speed[stance])
            limb["stance_foot_site_travel_m"] = float(np.sum(speed[stance])*dt)
        result["limbs"][foot] = limb
    touchdown = {f: np.flatnonzero(contacts[1:, j] & ~contacts[:-1, j])+1 for j, f in enumerate(FEET)}
    phase = {}
    for hind, fore in (("HL", "FL"), ("HR", "FR")):
        vals, diagonal = [], []
        opposite = "FR" if hind == "HL" else "FL"
        for c in strides[hind]:
            indices = touchdown[fore][(touchdown[fore] >= c["start"]) & (touchdown[fore] < c["end"])]
            if len(indices) == 1:
                vals.append((t[indices[0]]-t[c["start"]])/c["period_s"])
            indices = touchdown[opposite][(touchdown[opposite] >= c["start"]) & (touchdown[opposite] < c["end"])]
            if len(indices) == 1:
                p = (t[indices[0]]-t[c["start"]])/c["period_s"]
                diagonal.append(min(p, 1-p))
        phase[hind+"_to_"+fore] = circular_summary(vals)
        phase[hind+"_diagonal_pair_separation_cycle"] = numeric_summary(diagonal)
    result["limb_phase"] = phase
    result["diagonality_definition"] = "Circular temporal separation of diagonal touchdown pairs; 0=simultaneous. Not an independently validated scalar index."
    orders = Counter()
    for c in strides["HR"]:
        events = [(int(i), f) for f in FEET for i in touchdown[f] if c["start"] <= i < c["end"]]
        if len(events) == 4 and len(set(f for _, f in events)) == 4:
            groups = ["+".join(sorted(f for i, f in events if i == index)) for index in sorted(set(i for i, _ in events))]
            orders[" -> ".join(groups)] += 1
    result["observed_footfall_orders_HR_anchored"] = dict(orders)
    result["contact_quality"] = {
        "all_feet_have_complete_cycles": all(bool(strides[f]) for f in FEET),
        "HR_reference_cycles": len(strides["HR"]),
        "reference_cycles_with_one_touchdown_each_foot": int(sum(orders.values())),
        "note": "No-cycle limbs are unmeasurable, not a valid zero-frequency gait. Sub-control-step contacts are not resolved."}
    result["touchdown_events"] = [{"time_s": float(t[i]), "feet": [f for f in FEET if i in touchdown[f]]}
                                 for i in sorted(set(int(i) for f in FEET for i in touchdown[f] if t[i]>=settle_s))]
    for key, output_key in (("hip_height_m", "hip_height_svl"), ("shoulder_height_m", "shoulder_height_svl")):
        if key in s:
            result[output_key] = numeric_summary(np.asarray(s[key])[keep]/svl)
    for key in ("trunk_pitch_deg", "trunk_up_z", "forward_speed_m_s"):
        if key in s:
            result[key] = numeric_summary(np.asarray(s[key])[keep])
    if "belly_force_N" in s:
        result["belly_contact_time_fraction"] = float(np.mean(np.any(np.asarray(s["belly_force_N"])[keep] > threshold, axis=1)))
    if "tail_floor_contact" in s:
        result["tail_floor_contact_time_fraction"] = float(np.mean(np.asarray(s["tail_floor_contact"])[keep]))
    for key, destination in (("hinge_position_deg", "joint_angles"), ("anatomical_proxy_deg", "anatomical_proxies")):
        names = meta.get("hinge_names" if key.startswith("hinge") else "anatomical_proxy_names", [])
        if key not in s:
            continue
        arr = np.asarray(s[key], dtype=float)
        for j, name in enumerate(names):
            a = arr[:, j]
            suffix = name[-1] if name.endswith(("_L", "_R")) else None
            fore = name.startswith(("shoulder", "humerus", "elbow", "wrist"))
            foot = (("F" if fore else "H")+suffix) if suffix else "HR"
            cyc = strides[foot]
            result[destination][name] = {
                "whole_window": _window_angles(a, [(first, len(t)-1)]),
                "per_complete_stride": _window_angles(a, [(c["start"], c["end"]) for c in cyc]),
                "per_stance": _window_angles(a, [(c["start"], c["off"]-1) for c in cyc]),
                "at_touchdown_deg": numeric_summary([a[c["start"]] for c in cyc]),
                "cycle_reference": foot,
                "convention": "MuJoCo scalar hinge displacement" if destination=="joint_angles" else "Unvalidated geometric landmark proxy; see metadata"}
    work = {}
    for kind in ("positive", "negative", "abs"):
        key = "mechanical_work_"+kind+"_J"
        if key in s:
            arr = np.asarray(s[key], dtype=float)
            if np.all(np.isfinite(arr[1:][interval_keep])):
                val = float(np.sum(arr[1:][interval_keep]))
                work[kind+"_J"] = val
                work[kind+"_J_per_path_m"] = val/distance if distance > 1e-9 else None
    result["actuator_mechanical_work"] = work or {"status": "unavailable; requires physics-substep work logger"}
    result["deferred"]["COM_energy_recovery"] = "Not computed from trunk proxy; requires whole-model COM and a matched speed protocol."
    result["deferred"]["strike_shake"] = "50 Hz gait trace cannot resolve strike kinematics; log >=500 Hz before filtering."
    return result


def filtered_nose_kinematics(time_s, nose_xyz, cutoff_hz=50, output_hz=500, order=4):
    """Offline evaluator only; never synthesize high-rate evidence by upsampling.

    Butterworth order is an engineering choice requiring explicit reporting.
    Downsampling uses an anti-aliasing polyphase filter before zero-lag filtering.
    """
    t, dt = validate_times(time_s)
    fs = 1/dt
    if fs < output_hz*(1-1e-6):
        raise ValueError("Input acquisition must be >= output rate; upsampling cannot recover strikes.")
    if not 0 < cutoff_hz < min(fs, output_hz)/2:
        raise ValueError("Cutoff must be below both Nyquist frequencies.")
    try:
        from scipy.signal import butter, resample_poly, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError("High-rate strike filtering requires the optional scipy dependency; gait metrics do not.") from exc
    ratio = Fraction(output_hz/fs).limit_denominator(10000)
    x = np.asarray(nose_xyz, dtype=float)
    if x.shape != (len(t), 3) or not np.isfinite(x).all():
        raise ValueError("Expected finite nose positions[N,3].")
    y = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
    y = sosfiltfilt(butter(order, cutoff_hz, fs=output_hz, output="sos"), y, axis=0)
    tout = t[0]+np.arange(len(y))/output_hz
    valid = tout <= t[-1]+1e-10
    tout, y = tout[valid], y[valid]
    velocity = np.gradient(y, 1/output_hz, axis=0)
    acceleration = np.gradient(velocity, 1/output_hz, axis=0)
    return {"time_s": tout, "position_m": y, "velocity_m_s": velocity, "acceleration_m_s2": acceleration,
            "filter": {"order": order, "cutoff_hz": cutoff_hz, "output_hz": output_hz,
                       "edge_warning": "Do not interpret filtered endpoint peaks; retain pre/post event padding."}}


class TraceRecorder:
    """Read state only; no changes to dynamics, sensors, or controller."""
    def __init__(self, env, metadata=None):
        import mujoco
        self.env = env
        m = env.model
        self.site_ids = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
                         for name in FOOT_SITES+("nose_tip", "vent", "tail_tip", "hip_L", "hip_R", "shoulder_L", "shoulder_R", "pelvis_center", "pectoral_center", "mid_back")}
        self.body_ids = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
        self.sensor_ids = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in TOUCH_SENSORS}
        self.hinge_ids = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        for name in FOOT_SITES+("nose_tip", "vent", "tail_tip"):
            if self.site_ids[name] < 0:
                raise ValueError("Required site missing: "+name)
        if any(i < 0 for i in self.sensor_ids.values()):
            raise ValueError("Missing required touch sensor.")
        # Neutral skeleton SVL must not shrink as the animal curls or looks back.
        neutral = mujoco.MjData(m)
        mujoco.mj_forward(m, neutral)
        svl = float(np.linalg.norm(neutral.site_xpos[self.site_ids["nose_tip"]]-neutral.site_xpos[self.site_ids["vent"]]))
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in self.hinge_ids]
        self.metadata = {"schema_version": 1, "svl_m": svl, "svl_definition": "nose_tip to vent in model zero-joint reference pose, fixed across episode",
                         "contact_threshold_N": env.contact_threshold, "foot_order": list(FEET), "hinge_names": names,
                         "control_dt_s": env.dt, "physics_dt_s": m.opt.timestep,
                         "anatomical_proxy_names": [name+"_"+side for side in ("L", "R") for name in
                                                    ("femur_depression", "femur_retraction", "knee", "ankle", "humerus_depression", "humerus_retraction", "elbow", "wrist")],
                         "anatomical_proxy_convention": {
                             "depression": "atan2(-segment z, horizontal length) in local girdle frame",
                             "retraction": "atan2(-segment x, abs(segment y)) in local girdle frame",
                             "knee_elbow": "3D included angle between neighbouring segment joint centres; 180 means straight",
                             "ankle_wrist": "3D included angle using foot-zone centre as distal proxy; not exact digit landmark",
                             "status": "Unvalidated against paper conventions; never compare raw hinge maxima to biological included angles"},
                         "measurement_site_fallbacks": [n for n in ("hip_L", "hip_R", "shoulder_L", "shoulder_R", "pelvis_center", "pectoral_center", "mid_back") if self.site_ids[n]<0]}
        self.metadata.update(metadata or {})
        self.samples = {}

    def _site(self, name, fallback_body=None):
        index = self.site_ids[name]
        return self.env.data.site_xpos[index].copy() if index>=0 else self.env.data.xpos[self.body_ids[fallback_body]].copy()

    def record(self, info=None):
        info = info or {}
        e, d, m = self.env, self.env.data, self.env.model
        rotation = d.xmat[e._trunk].reshape(3, 3)
        forces = [float(d.sensordata[m.sensor_adr[self.sensor_ids[n]]]) for n in TOUCH_SENSORS]
        angles = []
        for side in ("L", "R"):
            for proximal, middle, distal, girdle, site in (("femur", "tibia", "pes", "pelvis", "footzone_hind_"),
                                                         ("humerus", "forearm", "manus", "trunk_anterior", "footzone_fore_")):
                a, b, c = [d.xpos[self.body_ids[n+"_"+side]] for n in (proximal, middle, distal)]
                distal_proxy = self._site(site+side)
                r = d.xmat[self.body_ids[girdle]].reshape(3, 3)
                vec = r.T @ (b-a)
                angles.extend([float(np.degrees(np.arctan2(-vec[2], np.linalg.norm(vec[:2])))),
                               float(np.degrees(np.arctan2(-vec[0], abs(vec[1])))),
                               _angle(a-b, c-b), _angle(b-c, distal_proxy-c)])
        tail_ids = {i for name, i in self.body_ids.items() if name and name.startswith("tail")}
        tail_contact = any((int(m.geom_bodyid[c.geom1]) in tail_ids and int(m.geom_bodyid[c.geom2])==0) or
                           (int(m.geom_bodyid[c.geom2]) in tail_ids and int(m.geom_bodyid[c.geom1])==0)
                           for c in d.contact[:d.ncon])
        sample = {"time_s": float(d.time), "trunk_position_m": d.xpos[e._trunk].copy(),
                  "trunk_quaternion_wxyz": d.xquat[e._trunk].copy(), "qpos": d.qpos.copy(), "qvel": d.qvel.copy(),
                  "hinge_position_deg": np.degrees(d.qpos[m.jnt_qposadr[self.hinge_ids]]),
                  "foot_force_N": forces[:4], "belly_force_N": forces[4:],
                  "foot_position_m": [self._site(n) for n in FOOT_SITES],
                  "nose_tip_position_m": self._site("nose_tip"), "tail_tip_position_m": self._site("tail_tip"),
                  "pelvis_position_m": self._site("pelvis_center", "pelvis"),
                  "mid_back_position_m": self._site("mid_back", "trunk_middle"),
                  "hip_height_m": [self._site("hip_"+side, "femur_"+side)[2] for side in ("L", "R")],
                  "shoulder_height_m": [self._site("shoulder_"+side, "humerus_"+side)[2] for side in ("L", "R")],
                  "trunk_pitch_deg": float(np.degrees(np.arctan2(rotation[2, 0], np.linalg.norm(rotation[:2, 0])))),
                  "trunk_up_z": float(rotation[2, 2]), "forward_speed_m_s": float(e._s("vel_trunk")[0]),
                  "tail_floor_contact": tail_contact, "anatomical_proxy_deg": angles,
                  "fallen": bool(info.get("fallen", False))}
        for kind in ("positive", "negative", "abs"):
            key = "mechanical_work_"+kind+"_J"
            sample[key] = info.get(key, 0 if not self.samples else None)
        for key, value in sample.items():
            self.samples.setdefault(key, []).append(value)

    def as_dict(self):
        return {"metadata": self.metadata, "samples": self.samples}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    """Artifact serialization, not a source-file rewrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    from common.provenance import parameter_value
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path)
    p.add_argument("--vecnormalize", type=Path)
    p.add_argument("--zero-residual", action="store_true", help="Use the CPG base and its contact reflex with zero policy action.")
    p.add_argument("--zero-tail-drive", action="store_true", help="Zero tail tendon commands only; NOT mechanical restriction.")
    p.add_argument("--xml", type=Path, default=REPO/"morphology/gecko_body_r.xml")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--duration", type=float, default=parameter_value("evaluation_duration_s"))
    p.add_argument("--settle", type=float, default=parameter_value("evaluation_settle_s"))
    p.add_argument("--debounce", type=float, default=parameter_value("contact_debounce_s"))
    p.add_argument("--contact-thresh", type=float, default=parameter_value("legacy_contact_threshold_N"))
    p.add_argument("--reset-noise", type=float, default=0., help="0 is deterministic calibration; nonzero explicitly changes the repeat protocol.")
    p.add_argument("--residual-scale", type=float, default=.25)
    p.add_argument("--front-stance-press", type=float, default=.40)
    p.add_argument("--front-swing-lift", type=float, default=.40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path)
    p.add_argument("--video", type=Path, help="Stream the first episode to this MP4.")
    p.add_argument("--video-every", type=int, default=1, help="Render every N control steps; playback fps changes accordingly.")
    args = p.parse_args(argv)
    if args.episodes < 1 or args.duration <= args.settle or args.settle < 0 or args.video_every < 1 or args.reset_noise < 0:
        p.error("Need positive episode count, duration > settle >= 0, video-every >= 1 and reset-noise >= 0.")
    if not args.zero_residual and (args.model is None or args.vecnormalize is None):
        p.error("Frozen policy evaluation requires both --model and --vecnormalize.")
    if platform.system()=="Linux" and not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from envs.gecko_walk_env import GeckoWalkEnv
    env = GeckoWalkEnv(xml_path=args.xml, control_mode="cpg_residual", max_steps=1,
                       contact_thresh=args.contact_thresh, reset_noise=args.reset_noise,
                       residual_scale=args.residual_scale, front_stance_press=args.front_stance_press,
                       front_swing_lift=args.front_swing_lift)
    if not np.isclose(env.dt, .02):
        env.close()
        p.error("CLI protocol expects 50 Hz; XML timestep/frame_skip changed.")
    steps = int(round(args.duration/env.dt))
    if not np.isclose(steps*env.dt, args.duration):
        env.close()
        p.error("Duration must be an integer number of control steps.")
    env.max_steps = steps
    floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0 or not np.isclose(env.model.geom_friction[floor, 0], .9):
        env.close()
        p.error("Calibration requires a named floor with friction .9; evaluator does not silently edit it.")
    normalizer = None
    policy = None
    if not args.zero_residual:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        wrapper = DummyVecEnv([lambda: env])
        normalizer = VecNormalize.load(str(args.vecnormalize), wrapper)
        normalizer.training = False
        normalizer.norm_reward = False
        policy = PPO.load(str(args.model), device=args.device)
        if policy.observation_space.shape != env.observation_space.shape or policy.action_space.shape != env.action_space.shape:
            env.close()
            raise ValueError("Checkpoint observation/action shapes do not match this body/controller.")
    if args.zero_tail_drive:
        original = env.cpg.compute
        ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("tail_bend_L", "tail_bend_R")]
        if min(ids)<0:
            raise ValueError("Tail-drive ablation requires both tail tendon actuators.")
        def zero_tail(*a, **kw):
            ctrl = original(*a, **kw)
            ctrl[ids] = 0
            return ctrl
        env.cpg.compute = zero_tail
    meta = {"xml_sha256": sha256(args.xml), "model_sha256": sha256(args.model) if policy else None,
            "normalizer_sha256": sha256(args.vecnormalize) if normalizer else None,
            "controller": "zero residual with contact reflex" if args.zero_residual else "frozen PPO residual",
            "ablation": "zero tail tendon drive; joints remain mechanically free" if args.zero_tail_drive else "none",
            "friction": float(env.model.geom_friction[floor, 0]), "temperature_C_context_only": parameter_value("calibration_temperature_C"),
            "reset_noise": args.reset_noise, "requested_duration_s": args.duration,
            "goal_distance_m": parameter_value("evaluation_goal_distance_m"),
            "residual_scale": args.residual_scale, "front_stance_press": args.front_stance_press,
            "front_swing_lift": args.front_swing_lift,
            "calibration_scenario": "fixed world heading and distant goal, level floor, no prey, no randomization except declared reset noise"}
    report = {"schema_version": 1, "protocol": meta, "episodes": [],
              "claim": "Instrumented simulation diagnostics; no biological validation claimed.",
              "repeat_note": "Different seeds with reset_noise=0 repeat the same deterministic experiment, not independent animal samples."}
    from common.provenance import load_registry
    registry = load_registry()["entries"]
    reference_names = ("hind_duty_factor", "hind_duty_factor_range", "fore_duty_factor", "fore_duty_factor_tolerance",
                       "limb_phase_range", "stride_length_svl_range", "walking_speed_m_s_range",
                       "trunk_pitch_deg", "trunk_pitch_sem_deg", "hip_height_svl", "shoulder_height_svl")
    report["literature_context"] = {name: registry[name] for name in reference_names}
    report["literature_comparison_warning"] = (
        "Reference values retain source/species/provenance and are context, not acceptance gates. "
        "Multi-study or residualized values and animal SEMs cannot certify this raw single-speed trace; "
        "anatomical landmark conventions still require validation.")
    writer = None
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed+episode)
            env.target = env.data.xpos[env._trunk][:2].copy()+np.array([meta["goal_distance_m"], 0.])
            _, env._prev_dist, _ = env._target_egocentric()
            obs = env._obs()
            recorder = TraceRecorder(env, {**meta, "episode": episode, "seed": args.seed+episode})
            recorder.record()
            if episode == 0 and args.video:
                import imageio.v2 as imageio
                args.video.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(args.video), fps=1/(env.dt*args.video_every), quality=8, macro_block_size=1)
                writer.append_data(env.render())
            reward_sum = 0.
            terminated = truncated = False
            for step in range(steps):
                if policy is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = policy.predict(normalizer.normalize_obs(obs[None, :]), deterministic=True)
                    action = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                recorder.record(info)
                reward_sum += reward
                if writer and (step+1)%args.video_every==0:
                    writer.append_data(env.render())
                if terminated or truncated:
                    break
            if writer:
                writer.close()
                writer = None
            trace = recorder.as_dict()
            status = {"episode": episode, "seed": args.seed+episode, "terminated": terminated,
                      "truncated": truncated, "simulated_seconds": float(env.data.time),
                      "completed_requested_duration": bool(not terminated and np.isclose(env.data.time, args.duration)),
                      "reward_sum_diagnostic_only": reward_sum}
            if env.data.time <= args.settle:
                result = {"status": "insufficient_post_settling_data", "reason": "Episode ended before scoring interval; counted as failure, not dropped."}
            else:
                result = analyze_trace(trace, args.settle, args.debounce)
            from eval.ethogram import score_ethogram
            status.update({"gait": result, "ethogram": score_ethogram(trace)})
            report["episodes"].append(status)
            if args.trace_dir:
                write_json(args.trace_dir/f"episode_{episode:03d}.json", trace)
            write_json(args.output, report)  # Preserve completed episodes if interrupted.
            print(f"episode={episode} seconds={env.data.time:.2f} fallen={terminated} speed={result.get('mean_path_speed_m_s')} gait_status={result['status']}", flush=True)
        valid = [x["gait"] for x in report["episodes"] if x["gait"]["status"]=="measured"]
        report["summary"] = {"requested_episodes": args.episodes, "completed_episodes": sum(x["completed_requested_duration"] for x in report["episodes"]),
                             "fall_episodes": sum(x["terminated"] for x in report["episodes"]),
                             "scorable_episodes_including_partial": len(valid),
                             "total_simulated_seconds": sum(x["simulated_seconds"] for x in report["episodes"]),
                             "episode_mean_path_speed_m_s": numeric_summary([x["mean_path_speed_m_s"] for x in valid]),
                             "note": "Scorable partial episodes remain labelled; do not hide early falls by averaging only survivors."}
        write_json(args.output, report)
    finally:
        if writer:
            writer.close()
        if normalizer:
            normalizer.close()
        else:
            env.close()
    return report


if __name__ == "__main__":
    main()
