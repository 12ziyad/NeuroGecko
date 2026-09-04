"""Fourteen explicit morphology gates, kept separate from software unit tests.

Run ``python -m common.morphology_audit --strict --json report.json``.
The baseline is EXPECTED TO FAIL biological/engineering acceptance gates.
Without --strict this tool reports measurements; --strict exits 1 on any failed
gate. Passing the unit tests validates the instruments, not the animal model.

Primary-source checks: Fuller et al. 2011, doi:10.1016/j.zool.2010.11.003,
Table 1; Rawat et al. 2019, doi:10.17161/randa.v26i1.14342, Table 1.
Other target transcriptions and every engineering tolerance are identified in
config/proxies.yaml. A neutral stance is NOT a walking-kinematics validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from common.provenance import DEFAULT_REGISTRY, load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "morphology" / "gecko_body_r.xml"
MEASUREMENT_SITES = (
    "hip_L", "hip_R", "shoulder_L", "shoulder_R", "pelvis_center",
    "pectoral_center", "mid_back",
)
FOOT_SITES = (
    "footzone_fore_L", "footzone_fore_R", "footzone_hind_L", "footzone_hind_R",
)


@dataclass(frozen=True)
class GateSpec:
    id: str
    label: str
    target_ref: str
    tolerance_ref: str | None = None
    range_ref: str | None = None
    mode: str = "interval"


GATE_SPECS = (
    GateSpec("total_mass_kg", "Total mass", "body_mass_kg", "body_mass_tolerance_kg"),
    GateSpec("tail_mass_fraction", "Tail mass fraction", "tail_mass_fraction", range_ref="tail_mass_fraction_range"),
    GateSpec("tail_length_svl", "Tail chain / SVL", "tail_length_svl", "tail_length_svl_tolerance"),
    GateSpec("com_intact_svl", "Intact CoM behind nose / SVL", "com_intact_svl", "com_svl_tolerance"),
    GateSpec("com_tail_excluded_svl", "Tail-excluded CoM / SVL", "com_tail_excluded_svl", "com_svl_tolerance"),
    GateSpec("hip_height_svl", "Settled hip height / SVL", "hip_height_svl", "girdle_height_svl_tolerance"),
    GateSpec("shoulder_height_svl", "Settled shoulder height / SVL", "shoulder_height_svl", "girdle_height_svl_tolerance"),
    GateSpec("shoulder_hip_ratio", "Shoulder / hip height", "shoulder_hip_ratio", "shoulder_hip_ratio_tolerance"),
    GateSpec("femur_tibia_ratio", "Femur / tibia length", "femur_tibia_ratio", range_ref="femur_tibia_ratio_range"),
    GateSpec("hindlimb_length_svl", "Hindlimb chain / SVL", "hindlimb_length_svl", "hindlimb_length_svl_tolerance"),
    GateSpec("hip_proret_range_deg", "Hip pro/retraction range", "hip_proret_min_range_deg", mode="minimum"),
    GateSpec("hip_rot_range_deg", "Hip rotation range", "hip_rot_max_range_deg", mode="maximum"),
    GateSpec("head_width_svl", "Skin head width / SVL", "head_width_svl", "head_width_svl_tolerance"),
    GateSpec("interorbital_distance_m", "Inner-eye gap", "interorbital_distance_m", "interorbital_distance_tolerance_m"),
)


def weighted_com(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Mass-weighted CoM; zero-mass world/visual components are harmless."""
    weights = np.asarray(masses, dtype=float)
    points = np.asarray(positions, dtype=float)
    if points.shape != (len(weights), 3):
        raise ValueError("positions must be (number of masses, 3)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Masses must be finite and nonnegative")
    if not np.all(np.isfinite(points)) or weights.sum() <= 0:
        raise ValueError("Finite positions and positive total mass required")
    return (weights[:, None] * points).sum(axis=0) / weights.sum()


def behind_nose_svl(nose: np.ndarray, vent: np.ndarray, point: np.ndarray) -> float:
    """Project onto the neutral snout-to-vent axis, invariant to world heading."""
    caudal = np.asarray(vent, dtype=float) - np.asarray(nose, dtype=float)
    svl = float(np.linalg.norm(caudal))
    if not math.isfinite(svl) or svl <= 0:
        raise ValueError("SVL must be positive and finite")
    return float(np.dot(np.asarray(point) - nose, caudal) / (svl * svl))


def sphere_surface_gap(centers: np.ndarray, radii: np.ndarray) -> float:
    """Shortest inner gap of two spherical visual eyes (a landmark proxy)."""
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.shape != (2, 3) or radii.shape != (2,) or np.any(radii <= 0):
        raise ValueError("Exactly two positive-radius eyes are required")
    return float(np.linalg.norm(centers[0] - centers[1]) - radii.sum())


def _geom_ids(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    return np.flatnonzero(model.geom_bodyid == model.body(body_name).id)


def _local_rotation(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
    return rotation.reshape(3, 3)


def _support_radius_y(model: mujoco.MjModel, geom_id: int) -> float:
    """Exact support radius along local body y for the primitive head geoms."""
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    axis = _local_rotation(model, geom_id)[1]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[0] + abs(axis[2]) * size[1])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(axis * size))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(axis) @ size)
    raise ValueError("Unsupported head primitive; do not guess a head width")


def _width_of_geoms(model: mujoco.MjModel, geom_ids: list[int]) -> float:
    if not geom_ids:
        raise ValueError("No matching head geometry")
    bounds = [
        (model.geom_pos[g, 1] - _support_radius_y(model, g),
         model.geom_pos[g, 1] + _support_radius_y(model, g))
        for g in geom_ids
    ]
    return float(max(hi for _, hi in bounds) - min(lo for lo, _ in bounds))


def _hindlimb_lengths(model: mujoco.MjModel, side: str) -> dict[str, float]:
    """Joint-center chain plus fourth digit, excluding the contact claw.

    Fuller Table1 includes tarsal+metatarsal and fourth toe. This MJCF has no
    metatarsal joint; its fourth visual digit is the fourth of five capsule
    digits in XML order. We report that mapping as an engineering proxy.
    """
    femur = float(np.linalg.norm(model.body(f"tibia_{side}").pos))
    tibia = float(np.linalg.norm(model.body(f"pes_{side}").pos))
    digits = [int(g) for g in _geom_ids(model, f"pes_{side}")
              if model.geom_group[g] == 1
              and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE]
    if len(digits) != 5:
        raise ValueError("Expected five visual hind-foot digits; update landmark mapping")
    digit = digits[3]
    offset = _local_rotation(model, digit)[:, 2] * model.geom_size[digit, 1]
    endpoints = np.stack((model.geom_pos[digit] - offset, model.geom_pos[digit] + offset))
    proximal = endpoints[np.argmin(np.linalg.norm(endpoints, axis=1))]
    ankle_to_digit = float(np.linalg.norm(proximal))
    toe = float(2 * model.geom_size[digit, 1])
    return {"femur_m": femur, "tibia_m": tibia,
            "ankle_to_fourth_digit_m": ankle_to_digit, "fourth_digit_m": toe,
            "total_m": femur + tibia + ankle_to_digit + toe}


def _joint_actuator_ranges(model: mujoco.MjModel, name: str) -> dict[str, float]:
    joint = model.joint(name)
    actuator = model.actuator(name)
    if not model.jnt_limited[joint.id] or not model.actuator_ctrllimited[actuator.id]:
        raise ValueError(f"{name}: expected limited joint and control interval")
    if model.actuator_trntype[actuator.id] != mujoco.mjtTrn.mjTRN_JOINT:
        raise ValueError(f"{name}: expected a direct joint position actuator")
    if model.actuator_trnid[actuator.id, 0] != joint.id:
        raise ValueError(f"{name}: actuator is bound to a different joint")
    # Compiler converts joint limits to radians; position ctrlrange is already rad.
    return {"joint_deg": float(np.rad2deg(np.ptp(joint.range))),
            "actuator_deg": float(np.rad2deg(np.ptp(actuator.ctrlrange)))}


def evaluate_gates(metrics: dict[str, Any], entries: dict[str, Any]) -> list[dict[str, Any]]:
    """All bilateral/range components must pass; averaging cannot hide one side."""
    result = []
    for spec in GATE_SPECS:
        source = entries[spec.target_ref]
        target = float(source["value"])
        lo: float | None = None
        hi: float | None = None
        refs = [spec.target_ref]
        if spec.mode == "minimum":
            lo = target
        elif spec.mode == "maximum":
            hi = target
        elif spec.range_ref:
            lo, hi = map(float, entries[spec.range_ref]["value"])
            refs.append(spec.range_ref)
        else:
            tolerance = float(entries[spec.tolerance_ref]["value"])
            lo, hi = target - tolerance, target + tolerance
            refs.append(spec.tolerance_ref)
        actual = metrics[spec.id]
        values = list(actual.values()) if isinstance(actual, dict) else [actual]
        component_passes = [math.isfinite(float(v)) and
                            (lo is None or float(v) >= lo - 1e-12) and
                            (hi is None or float(v) <= hi + 1e-12) for v in values]
        result.append({"id": spec.id, "label": spec.label, "actual": actual,
                       "target": target, "lower": lo, "upper": hi,
                       "units": source["units"], "passed": bool(all(component_passes)),
                       "provenance_refs": refs, "source": source["source"],
                       "confidence": source["confidence"], "notes": source["notes"]})
    return result


def collect_morphology_metrics(
    xml_path: str | Path = DEFAULT_XML,
    settle_seconds: float | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Compile, measure neutral morphology, then settle at zero control targets."""
    xml_path, registry_path = Path(xml_path).resolve(), Path(registry_path).resolve()
    registry = load_registry(registry_path)
    entries = registry["entries"]
    if settle_seconds is None:
        settle_seconds = float(entries["morphology_settle_s"]["value"])
    if not math.isfinite(settle_seconds) or settle_seconds <= 0:
        raise ValueError("settle_seconds must be positive and finite")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral").id)
    mujoco.mj_forward(model, data)
    for site in MEASUREMENT_SITES:
        if model.site(site).group != 4:
            raise ValueError(f"{site}: measurement site must be in hidden group 4")
    nose, vent = data.site("nose_tip").xpos.copy(), data.site("vent").xpos.copy()
    svl = float(np.linalg.norm(nose - vent))
    if svl <= 0:
        raise ValueError("Invalid neutral-pose snout-vent length")
    tail_ids = [model.body(f"tail{i}").id for i in range(1, 6)]
    tail_length = sum(float(np.linalg.norm(model.body(f"tail{i}").pos))
                      for i in range(2, 6))
    tail_length += float(np.linalg.norm(model.site("tail_tip").pos))
    intact_com = weighted_com(model.body_mass, data.xipos)
    retained = np.ones(model.nbody, dtype=bool)
    retained[tail_ids] = False
    no_tail_com = weighted_com(model.body_mass[retained], data.xipos[retained])
    head_geoms = _geom_ids(model, "head")
    skin_id, eye_id = model.material("skin").id, model.material("eye").id
    skin_geoms = [int(g) for g in head_geoms if model.geom_group[g] == 1 and model.geom_matid[g] == skin_id]
    collision_geoms = [int(g) for g in head_geoms if model.geom_group[g] == 3]
    eyes = [int(g) for g in head_geoms if model.geom_matid[g] == eye_id]
    if len(eyes) != 2 or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_SPHERE for g in eyes):
        raise ValueError("Expected exactly two spherical eye geoms")
    limbs = {side: _hindlimb_lengths(model, side) for side in ("L", "R")}
    ranges = {name: _joint_actuator_ranges(model, name)
              for side in ("L", "R") for name in (f"hip_proret_{side}", f"hip_rot_{side}")}
    metrics: dict[str, Any] = {
        "total_mass_kg": float(model.body_mass.sum()),
        "tail_mass_fraction": float(model.body_mass[tail_ids].sum() / model.body_mass.sum()),
        "tail_length_svl": tail_length / svl,
        "com_intact_svl": behind_nose_svl(nose, vent, intact_com),
        "com_tail_excluded_svl": behind_nose_svl(nose, vent, no_tail_com),
        "femur_tibia_ratio": {side: lengths["femur_m"] / lengths["tibia_m"] for side, lengths in limbs.items()},
        "hindlimb_length_svl": {side: lengths["total_m"] / svl for side, lengths in limbs.items()},
        "hip_proret_range_deg": {f"{side}_{kind}": val for side in ("L", "R")
                                   for kind, val in ranges[f"hip_proret_{side}"].items()},
        "hip_rot_range_deg": {f"{side}_{kind}": val for side in ("L", "R")
                                for kind, val in ranges[f"hip_rot_{side}"].items()},
        "head_width_svl": _width_of_geoms(model, skin_geoms) / svl,
        "interorbital_distance_m": sphere_surface_gap(model.geom_pos[eyes], model.geom_size[eyes, 0]),
    }
    data.ctrl[:] = 0
    steps = max(1, math.ceil(settle_seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        raise RuntimeError("Model became nonfinite during neutral settling")
    hip_z = {side: float(data.site(f"hip_{side}").xpos[2]) for side in ("L", "R")}
    shoulder_z = {side: float(data.site(f"shoulder_{side}").xpos[2]) for side in ("L", "R")}
    if any(z <= 0 for z in hip_z.values()):
        raise RuntimeError("Hip landmark reached or crossed ground during settling")
    metrics.update({
        "hip_height_svl": {side: z / svl for side, z in hip_z.items()},
        "shoulder_height_svl": {side: z / svl for side, z in shoulder_z.items()},
        "shoulder_hip_ratio": {side: shoulder_z[side] / hip_z[side] for side in ("L", "R")},
    })
    gates = evaluate_gates(metrics, entries)
    touches = {model.sensor(s).name: float(data.sensordata[model.sensor_adr[s]])
               for s in range(model.nsensor) if model.sensor_type[s] == mujoco.mjtSensor.mjSENS_TOUCH}
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(xml_path),
        "model_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "registry_path": str(registry_path),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "protocol": {"neutral_keyframe": "neutral", "settle_seconds_requested": settle_seconds,
                     "settle_seconds_actual": float(data.time), "settle_steps": steps,
                     "actuator_targets": "all zero (position targets, not zero applied force)",
                     "ground": "model floor z=0; no environmental randomization",
                     "com": "neutral-pose body-mass weighted axial projection; tail exclusion is mathematical",
                     "bilateral_gates": "both sides and both joint/actuator ranges must pass",
                     "claim": "engineering morphology audit; not walking or kinematic validation"},
        "structural_counts": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                              "hinge_joints": int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)),
                              "sites": model.nsite, "sensors": model.nsensor},
        "metrics": metrics,
        "diagnostics": {"svl_m": svl, "tail_chain_length_m": tail_length,
                        "neutral_com_m": intact_com.tolist(), "neutral_tail_excluded_com_m": no_tail_com.tolist(),
                        "hindlimb_segments": limbs,
                        "head_collision_width_m": _width_of_geoms(model, collision_geoms),
                        "head_visual_skin_width_m": _width_of_geoms(model, skin_geoms),
                        "eye_center_distance_m": float(np.linalg.norm(model.geom_pos[eyes[0]] - model.geom_pos[eyes[1]])),
                        "settled_hip_z_m": hip_z, "settled_shoulder_z_m": shoulder_z,
                        "settled_max_abs_qvel": float(np.max(np.abs(data.qvel))),
                        "settled_touch_forces_N": touches},
        "corrections_and_limits": [
            "The model has 32 hinge joints, not the 26 asserted in parts of the supplied documents.",
            "Fuller Table1 femur/tibia is 1.54/1.51=1.0199, not 0.98. The requested 0.90–1.06 pass band is retained explicitly.",
            "Fuller Table1 tibia SEM is 0.02 cm, not the 0.04 cm transcribed in the supplied scorecard.",
            "Interorbital distance is approximated by the inner spherical-eye gap, not center-to-center separation.",
            "The source limb total includes fourth toe. MJCF ankle-to-digit geometry is a proxy, not a measured metatarsal segment.",
            "Static zero-control heights are engineering gates using walking reference values; they do not establish dynamic posture validity.",
            "A root spawn-height edit alone cannot set the eventual mechanically settled hip height.",
            "Gate tolerances are engineering choices. Passing them is not a statistical claim about individual animals.",
            "Tail CoM exclusion holds all remaining segments in the same neutral pose and is not an autotomy behavior experiment.",
        ],
        "gates": gates,
        "passed_count": sum(gate["passed"] for gate in gates),
        "gate_count": len(gates),
        "all_passed": all(gate["passed"] for gate in gates),
    }


def format_table(report: dict[str, Any]) -> str:
    lines = ["Morphology engineering audit (not kinematic validation)",
             f"Model: {report['model_path']}",
             f"Model SHA256: {report['model_sha256']}",
             "Actual shows min..max when a gate has multiple components.",
             f"{'Gate':<34} {'Actual':>18} {'Accepted interval':>22}  Result"]
    for gate in report["gates"]:
        actual = gate["actual"]
        if isinstance(actual, dict):
            vals = list(actual.values())
            actual_text = f"{min(vals):.6g}..{max(vals):.6g}"
        else:
            actual_text = f"{actual:.6g}"
        lo, hi = gate["lower"], gate["upper"]
        interval = (f"<= {hi:.6g}" if lo is None else
                    f">= {lo:.6g}" if hi is None else f"[{lo:.6g}, {hi:.6g}]")
        lines.append(f"{gate['label']:<34} {actual_text:>18} {interval:>22}  {'PASS' if gate['passed'] else 'FAIL'}")
    lines.append(f"\n{report['passed_count']}/{report['gate_count']} morphology gates passed.")
    lines.append("Software unit-test success does NOT mean these morphology gates pass.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--settle-seconds", type=float, default=None)
    parser.add_argument("--json", type=Path, help="Write complete machine-readable report")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if ANY morphology gate fails")
    args = parser.parse_args(argv)
    report = collect_morphology_metrics(args.xml, args.settle_seconds, args.registry)
    print(format_table(report))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"\nJSON report: {args.json.resolve()}")
    return int(args.strict and not report["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
