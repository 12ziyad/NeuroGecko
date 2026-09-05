"""Read-only frozen-trunk diagnostics of commanded collision-foot kinematics.

This is inverse position-actuator mapping plus forward kinematics, NOT a dynamic
rollout or a contact/load predictor. Root, spine, other limbs and passive joints
remain at the saved stand keyframe. No model or controller parameter is edited.

python -m utils.controller_kinematics --profile lab --json diagnosis.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from common.gait_config import FOOT_ORDER
from common.morphology_audit import REPO_ROOT
from envs.cpg_residual_controller import CPGResidualController

DEFAULT_XML = REPO_ROOT / "morphology" / "gecko_body_lab_v2.xml"
FOOT_BODIES = {"HL": "pes_L", "FL": "manus_L", "HR": "pes_R", "FR": "manus_R"}
FEEDBACK_MODES = ("none", "loaded", "airborne")


def geom_support_radius(geom_type, size, local_direction):
    """Exact support radius along a unit direction for supported primitives."""
    size, direction = np.asarray(size, float), np.asarray(local_direction, float)
    if size.shape != (3,) or direction.shape != (3,) or not np.isfinite(size).all():
        raise ValueError("Expected finite three-component size and direction.")
    if not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1.):
        raise ValueError("Support direction must be a finite unit vector.")
    kind = mujoco.mjtGeom
    if geom_type == kind.mjGEOM_BOX:
        return float(np.dot(np.abs(direction), size))
    if geom_type == kind.mjGEOM_CAPSULE:
        return float(size[0] + size[1] * abs(direction[2]))
    if geom_type == kind.mjGEOM_SPHERE:
        return float(size[0])
    if geom_type == kind.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(direction * size))
    if geom_type == kind.mjGEOM_CYLINDER:
        return float(size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2]))
    raise ValueError(f"Unsupported collision primitive {geom_type}; no bounding-box approximation allowed.")


def geom_plane_clearance(model, data, geom_id, plane_point, plane_normal):
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    radius = geom_support_radius(model.geom_type[geom_id], model.geom_size[geom_id],
                                 rotation.T @ np.asarray(plane_normal))
    return float(np.dot(data.geom_xpos[geom_id] - plane_point, plane_normal) - radius)


class FrozenLimbKinematics:
    def __init__(self, xml_path=DEFAULT_XML, profile="lab"):
        self.xml_path = Path(xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key < 0:
            raise ValueError("A saved stand keyframe is required.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)
        self.stand_qpos = self.data.qpos.copy()
        self.stand_actuator_length = self.data.actuator_length.copy()
        self.controller = CPGResidualController(self.model, gait_profile=profile, verbose=False)
        self.foot_actuators = {foot: [aid for aid, limb, _ in self.controller.entries if limb == foot]
                               for foot in FOOT_ORDER}
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if self.floor_id < 0 or self.model.geom_type[self.floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("The floor must be an explicit plane geom named floor.")
        self.plane_point = self.data.geom_xpos[self.floor_id].copy()
        self.plane_normal = self.data.geom_xmat[self.floor_id].reshape(3, 3)[:, 2].copy()
        trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_middle")
        self.body_forward = self.data.xmat[trunk].reshape(3, 3)[:, 0].copy()
        self.foot_geoms = {}
        for foot, body_name in FOOT_BODIES.items():
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            ids = []
            for gid in range(self.model.ngeom):
                # Include all descendants and only geoms collision-compatible with floor.
                ancestor = int(self.model.geom_bodyid[gid])
                while ancestor not in (0, bid):
                    ancestor = int(self.model.body_parentid[ancestor])
                compatible = ((self.model.geom_contype[gid] & self.model.geom_conaffinity[self.floor_id])
                              or (self.model.geom_contype[self.floor_id] & self.model.geom_conaffinity[gid]))
                if ancestor == bid and compatible:
                    ids.append(gid)
            if not ids:
                raise ValueError(f"No floor-compatible collision geoms for {foot}.")
            self.foot_geoms[foot] = ids

    def time_for_phase(self, foot, phase):
        if not 0 <= phase < 1:
            raise ValueError("Local phase must be in [0, 1).")
        delay = (self.controller.phase[foot] if self.controller.gait_profile == "lab"
                 else -self.controller.phase[foot])
        return ((float(phase) + delay) % 1.) / self.controller.freq

    def command_at(self, foot, phase, feedback="none"):
        if feedback not in FEEDBACK_MODES:
            raise ValueError(f"Unknown feedback mode {feedback}.")
        contacts = None if feedback == "none" else {"FL": feedback == "loaded", "FR": feedback == "loaded"}
        return self.controller.compute(np.zeros(self.model.nu), self.time_for_phase(foot, phase),
                                       front_contact=contacts)

    def pose(self, foot, command):
        """Set only this limb's joints to zero-velocity position-servo targets."""
        command = np.asarray(command, float)
        if command.shape != (self.model.nu,) or not np.isfinite(command).all():
            raise ValueError("Command must be a finite nu-vector.")
        self.data.qpos[:] = self.stand_qpos
        self.data.qvel[:] = 0.
        applied = {}
        for aid in self.foot_actuators[foot]:
            model = self.model
            if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError("Diagnostic requires direct joint position actuators on limbs.")
            jid = int(model.actuator_trnid[aid, 0])
            if model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("Diagnostic supports scalar joints only.")
            gain, bias = model.actuator_gainprm[aid], model.actuator_biasprm[aid]
            if gain[0] <= 0 or not np.allclose([bias[0], bias[1]], [0., -gain[0]], atol=1e-14):
                raise ValueError("Expected a position-servo gain/bias contract.")
            gear = float(model.actuator_gear[aid, 0])
            if gear == 0 or np.any(model.actuator_gear[aid, 1:] != 0):
                raise ValueError("Expected a nonzero scalar joint gear.")
            target = float(command[aid])
            if model.actuator_ctrllimited[aid] and not (model.actuator_ctrlrange[aid, 0] - 1e-12 <= target
                                                        <= model.actuator_ctrlrange[aid, 1] + 1e-12):
                raise ValueError("Command outside ctrlrange; clip explicitly before diagnostic.")
            qadr = int(model.jnt_qposadr[jid])
            self.data.qpos[qadr] = self.stand_qpos[qadr] + (target - self.stand_actuator_length[aid]) / gear
            applied[model.actuator(aid).name] = {"ctrl": target, "qpos_rad": float(self.data.qpos[qadr]),
                                                  "normalized": float((target - self.controller.neutral[aid]) / self.controller.half[aid])}
        self.data.ctrl[:] = command
        mujoco.mj_forward(self.model, self.data)
        surfaces = []
        for gid in self.foot_geoms[foot]:
            surfaces.append({"geom_id": gid, "geom_name": self.model.geom(gid).name or None,
                             "type": int(self.model.geom_type[gid]),
                             "clearance_m": geom_plane_clearance(self.model, self.data, gid,
                                                                   self.plane_point, self.plane_normal)})
        lowest = min(surfaces, key=lambda item: item["clearance_m"])
        pad = next((gid for gid in self.foot_geoms[foot] if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX), None)
        if pad is None:
            raise ValueError("An explicit collision pad box is required for the fore-aft diagnostic.")
        return {"minimum_collision_clearance_m": lowest["clearance_m"], "lowest_geom_id": lowest["geom_id"],
                "pad_center_position_m": self.data.geom_xpos[pad].tolist(),
                "pad_forward_m": float(np.dot(self.data.geom_xpos[pad], self.body_forward)),
                "collision_surfaces": surfaces, "actuator_targets": applied}

    def at_phase(self, foot, phase, feedback="none", overrides=None):
        command = self.command_at(foot, phase, feedback)
        for name, normalized in (overrides or {}).items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid not in self.foot_actuators[foot] or not math.isfinite(normalized) or not -1 <= normalized <= 1:
                raise ValueError("Overrides must be bounded normalized commands on the selected foot.")
            command[aid] = self.controller.neutral[aid] + normalized * self.controller.half[aid]
        result = self.pose(foot, command)
        result.update({"local_phase_cycle": float(phase), "time_s": self.time_for_phase(foot, phase), "feedback": feedback})
        return result


def collect_diagnosis(xml_path=DEFAULT_XML, profile="lab", samples=201):
    if samples < 11:
        raise ValueError("At least 11 samples are required.")
    probe = FrozenLimbKinematics(xml_path, profile)
    feet = {}
    for foot in FOOT_ORDER:
        stance = probe.controller.stance_for(foot)
        phases = {"touchdown": 0., "mid_stance": stance / 2., "liftoff_before": stance - 1e-9,
                  "mid_swing": (stance + 1.) / 2., "cycle_end_before": 1. - 1e-9}
        modes = {}
        for mode in FEEDBACK_MODES:
            landmarks = {name: probe.at_phase(foot, phase, mode) for name, phase in phases.items()}
            modes[mode] = {"landmarks": landmarks,
                           "mid_swing_minus_mid_stance_clearance_m": landmarks["mid_swing"]["minimum_collision_clearance_m"] - landmarks["mid_stance"]["minimum_collision_clearance_m"],
                           "stance_pad_forward_travel_m": landmarks["liftoff_before"]["pad_forward_m"] - landmarks["touchdown"]["pad_forward_m"]}
        lift_name = ("knee_" if foot.startswith("H") else "elbow_") + foot[1]
        lift_sweep = []
        for normalized in np.linspace(-.8, .8, 9):
            row = probe.at_phase(foot, phases["mid_swing"], overrides={lift_name: float(normalized)})
            lift_sweep.append({"normalized_lift_command": float(normalized), "command_rad": row["actuator_targets"][lift_name]["qpos_rad"],
                               "minimum_collision_clearance_m": row["minimum_collision_clearance_m"], "lowest_geom_id": row["lowest_geom_id"]})
        cycle = []
        for phase in np.linspace(0., 1., samples, endpoint=False):
            row = probe.at_phase(foot, float(phase))
            cycle.append({"phase": float(phase), "commanded_stance": bool(phase < stance),
                          "minimum_collision_clearance_m": row["minimum_collision_clearance_m"],
                          "pad_forward_m": row["pad_forward_m"]})
        engineering_probes = []
        if foot.startswith("H"):
            for multiplier in (-1., -2.):
                raw_target = multiplier * probe.controller.amp["lift"]
                applied = float(np.clip(raw_target, -1., 1.))
                row = probe.at_phase(foot, phases["mid_swing"], overrides={lift_name: applied})
                engineering_probes.append({"case": "hind_lift_multiplier", "multiplier": multiplier,
                                           "raw_normalized_lift": raw_target, "applied_normalized_lift": applied,
                                           "clipped": raw_target != applied,
                                           "mid_swing_clearance_m": row["minimum_collision_clearance_m"],
                                           "delta_from_current_mid_stance_m": row["minimum_collision_clearance_m"] - modes["none"]["landmarks"]["mid_stance"]["minimum_collision_clearance_m"]})
        else:
            sprawl_name = "shoulder_sprawl_" + foot[1]
            sprawl_id = mujoco.mj_name2id(probe.model, mujoco.mjtObj.mjOBJ_ACTUATOR, sprawl_name)
            for tuck_rad in (probe.controller.shoulder_sprawl_tuck, 0.):
                normalized_tuck = (-1 if foot[1] == "L" else 1) * tuck_rad / probe.controller.half[sprawl_id]
                for press in (-.4, -.2, 0.):
                    stance_row = probe.at_phase(foot, phases["mid_stance"], overrides={lift_name: press, sprawl_name: normalized_tuck})
                    for swing_delta in (-.8, -1.2):
                        raw_target = press + swing_delta
                        applied = float(np.clip(raw_target, -1., 1.))
                        row = probe.at_phase(foot, phases["mid_swing"], overrides={lift_name: applied, sprawl_name: normalized_tuck})
                        engineering_probes.append({"case": "continuous_front_lift", "tuck_rad": tuck_rad,
                                                   "stance_press": press, "swing_delta": swing_delta,
                                                   "raw_normalized_lift": raw_target, "applied_normalized_lift": applied,
                                                   "clipped": raw_target != applied,
                                                   "mid_stance_clearance_m": stance_row["minimum_collision_clearance_m"],
                                                   "mid_swing_clearance_m": row["minimum_collision_clearance_m"],
                                                   "swing_minus_stance_m": row["minimum_collision_clearance_m"] - stance_row["minimum_collision_clearance_m"]})
        feet[foot] = {"commanded_stance_fraction": stance, "touchdown_global_cycle": probe.time_for_phase(foot, 0.) * probe.controller.freq,
                      "feedback_modes": modes, "lift_sensitivity": lift_sweep, "cycle_samples": cycle,
                      "engineering_parameter_probes": engineering_probes}
    return {"schema_version": 1, "xml_path": str(probe.xml_path),
            "xml_raw_sha256": hashlib.sha256(probe.xml_path.read_bytes()).hexdigest(),
            "controller_raw_sha256": hashlib.sha256((REPO_ROOT / "envs" / "cpg_residual_controller.py").read_bytes()).hexdigest(),
            "gait_profile": profile, "frequency_hz": probe.controller.freq, "feet": feet,
            "protocol": {"pose": "saved stand; all non-selected-limb and passive joints fixed", "ground": "actual floor plane", "clearance": "minimum exact support of every floor-compatible collision geom in the foot subtree", "fore_aft": "collision-pad center projected onto frozen trunk +X", "sensitivity_grid": "9 equally spaced normalized targets from -0.8 to +0.8; INVENTED diagnostic grid, not biological parameters", "parameter_probes": "INVENTED bounded diagnostic targets only, not controller changes: hind multipliers -1/-2; front press -0.4/-0.2/0 plus continuous swing delta -0.8/-1.2, current/zero tuck. Explicit ctrl clipping is reported.", "samples": samples},
            "limitations": ["Command targets are not achieved dynamic poses. Static overlap is not measured contact force.",
                            "Margin, soft contacts, torque/velocity limits, passive-joint response, body motion and servo lag are not simulated.",
                            "Feedback modes are conditional branches; actual rollout switches with measured front load.",
                            "A continuous periodic trajectory has equal-and-opposite net stance/swing return displacement; contact release, timing and path shape can differ."]}


def format_table(report):
    lines = [f"Frozen collision-foot targets: {report['gait_profile']} | {report['xml_path']}",
             "foot feedback   stance mm   swing mm   delta mm   stance pad X mm"]
    for foot, result in report["feet"].items():
        for mode, values in result["feedback_modes"].items():
            land = values["landmarks"]
            lines.append(f"{foot:4} {mode:8} {1000*land['mid_stance']['minimum_collision_clearance_m']:10.3f} {1000*land['mid_swing']['minimum_collision_clearance_m']:10.3f} {1000*values['mid_swing_minus_mid_stance_clearance_m']:10.3f} {1000*values['stance_pad_forward_travel_m']:17.3f}")
    lines.append("Negative clearance is commanded geometric overlap, NOT a measured force or achieved trajectory.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--profile", choices=("legacy", "lab"), default="lab")
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = collect_diagnosis(args.xml, args.profile, args.samples)
    print(format_table(report))
    if args.json:
        if args.json.exists():
            raise FileExistsError("Refusing to overwrite existing evidence; choose a new output path.")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
