"""Opt-in frozen-pose knee/ankle compensation, independent of any controller.

This deterministic forward-kinematics inverse solve is an engineering hypothesis,
not measured gecko anatomy, a dynamics prediction, or a learned policy. A table
is built once over each hip's actual actuator range; runtime is interpolation.
The model is never modified. Callers must explicitly enable and apply offsets.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import mujoco
import numpy as np


# Every limb spec below lists its sprawl actuator first.
SPRAWL_INDEX = 0


def swing_blend(local_phase, stance_fraction):
    """One through stance, smooth zero at mid-swing, one again at touchdown.

    The weight and its derivative join at the boundaries. The caller's original
    triangular hip sweep remains only position-continuous, not velocity-smooth.
    """
    if not math.isfinite(local_phase) or not 0 <= local_phase <= 1:
        raise ValueError("local_phase must be a finite fraction in [0, 1].")
    if not math.isfinite(stance_fraction) or not 0 < stance_fraction < 1:
        raise ValueError("stance_fraction must be strictly between zero and one.")
    if local_phase <= stance_fraction or local_phase == 1:
        return 1.
    fraction = (local_phase - stance_fraction) / (1. - stance_fraction)
    return math.cos(math.pi * fraction) ** 2


def _support_radius(kind, size, direction):
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(direction) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(size[0] + size[1] * abs(direction[2]))
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return float(np.linalg.norm(size * direction))
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(size[0] * np.linalg.norm(direction[:2]) + size[1] * abs(direction[2]))
    raise ValueError("Unsupported collision primitive; a bounding-box proxy is not used.")


class StanceCompensator:
    """Minimum-norm Jacobian updates to a target collision-foot height.

    Defaults are INVENTED engineering settings: target -0.6 mm, 129 table nodes,
    at most 0.35 rad knee/ankle offsets. Both joints have unit-weight updates;
    this is not a unique biomechanical inverse solution. The initial neutral
    limb has zero sprawl/rotation/knee/ankle actuator targets. Root, spine and
    passive joints stay at stand. Nonzero 'other' control waves invalidate that
    reference assumption and must be disabled or separately assessed by caller.

    Integration: after computing a lab base command, pass the actual hip command
    (the method clips to its real range), phase and duty to offsets(); add the
    returned knee/ankle control deltas before the usual final actuator clipping.
    Existing swing commands must remain. At mid-swing these offsets are zero.
    """

    def __init__(self, model, target_clearance_m=-.0006, table_size=129,
                 max_offset_rad=.70, feet=("HL", "HR"), sprawl_nodes=None,
                 sprawl_limit_rad=None):
        # The frozen band is twice the target, floored at the original -1.2 mm so
        # the default is bit-for-bit unchanged. A deeper target is an explicit
        # experiment: the table is solved against the STAND root pose, and a
        # walking gecko rides higher than a standing one, so the solved foot
        # lands short of the floor by whatever that height difference is.
        # Session 4 measured the shoulder at 0.125 SVL against a published
        # 0.11 SVL, with the forefoot sitting ~3.1 mm high when commanded down.
        # A scalar applies to every foot; a mapping sets each foot separately, so
        # the forelimb can be aimed deeper than the hindlimb. Session 4 measured
        # the two girdles missing their published heights by different amounts
        # (shoulder 0.125 vs 0.11 SVL, hip 0.158 vs 0.15), so one number for all
        # four feet is the wrong shape for this correction.
        if isinstance(target_clearance_m, dict):
            per_foot = {f: float(v) for f, v in target_clearance_m.items()}
        else:
            per_foot = None
        scalar = -.0006 if per_foot is not None else target_clearance_m
        for value in ([scalar] if per_foot is None else list(per_foot.values())):
            if not math.isfinite(value) or not -.008 <= value <= 0:
                raise ValueError("Every target must lie inside the [-8 mm, 0] frozen stance band.")
        if isinstance(table_size, bool) or not isinstance(table_size, int) or table_size < 17:
            raise ValueError("table_size must be an integer of at least 17.")
        if not math.isfinite(max_offset_rad) or not 0 < max_offset_rad <= math.pi:
            raise ValueError("max_offset_rad must be finite, positive, and at most pi.")
        self.model = model
        self.target_clearance_m = per_foot if per_foot is not None else float(scalar)
        deepest = min(per_foot.values()) if per_foot is not None else float(scalar)
        self._band_low = min(-.0012, 2. * deepest)
        self._per_foot_target = per_foot
        self.max_offset_rad = float(max_offset_rad)
        self.table_size = table_size
        # Sprawl moves the collision foot a long way -- measured 0.42 mm per
        # degree on this body, i.e. +/-8.5 mm across the actuator range -- so a
        # table keyed on the fore-aft drive alone is only valid while the sprawl
        # command sits at neutral. sprawl_nodes=None keeps exactly that original
        # one-dimensional table; an integer builds a second table axis over the
        # sprawl actuator's own range so a moving sprawl stays compensated.
        if sprawl_nodes is not None:
            if isinstance(sprawl_nodes, bool) or not isinstance(sprawl_nodes, int) or sprawl_nodes < 3:
                raise ValueError("sprawl_nodes must be an integer of at least 3, or None.")
        self.sprawl_nodes = sprawl_nodes
        # The knee/ankle can only buy back so much height: at large sprawl the
        # foot leaves the reachable band entirely and the solve has no answer.
        # The caller therefore declares the sprawl band it intends to command,
        # and the controller must clamp its amplitude to it rather than relying
        # on table-edge clamping, which would silently stop compensating.
        if sprawl_limit_rad is not None and (not math.isfinite(sprawl_limit_rad) or sprawl_limit_rad <= 0):
            raise ValueError("sprawl_limit_rad must be finite and positive, or None.")
        self.sprawl_limit_rad = None if sprawl_limit_rad is None else float(sprawl_limit_rad)
        feet = tuple(feet)
        if not feet or any(f not in ("HL", "HR", "FL", "FR") for f in feet):
            raise ValueError("feet must be a non-empty subset of HL/HR/FL/FR.")
        if per_foot is not None and set(per_foot) - set(feet):
            raise ValueError("Per-foot stance targets name a foot that is not compensated.")
        self._data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key < 0:
            raise ValueError("A saved stand keyframe is required.")
        mujoco.mj_resetDataKeyframe(model, self._data, key)
        mujoco.mj_forward(model, self._data)
        self._stand_qpos = self._data.qpos.copy()
        self._stand_lengths = self._data.actuator_length.copy()
        self._neutral = np.mean(model.actuator_ctrlrange, axis=1)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor < 0 or model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("An explicit plane geom named floor is required.")
        self._plane_point = self._data.geom_xpos[floor].copy()
        self._plane_normal = self._data.geom_xmat[floor].reshape(3, 3)[:, 2].copy()
        self._info = {}
        tables = {}
        fit_diagnostics = {}
        # Per-limb spec: (posture joints..., driving joint, free compensating joints...).
        # The driving joint is the fore-aft sweep whose command keys the table; the
        # free joints are the ones the solver may move. The hind limb has two free
        # joints (knee, ankle); the forelimb has one (elbow) because this MJCF has
        # no wrist actuator, so its solve is one-dimensional. Everything downstream
        # is written for a variable number of free joints.
        limb_specs = {
            "HL": (["hip_sprawl_L", "hip_proret_L", "hip_rot_L", "knee_L", "ankle_L"], 1, 3, "pes_L"),
            "HR": (["hip_sprawl_R", "hip_proret_R", "hip_rot_R", "knee_R", "ankle_R"], 1, 3, "pes_R"),
            "FL": (["shoulder_sprawl_L", "shoulder_proret_L", "elbow_L"], 1, 2, "manus_L"),
            "FR": (["shoulder_sprawl_R", "shoulder_proret_R", "elbow_R"], 1, 2, "manus_R"),
        }
        for foot in feet:
            names, drive_index, free_start, body_name = limb_specs[foot]
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
            if min(ids) < 0:
                raise ValueError(f"Expected all named {foot} joint actuators: {names}.")
            for aid in ids:
                jid = int(model.actuator_trnid[aid, 0])
                if (model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                        or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
                        or model.actuator_gear[aid, 0] != 1.
                        or np.any(model.actuator_gear[aid, 1:] != 0)
                        or not model.actuator_ctrllimited[aid]):
                    raise ValueError("Requires range-limited, unit-gear direct hinge actuators.")
                gain, bias = model.actuator_gainprm[aid], model.actuator_biasprm[aid]
                if gain[0] <= 0 or not np.allclose([bias[0], bias[1]], [0., -gain[0]], atol=1e-14):
                    raise ValueError("Expected position-servo gain and bias.")
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geoms = []
            for gid in range(model.ngeom):
                ancestor = int(model.geom_bodyid[gid])
                while ancestor not in (0, bid):
                    ancestor = int(model.body_parentid[ancestor])
                compatible = ((model.geom_contype[gid] & model.geom_conaffinity[floor])
                              or (model.geom_contype[floor] & model.geom_conaffinity[gid]))
                if ancestor == bid and compatible:
                    geoms.append(gid)
            if not geoms:
                raise ValueError("No floor-compatible collision foot geoms.")
            offset_bounds = model.actuator_ctrlrange[ids[free_start:]] - self._neutral[ids[free_start:], None]
            offset_bounds[:, 0] = np.maximum(offset_bounds[:, 0], -max_offset_rad)
            offset_bounds[:, 1] = np.minimum(offset_bounds[:, 1], max_offset_rad)
            self._info[foot] = {"ids": ids, "geoms": geoms, "offset_bounds": offset_bounds,
                                "drive_index": drive_index, "free_start": free_start,
                                "free_count": len(ids) - free_start}
            hip_grid = np.linspace(*model.actuator_ctrlrange[ids[drive_index]], table_size)
            if sprawl_nodes is None:
                sprawl_grid = np.zeros(1)
            else:
                low, high = model.actuator_ctrlrange[ids[SPRAWL_INDEX]] - self._neutral[ids[SPRAWL_INDEX]]
                if self.sprawl_limit_rad is not None:
                    low = max(low, -self.sprawl_limit_rad)
                    high = min(high, self.sprawl_limit_rad)
                sprawl_grid = np.linspace(low, high, sprawl_nodes)
            solutions, errors = [], []
            for sprawl in sprawl_grid:
                column = []
                for hip in hip_grid:
                    offsets, error = self._solve(foot, float(hip), float(sprawl))
                    column.append(offsets)
                    errors.append(error)
                solutions.append(np.asarray(column))
            # (n_drive, n_sprawl, n_free); the one-node case is the original table.
            table = np.stack(solutions, axis=1)
            table.setflags(write=False)
            hip_grid.setflags(write=False)
            sprawl_grid.setflags(write=False)
            tables[foot] = table
            self._info[foot]["hip_grid"] = hip_grid
            self._info[foot]["sprawl_grid"] = sprawl_grid
            fit_diagnostics[foot] = {
                "max_abs_table_height_error_m": float(np.max(np.abs(errors))),
                "free_joints": [names[i] for i in range(free_start, len(names))],
                "offset_ranges_rad": [[float(np.min(table[:, :, c])), float(np.max(table[:, :, c]))]
                                      for c in range(table.shape[2])],
                "offset_bound_rad": [[float(offset_bounds[c, 0]), float(offset_bounds[c, 1])]
                                     for c in range(offset_bounds.shape[0])],
                "max_adjacent_offset_change_rad": float(np.max(np.abs(np.diff(table, axis=0)))),
                "sprawl_nodes": 1 if sprawl_nodes is None else sprawl_nodes,
                "sprawl_grid_rad": [float(v) for v in sprawl_grid],
                "collision_geom_count": len(geoms),
            }
        self.tables = MappingProxyType(tables)
        self.diagnostics = {"method": "bounded minimum-norm Jacobian updates; no SciPy or stochastic search",
                            "target_clearance_m": self.target_clearance_m,
                            "frozen_band_low_m": self._band_low, "table_size": table_size,
                            "sprawl_limit_rad": self.sprawl_limit_rad,
                            "max_offset_rad": self.max_offset_rad, "fit": fit_diagnostics,
                            "assumptions": ["saved stand root/spine/passive joints", "other-amplitude zero",
                                            "zero knee/ankle base stance targets", "exact collision primitives; no force prediction"]}
        for foot in self.tables:
            hip_grid = self._info[foot]["hip_grid"]
            sprawl_grid = self._info[foot]["sprawl_grid"]
            grid = np.linspace(hip_grid[0], hip_grid[-1], 4 * (table_size-1) + 1)
            # Check between sprawl nodes too, not only on them; interpolation
            # error is largest halfway between rows.
            if sprawl_grid.size > 1:
                mid = .5 * (sprawl_grid[:-1] + sprawl_grid[1:])
                sprawl_check = np.concatenate((sprawl_grid, mid))
            else:
                sprawl_check = sprawl_grid
            heights = [self.clearance(foot, float(hip),
                                      *self.offsets(foot, float(hip), 0., .78, float(sprawl)),
                                      sprawl_ctrl_rad=float(sprawl))
                       for sprawl in sprawl_check for hip in grid]
            self.diagnostics["fit"][foot]["dense_interpolation_clearance_range_m"] = [float(min(heights)), float(max(heights))]
            if min(heights) < self._band_low - 1e-9 or max(heights) > 1e-9:
                raise ValueError(f"{foot} interpolation violates frozen stance band: {min(heights)}, {max(heights)}")

    def target_for(self, foot):
        """This foot's frozen-pose target depth, scalar or per-foot."""
        if self._per_foot_target is None:
            return self.target_clearance_m
        return self._per_foot_target.get(foot, -.0006)

    def clearance(self, foot, hip_ctrl_rad, *free_offsets_rad, sprawl_ctrl_rad=0.):
        """Read-only model diagnostic; private scratch data is not thread-safe."""
        info = self._info[foot]
        ids = info["ids"]
        commands = self._neutral[ids].copy()
        commands[info["drive_index"]] = hip_ctrl_rad
        commands[SPRAWL_INDEX] += sprawl_ctrl_rad
        for slot, value in enumerate(free_offsets_rad):
            commands[info["free_start"] + slot] += value
        self._data.qpos[:] = self._stand_qpos
        for aid, command in zip(ids, commands):
            jid = self.model.actuator_trnid[aid, 0]
            qadr = self.model.jnt_qposadr[jid]
            self._data.qpos[qadr] = self._stand_qpos[qadr] + command - self._stand_lengths[aid]
        mujoco.mj_kinematics(self.model, self._data)
        result = math.inf
        for gid in info["geoms"]:
            rotation = self._data.geom_xmat[gid].reshape(3, 3)
            radius = _support_radius(self.model.geom_type[gid], self.model.geom_size[gid], rotation.T @ self._plane_normal)
            height = float((self._data.geom_xpos[gid] - self._plane_point) @ self._plane_normal) - radius
            result = min(result, height)
        return result

    def _solve(self, foot, hip, sprawl=0.):
        # Numerical controls: 100 microradian differences, 2 um target residual,
        # bounded 0.05-rad updates, at most 60 iterations and 12 backtrack trials.
        # These are deterministic solver tolerances, not biological parameters.
        #
        # Tolerance provenance: the earlier 10 nm residual was unreachable, not
        # because the offset bounds bind (at hip 0.3068 rad the reachable band is
        # +0.03 to -7.72 mm around a -0.60 mm target) but because a 10 urad probe
        # moves the foot only ~90 nm, so the finite-difference gradient could not
        # resolve the requested residual and the backtracking line search stalled
        # at ~18 um. 2 um is 0.3% of the 0.6 mm headroom between the target and
        # either edge of the declared [-1.2 mm, 0] stance band, and is far below
        # anything the contact solver resolves. The post-build dense-interpolation
        # check against that band is unchanged and remains the real guard.
        x = np.zeros(self._info[foot]["free_count"])
        bounds = self._info[foot]["offset_bounds"]
        for _ in range(60):
            error = self.clearance(foot, hip, *x, sprawl_ctrl_rad=sprawl) - self.target_for(foot)
            if abs(error) <= 2e-6:
                return x, error
            gradient = np.zeros(x.size)
            for axis in range(x.size):
                plus, minus = x.copy(), x.copy()
                plus[axis] += 1e-4
                minus[axis] -= 1e-4
                gradient[axis] = (self.clearance(foot, hip, *plus, sprawl_ctrl_rad=sprawl) - self.clearance(foot, hip, *minus, sprawl_ctrl_rad=sprawl)) / 2e-4
            denominator = float(gradient @ gradient)
            if denominator < 1e-14:
                break
            step = -error * gradient / denominator
            step *= min(1., .05 / max(float(np.max(np.abs(step))), 1e-15))
            improved = False
            for backtrack in range(12):
                candidate = np.clip(x + step * 2.**(-backtrack), bounds[:, 0], bounds[:, 1])
                candidate_error = self.clearance(foot, hip, *candidate, sprawl_ctrl_rad=sprawl) - self.target_for(foot)
                if abs(candidate_error) < abs(error):
                    x, improved = candidate, True
                    break
            if not improved:
                break
        error = self.clearance(foot, hip, *x, sprawl_ctrl_rad=sprawl) - self.target_for(foot)
        if abs(error) > 2e-6:
            # The Newton loop stalls where clearance() switches which collision
            # geom is lowest: that min() is continuous but not differentiable, so
            # the finite-difference gradient is one-sided at the kink and the
            # line search cannot improve. Clearance is monotone decreasing along
            # +[knee, ankle] (probed at hip 0.3068 rad: (0,0) -> -0.68 mm,
            # (0.35,0.35) -> -7.72 mm), so bisect along that direction instead.
            # Bisection needs no derivative and is unaffected by the kink.
            bounds = self._info[foot]["offset_bounds"]
            direction = np.ones(x.size) / math.sqrt(float(x.size))
            lo_scale = float(np.min(bounds[:, 0] / direction))
            hi_scale = float(np.min(bounds[:, 1] / direction))
            at = lambda s: self.clearance(foot, hip, *np.clip(s * direction, bounds[:, 0], bounds[:, 1]), sprawl_ctrl_rad=sprawl) - self.target_for(foot)
            lo_error, hi_error = at(lo_scale), at(hi_scale)
            if lo_error * hi_error <= 0.:
                for _ in range(200):
                    mid_scale = .5 * (lo_scale + hi_scale)
                    mid_error = at(mid_scale)
                    if abs(mid_error) <= 2e-6 or hi_scale - lo_scale < 1e-12:
                        break
                    if mid_error * lo_error > 0.:
                        lo_scale, lo_error = mid_scale, mid_error
                    else:
                        hi_scale, hi_error = mid_scale, mid_error
                candidate = np.clip(mid_scale * direction, bounds[:, 0], bounds[:, 1])
                if abs(mid_error) < abs(error):
                    x, error = candidate, mid_error
        if abs(error) > 1e-5:
            if self.sprawl_nodes is None:
                raise ValueError(f"Unreachable frozen height for {foot}, hip={hip}: residual={error} m")
            # With a sprawl axis the target is genuinely out of reach in the
            # corners: sprawling the limb under the body drops the foot further
            # than the knee and ankle can lift it back. That is not an error --
            # the foot simply presses deeper into the floor, which is what a
            # loaded foot does. Take the closest reachable pose and record the
            # residual; the dense band check below is still the real guard.
            x = np.clip(np.full(x.size, min(hi_scale, max(lo_scale, 0.)) / math.sqrt(float(x.size))
                                if False else 0.), bounds[:, 0], bounds[:, 1])
            best, best_error = None, math.inf
            for scale in np.linspace(lo_scale, hi_scale, 65):
                candidate = np.clip(scale * direction, bounds[:, 0], bounds[:, 1])
                candidate_error = self.clearance(foot, hip, *candidate, sprawl_ctrl_rad=sprawl) - self.target_for(foot)
                if abs(candidate_error) < abs(best_error):
                    best, best_error = candidate, candidate_error
            x, error = best, best_error
        return x, error

    def offsets(self, foot, hip_ctrl_rad, local_phase, stance_fraction, sprawl_ctrl_rad=0.):
        """Return knee/ankle deltas in actuator radians, with smooth swing fade.

        `sprawl_ctrl_rad` is the sprawl command as an offset from neutral. It is
        ignored unless the compensator was built with a sprawl axis, so a caller
        that does not move sprawl gets the original one-dimensional answer.
        """
        if foot not in self.tables or not math.isfinite(hip_ctrl_rad):
            raise ValueError("Expected a compensated foot and a finite actual hip command.")
        if not math.isfinite(sprawl_ctrl_rad):
            raise ValueError("Sprawl command must be finite.")
        weight = swing_blend(local_phase, stance_fraction)
        table = self.tables[foot]
        hip_grid = self._info[foot]["hip_grid"]
        sprawl_grid = self._info[foot]["sprawl_grid"]
        if sprawl_grid.size == 1:
            plane = table[:, 0, :]
        else:
            # Linear blend between the two bracketing sprawl rows; np.interp
            # clamps at the ends, matching real ctrlrange clipping.
            position = float(np.interp(sprawl_ctrl_rad, sprawl_grid,
                                       np.arange(sprawl_grid.size, dtype=float)))
            low = int(math.floor(position))
            high = min(low + 1, sprawl_grid.size - 1)
            blend = position - low
            plane = (1. - blend) * table[:, low, :] + blend * table[:, high, :]
        return tuple(float(weight * np.interp(hip_ctrl_rad, hip_grid, plane[:, index]))
                     for index in range(plane.shape[1]))


# Backwards-compatible name for the original hind-only class.
HindStanceCompensator = StanceCompensator
