"""Kinematic instrument contracts, not claims that its controller walks."""
import hashlib
import json
import unittest

import mujoco
import numpy as np

from common.morphology_audit import DEFAULT_XML as LEGACY_XML
from utils.controller_kinematics import (
    DEFAULT_XML, FOOT_ORDER, FrozenLimbKinematics, collect_diagnosis,
    geom_plane_clearance, geom_support_radius,
)


class CollisionSupportTests(unittest.TestCase):
    def test_rotated_box_support_matches_all_corners(self):
        direction = np.array([1., 2., -3.]) / np.sqrt(14.)
        size = np.array([.003, .004, .002])
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        self.assertAlmostEqual(geom_support_radius(mujoco.mjtGeom.mjGEOM_BOX, size, direction),
                               np.max(corners @ direction), places=15)

    def test_capsule_support_includes_radius_and_axial_endpoint(self):
        self.assertAlmostEqual(geom_support_radius(mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                   [.002, .01, 0], [0, 0, 1]), .012)
        self.assertAlmostEqual(geom_support_radius(mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                   [.002, .01, 0], [1, 0, 0]), .002)

    def test_ellipsoid_and_cylinder_support(self):
        kind = mujoco.mjtGeom
        self.assertAlmostEqual(geom_support_radius(kind.mjGEOM_ELLIPSOID, [.003, .004, .002], [0, 1, 0]), .004)
        self.assertAlmostEqual(geom_support_radius(kind.mjGEOM_CYLINDER, [.003, .004, 0], [0, 0, 1]), .004)

    def test_unknown_shapes_and_nonunit_directions_fail_explicitly(self):
        with self.assertRaises(ValueError):
            geom_support_radius(mujoco.mjtGeom.mjGEOM_MESH, [.1, .1, .1], [0, 0, 1])
        with self.assertRaises(ValueError):
            geom_support_radius(mujoco.mjtGeom.mjGEOM_BOX, [.1, .1, .1], [0, 0, 2])


class FrozenPoseTests(unittest.TestCase):
    def setUp(self):
        self.probe = FrozenLimbKinematics()

    def test_only_selected_actuated_limb_joints_move(self):
        probe = self.probe
        untouched = np.ones(probe.model.nq, dtype=bool)
        for aid in probe.foot_actuators["HL"]:
            untouched[probe.model.jnt_qposadr[probe.model.actuator_trnid[aid, 0]]] = False
        probe.at_phase("HL", .9)
        np.testing.assert_array_equal(probe.data.qpos[untouched], probe.stand_qpos[untouched])
        np.testing.assert_array_equal(probe.data.qvel, np.zeros(probe.model.nv))
        self.assertEqual(probe.data.time, 0.)

    def test_target_mapping_is_position_servo_length_not_normalized_action(self):
        probe = self.probe
        row = probe.at_phase("HL", .9, overrides={"knee_L": -.5})
        aid = probe.model.actuator("knee_L").id
        expected = -.5 * probe.controller.half[aid]
        self.assertAlmostEqual(row["actuator_targets"]["knee_L"]["qpos_rad"], expected)
        self.assertAlmostEqual(probe.data.actuator_length[aid], expected)
        self.assertAlmostEqual(probe.data.actuator_force[aid], 0., places=13)

    def test_every_included_geom_can_collide_with_floor_and_sites_are_irrelevant(self):
        probe = self.probe
        for foot in FOOT_ORDER:
            self.assertEqual(len(probe.foot_geoms[foot]), 6)  # one pad and five claws
            for gid in probe.foot_geoms[foot]:
                self.assertNotEqual(probe.model.geom_contype[gid], 0)
            before = probe.at_phase(foot, .5)["minimum_collision_clearance_m"]
            probe.model.site_pos[:] += 100.
            after = probe.at_phase(foot, .5)["minimum_collision_clearance_m"]
            self.assertEqual(before, after)

    def test_minimum_includes_claws_not_only_pad(self):
        probe = self.probe
        row = probe.at_phase("HL", .9, overrides={"knee_L": .6})
        pad = next(gid for gid in probe.foot_geoms["HL"] if probe.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX)
        pad_clearance = geom_plane_clearance(probe.model, probe.data, pad, probe.plane_point, probe.plane_normal)
        self.assertLess(row["minimum_collision_clearance_m"], pad_clearance)

    def test_phase_round_trip_for_both_profile_conventions(self):
        for profile, xml in (("lab", DEFAULT_XML), ("legacy", LEGACY_XML)):
            probe = FrozenLimbKinematics(xml, profile)
            for foot in FOOT_ORDER:
                for phase in (.02, .31, .91):
                    self.assertAlmostEqual(probe.controller.foot_phase_fraction(foot, probe.time_for_phase(foot, phase)), phase)

    def test_invalid_override_cannot_change_other_limb_or_exceed_range(self):
        for overrides in ({"knee_R": .2}, {"knee_L": 1.1}, {"knee_L": float("nan")}):
            with self.assertRaises(ValueError):
                self.probe.at_phase("HL", .9, overrides=overrides)

    def test_files_and_compiled_model_not_modified_by_measurements(self):
        probe = self.probe
        before = hashlib.sha256(DEFAULT_XML.read_bytes()).hexdigest()
        mass = probe.model.body_mass.copy()
        for foot in FOOT_ORDER:
            probe.at_phase(foot, .91)
        self.assertEqual(before, hashlib.sha256(DEFAULT_XML.read_bytes()).hexdigest())
        np.testing.assert_array_equal(mass, probe.model.body_mass)

    def test_complete_report_is_finite_and_labels_conditional_feedback(self):
        report = collect_diagnosis(samples=11)
        json.dumps(report, allow_nan=False)
        self.assertEqual(set(report["feet"]), set(FOOT_ORDER))
        for result in report["feet"].values():
            self.assertEqual(set(result["feedback_modes"]), {"none", "loaded", "airborne"})
            self.assertEqual(len(result["lift_sensitivity"]), 9)


if __name__ == "__main__":
    unittest.main()
