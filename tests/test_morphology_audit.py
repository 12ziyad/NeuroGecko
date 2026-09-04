"""Software contracts, NOT assertions that the current body passes biology.

The 14 biological/engineering gates are run separately and fail explicitly:
    python -m common.morphology_audit --strict --json report.json
Run these tests with:
    python -m unittest discover -s tests -p test_morphology_audit.py -v
"""

import copy
import json
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from common.morphology_audit import (
    DEFAULT_XML, FOOT_SITES, GATE_SPECS, MEASUREMENT_SITES,
    behind_nose_svl, collect_morphology_metrics, evaluate_gates,
    format_table, sphere_surface_gap, weighted_com,
)
from common.provenance import load_registry, parameter_value


class MeasurementMathTests(unittest.TestCase):
    def test_mass_weighted_com_and_tail_exclusion(self):
        masses = np.array([0., 3., 1.])
        positions = np.array([[100., 0., 0.], [0., 0., 0.], [4., 0., 0.]])
        np.testing.assert_array_equal(weighted_com(masses, positions), [1., 0., 0.])
        np.testing.assert_array_equal(weighted_com(masses[:2], positions[:2]), [0., 0., 0.])

    def test_com_projection_is_heading_independent(self):
        nose, vent, point = np.array([2., 3., 4.]), np.array([2., 1., 4.]), np.array([9., 2., 4.])
        self.assertAlmostEqual(behind_nose_svl(nose, vent, point), .5)
        self.assertAlmostEqual(behind_nose_svl(np.zeros(3), np.array([-2., 0., 0.]), np.array([-1., 7., 0.])), .5)

    def test_interorbital_is_inner_gap_not_center_distance(self):
        centers = np.array([[0., .0083, 0.], [0., -.0083, 0.]])
        self.assertAlmostEqual(sphere_surface_gap(centers, [.0033, .0033]), .010)

    def test_invalid_measurements_raise(self):
        with self.assertRaises(ValueError):
            weighted_com(np.array([0.]), np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            behind_nose_svl(np.zeros(3), np.zeros(3), np.ones(3))


class AuditContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = collect_morphology_metrics()
        cls.registry = load_registry()

    def test_fourteen_explicit_boolean_gates(self):
        self.assertEqual(len(GATE_SPECS), 14)
        self.assertEqual(len(self.report["gates"]), 14)
        self.assertEqual(self.report["passed_count"], sum(g["passed"] for g in self.report["gates"]))
        for gate in self.report["gates"]:
            self.assertIs(type(gate["passed"]), bool)
            self.assertTrue(gate["provenance_refs"])
        self.assertIn("not kinematic validation", format_table(self.report))
        json.dumps(self.report, allow_nan=False)

    def test_one_side_failure_cannot_be_hidden_by_mean(self):
        metrics = copy.deepcopy(self.report["metrics"])
        metrics["hip_height_svl"] = {"L": .15, "R": .19}
        gates = evaluate_gates(metrics, self.registry["entries"])
        self.assertFalse(next(g for g in gates if g["id"] == "hip_height_svl")["passed"])

    def test_registry_metadata_and_corrected_ratio(self):
        self.assertAlmostEqual(parameter_value("femur_tibia_ratio"), 1.54 / 1.51)
        self.assertEqual(parameter_value("femur_tibia_ratio_range"), [.90, 1.06])
        self.assertEqual(self.registry["entries"]["hip_rot_max_range_deg"]["source_species"], "Iguana_iguana")
        self.assertEqual(self.registry["entries"]["body_mass_tolerance_kg"]["species"], "INVENTED")

    def test_instrumentation_preserves_model_and_trajectory_exactly(self):
        """Remove ONLY the additive markers/group edits to reconstruct prior physics.

        No baseline target acceptance is asserted here. Both model variants get
        the same controls, and positions, velocities and sensor outputs must be
        bit-identical throughout the rollout, not merely approximately close.
        """
        root = ET.fromstring(DEFAULT_XML.read_text(encoding="utf-8"))
        found = set()
        for parent in root.iter():
            for child in list(parent):
                name = child.get("name")
                if child.tag == "site" and name in MEASUREMENT_SITES:
                    self.assertEqual(child.get("group"), "4")
                    parent.remove(child)
                    found.add(name)
                elif child.tag == "site" and name in FOOT_SITES:
                    self.assertEqual(child.get("group"), "4")
                    child.attrib.pop("group")
        self.assertEqual(found, set(MEASUREMENT_SITES))
        previous = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        current = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
        self.assertEqual((previous.nq, previous.nv, previous.nu), (current.nq, current.nv, current.nu))
        self.assertEqual((current.nq, current.nv, current.nu), (39, 38, 25))
        self.assertEqual(current.nsite - previous.nsite, 7)
        for field in ("body_mass", "body_inertia", "body_pos", "jnt_range", "actuator_gainprm", "actuator_ctrlrange", "geom_pos", "geom_size"):
            np.testing.assert_array_equal(getattr(previous, field), getattr(current, field), err_msg=field)
        old_data, new_data = mujoco.MjData(previous), mujoco.MjData(current)
        mujoco.mj_resetDataKeyframe(previous, old_data, previous.key("stand").id)
        mujoco.mj_resetDataKeyframe(current, new_data, current.key("stand").id)
        nsteps = int(round(2.0 / current.opt.timestep))
        for step in range(nsteps):
            targets = .15 * np.sin(step * .007 + np.arange(current.nu))
            old_data.ctrl[:] = targets
            new_data.ctrl[:] = targets
            mujoco.mj_step(previous, old_data)
            mujoco.mj_step(current, new_data)
            if step % 100 == 0 or step == nsteps - 1:
                np.testing.assert_array_equal(old_data.qpos, new_data.qpos)
                np.testing.assert_array_equal(old_data.qvel, new_data.qvel)
                np.testing.assert_array_equal(old_data.sensordata, new_data.sensordata)


if __name__ == "__main__":
    unittest.main()
