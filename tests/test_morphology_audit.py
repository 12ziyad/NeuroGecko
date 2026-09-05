"""Software contracts, NOT assertions that the current body passes biology.

The 14 biological/engineering gates are run separately and fail explicitly:
    python -m common.morphology_audit --strict --json report.json
Run these tests with:
    python -m unittest discover -s tests -p test_morphology_audit.py -v
"""

import copy
import json
from pathlib import Path
import tempfile
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
from utils.build_lab_morphology import (DEFAULT_OUTPUT, V2_OUTPUT, bounded_entropy_fit,
                                      canonical_source_sha256, canonical_source_text,
                                      extreme_masses, make_candidate)


def assert_generated_model_equivalent(testcase, generated, recorded):
    """Compare all model structure/attributes without requiring last-bit text.

    Geometric/dynamic numeric attributes get only serialization-scale tolerance
    (1e-14 absolute, 2e-13 relative). The settled stand qpos alone gets 1e-9
    absolute tolerance: one nanometre for translation / nanoradian for hinges.
    This tolerance applies ONLY to reproducibility tests, never research gates.
    """
    source_marker = f"canonical-LF SHA256 {canonical_source_sha256(DEFAULT_XML)}."
    testcase.assertIn(source_marker, generated)
    testcase.assertIn(source_marker, recorded)
    left, right = list(ET.fromstring(generated).iter()), list(ET.fromstring(recorded).iter())
    testcase.assertEqual(len(left), len(right), "XML element count changed")
    for index, (actual, expected) in enumerate(zip(left, right)):
        where = f"element {index}: {actual.tag} {actual.get('name', '')}"
        testcase.assertEqual(actual.tag, expected.tag, where)
        testcase.assertEqual(set(actual.attrib), set(expected.attrib), where)
        for attribute in actual.attrib:
            current_value, saved_value = actual.get(attribute), expected.get(attribute)
            if current_value == saved_value:
                continue
            try:
                current_numbers = np.array([float(x) for x in current_value.split()])
                saved_numbers = np.array([float(x) for x in saved_value.split()])
            except ValueError:
                testcase.assertEqual(current_value, saved_value, f"{where} attribute {attribute}")
                continue
            testcase.assertEqual(current_numbers.shape, saved_numbers.shape, f"{where} attribute {attribute}")
            settled_state = actual.tag == "key" and actual.get("name") == "stand" and attribute == "qpos"
            np.testing.assert_allclose(current_numbers, saved_numbers,
                                       atol=1e-9 if settled_state else 1e-14,
                                       rtol=0 if settled_state else 2e-13,
                                       err_msg=f"{where} attribute {attribute}")


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


class LabCandidateTests(unittest.TestCase):
    def test_generator_is_reproducible_and_does_not_change_legacy(self):
        before = DEFAULT_XML.read_bytes()
        generated, derivations = make_candidate()
        assert_generated_model_equivalent(self, generated, DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(before, DEFAULT_XML.read_bytes())
        self.assertAlmostEqual(derivations["actuator_scale"], .038 / .0612)

    def test_candidate_compiles_without_action_or_observation_topology_change(self):
        legacy = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
        candidate = mujoco.MjModel.from_xml_path(str(DEFAULT_OUTPUT))
        self.assertEqual((legacy.nq, legacy.nv, legacy.nu, legacy.nsensor),
                         (candidate.nq, candidate.nv, candidate.nu, candidate.nsensor))
        self.assertEqual(len(candidate.key("stand").qpos), candidate.nq)
        self.assertEqual([legacy.actuator(i).name for i in range(legacy.nu)],
                         [candidate.actuator(i).name for i in range(candidate.nu)])

    def test_regenerated_stand_stays_finite_with_all_feet_and_no_belly_contact(self):
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_OUTPUT))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        data.ctrl[:] = 0
        for _ in range(round(3.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        self.assertTrue(np.all(np.isfinite(data.qvel)))
        for foot in ("fore_L", "fore_R", "hind_L", "hind_R"):
            self.assertGreater(float(data.sensor(f"touch_{foot}").data[0]), 0.0)
        for region in ("mid", "post"):
            self.assertEqual(float(data.sensor(f"touch_belly_{region}").data[0]), 0.0)


class InverseMassCalibrationTests(unittest.TestCase):
    def test_entropy_solver_matches_known_two_mass_solution(self):
        masses, info = bounded_entropy_fit([1., 1.], [0., 1.], 2., .75, [.1, .1], [2., 2.])
        np.testing.assert_allclose(masses, [.5, 1.5], atol=1e-10)
        self.assertTrue(info["target_feasible"])

    def test_infeasible_mean_is_reported_not_silently_claimed(self):
        masses, info = bounded_entropy_fit([1., 1.], [0., 1.], 2., .05, [.5, .5], [1.5, 1.5])
        np.testing.assert_array_equal(masses, [1.5, .5])
        self.assertFalse(info["target_feasible"])
        self.assertEqual(info["feasible_centroid_range"], [.25, .75])

    def test_impossible_total_mass_raises_before_writing(self):
        with self.assertRaises(ValueError):
            extreme_masses([0., 1.], [.1, .1], [.5, .5], total=2.)

    def test_v2_changes_only_mass_inertia_and_regenerated_stand(self):
        v1 = mujoco.MjModel.from_xml_path(str(DEFAULT_OUTPUT))
        v2 = mujoco.MjModel.from_xml_path(str(V2_OUTPUT))
        for field in ("body_pos", "body_quat", "geom_pos", "geom_quat", "geom_size", "site_pos", "site_size", "jnt_range", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_ctrlrange"):
            np.testing.assert_array_equal(getattr(v1, field), getattr(v2, field), err_msg=field)
        self.assertEqual((v1.nq, v1.nv, v1.nu), (v2.nq, v2.nv, v2.nu))
        self.assertAlmostEqual(v1.body_mass.sum(), v2.body_mass.sum(), places=12)
        tails = [v2.body(f"tail{i}").id for i in range(1, 6)]
        self.assertAlmostEqual(v1.body_mass[tails].sum(), v2.body_mass[tails].sum(), places=12)
        ratios = v2.body_mass[1:] / v1.body_mass[1:]
        np.testing.assert_allclose(v2.body_inertia[1:], v1.body_inertia[1:] * ratios[:, None], rtol=1e-12, atol=1e-18)
        self.assertTrue(np.all(v2.body_inertia[1:] > 0))
        self.assertTrue(np.all(np.isfinite(v2.body_inertia)))
        self.assertTrue(np.all(2 * v2.body_inertia[1:].max(axis=1) <= v2.body_inertia[1:].sum(axis=1) + 1e-16))
        for part in ("humerus", "forearm", "manus", "femur", "tibia", "pes"):
            self.assertEqual(v2.body(f"{part}_L").mass[0], v2.body(f"{part}_R").mass[0])

    def test_v2_reproducible_and_exact_mean_limit_explicit(self):
        generated, derivations = make_candidate(fit_com=True)
        assert_generated_model_equivalent(self, generated, V2_OUTPUT.read_text(encoding="utf-8"))
        fitted = derivations["inverse_mass_calibration"]
        self.assertTrue(fitted["body_fit"]["target_feasible"])
        self.assertFalse(fitted["tail_fit"]["target_feasible"])
        self.assertGreater(fitted["geometry_lower_bound"]["minimum_first_segment_density_kg_m3"],
                           fitted["tail_fit"]["guard_density_kg_m3"])


class CrossPlatformReproducibilityTests(unittest.TestCase):
    def test_lf_and_crlf_sources_generate_identical_model_text(self):
        canonical = canonical_source_text(DEFAULT_XML)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for name, newline in (("lf", "\n"), ("crlf", "\r\n")):
                folder = root / name
                folder.mkdir()
                source = folder / DEFAULT_XML.name
                source.write_bytes(canonical.replace("\n", newline).encode("utf-8"))
                sources.append(source)
            self.assertEqual(canonical_source_sha256(sources[0]), canonical_source_sha256(sources[1]))
            first, first_evidence = make_candidate(source=sources[0])
            second, second_evidence = make_candidate(source=sources[1])
            self.assertEqual(first, second)
            self.assertEqual(first_evidence["source_canonical_lf_sha256"], second_evidence["source_canonical_lf_sha256"])
            self.assertNotEqual(first_evidence["source_raw_file_sha256"], second_evidence["source_raw_file_sha256"])

    def test_model_comparison_rejects_geometry_changes(self):
        recorded = DEFAULT_OUTPUT.read_text(encoding="utf-8")
        changed = ET.fromstring(recorded)
        geom = next(g for g in changed.iter("geom") if g.get("mass") and float(g.get("mass")) > 0)
        geom.set("mass", str(float(geom.get("mass")) * 1.01))
        # Preserve the hash marker when serializing without comments; the test
        # must fail on the changed physical attribute, not missing provenance.
        marker = f"<!-- canonical-LF SHA256 {canonical_source_sha256(DEFAULT_XML)}. -->\n"
        with self.assertRaises(AssertionError):
            assert_generated_model_equivalent(self, marker + ET.tostring(changed, encoding="unicode"), recorded)

    def test_model_comparison_tolerates_only_tiny_stand_roundoff(self):
        recorded = DEFAULT_OUTPUT.read_text(encoding="utf-8")
        changed = ET.fromstring(recorded)
        key = changed.find(".//key[@name='stand']")
        state = np.fromstring(key.get("qpos"), sep=" ")
        state[0] += 1e-11
        key.set("qpos", " ".join(map(str, state)))
        marker = f"<!-- canonical-LF SHA256 {canonical_source_sha256(DEFAULT_XML)}. -->\n"
        assert_generated_model_equivalent(self, marker + ET.tostring(changed, encoding="unicode"), recorded)
        state[0] += 1e-5
        key.set("qpos", " ".join(map(str, state)))
        with self.assertRaises(AssertionError):
            assert_generated_model_equivalent(self, marker + ET.tostring(changed, encoding="unicode"), recorded)


if __name__ == "__main__":
    unittest.main()
