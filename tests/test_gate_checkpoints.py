"""Contracts for the checkpoint gate scoreboard.

These test the loop-and-rank layer only. The gate values themselves come from
`eval.session2_controller.gate2`, which has its own tests; nothing here asserts
a simulated number, and a synthetic gate payload is not simulation evidence.
"""
import json
from pathlib import Path
import tempfile
import unittest

from tools.gate_checkpoints import (
    base_flags_from_config, check_values, markdown, step_number, tiebreak_distance,
)

SPEC = {"minimum_forward_m_s": .04, "minimum_net_path": .5, "maximum_hind_swing_load": .1,
        "minimum_front_stance_load": .65, "hind_duty_target": .78, "hind_duty_tolerance": .05,
        "limb_phase_target": .435, "limb_phase_tolerance": .03}


def gate(forward=.05, net=.8, swing=(.02, .03), stance=(.70, .72),
         duty=(.78, .77), phase=(.435, .44)):
    checks = {
        "signed_forward": {"value_m_s": forward, "pass": forward >= .04},
        "net_path": {"value": net, "pass": net >= .5},
        "hind_swing_load": {"values": {"HL": swing[0], "HR": swing[1]},
                            "pass": all(v < .1 for v in swing)},
        "front_stance_load": {"values": {"FL": stance[0], "FR": stance[1]},
                              "pass": all(v >= .65 for v in stance)},
        "hind_duty": {"values": {"HL": duty[0], "HR": duty[1]},
                      "pass": all(abs(v - .78) <= .05 for v in duty)},
        "limb_phase": {"values": {"HL_to_FL": phase[0], "HR_to_FR": phase[1]},
                       "pass": all(abs((v - .435 + .5) % 1 - .5) <= .03 for v in phase)},
    }
    return {"requirements": SPEC, "checks": checks,
            "pass": all(c["pass"] for c in checks.values()),
            "completed_without_fall": True, "timing_and_acquisition_valid": True}


class WorstLimbReporting(unittest.TestCase):
    def test_a_must_stay_below_check_reports_its_highest_limb(self):
        # Reporting the minimum would flatter the limb that actually fails.
        values = check_values(gate(swing=(.02, .40)))
        self.assertAlmostEqual(values["hind_swing_load"], .40)

    def test_a_must_exceed_check_reports_its_lowest_limb(self):
        values = check_values(gate(stance=(.90, .30)))
        self.assertAlmostEqual(values["front_stance_load"], .30)

    def test_a_band_check_reports_the_limb_furthest_outside(self):
        values = check_values(gate(duty=(.78, .60)))
        self.assertAlmostEqual(values["hind_duty"], .60)

    def test_an_all_passing_check_still_reports_a_real_limb(self):
        values = check_values(gate(duty=(.77, .79)))
        self.assertIn(values["hind_duty"], (.77, .79))

    def test_missing_limb_values_are_none_not_zero(self):
        payload = gate()
        payload["checks"]["hind_duty"]["values"] = {"HL": None, "HR": None}
        self.assertIsNone(check_values(payload)["hind_duty"])


class TiebreakOrdering(unittest.TestCase):
    def test_all_six_inside_their_bands_scores_zero(self):
        self.assertEqual(tiebreak_distance(gate()), 0.0)

    def test_distance_grows_with_the_worst_limb(self):
        near = tiebreak_distance(gate(stance=(.64, .70)))
        far = tiebreak_distance(gate(stance=(.30, .70)))
        self.assertGreater(far, near)
        self.assertGreater(near, 0.0)

    def test_limb_phase_distance_is_circular(self):
        # 0.915 and 0.955 sit the same distance (0.48) either side of a 0.435
        # target once the cycle wraps, so they must score identically. A linear
        # gap would read 0.48 and 0.52 and rank one above the other.
        near = tiebreak_distance(gate(phase=(.915, .435)))
        far = tiebreak_distance(gate(phase=(.955, .435)))
        self.assertAlmostEqual(near, far)
        self.assertGreater(near, 0.0)

    def test_an_unmeasurable_check_sorts_last(self):
        payload = gate()
        payload["checks"]["limb_phase"]["values"] = {"HL_to_FL": None, "HR_to_FR": None}
        self.assertEqual(tiebreak_distance(payload), float("inf"))


class ProtocolReconstruction(unittest.TestCase):
    def test_evaluation_flags_come_from_the_runs_own_config(self):
        flags = base_flags_from_config(
            {"xml_path_resolved": "body.xml", "gait_profile": "lab",
             "requested_hind_stance_compensation": "all", "residual_scale": .25}, 20., 3.)
        self.assertIn("--hind-stance-compensation", flags)
        self.assertEqual(flags[flags.index("--hind-stance-compensation") + 1], "all")
        self.assertEqual(flags[flags.index("--xml") + 1], "body.xml")
        self.assertEqual(flags[flags.index("--gait-profile") + 1], "lab")
        self.assertEqual(flags[flags.index("--duration") + 1], "20.0")

    def test_compensation_is_omitted_when_the_run_did_not_use_it(self):
        flags = base_flags_from_config(
            {"xml_path_resolved": "body.xml", "gait_profile": "legacy",
             "requested_hind_stance_compensation": False}, 20., 3.)
        self.assertNotIn("--hind-stance-compensation", flags)

    def test_hind_only_compensation_is_not_promoted_to_all(self):
        flags = base_flags_from_config(
            {"xml_path_resolved": "b.xml", "gait_profile": "lab",
             "requested_hind_stance_compensation": True}, 20., 3.)
        self.assertEqual(flags[flags.index("--hind-stance-compensation") + 1], "hind")


class Tabulation(unittest.TestCase):
    def test_the_baseline_row_is_labelled_not_numbered(self):
        rows = [{"step": -1, "values": check_values(gate()), "passed": {k: True for k in
                 ("signed_forward", "net_path", "hind_swing_load", "front_stance_load",
                  "hind_duty", "limb_phase")}, "gates_passed": 6, "stride_period_cv": .01,
                 "note": ""}]
        text = markdown(rows, "run", {})
        self.assertIn("base (policy off)", text)
        self.assertIn("6/6", text)

    def test_a_failed_row_is_kept_and_marked(self):
        text = markdown([{"step": 100, "error": "did not converge"}], "run", {})
        self.assertIn("FAILED", text)
        self.assertIn("100", text)

    def test_reward_is_not_a_column(self):
        self.assertNotIn("reward |", markdown([], "run", {}).lower())

    def test_step_numbers_sort_numerically_not_lexically(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ["step-100000", "step-2000000", "step-300000"]
            paths = [Path(tmp) / n for n in names]
            for path in paths:
                path.mkdir()
            self.assertEqual([step_number(p) for p in sorted(paths, key=step_number)],
                             [100_000, 300_000, 2_000_000])

    def test_final_bundle_suffix_still_parses(self):
        self.assertEqual(step_number(Path("step-256-final")), 256)


class BestTrainedSelection(unittest.TestCase):
    """step-0 is published before learning; crediting it to training is a lie."""

    def test_step_zero_is_not_eligible_as_the_best_trained_checkpoint(self):
        import inspect
        from tools import gate_checkpoints
        source = inspect.getsource(gate_checkpoints.main)
        self.assertIn('r["step"] > 0', source)
        self.assertNotIn('r["step"] >= 0', source)

    def test_an_irregular_gait_cannot_be_selected_however_many_gates_it_passes(self):
        # Session 3g: a harness rewarded for driving stride CV 0.0149 -> 0.2713
        # reported a limb phase the official measure disagreed with. Run 2b
        # reproduced the same shape: front stance load passed only on rows whose
        # CV was 0.16-0.38 against a 0.10 ceiling.
        import inspect
        from tools import gate_checkpoints
        from tools.fit_gate2_official import CV_CEILING
        source = inspect.getsource(gate_checkpoints.main)
        self.assertIn("CV_CEILING", source)
        self.assertIn('r["stride_period_cv"] <= CV_CEILING', source)
        self.assertEqual(gate_checkpoints.CV_CEILING, CV_CEILING)

    def test_the_ceiling_is_the_fitters_ceiling_not_a_second_definition(self):
        from tools import gate_checkpoints, fit_gate2_official
        self.assertIs(gate_checkpoints.CV_CEILING, fit_gate2_official.CV_CEILING)


if __name__ == "__main__":
    unittest.main()
