"""Engineering gate contracts; synthetic numbers are not simulation evidence."""
import copy
import unittest
from eval.session2_controller import gate2


class GateContracts(unittest.TestCase):
    # Session 4 replaced the unconditional lab block with an evidence contract.
    # Gate 2 scores the base with the policy OFF, so it gates the foundation and
    # not the trained walker; what must be proved is that the base being trained
    # on is the corrected one, and that it is recorded. See docs/BLOCKED.md.
    BASE = "artifacts/evidence/session4/gate2_lab_base_session4.json"
    PRE_SESSION4 = "artifacts/evidence/session3/gate2_all_limbs.json"
    LAB_BODY = "morphology/gecko_body_lab_v2.xml"

    def readiness(self, pre_session4_registry=False, **overrides):
        """pre_session4_registry rolls the coupling back to its no-op value so a
        Session 3 report is legitimately comparable, which is the only way to
        exercise the older rejection reasons now that the channel is live."""
        import argparse
        from unittest.mock import patch
        from common.provenance import parameter_value
        from train.train_walk_ppo import require_lab_training_readiness
        defaults = dict(gait_profile="lab", lab_base_evidence=self.BASE,
                        xml_path=self.LAB_BODY, hind_stance_compensation="all")
        defaults.update(overrides)
        args = argparse.Namespace(**defaults)
        if not pre_session4_registry:
            return require_lab_training_readiness(args)
        registry = dict(parameter_value("lab_base_parameters"))
        registry["tail_hindlimb_coupling"] = 0.0
        # Only the controller parameters are rolled back; every other registry
        # lookup must still reach the real registry.
        rolled_back = lambda name: registry if name == "lab_base_parameters" else parameter_value(name)
        with patch("train.train_walk_ppo.parameter_value", side_effect=rolled_back):
            return require_lab_training_readiness(args)

    def test_legacy_training_needs_no_evidence_and_is_unchanged(self):
        self.assertIsNone(self.readiness(gait_profile="legacy", lab_base_evidence=None,
                                         xml_path=None, hind_stance_compensation="off"))

    def test_lab_training_refuses_without_measured_base_evidence(self):
        with self.assertRaisesRegex(ValueError, "lab-base-evidence"):
            self.readiness(lab_base_evidence=None)

    def test_lab_training_refuses_the_default_legacy_body(self):
        with self.assertRaisesRegex(ValueError, "explicit --xml-path"):
            self.readiness(xml_path=None)

    def test_evidence_must_have_measured_the_body_being_trained(self):
        with self.assertRaisesRegex(ValueError, "different body"):
            self.readiness(xml_path="morphology/gecko_body_r.xml")

    def test_evidence_must_match_the_requested_stance_compensation(self):
        with self.assertRaisesRegex(ValueError, "stance compensation"):
            self.readiness(hind_stance_compensation="off")
        with self.assertRaisesRegex(ValueError, "stance compensation"):
            self.readiness(pre_session4_registry=True, hind_stance_compensation="off",
                           lab_base_evidence=self.PRE_SESSION4)

    def test_an_uncorrected_base_is_refused_on_its_hind_duty(self):
        # The scope of compensation is not recorded by realism_metrics, so hind
        # duty is what separates hind-only from all-limb on this body.
        with self.assertRaisesRegex(ValueError, "hind duty"):
            self.readiness(pre_session4_registry=True,
                           lab_base_evidence="artifacts/evidence/session3/gate2_with_compensation.json")

    def test_a_different_controller_parameter_set_is_refused(self):
        with self.assertRaisesRegex(ValueError, "different effective lab controller"):
            self.readiness(lab_base_evidence="artifacts/evidence/session3/gate2_ADOPTED.json")

    def test_a_short_window_measurement_is_refused(self):
        # Session 3h: limb phase reads bimodally at 12 s on this configuration.
        with self.assertRaisesRegex(ValueError, "duration"):
            self.readiness(pre_session4_registry=True,
                           lab_base_evidence="artifacts/evidence/session3/gate2_baseline12.json")

    def test_evidence_predating_a_live_channel_is_refused(self):
        """Session 4 turned tail_hindlimb_coupling on, so a report measured
        before it existed no longer proves what base the run will use, even
        though the intact gait is bit-identical."""
        with self.assertRaisesRegex(ValueError, "no-op"):
            self.readiness(lab_base_evidence=self.PRE_SESSION4)

    def test_the_session4_base_reproduces_the_session3_measurements_exactly(self):
        import json
        old = json.loads(open(self.PRE_SESSION4, encoding="utf-8").read())["episodes"][0]["gait"]
        new = json.loads(open(self.BASE, encoding="utf-8").read())["episodes"][0]["gait"]
        number = lambda v: v["mean"] if isinstance(v, dict) else v
        for key in ("forward_speed_m_s", "net_displacement_m", "distance_path_m"):
            self.assertAlmostEqual(number(old[key]), number(new[key]), places=12, msg=key)
        for foot in ("HL", "HR"):
            self.assertAlmostEqual(number(old["limbs"][foot]["duty_factor"]),
                                   number(new["limbs"][foot]["duty_factor"]), places=12)

    def test_the_accepted_contract_records_what_is_still_unmet(self):
        contract = self.readiness()
        self.assertIn("limb_phase", contract["report_derivable_gates_unmet"])
        self.assertEqual(contract["hind_stance_compensation"], "all")
        self.assertTrue(contract["evidence_sha256"])
        # Front/hind contact loads need the trace and are deliberately not
        # re-derived here; that limitation is carried, not hidden.
        self.assertIn("loads", contract["loads_not_rederived"])

    def test_older_evidence_is_accepted_only_via_the_controller_defaults(self):
        # spine_amp/tail_amp/tail_phase_lag entered the registry in Session 3g at
        # the controller's existing defaults, so a pre-3g report omits them.
        contract = self.readiness()
        # The Session 4 base was measured against the current registry, so
        # nothing has to be defaulted in on its behalf.
        self.assertEqual(contract["parameters_defaulted_in_older_evidence"], [])

    def test_a_later_channel_at_a_non_no_op_value_is_refused(self):
        # A report predating a channel proves nothing about a run that turns it
        # on, so acceptance must fail rather than assume the older run matched.
        import argparse
        from unittest.mock import patch
        from train.train_walk_ppo import require_lab_training_readiness
        from common.provenance import parameter_value
        registry = dict(parameter_value("lab_base_parameters"))
        registry["hind_sprawl_amplitude"] = 0.25
        rolled = lambda name: registry if name == "lab_base_parameters" else parameter_value(name)
        with patch("train.train_walk_ppo.parameter_value", side_effect=rolled):
            # The pre-Session-4 report omits the key entirely, which is the
            # path this guard exists for.
            with self.assertRaisesRegex(ValueError, "no-op"):
                require_lab_training_readiness(argparse.Namespace(
                    gait_profile="lab", lab_base_evidence=self.PRE_SESSION4,
                    xml_path=self.LAB_BODY, hind_stance_compensation="all"))

    def setUp(self):
        self.result={"limbs":{f:{"duty_factor":{"mean":.78}} for f in ("HL","HR")},
                     "limb_phase":{f:{"mean_cycle":.435} for f in ("HL_to_FL","HR_to_FR")},
                     "contact_quality":dict(entrainment_pass=True,single_digit_period_cv_pass=True,acquisition_at_least_200_hz=True)}
        self.diagnosis={"feet":{f:{"commanded_swing":{"raw_loaded_fraction":.09},"commanded_stance":{"raw_loaded_fraction":.66}} for f in ("HL","FL","HR","FR")},
                        "motion":{"signed_body_forward_speed_m_s":{"mean":.04},"net_path_ratio":.5}}

    def test_all_requirements_and_completion_needed(self):
        self.assertTrue(gate2({},self.result,self.diagnosis,True)["pass"])
        self.assertFalse(gate2({},self.result,self.diagnosis,False)["pass"])
        for key in self.result["contact_quality"]:
            result=copy.deepcopy(self.result)
            result["contact_quality"][key]=False
            self.assertFalse(gate2({},result,self.diagnosis,True)["pass"])

    def test_a_bad_individual_limb_cannot_hide_in_average(self):
        self.diagnosis["feet"]["HR"]["commanded_swing"]["raw_loaded_fraction"] = .1
        gate=gate2({},self.result,self.diagnosis,True)
        self.assertFalse(gate["checks"]["hind_swing_load"]["pass"])
        self.assertFalse(gate["pass"])

    def test_missing_speed_is_failure_not_zero_or_exception(self):
        self.diagnosis["motion"]["signed_body_forward_speed_m_s"]["mean"]=None
        self.assertFalse(gate2({},self.result,self.diagnosis,True)["pass"])
