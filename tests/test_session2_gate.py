"""Engineering gate contracts; synthetic numbers are not simulation evidence."""
import copy
import unittest
from eval.session2_controller import gate2


class GateContracts(unittest.TestCase):
    def test_failed_lab_gate_blocks_training_without_starting_a_learner(self):
        from train.train_walk_ppo import require_lab_training_readiness
        with self.assertRaisesRegex(ValueError, "Gate 2 failed"):
            require_lab_training_readiness("lab")
        self.assertIsNone(require_lab_training_readiness("legacy"))

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
