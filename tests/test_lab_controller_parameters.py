"""Opt-in mechanical controls and immutable Session 2 baseline contracts."""
import json
import unittest

import mujoco
import numpy as np

from common.gait_config import GaitProfile, FOOT_ORDER
from common.morphology_audit import DEFAULT_XML as LEGACY_XML, REPO_ROOT
from envs.cpg_residual_controller import CPGResidualController
from utils.controller_kinematics import DEFAULT_XML, FrozenLimbKinematics


# Versioned compatibility fixture, independent of later selected registry defaults.
BASELINE_PARAMETERS = {
    "mirror_left_fa": False, "hind_lift_multiplier": 1.,
    "front_stance_press": .4, "front_stance_press_fr": .5,
    "front_swing_lift": .4, "front_stance_seek": .12,
    "front_seek_relax": .35, "shoulder_sprawl_tuck": .3,
    "fore_hind_amplitude_ratio": 8/9, "hind_fa_amplitude": .68,
    "heading_gain": 0., "continuous_front_lift": False,
    "other_amplitude": .15,
    "other_amplitude": .15,
}
BASELINE_PROFILE = GaitProfile("lab", 1.1888, (0., .44, .5, .94), (.765, .7, .765, .7))


class LabParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))

    def controller(self, **parameters):
        return CPGResidualController(self.model, gait_profile=BASELINE_PROFILE, verbose=False,
                                     lab_parameters={**BASELINE_PARAMETERS, **parameters})

    def phase_time(self, controller, foot, phase):
        return ((phase + controller.phase[foot]) % 1.) / controller.freq

    def test_explicit_old_lab_controls_match_committed_pre_edit_landmarks_exactly(self):
        evidence = json.loads((REPO_ROOT / "artifacts/evidence/session2_kinematics_baseline_lab.json").read_text())
        controller = self.controller()
        for foot in FOOT_ORDER:
            for mode, results in evidence["feet"][foot]["feedback_modes"].items():
                contact = None if mode == "none" else {"FL": mode == "loaded", "FR": mode == "loaded"}
                for landmark in results["landmarks"].values():
                    current = controller.compute(np.zeros(self.model.nu), landmark["time_s"], front_contact=contact)
                    for name, saved in landmark["actuator_targets"].items():
                        self.assertEqual(current[self.model.actuator(name).id], saved["ctrl"], name)

    def test_left_fa_mirror_changes_only_left_protraction_commands(self):
        old, changed = self.controller(), self.controller(mirror_left_fa=True)
        left_ids = {self.model.actuator(name).id for name in ("hip_proret_L", "shoulder_proret_L")}
        for time in np.arange(30) * .017:
            before, after = old.base_ctrl(time), changed.base_ctrl(time)
            for aid in range(self.model.nu):
                self.assertEqual(after[aid], -before[aid] if aid in left_ids else before[aid])

    def test_requested_ratio_is_angular_not_identical_normalized_amplitude(self):
        controller = self.controller(fore_hind_amplitude_ratio=.54)
        for side in ("L", "R"):
            amplitudes = []
            for foot, actuator in (("H" + side, "hip_proret_" + side), ("F" + side, "shoulder_proret_" + side)):
                aid = self.model.actuator(actuator).id
                td = controller.base_ctrl(self.phase_time(controller, foot, 0.))[aid]
                off = controller.base_ctrl(self.phase_time(controller, foot, controller.stance_for(foot)))[aid]
                amplitudes.append(abs(td - off))
            self.assertAlmostEqual(amplitudes[1] / amplitudes[0], .54)

    def test_zero_other_amplitude_changes_only_mapped_other_channels(self):
        original = self.controller(shoulder_sprawl_tuck=0.)
        changed = self.controller(shoulder_sprawl_tuck=0., other_amplitude=0.)
        other_ids = {aid for aid, _, role in original.entries if role == "other"}
        self.assertEqual(len(other_ids), 8)  # three hind channels per side, two shoulders
        for time in np.arange(30) * .017:
            before, after = original.base_ctrl(time), changed.base_ctrl(time)
            for aid in range(self.model.nu):
                self.assertEqual(after[aid], changed.neutral[aid] if aid in other_ids else before[aid])

    def test_signed_hind_lift_and_total_ctrl_clipping(self):
        controller = self.controller(hind_lift_multiplier=-2.)
        for foot in ("HL", "HR"):
            time = self.phase_time(controller, foot, (1+controller.stance_for(foot))/2)
            aid = self.model.actuator("knee_" + foot[1]).id
            self.assertLess(controller.base_ctrl(time)[aid], controller.lo[aid])
            self.assertEqual(controller.compute(np.zeros(self.model.nu), time)[aid], controller.lo[aid])

    def test_continuous_front_option_joins_both_boundaries_with_nonzero_press(self):
        controller = self.controller(continuous_front_lift=True, front_stance_press=.1,
                                     front_stance_press_fr=.2, front_swing_lift=-.8)
        for foot in ("FL", "FR"):
            aid = self.model.actuator("elbow_" + foot[1]).id
            for contact in (None, {"FL": True, "FR": True}, {"FL": False, "FR": False}):
                # This mode intentionally disables seek/relax; the stance target
                # must not jump when a contact-dependent branch changes.
                for boundary in (0., controller.stance_for(foot)):
                    phases = ((boundary-1e-10)%1, boundary, (boundary+1e-10)%1)
                    values = [controller.base_ctrl(self.phase_time(controller, foot, phase), front_contact=contact)[aid]
                              for phase in phases]
                    self.assertLess(max(values)-min(values), 2e-9)

    def test_heading_sign_and_zero_gain_preservation(self):
        base = self.controller(mirror_left_fa=True)
        steered = self.controller(mirror_left_fa=True, heading_gain=.5)
        for time in (.1, .23):
            np.testing.assert_array_equal(base.base_ctrl(time), base.base_ctrl(time, heading_error=10.))
            a = base.base_ctrl(time)
            b = steered.base_ctrl(time, heading_error=1.)
            for foot in FOOT_ORDER:
                name = ("hip_proret_" if foot.startswith("H") else "shoulder_proret_") + foot[1]
                aid = self.model.actuator(name).id
                self.assertAlmostEqual(b[aid], a[aid] * (.5 if foot.endswith("L") else 1.5))

    def test_parameter_validation_and_read_only_snapshot(self):
        for invalid in ({"unknown": 1.}, {"mirror_left_fa": 1}, {"hind_lift_multiplier": float("nan")},
                        {"heading_gain": -1.}, {"hind_fa_amplitude": -.1}):
            with self.assertRaises(ValueError):
                self.controller(**invalid)
        controller = self.controller()
        with self.assertRaises(TypeError):
            controller.lab_parameters["hind_lift_multiplier"] = -1.
        with self.assertRaises(ValueError):
            controller.base_ctrl(.1, heading_error=float("nan"))

    def test_legacy_rejects_lab_dictionary_and_ignores_new_heading_channel(self):
        model = mujoco.MjModel.from_xml_path(str(LEGACY_XML))
        with self.assertRaises(ValueError):
            CPGResidualController(model, verbose=False, lab_parameters={})
        legacy = CPGResidualController(model, verbose=False)
        np.testing.assert_array_equal(legacy.compute(np.zeros(model.nu), .3),
                                      legacy.compute(np.zeros(model.nu), .3, heading_error=10.))

    def test_proposed_sign_tuck_lift_controls_improve_frozen_clearance_not_gait_claim(self):
        probe = FrozenLimbKinematics()
        probe.controller = CPGResidualController(probe.model, gait_profile=BASELINE_PROFILE, verbose=False,
            lab_parameters={**BASELINE_PARAMETERS, "mirror_left_fa": True, "hind_lift_multiplier": -4/3,
                            "shoulder_sprawl_tuck": 0., "front_stance_press": 0., "front_stance_press_fr": 0.,
                            "continuous_front_lift": True, "front_swing_lift": -.8})
        for foot in FOOT_ORDER:
            stance = probe.controller.stance_for(foot)
            stand = probe.at_phase(foot, stance/2)["minimum_collision_clearance_m"]
            swing = probe.at_phase(foot, (1+stance)/2)["minimum_collision_clearance_m"]
            self.assertGreater(swing, stand)
            self.assertGreater(swing, 0.)
            touchdown = probe.at_phase(foot, 0.)["pad_forward_m"]
            liftoff = probe.at_phase(foot, stance-1e-9)["pad_forward_m"]
            self.assertLess(liftoff, touchdown)


if __name__ == "__main__":
    unittest.main()
