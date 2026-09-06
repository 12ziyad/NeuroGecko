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

    # ---- Session 4: sprawl drive and tail-hindlimb coupling ----------------

    def test_zero_sprawl_amplitude_changes_no_command_at_all(self):
        """The channel is additive, so off must be bit-identical to absent."""
        old = self.controller()
        new = self.controller(hind_sprawl_amplitude=0., fore_sprawl_amplitude=0.)
        for time in np.arange(40) * .013:
            np.testing.assert_array_equal(old.base_ctrl(time), new.base_ctrl(time))

    def test_sprawl_amplitude_moves_only_the_sprawl_actuators(self):
        old = self.controller()
        new = self.controller(hind_sprawl_amplitude=.5, fore_sprawl_amplitude=.5)
        sprawl = {self.model.actuator(n).id for n in
                  ("hip_sprawl_L", "hip_sprawl_R", "shoulder_sprawl_L", "shoulder_sprawl_R")}
        moved = set()
        for time in np.arange(40) * .013:
            for i, (a, b) in enumerate(zip(old.base_ctrl(time), new.base_ctrl(time))):
                if a != b:
                    moved.add(i)
        self.assertTrue(moved, "a nonzero sprawl amplitude must change something")
        self.assertEqual(moved - sprawl, set())

    def test_sprawl_is_a_bounded_sinusoid_of_the_limb_phase(self):
        controller = self.controller(hind_sprawl_amplitude=.5)
        values = [controller.sprawl_signal("HL", t / 200.) for t in range(200)]
        self.assertLessEqual(max(abs(v) for v in values), .5 + 1e-12)
        self.assertGreater(max(values), .45)
        self.assertLess(min(values), -.45)

    def test_tail_coupling_is_exactly_neutral_at_the_reference_amplitude(self):
        """Enabling the coupling must not touch the intact animal."""
        from common.provenance import parameter_value
        reference = float(parameter_value("lab_base_parameters")["tail_amp"])
        controller = self.controller(tail_hindlimb_coupling=.36, tail_amp=reference)
        self.assertEqual(controller.tail_coupling_scale(), 1.)
        plain = self.controller(tail_hindlimb_coupling=0., tail_amp=reference)
        for time in np.arange(40) * .013:
            np.testing.assert_array_equal(plain.base_ctrl(time), controller.base_ctrl(time))

    def test_removing_tail_drive_weakens_only_hind_protraction(self):
        """Caudofemoralis retracts the femur; the forelimb has no such muscle."""
        intact = self.controller(tail_hindlimb_coupling=.36)
        restricted = self.controller(tail_hindlimb_coupling=.36, tail_amp=0.)
        self.assertAlmostEqual(restricted.tail_coupling_scale(), .64)
        hind = {self.model.actuator(n).id for n in ("hip_proret_L", "hip_proret_R")}
        fore = {self.model.actuator(n).id for n in ("shoulder_proret_L", "shoulder_proret_R")}
        hind_moved = fore_moved = False
        for time in np.arange(40) * .013:
            a, b = intact.base_ctrl(time), restricted.base_ctrl(time)
            for i in hind:
                hind_moved |= a[i] != b[i]
            for i in fore:
                fore_moved |= a[i] != b[i]
        self.assertTrue(hind_moved, "hind protraction must weaken")
        self.assertFalse(fore_moved, "the forelimb must be untouched")

    def test_zero_coupling_leaves_the_tail_a_passive_pendulum(self):
        intact = self.controller(tail_hindlimb_coupling=0.)
        restricted = self.controller(tail_hindlimb_coupling=0., tail_amp=0.)
        self.assertEqual(restricted.tail_coupling_scale(), 1.)
        for time in np.arange(20) * .013:
            for i in (self.model.actuator("hip_proret_L").id,):
                self.assertEqual(intact.base_ctrl(time)[i], restricted.base_ctrl(time)[i])

    def test_a_moving_sprawl_refuses_a_sprawl_blind_stance_table(self):
        """0.42 mm of foot height per degree; a blind table stops compensating."""
        with self.assertRaisesRegex(ValueError, "sprawl-aware"):
            CPGResidualController(self.model, gait_profile=BASELINE_PROFILE, verbose=False,
                                  lab_parameters={**BASELINE_PARAMETERS, "other_amplitude": 0.,
                                                  "hind_sprawl_amplitude": .3},
                                  hind_stance_compensation="all")

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
