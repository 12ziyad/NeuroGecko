"""Engineered body-aware lab targets, explicitly separate from legacy rewards."""
import json
import unittest
from pathlib import Path

import mujoco
import numpy as np

from common.provenance import parameter_value
from envs.gecko_walk_env import DEFAULT_XML, GeckoWalkEnv
from rewards.walk_reward import DEFAULTS, WalkReward, lab_reward_calibration


LAB_XML = Path(DEFAULT_XML).with_name("gecko_body_lab.xml")


class LabRewardCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(LAB_XML))

    def test_model_weight_not_skin_sensitivity_sets_force_scale(self):
        c = lab_reward_calibration(self.model)
        expected = np.sum(self.model.body_mass[1:]) * 9.81 * parameter_value("lab_contact_bodyweight_fraction")
        self.assertAlmostEqual(c["contact_threshold_N"], expected)
        self.assertAlmostEqual(c["reward_overrides"]["front_load_force_scale"], expected)
        self.assertIn("NOT skin tactile", lab_reward_calibration.__doc__)
        self.model.body_mass[1:] *= 2
        doubled = lab_reward_calibration(self.model)
        self.assertAlmostEqual(doubled["contact_threshold_N"], 2*expected)
        self.assertEqual(doubled["reward_overrides"]["trunk_height_target"], c["reward_overrides"]["trunk_height_target"])

    def test_new_body_force_preserves_legacy_bodyweight_fraction(self):
        c = lab_reward_calibration(self.model)
        old = lab_reward_calibration(mujoco.MjModel.from_xml_path(str(DEFAULT_XML)))
        self.assertAlmostEqual(old["contact_threshold_N"], .0564)
        self.assertAlmostEqual(c["contact_threshold_N"], .0564*.038/.0612)

    def test_height_guard_uses_both_girdle_offsets_and_svl(self):
        c = lab_reward_calibration(self.model)
        inputs, guard = c["model_inputs"], c["height_guard"]
        implied = [parameter_value(g+"_height_svl")*inputs["neutral_svl_m"]-
                   inputs["stand_girdle_offsets_from_root_m"][g] for g in ("hip", "shoulder")]
        self.assertAlmostEqual(guard["nominal_root_target_m"], np.mean(implied))
        w = c["reward_overrides"]
        self.assertGreater(w["trunk_height_target"], w["trunk_height_min"])
        self.assertLess(w["trunk_height_target"], DEFAULTS["trunk_height_min"])

    def test_lab_neutral_not_penalized_by_legacy_height_guard(self):
        env = GeckoWalkEnv(xml_path=LAB_XML, gait_profile="lab", reset_noise=0)
        try:
            env.reset(seed=1)
            _, dist, head = env._target_egocentric()
            metrics, _ = env._step_metrics(dist, head, 1., False, False)
            # Exercise a fully open speed gate without physically moving the
            # stand pose: this isolates its height penalty from gait/contact.
            metrics.update(progress=.09, forward_speed=.09)
            _, info = env.reward_fn(env, np.zeros(env.nu), metrics)
            self.assertEqual(info["clean_gait_speed_gate"], 1.)
            self.assertEqual(info["trunk_support_score"], 1.)
            self.assertEqual(info["r_low_trunk"], 0.)
            _, old_info = WalkReward()(env, np.zeros(env.nu), metrics)
            self.assertEqual(old_info["trunk_support_score"], 0.)
            self.assertLess(old_info["r_low_trunk"], 0.)
        finally:
            env.close()

    def test_stride_frequency_and_speed_are_one_consistent_choice(self):
        c = lab_reward_calibration(self.model)
        w = c["reward_overrides"]
        svl = c["model_inputs"]["neutral_svl_m"]
        self.assertAlmostEqual(w["target_speed"], .72*svl*1.1888)
        self.assertTrue(.09 <= w["target_speed"] <= .10)
        self.assertGreater(w["target_speed"], w["speed_floor"])
        self.assertGreater(w["speed_track_sigma"], 0)
        self.assertEqual(w["slow_penalty"], 0)
        self.assertEqual(w["front_duty_fl_target"], .70)
        self.assertEqual(w["front_duty_fr_target"], .70)

    def test_lab_env_resolves_defaults_and_reports_superseded_preset(self):
        cfg = dict(target_speed=.12, trunk_height_min=.028, slow_penalty=10., speed_track=6., progress=13.)
        original = dict(cfg)
        env = GeckoWalkEnv(xml_path=LAB_XML, gait_profile="lab", reward_cfg=cfg)
        try:
            self.assertEqual(cfg, original)
            self.assertEqual(env.reward_fn.w["progress"], 13.)
            self.assertEqual(env.reward_fn.w["speed_track"], 6.)
            self.assertEqual(env.reward_fn.w["slow_penalty"], 0.)
            self.assertNotEqual(env.reward_fn.w["target_speed"], cfg["target_speed"])
            self.assertEqual(set(env.reward_calibration["superseded_reward_cfg"]), {"target_speed", "trunk_height_min", "slow_penalty"})
            self.assertEqual(env.contact_threshold, env.reward_calibration["contact_threshold_N"])
            json.dumps(env.reward_calibration, allow_nan=False)
        finally:
            env.close()

    def test_explicit_lab_threshold_honored_not_reused_as_load_scale(self):
        env = GeckoWalkEnv(xml_path=LAB_XML, gait_profile="lab", contact_thresh=.001)
        try:
            self.assertEqual(env.contact_threshold, .001)
            self.assertEqual(env.reward_calibration["contact_threshold_source"], "explicit caller override")
            self.assertNotEqual(env.reward_fn.w["front_load_force_scale"], .001)
        finally:
            env.close()

    def test_bad_lab_threshold_fails(self):
        for threshold in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                GeckoWalkEnv(xml_path=LAB_XML, gait_profile="lab", contact_thresh=threshold)

    def test_legacy_defaults_and_explicit_reward_cfg_untouched(self):
        env = GeckoWalkEnv(reward_cfg={"target_speed": .13, "slow_penalty": 2.})
        try:
            self.assertEqual(env.contact_threshold, 1e-6)
            self.assertEqual(env.reward_fn.w["target_speed"], .13)
            self.assertEqual(env.reward_fn.w["slow_penalty"], 2.)
            self.assertEqual(env.reward_fn.w["trunk_height_min"], .028)
            self.assertEqual(env.reward_fn.w["front_load_force_scale"], .0564)
            self.assertEqual(env.reward_fn.w["front_duty_fl_target"], .45)
            self.assertEqual(env.reward_fn.w["front_duty_fr_target"], .35)
        finally:
            env.close()

    def test_calibration_does_not_mutate_model_or_existing_data(self):
        data = mujoco.MjData(self.model)
        data.qpos[0] = .4
        arrays = [self.model.qpos0, self.model.key_qpos, self.model.body_mass, self.model.opt.gravity, data.qpos, data.qvel]
        before = [a.copy() for a in arrays]
        lab_reward_calibration(self.model)
        for original, actual in zip(before, arrays):
            np.testing.assert_array_equal(original, actual)

    def test_unsupported_geometry_gravity_and_profile_fail_fast(self):
        with self.assertRaises(ValueError):
            lab_reward_calibration(self.model, "legacy")
        self.model.opt.gravity[0] = 1.
        with self.assertRaisesRegex(ValueError, "vertical gravity"):
            lab_reward_calibration(self.model)
        minimal = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".01"/></body></worldbody></mujoco>')
        with self.assertRaisesRegex(ValueError, "nose_tip"):
            lab_reward_calibration(minimal)


if __name__ == "__main__":
    unittest.main()
