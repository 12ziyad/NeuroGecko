"""Mechanical accounting and frozen-control tests, not learning-response claims."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common.energetics import actuator_power, actuator_work, mean_absolute_actuator_power
from common.provenance import parameter_value
from envs.gecko_walk_env import DEFAULT_XML, GeckoWalkEnv
from rewards.walk_reward import DEFAULTS, WalkReward


class TotalPowerAccountingTests(unittest.TestCase):
    def test_opposing_actuator_power_does_not_cancel(self):
        self.assertEqual(actuator_power([3., -3.], [2., 2.]), (6., -6., 12.))
        work = actuator_work([3., -3.], [2., 2.], .02)
        self.assertEqual(mean_absolute_actuator_power(work[2], .02), 12.)
        self.assertEqual(sum([6., -6.]), 0.)  # Wrong net-power charge would vanish.

    def test_opposing_substep_work_does_not_cancel(self):
        first = actuator_work([2.], [3.], .01)
        second = actuator_work([2.], [-3.], .01)
        self.assertEqual(mean_absolute_actuator_power(first[2]+second[2], .02), 6.)

    def test_generalized_velocity_dimension_is_not_actuator_velocity(self):
        with self.assertRaises(ValueError):
            actuator_power(np.ones(25), np.ones(38))

    def test_static_force_is_not_mechanical_power(self):
        self.assertEqual(actuator_power([50., 80.], [0., 0.]), (0., 0., 0.))

    def test_invalid_work_time_and_alpha_fail_closed(self):
        for work, interval in ((-1., .02), (float("nan"), .02), (1., 0.), (1., float("inf"))):
            with self.assertRaises(ValueError):
                mean_absolute_actuator_power(work, interval)
        for dt in (0., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                actuator_work([1.], [1.], dt)
        for alpha in (-.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                WalkReward({"total_power_effort_alpha": alpha})

    def test_zero_residual_still_pays_for_total_base_work(self):
        reward = WalkReward({"total_power_effort_alpha": .5})
        env = SimpleNamespace(gait_profile="lab", dt=.02, residual=np.zeros(25))
        terms = reward.total_power_effort_terms(env, {"mechanical_work_abs_J": .08})
        self.assertEqual(terms["mechanical_power_abs_mean_W"], 4.)
        self.assertEqual(terms["r_total_power_effort"], -2.)
        # Same measured total work pays the same, irrespective of decomposition.
        env.residual[:] = 1.
        self.assertEqual(reward.total_power_effort_terms(env, {"mechanical_work_abs_J": .08}), terms)

    def test_zero_work_does_not_charge_residual_magnitude(self):
        reward = WalkReward({"total_power_effort_alpha": .5})
        env = SimpleNamespace(gait_profile="lab", dt=.02, residual=np.ones(25))
        terms = reward.total_power_effort_terms(env, {"mechanical_work_abs_J": 0.})
        self.assertEqual(terms["r_total_power_effort"], 0.)

    def test_offline_alpha_ramp_only_changes_charge_for_frozen_samples(self):
        reward = WalkReward()
        env = SimpleNamespace(gait_profile="lab", dt=.02)
        powers, charges = [], []
        for alpha in (0., .25, .5):  # Synthetic engineering test inputs.
            reward.set_total_power_effort_alpha(alpha)
            terms = reward.total_power_effort_terms(env, {"mechanical_work_abs_J": .08})
            powers.append(terms["mechanical_power_abs_mean_W"])
            charges.append(terms["r_total_power_effort"])
        self.assertEqual(powers, [4., 4., 4.])
        self.assertEqual(charges, [0., -1., -2.])
        # With frozen samples power/residual do not respond to alpha. Their
        # correlation is undefined if constant; this cannot pass learning Gate1.

    def test_legacy_has_no_default_effort_and_rejects_enabling_it(self):
        self.assertNotIn("total_power_effort_alpha", DEFAULTS)
        env = SimpleNamespace(gait_profile="legacy", dt=.02)
        self.assertIsNone(WalkReward().total_power_effort_terms(env, {}))
        with self.assertRaises(ValueError):
            WalkReward({"total_power_effort_alpha": .1}).total_power_effort_terms(env, {})

    def test_missing_integrated_work_does_not_fallback_to_action_or_endpoint(self):
        reward = WalkReward({"total_power_effort_alpha": .5})
        with self.assertRaises(ValueError):
            reward.total_power_effort_terms(SimpleNamespace(gait_profile="lab", dt=.02), {})


class FrozenControlEffortTests(unittest.TestCase):
    def test_lab_default_uses_registry_and_alpha_ramp_does_not_change_frozen_controls(self):
        kwargs = dict(xml_path=Path(DEFAULT_XML).with_name("gecko_body_lab_v2.xml"),
                      control_mode="cpg_residual", gait_profile="lab", reset_noise=0)
        no_charge, ramp = GeckoWalkEnv(**kwargs), GeckoWalkEnv(**kwargs)
        try:
            self.assertEqual(ramp.reward_fn.w["total_power_effort_alpha"],
                             parameter_value("lab_total_power_effort_alpha"))
            self.assertGreater(ramp.reward_fn.w["total_power_effort_alpha"], 0.)
            no_charge.reward_fn.set_total_power_effort_alpha(0.)
            no_charge.reset(seed=0)
            ramp.reset(seed=0)
            charged_nonzero_base_work = False
            for index in range(12):
                alpha = float(index)/1000.  # Explicit synthetic, offline ramp.
                ramp.reward_fn.set_total_power_effort_alpha(alpha)
                left = no_charge.step(np.zeros(no_charge.nu))
                right = ramp.step(np.zeros(ramp.nu))
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(no_charge.data.qpos, ramp.data.qpos)
                self.assertEqual(left[4]["mechanical_work_abs_J"], right[4]["mechanical_work_abs_J"])
                expected = -alpha * right[4]["mechanical_work_abs_J"] / ramp.dt
                self.assertAlmostEqual(right[4]["r_total_power_effort"], expected)
                self.assertAlmostEqual(right[1]-left[1], expected)
                charged_nonzero_base_work |= expected < 0
            self.assertTrue(charged_nonzero_base_work)
        finally:
            no_charge.close()
            ramp.close()

    def test_legacy_default_and_explicit_zero_are_bitwise_equal(self):
        kwargs = dict(control_mode="cpg_residual", gait_profile="legacy", reset_noise=0,
                      contact_thresh=.0564)
        baseline = GeckoWalkEnv(**kwargs)
        explicit_zero = GeckoWalkEnv(**kwargs, reward_cfg={"total_power_effort_alpha": 0.})
        try:
            baseline.reset(seed=0)
            explicit_zero.reset(seed=0)
            for _ in range(12):
                left = baseline.step(np.zeros(baseline.nu))
                right = explicit_zero.step(np.zeros(explicit_zero.nu))
                self.assertEqual(left[1].hex(), right[1].hex())
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(baseline.data.qpos, explicit_zero.data.qpos)
                self.assertNotIn("r_total_power_effort", left[4])
        finally:
            baseline.close()
            explicit_zero.close()


if __name__ == "__main__":
    unittest.main()
