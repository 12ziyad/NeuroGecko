"""Opt-in timing tests and checkpoint compatibility regression fixtures.

Fixtures were captured before the gait-profile refactor on the unchanged legacy
body, MuJoCo3.9. The local80-step qpos+qvel+obs+reward SHA256 also matched exactly
before/after: 607df971c3746ea0c4c467cb14888546c23aef8615e9c20cb33787629410461c.
That platform-specific digest is documented, not a cross-platform numerical gate.
"""
import unittest
from dataclasses import FrozenInstanceError

import mujoco
import numpy as np

from common.gait_config import FOOT_ORDER, LOCKED_FREQUENCY_HZ, get_gait_profile
from envs.cpg_residual_controller import CPGResidualController
from envs.gecko_walk_env import DEFAULT_XML, GeckoWalkEnv, _GAIT_FEET
from rewards.gait_prior import LateralSequenceCPG


class GaitProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))

    def test_registry_lab_profile(self):
        profile = get_gait_profile("lab")
        self.assertEqual(profile.frequency_hz, LOCKED_FREQUENCY_HZ)
        self.assertEqual(profile.touchdown_delays, {"HL":0., "FL":.44, "HR":.5, "FR":.94})
        self.assertEqual(profile.stance_by_foot, {"HL":.765, "FL":.70, "HR":.765, "FR":.70})
        with self.assertRaises(FrozenInstanceError):
            profile.frequency_hz = 2

    def test_contacts_identical_at_1000_control_samples(self):
        controller = CPGResidualController(self.model, gait_profile="lab", verbose=False)
        reward = LateralSequenceCPG(gait_profile="lab")
        for t in np.arange(1000)*.02:
            np.testing.assert_array_equal(controller.commanded_contact_array(t), reward.target_contact_array(t))
            # Also test a non-default caller order, so labels never silently swap.
            reverse = tuple(reversed(FOOT_ORDER))
            np.testing.assert_array_equal(controller.commanded_contact_array(t, reverse), reward.target_contact_array(t, reverse))

    def test_exact_touchdown_liftoff_and_cycle_wrap(self):
        controller = CPGResidualController(self.model, gait_profile="lab", verbose=False)
        reward = LateralSequenceCPG(gait_profile="lab")
        profile = controller.profile
        for foot in FOOT_ORDER:
            delay = profile.touchdown_delays[foot]
            duty = profile.stance_for(foot)
            for cycle in (0,1,5):
                td = (cycle+delay)/profile.frequency_hz
                off = (cycle+delay+duty)/profile.frequency_hz
                self.assertEqual(controller.foot_phase_fraction(foot,td), 0)
                self.assertEqual(controller.commanded_contacts(td)[foot], 1)
                self.assertEqual(controller.commanded_contacts(off)[foot], 0)
                for t in (td, off, td-1e-7, td+1e-7, off-1e-7, off+1e-7):
                    self.assertEqual(controller.commanded_contacts(t)[foot], reward.target_contacts(t)[foot])

    def test_footfall_order_RH_RF_LH_LF(self):
        profile = get_gait_profile("lab")
        # Anchor the cycle at right hind touchdown and sort normalized delays.
        delays = profile.touchdown_delays
        order = sorted(FOOT_ORDER, key=lambda foot:(delays[foot]-delays["HR"])%1)
        self.assertEqual(order, ["HR", "FR", "HL", "FL"])
        # Independently observe transitions on a dense synthetic clock.
        previous = {foot:profile.contact(foot, (.5-1e-5)/profile.frequency_hz) for foot in FOOT_ORDER}
        observed = []
        for frac in np.arange(10000)/10000+.5:
            current = {foot:profile.contact(foot,frac/profile.frequency_hz) for foot in FOOT_ORDER}
            observed.extend(foot for foot in FOOT_ORDER if current[foot] and not previous[foot])
            previous = current
        self.assertEqual(observed, ["HR", "FR", "HL", "FL"])

    def test_lab_changes_do_not_add_observation_channels(self):
        legacy = LateralSequenceCPG()
        lab = LateralSequenceCPG(gait_profile="lab")
        for t in np.arange(1000)*.02:
            np.testing.assert_array_equal(legacy.phase_observation(t), lab.phase_observation(t))
        self.assertEqual(_GAIT_FEET, FOOT_ORDER)

    def test_lab_conflicting_overrides_rejected(self):
        for kwargs in ({"freq":2.}, {"stance":.62}, {"phase":{"HL":0.,"FL":.25,"HR":.5,"FR":.75}}):
            with self.assertRaises(ValueError):
                CPGResidualController(self.model, gait_profile="lab", verbose=False, **kwargs)
        for kwargs in ({"frequency_hz":2.}, {"stance_ratio":.62}, {"swing_ratio":.38}):
            with self.assertRaises(ValueError):
                LateralSequenceCPG(gait_profile="lab", **kwargs)
        with self.assertRaises(ValueError):
            get_gait_profile("unknown")

    def test_lab_phase_maps_read_only(self):
        controller = CPGResidualController(self.model, gait_profile="lab", verbose=False)
        reward = LateralSequenceCPG(gait_profile="lab")
        with self.assertRaises(TypeError):
            controller.phase["FL"] = .25
        with self.assertRaises(TypeError):
            reward.phase_offsets["FL"] = .25
        self.assertTrue(controller.anti_phase_ok()[1])

    def test_actual_step_reward_uses_executed_interval_at_boundaries(self):
        env = GeckoWalkEnv(control_mode="cpg_residual", gait_profile="lab", reset_noise=0)
        try:
            env.reset(seed=3)
            boundary_intervals = 0
            for _ in range(100):
                executed_time = env._cpg_t
                executed_contacts = env.cpg.commanded_contact_array(executed_time)
                obs, _, terminated, _, info = env.step(np.zeros(env.nu))
                np.testing.assert_array_equal(info["target_contacts"], executed_contacts)
                self.assertEqual(info["gait_target_time_s"], executed_time)
                # Observation clock remains current, not shifted to past reward time.
                np.testing.assert_array_equal(obs[-2:], env.gait.phase_observation(env._gait_time()))
                next_contacts = env.gait.target_contact_array(env._gait_time())
                boundary_intervals += int(not np.array_equal(next_contacts, executed_contacts))
                if terminated:
                    break
            self.assertGreater(boundary_intervals, 0, "Test must cross a stance boundary to detect the former off-by-one interval.")
            env.reset(seed=4)
            self.assertEqual(env._last_cpg_command_time_s, 0.)
        finally:
            env.close()


class LegacyFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))

    def test_pre_refactor_controller_fixture(self):
        c = CPGResidualController(self.model, verbose=False)
        times = (.123,.5)
        fixture = np.array([
            [.04840935659225807,.25080677658508393,.04840935659225807,0.,.06224061265393549,
             -.07898368707157895,-.409211056533558,-.07898368707157895,.1351488763543171,-.1015504732774737,
             -.31515189546,-.07850131311359998,.14660799999999996,.328235646376,.14628765905749327,
             .37075997684966955,0.,0.,0.,0.,.2861420879016462,0.,-.008187930628990814,.008187930628990814,0.],
            [-.08406290612903226,-.4355262701419355,-.08406290612903226,0.,-.10808089896774195,
             .06372701322580644,.33016689114838704,.06372701322580644,0.,.08193474619354837,
             -.29748018187499997,.01305506840000004,.4184892515732695,.30146607599999997,.007595676159999902,
             .649264,0.,0.,0.,0.,-.20122609345416997,0.,.11808747687600203,-.11808747687600203,0.]
        ])
        actual = np.array([c.base_ctrl(t,front_contact={"FL":True,"FR":False}) for t in times])
        # Tiny libm tolerance permits cross-platform sin rounding; local arrays
        # and the complete80-step state/reward digest matched bit-for-bit.
        np.testing.assert_allclose(actual, fixture, rtol=0, atol=5e-15)

    def test_pre_refactor_reward_and_phase_fixtures(self):
        prior = LateralSequenceCPG()
        np.testing.assert_array_equal(prior.target_contact_array(.123), [1,0,0,1])
        np.testing.assert_array_equal(prior.target_contact_array(.5), [1,1,1,0])
        np.testing.assert_array_equal(prior.phase_observation(.123), np.array([.7948391437530518,.6068202257156372],dtype=np.float32))
        np.testing.assert_array_equal(prior.phase_observation(.5), np.array([-.5589613914489746,-.8291937112808228],dtype=np.float32))

    def test_known_legacy_mismatch_is_intentionally_preserved(self):
        c = CPGResidualController(self.model, verbose=False)
        prior = LateralSequenceCPG()
        np.testing.assert_array_equal(c.commanded_contact_array(0), [1,1,1,0])
        np.testing.assert_array_equal(prior.target_contact_array(0), [1,0,1,1])

    def test_actual_step_legacy_reward_keeps_historical_end_time(self):
        env = GeckoWalkEnv(control_mode="cpg_residual", reset_noise=0)
        try:
            env.reset(seed=3)
            for _ in range(10):
                _, _, _, _, info = env.step(np.zeros(env.nu))
                self.assertEqual(info["gait_target_time_s"], env._gait_time())
                np.testing.assert_array_equal(info["target_contacts"], env.gait.target_contact_array(env._gait_time()))
        finally:
            env.close()

    def test_legacy_custom_overrides_remain_supported(self):
        c = CPGResidualController(self.model, freq=1., stance=.6, verbose=False)
        self.assertEqual(c.freq, 1.)
        self.assertEqual(c.stance, .6)
        prior = LateralSequenceCPG(frequency_hz=1., stance_ratio=.6, swing_ratio=.4)
        self.assertEqual(prior.frequency_hz, 1.)
        self.assertEqual(prior.stance_ratio, .6)

    def test_default_and_explicit_legacy_rollouts_identical(self):
        base = dict(control_mode="cpg_residual", reset_noise=0, contact_thresh=.0564, residual_scale=.25)
        a = GeckoWalkEnv(**base)
        b = GeckoWalkEnv(**base, gait_profile="legacy")
        try:
            a.reset(seed=17)
            b.reset(seed=17)
            for k in range(80):
                action = np.sin(np.arange(a.nu)+k*.07).astype(np.float32)*.1
                ao, ar, at, atr, ai = a.step(action)
                bo, br, bt, btr, bi = b.step(action)
                np.testing.assert_array_equal(ao,bo)
                np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
                np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
                self.assertEqual((ar,at,atr),(br,bt,btr))
        finally:
            a.close()
            b.close()

    def test_environment_shares_lab_profile_object_and_shapes(self):
        env = GeckoWalkEnv(control_mode="cpg_residual", gait_profile="lab")
        try:
            self.assertIs(env.gait.profile, env.cpg.profile)
            self.assertEqual(env.observation_space.shape, (92,))
            self.assertEqual(env.action_space.shape, (25,))
            obs,_ = env.reset(seed=1)
            obs,reward,terminated,truncated,info = env.step(np.zeros(25))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
