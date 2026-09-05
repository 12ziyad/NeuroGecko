"""Real physics acquisition, snapshot isolation, and honest failed entrainment."""
import unittest
import numpy as np
from envs.gecko_walk_env import GeckoWalkEnv
from realism_metrics import TraceRecorder, analyze_trace, assert_entrainment


class HighRateTraceTests(unittest.TestCase):
    def test_substeps_preserve_live_legacy_state_reward_and_observation(self):
        kwargs = dict(control_mode="cpg_residual", reset_noise=0, contact_thresh=.0564, residual_scale=.25)
        a, b = GeckoWalkEnv(**kwargs), GeckoWalkEnv(**kwargs)
        try:
            a.reset(seed=17); b.reset(seed=17)
            recorder = TraceRecorder(b, physics_substeps=5)
            recorder.record()
            total_work = 0.
            for k in range(80):
                action = np.sin(np.arange(a.nu)+k*.07).astype(np.float32)*.1
                ar, br = a.step(action), b.step(action)
                np.testing.assert_array_equal(ar[0], br[0])
                np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
                np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
                self.assertEqual(ar[1:4], br[1:4])
                total_work += br[4]["mechanical_work_abs_J"]
            s = recorder.samples
            self.assertEqual(len(s["time_s"]), 401)
            np.testing.assert_allclose(np.diff(s["time_s"]), .004, atol=1e-12)
            self.assertAlmostEqual(sum(s["mechanical_work_abs_J"]), total_work, places=12)
            self.assertEqual(recorder.metadata["commanded_frequency_hz"], 1.1888)
            self.assertEqual(s["command_time_s"][1:6], [0.]*5)
            self.assertEqual(s["command_time_s"][6:11], [.02]*5)
            # Distinct physical samples within a held control command, not interpolation.
            self.assertFalse(np.array_equal(s["qpos"][1], s["qpos"][2]))
            np.testing.assert_array_equal(s["actuator_control"][1], s["actuator_control"][5])
            recorder.detach()
            self.assertIsNone(b.physics_observer)
        finally:
            a.close(); b.close()

    def test_real_frequency_comparison_never_forces_chatter_to_pass(self):
        t = np.arange(2501)*.004
        for frequency, passes in ((1.1888, True), (4., False)):
            forces = np.tile(((t*frequency)%1 < .7)[:, None], (1,4)).astype(float)
            trace = {"metadata": {"svl_m":.106, "contact_threshold_N":.1, "commanded_frequency_hz":1.1888},
                     "samples": {"time_s":t, "trunk_position_m":np.column_stack((.04*t, 0*t, 0*t)), "foot_force_N":forces}}
            result = analyze_trace(trace, settle_s=1)
            self.assertEqual(result["contact_quality"]["entrainment_pass"], passes)
            if passes:
                assert_entrainment(result)
            else:
                with self.assertRaises(AssertionError):
                    assert_entrainment(result)
                self.assertGreater(result["limbs"]["HL"]["complete_strides"], 30)

    def test_bad_sampling_divisor_rejected(self):
        env = GeckoWalkEnv()
        try:
            with self.assertRaises(ValueError):
                TraceRecorder(env, physics_substeps=6)
        finally:
            env.close()
