import unittest
import numpy as np

from common.energetics import actuator_work
from envs.gecko_walk_env import GeckoWalkEnv


class EnergeticsTests(unittest.TestCase):
    def test_conjugate_work_does_not_cancel_actuators(self):
        self.assertEqual(actuator_work([2., -3.], [4., 2.], .5), (4., -3., 7.))

    def test_stationary_actuator_does_no_work(self):
        self.assertEqual(actuator_work([20., 30.], [0., 0.], .02), (0., 0., 0.))

    def test_invalid_power_is_rejected(self):
        with self.assertRaises(ValueError):
            actuator_work([float('nan')], [1], .02)

    def test_environment_reports_substep_work_without_obs_change(self):
        env = GeckoWalkEnv(control_mode='cpg_residual', reset_noise=0)
        try:
            obs, _ = env.reset(seed=4)
            self.assertEqual(obs.shape, (92,))
            _, _, _, _, info = env.step(np.zeros(env.nu))
            self.assertGreaterEqual(info['mechanical_work_positive_J'], 0)
            self.assertLessEqual(info['mechanical_work_negative_J'], 0)
            self.assertAlmostEqual(info['mechanical_work_abs_J'],
                                   info['mechanical_work_positive_J'] - info['mechanical_work_negative_J'])
            env.reset(seed=4)
            self.assertEqual(env._episode_path_m, 0)
            np.testing.assert_array_equal(env._episode_work, np.zeros(3))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
