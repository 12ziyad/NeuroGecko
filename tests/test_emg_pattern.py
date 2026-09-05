import unittest
import numpy as np

from common.emg_pattern import Burst, activation_prior, hindlimb_patterns, sample_hindlimbs
from common.gait_config import get_gait_profile


class EMGPatternTests(unittest.TestCase):
    def test_recorded_peak_and_boundaries(self):
        for bursts in hindlimb_patterns().values():
            for burst in bursts:
                self.assertAlmostEqual(float(burst.sample(burst.peak)), burst.amplitude)
                self.assertAlmostEqual(float(burst.sample(burst.onset)), 0.)
                self.assertAlmostEqual(float(burst.sample(burst.onset+burst.duration)), 0.)

    def test_negative_onset_wraps_across_touchdown(self):
        burst = hindlimb_patterns()['caudofemoralis'][0]
        self.assertGreater(float(burst.sample(.99)), 0.)
        self.assertGreater(float(burst.sample(0.)), 0.)
        self.assertEqual(float(burst.sample(.90)), 0.)

    def test_two_separate_gastrocnemius_bursts(self):
        self.assertEqual(len(hindlimb_patterns()['gastrocnemius']), 2)
        self.assertAlmostEqual(float(activation_prior('gastrocnemius', .3379)), .3849)
        self.assertAlmostEqual(float(activation_prior('gastrocnemius', .9474)), .5450)
        self.assertEqual(float(activation_prior('gastrocnemius', .80)), 0.)

    def test_periodic_finite_nonnegative(self):
        phase = np.linspace(-2., 3., 5001)
        for muscle in hindlimb_patterns():
            output = activation_prior(muscle, phase)
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue(((output >= 0.) & (output <= 1.)).all())
            np.testing.assert_allclose(output, activation_prior(muscle, phase+1.), atol=2e-14)

    def test_shared_clock_and_half_cycle_hindlimb_delay(self):
        profile = get_gait_profile('lab')
        time_s = .1401/profile.frequency_hz
        left = sample_hindlimbs(profile, time_s)['HL']
        right = sample_hindlimbs(profile, time_s+.5/profile.frequency_hz)['HR']
        self.assertAlmostEqual(left['caudofemoralis'], .5373)
        for muscle in left:
            self.assertAlmostEqual(left[muscle], right[muscle])
        with self.assertRaises(ValueError):
            sample_hindlimbs('legacy', 0.)

    def test_invalid_windows_and_inputs(self):
        for args in [(0., 0., .1, .5), (0., .2, .3, .5), (0., .5, .2, 2.), (float('nan'), .5, .2, .5)]:
            with self.assertRaises(ValueError):
                Burst(*args)
        with self.assertRaises(ValueError):
            Burst(0., .5, .2, .5).sample(float('nan'))


if __name__ == '__main__':
    unittest.main()
