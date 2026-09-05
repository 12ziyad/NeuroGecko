import unittest
import numpy as np

from utils.render_trace_video import frame_indices


class TraceReplayTests(unittest.TestCase):
    def test_50hz_trace_to_25fps_selects_existing_samples(self):
        times = np.arange(1001)*.02
        np.testing.assert_array_equal(frame_indices(times, 25, 20), np.arange(0, 1000, 2))

    def test_no_fake_extension_beyond_recording(self):
        with self.assertRaises(ValueError):
            frame_indices([0., .02, .04], 25, 4.)

    def test_invalid_clock_or_rate_rejected(self):
        for times, fps, duration in [([0., 0.], 25, 1.), ([0., float('nan')], 25, 1.),
                                      ([0., 1.], 0, 1.), ([0., 1.], 25, -1.),
                                      ([0., 1.], float('inf'), 1.)]:
            with self.assertRaises(ValueError):
                frame_indices(times, fps, duration)

    def test_nonintegral_frame_duration_rejected(self):
        with self.assertRaises(ValueError):
            frame_indices([0., 1.], 25, .03)


if __name__ == '__main__':
    unittest.main()
