"""Synthetic event-count tests, not biological training examples."""
import unittest

import numpy as np

from eval.ethogram import score_ethogram


def trace(duration=6., velocity=0., joint_name="knee_L"):
    t = np.arange(round(duration/.02)+1)*.02
    return {"metadata": {"svl_m": .1, "hinge_names": [joint_name]},
            "samples": {"time_s": t, "trunk_position_m": np.column_stack([velocity*t,0*t,0*t]),
                        "hinge_position_deg": np.zeros((len(t),1)),
                        "trunk_quaternion_wxyz": np.tile([1.,0.,0.,0.], (len(t),1))}}


class EthogramTests(unittest.TestCase):
    def test_continuous_long_walk_counts_once(self):
        result = score_ethogram(trace(12., .1))
        self.assertEqual(result["counts"], {"walk_around": 1})
        self.assertAlmostEqual(result["event_percentages"]["walk_around"], 100)

    def test_three_seconds_rest_counts_once_not_duration(self):
        for duration in (3., 6., 12.):
            result = score_ethogram(trace(duration))
            self.assertEqual(result["counts"], {"rest_proxy": 1})
        self.assertEqual(score_ethogram(trace(2.98))["counts"], {})

    def test_sub_svl_orientation_needs_leg_movement(self):
        sample = trace(4., .01)
        t = sample["samples"]["time_s"]
        sample["samples"]["hinge_position_deg"][:,0] = 10*t
        self.assertEqual(score_ethogram(sample)["counts"], {"orientation_change_proxy": 1})
        sample["samples"]["hinge_position_deg"][:] = 0
        self.assertEqual(score_ethogram(sample)["counts"], {"unclassified_movement": 1})

    def test_moving_head_prevents_false_rest(self):
        sample = trace(6., joint_name="head_yaw")
        sample["samples"]["hinge_position_deg"][:,0] = 10*sample["samples"]["time_s"]
        result = score_ethogram(sample)
        self.assertNotIn("rest_proxy", result["counts"])
        self.assertEqual(result["counts"], {"unclassified_movement": 1})

    def test_vertical_motion_is_not_rest(self):
        sample = trace()
        sample["samples"]["trunk_position_m"][:,2] = .01*sample["samples"]["time_s"]
        self.assertEqual(score_ethogram(sample)["counts"], {"unclassified_movement": 1})

    def test_missing_joints_do_not_prove_rest(self):
        sample = trace()
        sample["samples"].pop("hinge_position_deg")
        result = score_ethogram(sample)
        self.assertEqual(result["counts"], {"stationary_unknown_joint_activity": 1})

    def test_quaternion_sign_is_not_rotation(self):
        sample = trace()
        sample["samples"]["trunk_quaternion_wxyz"][::2] *= -1
        self.assertEqual(score_ethogram(sample)["counts"], {"rest_proxy": 1})

    def test_rest_separator_gives_distinct_bouts(self):
        sample = trace(12.)
        t = sample["samples"]["time_s"]
        x = .1*np.minimum(t,4)+.1*np.maximum(t-8,0)
        sample["samples"]["trunk_position_m"][:,0] = x
        result = score_ethogram(sample)
        self.assertEqual(result["counts"], {"walk_around": 2, "rest_proxy": 1})
        self.assertEqual(result["total_detected_events"], 3)

    def test_short_pause_does_not_recount_same_bout(self):
        sample = trace(8.)
        t = sample["samples"]["time_s"]
        sample["samples"]["trunk_position_m"][:,0] = .1*np.minimum(t,4)+.1*np.maximum(t-4.04,0)
        self.assertEqual(score_ethogram(sample)["counts"], {"walk_around": 1})

    def test_threshold_is_svl_not_total_length(self):
        self.assertEqual(score_ethogram(trace(1., .1))["counts"], {"walk_around": 1})
        self.assertEqual(score_ethogram(trace(1., .099))["counts"], {"unclassified_movement": 1})

    def test_missing_categories_remain_unavailable_not_zero(self):
        result = score_ethogram(trace())
        self.assertIn("foraging", result["unavailable_categories"])
        self.assertNotIn("foraging", result["counts"])
        self.assertIn("not time fraction", result["unit"])
        self.assertTrue(result["events"][0]["left_censored"])
        self.assertTrue(result["events"][0]["right_censored"])


if __name__ == "__main__":
    unittest.main()
