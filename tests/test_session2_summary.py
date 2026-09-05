"""Read-only synthetic steering checks; no simulator or learner."""
import copy
import unittest
import numpy as np
from eval.session2_summary import steering_from_trace


def trace_for_turn(angle=.4):
    t=np.arange(5001)*.004
    yaw=angle*t/20
    q=np.column_stack([np.cos(yaw/2),0*t,0*t,np.sin(yaw/2)])
    return {"metadata":{"commanded_frequency_hz":1.,"goal_angle_rad":angle},
            "samples":{"time_s":t.tolist(),"trunk_quaternion_wxyz":q.tolist(),
                       "trunk_position_m":np.column_stack([.04*t*np.cos(angle),.04*t*np.sin(angle),t*0]).tolist()}}


class SteeringSummaryTests(unittest.TestCase):
    def test_known_left_and_right_turns_and_error_reduction(self):
        for angle in (.4,-.4):
            r=steering_from_trace(trace_for_turn(angle))
            self.assertTrue(r["directional_check"]["turn_sign_matches_request"])
            self.assertTrue(r["directional_check"]["mean_absolute_goal_error_reduced_first_to_last_cycle"])
            self.assertAlmostEqual(r["displacement"]["after_3s"]["displacement_heading_rad"],angle)
            self.assertLess(r["windows"]["last_full_commanded_cycle"]["mean_absolute_goal_bearing_error_rad"],.02)

    def test_quaternion_double_cover_preserves_heading(self):
        trace=trace_for_turn()
        original=steering_from_trace(trace)
        trace["samples"]["trunk_quaternion_wxyz"]=(-np.asarray(trace["samples"]["trunk_quaternion_wxyz"])).tolist()
        changed=steering_from_trace(trace)
        self.assertEqual(original["windows"],changed["windows"])

    def test_partial_final_cycle_is_excluded(self):
        trace=trace_for_turn()
        trace["metadata"]["commanded_frequency_hz"]=1.1888
        r=steering_from_trace(trace)
        self.assertEqual(r["windows"]["first_full_commanded_cycle"]["cycle_index"],0)
        self.assertEqual(r["windows"]["last_full_commanded_cycle"]["cycle_index"],22)
        self.assertLess(r["windows"]["last_full_commanded_cycle"]["commanded_window_s"][1],20.)

    def test_body_heading_is_not_displacement_heading(self):
        trace=trace_for_turn()
        trace["samples"]["trunk_quaternion_wxyz"]=[[1,0,0,0]]*len(trace["samples"]["time_s"])
        r=steering_from_trace(trace)
        self.assertEqual(r["windows"]["last_full_commanded_cycle"]["mean_body_heading_rad"],0.)
        self.assertAlmostEqual(r["displacement"]["full_run"]["displacement_heading_rad"],.4)
        self.assertFalse(r["directional_check"]["turn_sign_matches_request"])

    def test_read_only_and_invalid_quaternions(self):
        trace=trace_for_turn()
        original=copy.deepcopy(trace)
        steering_from_trace(trace)
        self.assertEqual(original,trace)
        trace["samples"]["trunk_quaternion_wxyz"][1]=[0,0,0,0]
        with self.assertRaisesRegex(ValueError,"Zero-norm"):
            steering_from_trace(trace)


if __name__=="__main__":
    unittest.main()
