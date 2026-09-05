"""Synthetic instrumentation tests: these are not gecko validation data."""
import importlib.util
import json
import unittest

import numpy as np

from realism_metrics import (FEET, TraceRecorder, _json_default, analyze_trace,
                             circular_summary, complete_strides,
                             debounce_contacts, filtered_nose_kinematics)


def synthetic_trace(duration=6., dt=.02):
    t = np.arange(int(round(duration/dt))+1)*dt
    offsets = (0., .44, .50, .94)
    # Integer ticks remove binary-floating-point ambiguity at synthetic edges.
    cycles = np.mod(np.arange(len(t))[:, None]-np.round(np.asarray(offsets)/dt)[None, :], round(1/dt))*dt
    duty = np.array([.76, .70, .76, .70])
    force = (cycles < duty[None, :]-1e-10).astype(float)
    xyz = np.column_stack([.08*t, np.zeros(len(t)), np.full(len(t), .02)])
    return {"metadata": {"svl_m": .1, "contact_threshold_N": .5, "hinge_names": ["knee_L"]},
            "samples": {"time_s": t, "trunk_position_m": xyz, "foot_force_N": force,
                        "foot_position_m": np.zeros((len(t), 4, 3)),
                        "hinge_position_deg": (100*cycles[:, 0])[:, None]}}


class ContactTests(unittest.TestCase):
    def test_debounce_bounded_one_step_contact_and_flight(self):
        for signal in ([0,0,1,0,0], [1,1,0,1,1]):
            expected = np.full(5, bool(signal[0]))
            np.testing.assert_array_equal(debounce_contacts(signal, .02, .04), expected)

    def test_exact_40ms_is_retained(self):
        x = [0,0,1,1,0,0]
        np.testing.assert_array_equal(debounce_contacts(x, .02, .04), x)

    def test_boundary_runs_not_fabricated(self):
        x = [1,0,0,1]
        np.testing.assert_array_equal(debounce_contacts(x, .02, .04), x)

    def test_short_chatter_tie_break_is_deterministic(self):
        x = [0,0,1,0,1,1,0,0]
        a = debounce_contacts(x, .02, .04)
        np.testing.assert_array_equal(a, debounce_contacts(x, .02, .04))
        np.testing.assert_array_equal(a, debounce_contacts(a, .02, .04))

    def test_censored_cycles_are_excluded(self):
        trace = synthetic_trace()
        t = trace["samples"]["time_s"]
        contact = trace["samples"]["foot_force_N"][:, 0]>.5
        cycles = complete_strides(t, contact, settle_s=.5)
        self.assertEqual(len(cycles), 5)
        self.assertEqual(cycles[0]["start"], 50)  # Initial ongoing stance is not a new cycle.
        self.assertEqual(cycles[-1]["end"], 300)
        self.assertAlmostEqual(cycles[0]["period_s"], 1.)
        self.assertAlmostEqual(cycles[0]["duty_factor"], .76)

    def test_constant_contact_has_no_observed_strides(self):
        t = np.arange(201)*.02
        self.assertEqual(complete_strides(t, np.ones(len(t)), 0), [])

    def test_irregular_sampling_rejected(self):
        with self.assertRaisesRegex(ValueError, "uniform"):
            complete_strides([0,.02,.06], [0,1,0])


class GaitAnalysisTests(unittest.TestCase):
    def test_known_single_speed_metrics(self):
        result = analyze_trace(synthetic_trace(), settle_s=1.)
        self.assertAlmostEqual(result["mean_path_speed_m_s"], .08)
        self.assertAlmostEqual(result["limbs"]["HL"]["duty_factor"]["mean"], .76)
        self.assertAlmostEqual(result["limbs"]["FL"]["duty_factor"]["mean"], .70)
        self.assertAlmostEqual(result["limbs"]["HR"]["stride_length_svl"]["mean"], .8)
        self.assertAlmostEqual(result["limbs"]["HR"]["stride_frequency_hz"]["mean"], 1.)
        for key in ("HL_to_FL", "HR_to_FR"):
            self.assertAlmostEqual(result["limb_phase"][key]["mean_cycle"], .44)
        self.assertEqual(set(result["observed_footfall_orders_HR_anchored"]), {"HR -> FR -> HL -> FL"})

    def test_circular_mean_across_wrap(self):
        mean = circular_summary([.99,.01])["mean_cycle"]
        self.assertLess(min(mean, 1-mean), 1e-10)
        self.assertIsNone(circular_summary([])["mean_cycle"])

    def test_stance_slip_uses_observed_contact_only(self):
        trace = synthetic_trace()
        t = trace["samples"]["time_s"]
        trace["samples"]["foot_position_m"][:, :, 0] = .03*t[:, None]
        trace["samples"]["target_contacts"] = np.zeros((len(t), 4))  # Deliberately misleading commanded contacts.
        result = analyze_trace(trace, settle_s=1.)
        for foot in FEET:
            self.assertAlmostEqual(result["limbs"][foot]["stance_foot_site_speed_m_s"]["mean"], .03)

    def test_slip_excludes_touchdown_discontinuity(self):
        trace = synthetic_trace()
        # Positions may jump at touchdown; actual support intervals remain still.
        trace["samples"]["foot_position_m"][:, :, 0] = np.floor(trace["samples"]["time_s"][:, None])
        result = analyze_trace(trace, settle_s=1.)
        self.assertAlmostEqual(result["limbs"]["HL"]["stance_foot_site_speed_m_s"]["mean"], 0.)

    def test_stance_and_whole_stride_are_separate(self):
        result = analyze_trace(synthetic_trace(), settle_s=1.)
        knee = result["joint_angles"]["knee_L"]
        self.assertAlmostEqual(knee["per_stance"]["maximum_deg"]["mean"], 74.)
        self.assertAlmostEqual(knee["per_complete_stride"]["maximum_deg"]["mean"], 98.)
        self.assertEqual(knee["convention"], "MuJoCo scalar hinge displacement")

    def test_empty_cycles_remain_missing_not_zero_frequency(self):
        trace = synthetic_trace()
        trace["samples"]["foot_force_N"][:] = 1.
        result = analyze_trace(trace, settle_s=1.)
        self.assertIsNone(result["limbs"]["HL"]["stride_frequency_hz"]["mean"])
        self.assertEqual(result["limbs"]["HL"]["complete_strides"], 0)
        self.assertEqual(result["observed_footfall_orders_HR_anchored"], {})

    def test_excess_contact_cycles_warn_without_forcefitting(self):
        trace = synthetic_trace()
        trace["metadata"]["frequency_hz"] = 1.1888
        # 5 Hz touchdown cycles remain after 40 ms debounce: a deliberately
        # incompatible contact trace, not a biological or oscillator stride.
        ticks = np.arange(len(trace["samples"]["time_s"]))
        trace["samples"]["foot_force_N"][:] = ((ticks % 10) < 6)[:, None]
        result = analyze_trace(trace, settle_s=1.)
        self.assertEqual(len(result["contact_quality"]["warnings"]), 4)
        for foot in FEET:
            limb = result["limbs"][foot]
            diagnostic = limb["contact_cycle_diagnostic"]
            self.assertTrue(diagnostic["gross_frequency_mismatch_warning"])
            self.assertAlmostEqual(diagnostic["observed_contact_cycle_rate_hz"], 5.)
            self.assertAlmostEqual(limb["stride_frequency_hz"]["mean"], 5.)
            self.assertGreater(limb["complete_strides"], 20)
        self.assertIn("NOT verified biological", result["cycle_metric_semantics"])

    def test_slow_contact_cycles_warn_but_matched_and_unknown_do_not(self):
        trace = synthetic_trace()
        trace["metadata"]["frequency_hz"] = 3.
        result = analyze_trace(trace, settle_s=1.)
        self.assertEqual(len(result["contact_quality"]["warnings"]), 4)
        trace["metadata"]["frequency_hz"] = 1.
        matched = analyze_trace(trace, settle_s=1.)
        self.assertEqual(matched["contact_quality"]["warnings"], [])
        self.assertFalse(matched["limbs"]["HL"]["contact_cycle_diagnostic"]["gross_frequency_mismatch_warning"])
        del trace["metadata"]["frequency_hz"]
        unknown = analyze_trace(trace, settle_s=1.)
        self.assertIsNone(unknown["limbs"]["HL"]["contact_cycle_diagnostic"]["gross_frequency_mismatch_warning"])
        self.assertEqual(unknown["contact_quality"]["warnings"], [])

    def test_invalid_recorded_frequency_is_not_silently_replaced(self):
        trace = synthetic_trace()
        for frequency in (0., float("nan"), -1.):
            trace["metadata"]["frequency_hz"] = frequency
            with self.assertRaisesRegex(ValueError, "commanded frequency"):
                analyze_trace(trace, settle_s=1.)

    def test_coarse_gait_rejected(self):
        with self.assertRaisesRegex(ValueError, "50 Hz"):
            analyze_trace(synthetic_trace(dt=.04), settle_s=1.)

    def test_work_uses_postsettle_intervals(self):
        trace = synthetic_trace()
        trace["samples"]["mechanical_work_positive_J"] = [.2]*len(trace["samples"]["time_s"])
        result = analyze_trace(trace, settle_s=1.)
        self.assertAlmostEqual(result["actuator_mechanical_work"]["positive_J"], 50.)
        self.assertAlmostEqual(result["actuator_mechanical_work"]["positive_J_per_path_m"], 125.)

    def test_missing_work_and_low_rate_strikes_are_explicit(self):
        result = analyze_trace(synthetic_trace(), settle_s=1.)
        self.assertIn("unavailable", result["actuator_mechanical_work"]["status"])
        self.assertIn("500 Hz", result["deferred"]["strike_shake"])
        self.assertIn("requires descending drive", result["deferred"]["frequency_speed_regression"])

    def test_json_is_strict(self):
        result = analyze_trace(synthetic_trace(), settle_s=1.)
        json.dumps(result, default=_json_default, allow_nan=False)

    def test_bad_shapes_and_nonfinite_rejected(self):
        trace = synthetic_trace()
        trace["samples"]["foot_force_N"][3, 0] = np.nan
        with self.assertRaises(ValueError):
            analyze_trace(trace, settle_s=1.)


class StrikeFilterTests(unittest.TestCase):
    def test_50hz_cannot_be_upsampled_into_evidence(self):
        t = np.arange(251)*.02
        with self.assertRaisesRegex(ValueError, "upsampling"):
            filtered_nose_kinematics(t, np.zeros((len(t),3)))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "Optional scipy is not installed; no high-rate strike filtering performed")
    def test_high_rate_filter_linear_velocity(self):
        t = np.arange(2501)*.0008
        xyz = np.column_stack([.3*t, 0*t, 0*t])
        result = filtered_nose_kinematics(t, xyz)
        self.assertAlmostEqual(np.median(result["velocity_m_s"][100:-100,0]), .3, places=3)
        self.assertLessEqual(result["time_s"][-1], t[-1]+1e-10)

    def test_invalid_filter_nyquist(self):
        t = np.arange(2001)*.001
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            filtered_nose_kinematics(t, np.zeros((len(t),3)), cutoff_hz=300)


class RecorderIntegrationTests(unittest.TestCase):
    def test_records_all_hinges_and_does_not_mutate_dynamics(self):
        try:
            from envs.gecko_walk_env import GeckoWalkEnv
        except ImportError:
            self.skipTest("MuJoCo/Gym unavailable")
        a = GeckoWalkEnv(control_mode="cpg_residual", reset_noise=0)
        b = GeckoWalkEnv(control_mode="cpg_residual", reset_noise=0)
        try:
            a.reset(seed=7)
            b.reset(seed=7)
            recorder = TraceRecorder(a)
            recorder.record()
            self.assertEqual(len(recorder.metadata["hinge_names"]), 32)
            self.assertGreater(recorder.metadata["svl_m"], .10)
            before = a.data.qpos.copy()
            recorder.record()
            np.testing.assert_array_equal(before, a.data.qpos)
            for _ in range(3):
                _, _, _, _, info = a.step(np.zeros(a.nu))
                b.step(np.zeros(b.nu))
                recorder.record(info)
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
