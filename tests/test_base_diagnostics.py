"""Synthetic trace-only tests. This module imports no simulator or learner."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval.base_diagnostics import (FEET, SESSION1_LEGACY_REFERENCE, _debounce,
                                   diagnose_file, diagnose_trace, resolve_commands)


def synthetic_trace(dt=.004, duration=5., frequency=1.):
    t = np.arange(int(round(duration/dt))+1)*dt
    command_time = (np.ceil(t/.02-1e-9)-1)*.02
    command_time[0] = np.nan
    delays = np.array([0., .44, .5, .94])
    phase = (command_time[:, None]*frequency-delays)%1
    phase[np.minimum(phase, 1-phase) < 1e-12] = 0.
    stances = np.array([.78, .70, .78, .70])
    for j, ratio in enumerate(stances):
        phase[np.abs(phase[:, j]-ratio) < 1e-12, j] = ratio
    commands = np.where(np.isfinite(phase), phase < stances, np.nan)
    force = np.where(commands == 1, .2, .01)
    return {"metadata": {"foot_order": list(FEET), "contact_threshold_N": .05,
                         "control_dt_s": .02, "commanded_frequency_hz": frequency,
                         "commanded_stance_by_foot": dict(zip(FEET, stances.tolist()))},
            "samples": {"time_s": t.tolist(), "command_time_s": [None if not np.isfinite(x) else float(x) for x in command_time],
                        "commanded_phase_fraction": [None if i == 0 else phase[i].tolist() for i in range(len(t))],
                        "commanded_contacts": [None if i == 0 else commands[i].tolist() for i in range(len(t))],
                        "foot_force_N": force.tolist(), "forward_speed_m_s": [.08]*len(t),
                        "trunk_position_m": np.column_stack([t*.08, t*0, t*0+.02]).tolist()}}


def diagnose(trace, **kwargs):
    options = dict(settle_s=1., phase_bins=20, entrainment_tolerance_fraction=.10, period_cv_max=.10)
    options.update(kwargs)
    return diagnose_trace(trace, **options)


class CommandResolutionTests(unittest.TestCase):
    def test_recorded_held_clock_and_phase_are_used_not_instantaneous_time(self):
        trace = synthetic_trace()
        cmd = resolve_commands(trace)
        # At .016s the current phase clock would differ; held control is still0.
        self.assertEqual(cmd["time_s"][4], 0.)
        self.assertEqual(cmd["phase"][4, 0], 0.)
        self.assertTrue(np.isnan(cmd["contacts"][0]).all())
        self.assertEqual(cmd["provenance"]["recorded_phase_vs_contact_reference_mismatch_count"], 0)

    def test_old_legacy_clock_requires_explicit_historical_reference(self):
        trace = synthetic_trace(dt=.02)
        for field in ("command_time_s", "commanded_phase_fraction", "commanded_contacts"):
            del trace["samples"][field]
        del trace["metadata"]["commanded_frequency_hz"]
        del trace["metadata"]["commanded_stance_by_foot"]
        # Historical legacy reward targets are not controller contacts.
        trace["metadata"]["gait_profile"] = "legacy"
        trace["samples"]["gait_target_contacts"] = [[1, 1, 1, 1]]*len(trace["samples"]["time_s"])
        missing = resolve_commands(trace)
        self.assertTrue(np.isnan(missing["contacts"]).all())
        reference = resolve_commands(trace, SESSION1_LEGACY_REFERENCE)
        np.testing.assert_array_equal(reference["time_s"][1:4], [0., .02, .04])
        np.testing.assert_array_equal(reference["contacts"][1], [1., 1., 1., 0.])
        self.assertTrue(np.isnan(reference["contacts"][0]).all())
        self.assertIn("INFERRED", reference["provenance"]["command_time_source"])
        self.assertEqual(reference["frequency_hz"], 1.1888)

    def test_old_clock_inference_refuses_high_rate_guess(self):
        trace = synthetic_trace()
        for field in ("command_time_s", "commanded_phase_fraction", "commanded_contacts"):
            del trace["samples"][field]
        del trace["metadata"]["commanded_frequency_hz"]
        with self.assertRaisesRegex(ValueError, "one endpoint"):
            resolve_commands(trace, SESSION1_LEGACY_REFERENCE)

    def test_lab_old_reward_clock_is_explicitly_executed_command(self):
        trace = synthetic_trace()
        s, meta = trace["samples"], trace["metadata"]
        s["gait_target_time_s"] = s.pop("command_time_s")
        s["gait_target_contacts"] = s.pop("commanded_contacts")
        expected_phase = s.pop("commanded_phase_fraction")
        meta["gait_profile"] = "lab"
        meta["phase_offsets_cycle"] = dict(zip(FEET, [0., .44, .5, .94]))
        meta["frequency_hz"] = meta.pop("commanded_frequency_hz")
        result = resolve_commands(trace)
        np.testing.assert_allclose(result["phase"][1:], expected_phase[1:])
        self.assertIn("historical alias", result["provenance"]["frequency_source"])
        self.assertIn("lab Session1", result["provenance"]["command_time_source"])

    def test_reference_conflict_and_malformed_phase_fail(self):
        trace = synthetic_trace()
        with self.assertRaisesRegex(ValueError, "contradicts"):
            resolve_commands(trace, SESSION1_LEGACY_REFERENCE)
        trace["samples"]["commanded_phase_fraction"][1][0] = 1.01
        with self.assertRaisesRegex(ValueError, "local phases"):
            resolve_commands(trace)

    def test_recorded_phase_contact_disagreement_is_visible(self):
        trace = synthetic_trace()
        trace["samples"]["commanded_contacts"][5][0] = 0
        result = resolve_commands(trace)
        self.assertEqual(result["provenance"]["recorded_phase_vs_contact_reference_mismatch_count"], 1)
        self.assertEqual(result["contacts"][5, 0], 0)


class ForceDiagnosisTests(unittest.TestCase):
    def test_known_stance_swing_forces_and_real_high_rate(self):
        result = diagnose(synthetic_trace())
        self.assertAlmostEqual(result["sampling"]["sample_rate_hz"], 250.)
        self.assertEqual(result["sampling"]["endpoint_sample_count"], 1000)
        for foot in FEET:
            f = result["feet"][foot]
            self.assertEqual(f["commanded_stance"]["raw_loaded_fraction"], 1.)
            self.assertEqual(f["commanded_swing"]["raw_loaded_fraction"], 0.)
            self.assertAlmostEqual(f["commanded_stance"]["mean_scalar_touch_force_N"], .2)
            self.assertAlmostEqual(f["commanded_swing"]["mean_scalar_touch_force_N"], .01)
            self.assertAlmostEqual(f["debounced"]["observed_contact_cycle_rate_hz"], 1.)
            self.assertTrue(f["debounced"]["entrainment"]["passes_engineering_check"])

    def test_phase_bins_cover_all_known_postsettle_samples(self):
        result = diagnose(synthetic_trace())
        for foot in FEET:
            bins = result["feet"][foot]["phase_bins"]
            self.assertEqual(len(bins), 20)
            self.assertEqual(sum(b["sample_count"] for b in bins), 1000)
            self.assertEqual(bins[-1]["local_phase_end_exclusive"], 1.)

    def test_threshold_is_strict_and_never_uses_toe_height(self):
        trace = synthetic_trace()
        trace["samples"]["foot_force_N"] = [[.05]*4 for _ in trace["samples"]["time_s"]]
        trace["samples"]["foot_position_m"] = [[[0., 0., -1.]]*4 for _ in trace["samples"]["time_s"]]
        result = diagnose(trace)
        for foot in FEET:
            self.assertEqual(result["feet"][foot]["raw"]["loaded_fraction_including_censored"], 0.)
        self.assertFalse(result["force_semantics"]["toe_height_used_for_contact"])
        self.assertIn("NOT world-vertical", result["force_semantics"]["definition"])

    def test_raw_and_debounced_chatter_stay_distinct(self):
        trace = synthetic_trace()
        # 4 ms isolated touch inside commanded HL swing is removed, but raw
        # conditional swing statistics still expose this loaded sample.
        trace["samples"]["foot_force_N"][int(1.9/.004)][0] = .3
        result = diagnose(trace)
        f = result["feet"]["HL"]
        self.assertGreater(f["commanded_swing"]["raw_loaded_fraction"], 0.)
        self.assertEqual(f["commanded_swing"]["debounced_loaded_fraction"], 0.)
        self.assertGreater(f["raw"]["touchdown_count"], f["debounced"]["touchdown_count"])

    def test_retained_long_chatter_fails_entrainment_without_cycle_merging(self):
        trace = synthetic_trace()
        t = np.asarray(trace["samples"]["time_s"])
        # 100ms contact and100ms flight are both above40ms debounce.
        trace["samples"]["foot_force_N"] = np.repeat(np.where(np.arange(len(t))%50 < 25, .2, 0.)[:, None], 4, axis=1).tolist()
        result = diagnose(trace)
        for foot in FEET:
            f = result["feet"][foot]
            self.assertAlmostEqual(f["debounced"]["observed_contact_cycle_rate_hz"], 5.)
            self.assertFalse(f["debounced"]["entrainment"]["passes_engineering_check"])
            self.assertEqual(f["raw"]["complete_contact_cycle_count"], f["debounced"]["complete_contact_cycle_count"])
        self.assertFalse(result["observed_limb_phase"]["debounced"]["HL_to_FL"]["entrainment_supported"])

    def test_debounce_exact_threshold_and_censored_edges_retained(self):
        for signal in ([0,0,1,1,0,0], [1,0,0,1]):
            np.testing.assert_array_equal(_debounce(signal, .02, .04), signal)
        np.testing.assert_array_equal(_debounce([0,0,1,0,0], .02, .04), [0,0,0,0,0])
        np.testing.assert_array_equal(_debounce([1,1,0,1,1], .02, .04), [1,1,1,1,1])

    def test_no_complete_cycles_is_missing_not_pass_or_zero_frequency(self):
        trace = synthetic_trace()
        trace["samples"]["foot_force_N"] = [[.2]*4 for _ in trace["samples"]["time_s"]]
        f = diagnose(trace)["feet"]["HL"]["debounced"]
        self.assertEqual(f["complete_contact_cycle_count"], 0)
        self.assertIsNone(f["observed_contact_cycle_rate_hz"])
        self.assertIsNone(f["entrainment"]["passes_engineering_check"])

    def test_missing_commands_are_not_fabricated(self):
        trace = synthetic_trace()
        for field in ("command_time_s", "commanded_phase_fraction", "commanded_contacts"):
            del trace["samples"][field]
        result = diagnose(trace)
        for foot in FEET:
            f = result["feet"][foot]
            self.assertEqual(f["command_sample_coverage"], 0.)
            self.assertIsNone(f["commanded_swing"]["raw_loaded_fraction"])
            self.assertEqual(sum(b["sample_count"] for b in f["phase_bins"]), 0)

    def test_missing_frequency_keeps_observed_rate_but_no_entrainment_claim(self):
        trace = synthetic_trace()
        del trace["metadata"]["commanded_frequency_hz"]
        f = diagnose(trace)["feet"]["HL"]["debounced"]
        self.assertAlmostEqual(f["observed_contact_cycle_rate_hz"], 1.)
        self.assertIsNone(f["entrainment"]["passes_engineering_check"])

    def test_motion_separates_path_net_and_signed_local_velocity(self):
        trace = synthetic_trace(duration=5.)
        t = np.asarray(trace["samples"]["time_s"])
        trace["samples"]["trunk_position_m"] = np.column_stack([np.cos(2*np.pi*t), np.sin(2*np.pi*t), t*0]).tolist()
        trace["samples"]["forward_speed_m_s"] = [-.1]*len(t)
        result = diagnose(trace)
        self.assertGreater(result["motion"]["path_length_m"], 20.)
        self.assertLess(result["motion"]["net_path_ratio"], 1e-12)
        self.assertAlmostEqual(result["motion"]["signed_body_forward_speed_m_s"]["mean"], -.1)

    def test_shape_order_nonfinite_and_coarse_sampling_honesty(self):
        trace = synthetic_trace(dt=.02)
        self.assertIn("historical low-rate", diagnose(trace)["sampling"]["acquisition_status"])
        trace["metadata"]["foot_order"] = list(reversed(FEET))
        with self.assertRaisesRegex(ValueError, "foot_order"):
            diagnose(trace)
        trace = synthetic_trace()
        trace["samples"]["foot_force_N"][3][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            diagnose(trace)
        trace = synthetic_trace()
        trace["samples"]["time_s"][3] += .001
        with self.assertRaisesRegex(ValueError, "uniform"):
            diagnose(trace)

    def test_read_only_analysis_and_strict_json(self):
        trace = synthetic_trace()
        original = copy.deepcopy(trace)
        result = diagnose(trace)
        self.assertEqual(trace, original)
        json.dumps(result, allow_nan=False)

    def test_original_byte_sha_is_preserved(self):
        trace = synthetic_trace()
        raw = json.dumps(trace).encode()
        with tempfile.TemporaryDirectory(prefix="gecko_trace_test_") as folder:
            path = Path(folder)/"trace.json"
            path.write_bytes(raw)
            result = diagnose_file(path, settle_s=1., phase_bins=20, entrainment_tolerance_fraction=.10, period_cv_max=.10)
            self.assertEqual(result["original_trace"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
