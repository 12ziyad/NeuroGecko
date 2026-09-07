"""Tests for brain/spinal_cpg.py -- the leg rhythm.

The load-bearing test is the regression: with the coupling switched off, the
new oscillator must reproduce the clock the accepted walker uses. If that ever
drifts, a change in the gait gates could be the rewrite rather than the
biology, and the whole point of replacing the clock is lost.
"""

import importlib.util
import math
import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_scpg = _load("_spinal_cpg_test", "brain/spinal_cpg.py")
SpinalCPG = _scpg.SpinalCPG
FEET = _scpg.FEET


def _lab():
    from common.gait_config import get_gait_profile
    p = get_gait_profile("lab")
    return p, {f: p.touchdown_delays_cycle[i] for i, f in enumerate(FEET)}


class ReproducesTheAcceptedWalker(unittest.TestCase):
    """The regression that makes the rewrite safe."""

    def test_it_tracks_the_old_clock_with_the_coupling_off(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        worst = cpg.max_clock_deviation(seconds=60.0, dt_s=0.004)
        # A full minute of walking. The gait gates measure duty factors and
        # phase differences to two or three decimals; this is eleven orders
        # below that.
        self.assertLess(worst, 1e-9,
                        f"drifted {worst:.3e} of a stride from the clock")

    def test_bit_identity_is_NOT_claimed(self):
        """The build plan asked for a bit-identical trace. It is arithmetically
        unavailable over a run -- an accumulated phase and a closed-form one
        differ in the last bits, and the closed form snaps to exact cycle
        boundaries that an accumulator never lands on. Recorded rather than
        quietly downgraded.

        Note the two DO agree exactly for the first few steps, which is why
        this asserts over a run rather than after one step: an earlier version
        of this test checked a single step, found them equal, and would have
        been read as evidence for a claim that is false."""
        self.assertIn("not achievable", _scpg.__doc__.lower())
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        worst = cpg.max_clock_deviation(seconds=60.0, dt_s=0.004)
        self.assertGreater(worst, 0.0, "they do diverge, and the docs say so")
        self.assertLess(worst, 1e-9, "but far below anything the gates measure")

    def test_it_starts_where_the_clock_starts(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        for foot in FEET:
            self.assertAlmostEqual(cpg.phase_fraction(foot),
                                   cpg.clock_phase_fraction(foot, 0.0),
                                   places=12, msg=foot)

    def test_the_foot_order_comes_from_the_repository(self):
        """Restating it is how a silent index mismatch gets in. The canonical
        order is not the one anyone guesses."""
        from common.gait_config import FOOT_ORDER
        self.assertEqual(FEET, FOOT_ORDER)
        self.assertEqual(FEET, ("HL", "FL", "HR", "FR"))


class TheDescendingCommand(unittest.TestCase):
    """The thing that did not exist before, and the reason for the rewrite."""

    def test_frequency_is_settable(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        np.testing.assert_allclose(cpg.set_drive(frequency_hz=1.8), 1.8)

    def test_a_faster_command_produces_a_faster_rhythm(self):
        profile, delays = _lab()
        slow, fast = (SpinalCPG(profile.frequency_hz, delays) for _ in range(2))
        fast.set_drive(frequency_hz=2.4)
        for _ in range(50):
            slow.step(0.004)
            fast.step(0.004)
        # Over the same wall-clock, the driven one has advanced further.
        self.assertGreater(fast.phases()[0] - slow.phases()[0], 0.0)

    def test_turning_is_produced_by_left_right_asymmetry_alone(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        freqs = cpg.set_drive(frequency_hz=1.8, left_right_bias=0.2)
        left = [f for foot, f in zip(FEET, freqs) if foot.endswith("L")]
        right = [f for foot, f in zip(FEET, freqs) if foot.endswith("R")]
        self.assertTrue(all(l > r for l in left for r in right))
        # and a symmetric command is symmetric
        sym = cpg.set_drive(frequency_hz=1.8, left_right_bias=0.0)
        self.assertEqual(len(set(np.round(sym, 12))), 1)

    def test_the_command_to_frequency_mapping_is_not_invented_here(self):
        """set_drive takes hertz. Whatever turns a descending command into
        hertz belongs in the brainstem, with its own provenance -- putting it
        here would bury an invented constant in the cord."""
        self.assertIn("frequency_hz",
                      SpinalCPG.set_drive.__code__.co_varnames)
        self.assertIn("belongs in the brainstem", SpinalCPG.set_drive.__doc__)

    def test_invalid_commands_are_refused(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                cpg.set_drive(frequency_hz=bad)
        for bad in (1.5, -2.0, float("inf")):
            with self.assertRaises(ValueError):
                cpg.set_drive(left_right_bias=bad)


class LoadFeedback(unittest.TestCase):
    def test_it_is_off_by_default_and_declared_invented(self):
        profile, delays = _lab()
        self.assertEqual(SpinalCPG(profile.frequency_hz, delays).load_feedback,
                         0.0)
        self.assertIn("INVENTED", _scpg.__doc__)

    def test_with_the_gain_at_zero_loads_change_nothing(self):
        profile, delays = _lab()
        a = SpinalCPG(profile.frequency_hz, delays)
        b = SpinalCPG(profile.frequency_hz, delays)
        loads = {"HL": 1.0, "FL": 0.0, "HR": 1.0, "FR": 0.0}
        for _ in range(100):
            a.step(0.004)
            b.step(0.004, loads=loads)
        np.testing.assert_allclose(a.phases(), b.phases(), atol=1e-15)

    def test_with_the_gain_on_a_loaded_leg_is_held_back(self):
        profile, delays = _lab()
        free = SpinalCPG(profile.frequency_hz, delays, load_feedback=0.5)
        held = SpinalCPG(profile.frequency_hz, delays, load_feedback=0.5)
        loads = {"HL": 1.0, "FL": 0.0, "HR": 0.0, "FR": 0.0}
        for _ in range(100):
            free.step(0.004)
            held.step(0.004, loads=loads)
        self.assertNotAlmostEqual(free.phase_fraction("HL"),
                                  held.phase_fraction("HL"), places=6)
        # the unloaded legs are untouched
        self.assertAlmostEqual(free.phase_fraction("FL"),
                               held.phase_fraction("FL"), places=12)

    def test_bad_loads_are_refused(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        with self.assertRaises(ValueError):
            cpg.step(0.004, loads=[1.0, 2.0])
        with self.assertRaises(ValueError):
            cpg.step(0.004, loads=[1.0, 2.0, 3.0, float("nan")])


class TheSolverIsNotTheCallersRate(unittest.TestCase):
    """The basal ganglia cost two sessions to this exact mistake."""

    def test_the_answer_does_not_depend_on_the_control_rate(self):
        profile, delays = _lab()
        results = {}
        for dt in (0.02, 0.004, 0.001):
            cpg = SpinalCPG(profile.frequency_hz, delays)
            for _ in range(int(round(10.0 / dt))):
                cpg.step(dt)
            results[dt] = cpg.phases()
        reference = results[0.001]
        for dt, phases in results.items():
            np.testing.assert_allclose(phases, reference, atol=1e-9,
                                       err_msg=f"dt={dt}")

    def test_a_bad_timestep_is_refused(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        for bad in (0.0, -0.01, float("nan")):
            with self.assertRaises(ValueError):
                cpg.step(bad)


class Housekeeping(unittest.TestCase):
    def test_reset_returns_it_to_the_start(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        start = cpg.phases()
        for _ in range(100):
            cpg.step(0.004)
        cpg.reset()
        np.testing.assert_allclose(cpg.phases(), start, atol=1e-15)
        self.assertEqual(cpg.elapsed_s, 0.0)

    def test_phases_stay_in_the_unit_interval(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays, load_feedback=0.5)
        for i in range(2000):
            p = cpg.step(0.004, loads={f: float(i % 2) for f in FEET})
            self.assertTrue(np.all((p >= 0.0) & (p < 1.0)))

    def test_a_bad_frequency_is_refused_at_construction(self):
        _, delays = _lab()
        for bad in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                SpinalCPG(bad, delays)

    def test_state_is_reportable(self):
        profile, delays = _lab()
        cpg = SpinalCPG(profile.frequency_hz, delays)
        cpg.step(0.004)
        state = cpg.state()
        self.assertEqual(state["feet"], list(FEET))
        self.assertEqual(len(state["phase"]), 4)
        self.assertEqual(state["load_feedback"], 0.0)


if __name__ == "__main__":
    unittest.main()
