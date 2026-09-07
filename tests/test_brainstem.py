"""Tests for brain/brainstem.py -- the decision-to-command seam.

The brainstem carries the thinnest evidence of any module here, so as much of
this file guards the PROVENANCE as guards the behaviour: that the invented
constants are declared, that the absent gait channel stays absent, and that the
frequency band's dependence on an invented lock is on the record.
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


_sc = _load("_sc_for_brainstem_test", "brain/spinal_cpg.py")
_bs = _load("_brainstem_test", "brain/brainstem.py")
_gs = _load("_gs_for_brainstem_test", "brain/gecko_selector.py")
SpinalCPG, FEET = _sc.SpinalCPG, _sc.FEET
Brainstem = _bs.Brainstem
GeckoSelector = _gs.GeckoSelector


def _cord():
    from common.gait_config import get_gait_profile
    p = get_gait_profile("lab")
    return SpinalCPG(p.frequency_hz,
                     {f: p.touchdown_delays_cycle[i] for i, f in enumerate(FEET)})


class TurningIsAsymmetryInTheDrive(unittest.TestCase):
    """Published in direction: turning comes from left/right asymmetry in the
    descending drive, not from a separate steering command."""

    def test_a_target_on_the_left_speeds_the_right_side(self):
        """The sign. Getting it backwards makes an animal that turns away from
        everything it wants, and every downstream test would still pass."""
        cord = _cord()
        Brainstem(cord).step("hunt", 1.0, bearing_error_rad=+0.9)
        freq = dict(zip(FEET, cord._frequency))
        self.assertGreater(freq["HR"], freq["HL"])
        self.assertGreater(freq["FR"], freq["FL"])

    def test_a_target_on_the_right_speeds_the_left_side(self):
        cord = _cord()
        Brainstem(cord).step("hunt", 1.0, bearing_error_rad=-0.9)
        freq = dict(zip(FEET, cord._frequency))
        self.assertGreater(freq["HL"], freq["HR"])

    def test_straight_ahead_is_symmetric(self):
        cord = _cord()
        cmd = Brainstem(cord).step("hunt", 1.0, bearing_error_rad=0.0)
        self.assertAlmostEqual(cmd["turn"], 0.0, places=12)
        self.assertEqual(len(set(np.round(cord._frequency, 12))), 1)

    def test_the_turn_is_clipped(self):
        stem = Brainstem(_cord())
        for bearing in (math.pi / 2, -math.pi / 2, 3.0, -3.0):
            self.assertLessEqual(abs(stem.turn_for(bearing)), _bs.MAX_TURN)

    def test_a_bad_bearing_is_refused(self):
        with self.assertRaises(ValueError):
            Brainstem(_cord()).turn_for(float("nan"))


class DriveSetsSpeed(unittest.TestCase):
    def test_below_threshold_the_animal_stands_still(self):
        """Zero frequency is a real state, not a failure: a resting gecko has
        no stride frequency."""
        stem = Brainstem(_cord())
        self.assertEqual(stem.frequency_for(0.0), 0.0)
        self.assertEqual(stem.frequency_for(_bs.DRIVE_THRESHOLD), 0.0)
        self.assertGreater(stem.frequency_for(1.0), 0.0)

    def test_more_drive_means_a_faster_rhythm(self):
        """Published in DIRECTION only -- the salamander preparation this comes
        from steps an order of magnitude slower than a gecko, so the ordering
        transfers and the values do not."""
        stem = Brainstem(_cord())
        rates = [stem.frequency_for(d) for d in (0.3, 0.5, 0.7, 1.0)]
        self.assertEqual(rates, sorted(rates))
        self.assertTrue(all(a < b for a, b in zip(rates, rates[1:])))

    def test_the_band_is_centred_on_the_lock_not_the_published_frequency(self):
        """The honest problem, pinned. The cord is locked at 1.1888 Hz, which is
        INVENTED and exists to keep an old checkpoint loadable; the only
        published stride frequency for this species is 2.03 Hz. The band is
        anchored on the lock because the accepted walker runs there."""
        from common.gait_config import LOCKED_FREQUENCY_HZ
        stem = Brainstem(_cord())
        self.assertAlmostEqual(stem.base_hz, LOCKED_FREQUENCY_HZ, places=6)
        top = stem.frequency_for(1.0)
        self.assertLess(top, 2.03, "the band does not reach the published rate")
        self.assertIn("2.03", _bs.__doc__)

    def test_a_hesitant_decision_produces_a_hesitant_walk(self):
        """The urgency IS the gate value, so the strength of the decision
        survives into the movement. This is what a disinhibition model buys
        over a plain maximum."""
        stem = Brainstem(_cord())
        firm = stem.step("hunt", urgency=1.0)
        weak = stem.step("hunt", urgency=0.4)
        self.assertGreater(firm["frequency_hz"], weak["frequency_hz"])

    def test_non_locomotor_behaviours_do_not_drive_the_legs(self):
        stem = Brainstem(_cord())
        for behaviour in ("rest", "groom"):
            cmd = stem.step(behaviour, 1.0)
            self.assertFalse(cmd["stepping"], behaviour)
        self.assertFalse(stem.step(None)["stepping"])


class TheSeamWithTheSelector(unittest.TestCase):
    def test_it_drives_straight_from_the_basal_ganglia(self):
        sel = GeckoSelector()
        for _ in range(25):
            sel.step(hunger=0.9, prey_visible=1.0)
        stem = Brainstem(_cord())
        cmd = stem.step_from_selector(sel)
        self.assertEqual(cmd["behaviour"], "hunt")
        self.assertTrue(cmd["stepping"])

    def test_a_chosen_rest_stops_the_legs(self):
        sel = GeckoSelector()
        for _ in range(25):
            sel.step(fatigue=0.95)
        cmd = Brainstem(_cord()).step_from_selector(sel)
        self.assertEqual(cmd["behaviour"], "rest")
        self.assertFalse(cmd["stepping"])

    def test_choosing_nothing_stops_the_legs(self):
        sel = GeckoSelector()
        sel.step()
        cmd = Brainstem(_cord()).step_from_selector(sel)
        self.assertIsNone(cmd["behaviour"])
        self.assertFalse(cmd["stepping"])

    def test_the_urgency_it_uses_is_the_gate_of_the_chosen_channel(self):
        sel = GeckoSelector()
        for _ in range(25):
            sel.step(hunger=0.9, prey_visible=1.0)
        stem = Brainstem(_cord())
        cmd = stem.step_from_selector(sel)
        gate = float(sel.gates()[list(sel.channels).index(cmd["behaviour"])])
        self.assertAlmostEqual(cmd["drive"],
                               _bs.LOCOMOTOR[cmd["behaviour"]] * gate, places=5)


class ProvenanceIsOnTheRecord(unittest.TestCase):
    def test_the_invented_constants_are_declared(self):
        self.assertIn("INVENTED", _bs.__doc__)
        for name in ("DRIVE_THRESHOLD", "FREQUENCY_SPAN", "TURN_GAIN"):
            self.assertIn(name, Brainstem(_cord()).state()["invented"])

    def test_there_is_no_gait_selection_channel(self):
        """Deliberately absent. This species does not change footfall pattern
        with speed, so a gait switch would model something the animal does not
        do. The absence is a finding, and it is documented as one."""
        self.assertNotIn("gait_selection", dir(Brainstem))
        self.assertIn("DELIBERATELY ABSENT", _bs.__doc__)
        for word in ("44", "43"):        # the walking/running phase values
            self.assertIn(word, _bs.__doc__)

    def test_the_gecko_stimulation_work_is_cited_as_abstract_only(self):
        self.assertIn("Gekko gecko", _bs.__doc__)
        self.assertIn("abstract", _bs.__doc__.lower())

    def test_it_refuses_something_that_is_not_a_cord(self):
        with self.assertRaises(TypeError):
            Brainstem(object())

    def test_it_does_not_advance_the_cord_itself(self):
        """The walking controller owns the clock. A brainstem that also stepped
        the rhythm would double-advance it."""
        cord = _cord()
        before = cord.elapsed_s
        Brainstem(cord).step("hunt", 1.0)
        self.assertEqual(cord.elapsed_s, before)


if __name__ == "__main__":
    unittest.main()
