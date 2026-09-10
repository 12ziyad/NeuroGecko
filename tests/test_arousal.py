"""Tests for brain/arousal.py.

The load-bearing test is the one that checks the sleep cycle period is STILL
NULL. This animal is in the reptile sleep literature -- seven lizard species,
leopard gecko among them, n=2 -- so it is genuinely tempting to reach for the
bearded dragon's well-known cycle period and call the module finished. That
would put an agamid's number in a eublepharid's mouth, which is the exact
substitution config/proxies.yaml exists to make visible.
"""

import pathlib
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.arousal import Arousal, DAY_S                        # noqa: E402


class TheGapStaysOpen(unittest.TestCase):
    def test_the_sleep_cycle_period_is_still_null(self):
        """If someone fills this in, it must be because a leopard gecko was
        measured -- not because a bearded dragon was."""
        self.assertIsNone(Arousal().sleep_cycle_period_s)

    def test_the_registry_says_why(self):
        from common.provenance import load_registry
        entry = load_registry()["entries"]["sleep_cycle_period_s"]
        self.assertIsNone(entry["value"])
        self.assertEqual(entry["species"], "NOT_IN_CORPUS")
        self.assertIn("bearded dragon", entry["notes"])

    def test_the_module_declares_the_absence(self):
        gap = " ".join(Arousal().state()["not_in_corpus"])
        self.assertIn("sleep_cycle_period_s", gap)


class TheAnimalIsAwakeAtNight(unittest.TestCase):
    def test_it_sleeps_through_the_light_phase(self):
        a = Arousal()
        for hour in (0, 2, 4, 6, 8):
            self.assertEqual(a.arousal_at(hour * 3600), 0.0, f"{hour}:00")

    def test_arousal_peaks_after_dark(self):
        a = Arousal()
        dark = a.arousal_at(a.dusk_s + 3 * 3600)
        light = a.arousal_at(a.dusk_s - 6 * 3600)
        self.assertGreater(dark, light)
        self.assertEqual(dark, 1.0)

    def test_the_rise_begins_before_dark(self):
        """Crepuscular, not simply nocturnal. Preferred body temperature rises
        through the light phase and peaks toward its end, which is proposed as
        what initiates evening emergence -- so arousal must not switch on at
        dusk, it must already be rising."""
        a = Arousal()
        just_before = a.arousal_at(a.dusk_s - 0.5 * 3600)
        self.assertGreater(just_before, 0.0)
        self.assertLess(just_before, 1.0)

    def test_it_decays_toward_dawn(self):
        """Activity is higher in the first hour after lights-out than in the
        last stretch before dawn."""
        a = Arousal()
        early = a.arousal_at(a.dusk_s + 2 * 3600)
        late = a.arousal_at(a.dusk_s + 11 * 3600)
        self.assertGreater(early, late)

    def test_arousal_stays_in_range(self):
        a = Arousal()
        for i in range(0, int(DAY_S), 600):
            v = a.arousal_at(i)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_the_clock_wraps(self):
        a = Arousal()
        a.reset(time_of_day_s=DAY_S - 60)
        out = a.step(120)
        self.assertLess(out["time_of_day_h"], 1.0)


class ProvenanceIsOnTheRecord(unittest.TestCase):
    def test_the_thin_numbers_carry_their_sample_sizes(self):
        published = Arousal().state()["published"]
        self.assertEqual(published["onset_n"], 1)
        self.assertEqual(published["peak_window_n"], 18)

    def test_the_caveats_are_stated_not_implied(self):
        caveats = " ".join(Arousal().state()["caveats"])
        self.assertIn("n=1", caveats)
        self.assertIn("SD larger than its mean", caveats)
        self.assertIn("no field activity budget", caveats)

    def test_the_invented_shape_is_declared(self):
        self.assertIn("shape", " ".join(Arousal().state()["invented"]))

    def test_a_bad_photoperiod_is_refused(self):
        with self.assertRaises(ValueError):
            Arousal(dark_fraction=0.0)
        with self.assertRaises(ValueError):
            Arousal(dark_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
