"""Tests for brain/vomeronasal.py.

The load-bearing test is the one that checks smell REFUSES to give a bearing.
Everything about this module is shaped by a single gap in the literature: the
assay that produced its numbers presents a scented swab about a centimetre from
the animal's snout, in plain view. It shows the animal can tell cricket
chemicals from a control. It shows nothing whatever about localisation, and no
gecko has ever been shown to find prey by smell alone.

A gradient would be four lines of code and would be an invention wearing a
published number's clothes.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.vomeronasal import Vomeronasal                      # noqa: E402


class SmellSaysThatNeverWhere(unittest.TestCase):
    """The whole design, in three tests."""

    def test_it_never_returns_a_bearing(self):
        nose = Vomeronasal()
        for distance in (0.005, 0.05, 0.2, 0.5, 5.0, None):
            out = nose.step(prey_distance_m=distance)
            self.assertIsNone(out["bearing_deg"])

    def test_the_absence_is_asserted_not_merely_missing(self):
        """A missing key invites someone to add one. A stated refusal does
        not."""
        out = Vomeronasal().step(prey_distance_m=0.02)
        self.assertIn("bearing_deg", out)
        self.assertIn("NOT AVAILABLE", out["localisation"])
        self.assertIn("bearing", " ".join(Vomeronasal().state()["refuses"]))

    def test_the_module_says_why(self):
        import brain.vomeronasal as mod
        self.assertIn("NO GECKO HAS EVER BEEN SHOWN TO FIND PREY BY SMELL ALONE",
                      mod.__doc__)


class ThePublishedRates(unittest.TestCase):
    def test_the_two_measured_endpoints_are_reproduced(self):
        """3.0 flicks/min at an odourless control, 14.57 at cricket chemicals.
        n=7, this species."""
        nose = Vomeronasal()
        self.assertAlmostEqual(nose.flick_rate(0.0), 3.0, places=6)
        self.assertAlmostEqual(nose.flick_rate(1.0), 14.57, places=6)

    def test_prey_odour_raises_flicking_about_fivefold(self):
        nose = Vomeronasal()
        self.assertAlmostEqual(nose.flick_rate(1.0) / nose.flick_rate(0.0),
                               4.86, delta=0.05)

    def test_the_interpolation_is_declared_invented(self):
        """Only the two ends were measured."""
        invented = " ".join(Vomeronasal().state()["invented"])
        self.assertIn("interpolation", invented)


class DefenceIsGatedOnChemistry(unittest.TestCase):
    """0.21 to scent, 0.07 to sight, n=42, this species. Only the chemical
    term is significant: chi2 8.098 p<0.0044 against chi2<0.01 p>0.9."""

    def test_scent_alone_reproduces_the_published_probability(self):
        nose = Vomeronasal()
        self.assertAlmostEqual(nose.defensive_probability(1.0), 0.21, places=6)

    def test_sight_alone_does_almost_nothing(self):
        """The number that indicts a vision-first threat model. Seeing a
        predator with no scent must not by itself drive a defensive response."""
        nose = Vomeronasal()
        self.assertEqual(nose.defensive_probability(0.0, predator_visible=True), 0.0)

    def test_sight_amplifies_scent_rather_than_replacing_it(self):
        nose = Vomeronasal()
        scent_only = nose.defensive_probability(1.0)
        both = nose.defensive_probability(1.0, predator_visible=True)
        self.assertGreater(both, scent_only)
        self.assertAlmostEqual(both - scent_only, 0.07, places=6)

    def test_chemical_outweighs_visual_threefold(self):
        nose = Vomeronasal()
        self.assertAlmostEqual(nose.predator_chemical / nose.predator_visual,
                               3.0, delta=0.01)


class TheFalloffIsHonestlyLabelled(unittest.TestCase):
    def test_concentration_only_decreases(self):
        nose = Vomeronasal()
        values = [nose.concentration(d) for d in (0.01, 0.05, 0.1, 0.3, 0.55)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_beyond_range_is_nothing(self):
        nose = Vomeronasal()
        self.assertEqual(nose.concentration(nose.max_range_m + 0.01), 0.0)
        self.assertEqual(nose.concentration(None), 0.0)
        self.assertEqual(nose.concentration(float("nan")), 0.0)

    def test_the_falloff_is_declared_invented(self):
        """No odour plume has been measured around any prey item for any
        gecko, and a real plume in moving air is not a function of range."""
        invented = " ".join(Vomeronasal().state()["invented"])
        self.assertIn("falloff", invented)

    def test_a_bad_configuration_is_refused(self):
        with self.assertRaises(ValueError):
            Vomeronasal(max_range_m=0.001, reference_m=0.01)


if __name__ == "__main__":
    unittest.main()
