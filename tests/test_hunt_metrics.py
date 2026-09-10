"""Tests for brain/hunt_metrics.py.

Two things are being guarded. The first is that the hunt is scored by PATH
EFFICIENCY rather than time, which for this species is a published result and
not a preference: path length is significant across training (F=24.157,
p<0.0001, n=42) and latency is not (F=0.326, p=0.568). Geckos found a hidden
goal FASTER in complete darkness than after training, by moving twice as fast
along paths twice as long -- so a latency-scored task rewards a fast undirected
search with no spatial knowledge at all.

The second is that the oracle shadow can never reach the animal. Three
privileged channels in this project were believed off and were not (#26, #33,
#165), so this one refuses to be switchable rather than defaulting to safe.
"""

import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.hunt_metrics import HuntMetrics                      # noqa: E402


class TheShadowCannotReachTheAnimal(unittest.TestCase):
    def test_it_refuses_to_be_switched_off_shadow_mode(self):
        with self.assertRaises(ValueError):
            HuntMetrics(shadow_only=False)

    def test_every_oracle_field_is_namespaced(self):
        """A stray read has to be visible in a diff."""
        m = HuntMetrics()
        row = m.step(mouth_xy=(0., 0.), prey_xy=(0.2, 0.), true_bearing_deg=0.0,
                     eye_bearing_deg=3.0, eye_salience=0.5, speed_m_s=0.05)
        oracle = {"shadow_true_range_m", "shadow_true_bearing_deg",
                  "shadow_bearing_error_deg", "shadow_failure_mode"}
        self.assertTrue(oracle <= set(row))
        for key in row:
            if "true" in key or "bearing_error" in key:
                self.assertTrue(key.startswith("shadow_"), key)

    def test_the_module_says_what_it_must_not_become(self):
        import brain.hunt_metrics as mod
        self.assertIn("never into the animal", mod.__doc__)


class ItScoresDirectnessNotSpeed(unittest.TestCase):
    def test_a_straight_approach_scores_near_one(self):
        m = HuntMetrics()
        m.begin_pursuit((0.0, 0.0), (0.30, 0.0))
        for x in np.linspace(0.02, 0.20, 10):
            row = m.step(mouth_xy=(x, 0.0), prey_xy=(0.30, 0.0),
                         true_bearing_deg=0.0, speed_m_s=0.05)
        self.assertGreater(row["path_efficiency"], 0.95)

    def test_a_thrash_that_arrives_scores_near_zero(self):
        """The darkness result, in miniature: getting there fast by covering
        ground is not the same as getting there directly."""
        m = HuntMetrics()
        m.begin_pursuit((0.0, 0.0), (0.30, 0.0))
        for i in range(60):
            x = 0.003 * i
            y = 0.05 * np.sin(i)          # wandering wildly sideways
            row = m.step(mouth_xy=(x, y), prey_xy=(0.30, 0.0),
                         true_bearing_deg=0.0, speed_m_s=0.20)
        self.assertLess(row["path_efficiency"], 0.35)

    def test_efficiency_is_undefined_before_the_animal_moves(self):
        m = HuntMetrics()
        m.begin_pursuit((0.0, 0.0), (0.30, 0.0))
        self.assertIsNone(m.path_efficiency((0.0, 0.0), (0.30, 0.0)))

    def test_speed_alone_does_not_raise_the_score(self):
        """Two animals covering the same route, one twice as fast, must score
        the same. Time is not in the measure."""
        def walk(step):
            m = HuntMetrics(); m.begin_pursuit((0.0, 0.0), (0.30, 0.0))
            row = None
            for x in np.arange(step, 0.20, step):
                row = m.step(mouth_xy=(x, 0.0), prey_xy=(0.30, 0.0),
                             true_bearing_deg=0.0, speed_m_s=step * 50)
            return row["path_efficiency"]
        self.assertAlmostEqual(walk(0.004), walk(0.008), delta=0.02)


class ItSeparatesTheTwoFailures(unittest.TestCase):
    """The reason the shadow exists: 'never saw it' and 'saw it and could not
    get there' produce the same capture rate and need opposite fixes."""

    def test_no_detection_is_labelled_as_such(self):
        m = HuntMetrics()
        row = m.step(mouth_xy=(0., 0.), prey_xy=(0.2, 0.), true_bearing_deg=10.0,
                     eye_bearing_deg=None, eye_salience=0.0, speed_m_s=0.05)
        self.assertEqual(row["shadow_failure_mode"], "not detected")

    def test_a_wrong_bearing_is_distinguished_from_no_bearing(self):
        m = HuntMetrics()
        row = m.step(mouth_xy=(0., 0.), prey_xy=(0.2, 0.), true_bearing_deg=10.0,
                     eye_bearing_deg=70.0, eye_salience=0.4, speed_m_s=0.05)
        self.assertEqual(row["shadow_failure_mode"], "detected, bearing wrong")

    def test_a_usable_bearing_is_recognised(self):
        m = HuntMetrics()
        row = m.step(mouth_xy=(0., 0.), prey_xy=(0.2, 0.), true_bearing_deg=10.0,
                     eye_bearing_deg=16.0, eye_salience=0.4, speed_m_s=0.05)
        self.assertEqual(row["shadow_failure_mode"], "detected, bearing usable")

    def test_the_summary_counts_all_three(self):
        m = HuntMetrics()
        rows = [
            m.step(mouth_xy=(0., 0.), prey_xy=(0.2, 0.), true_bearing_deg=0.0,
                   eye_salience=0.0, speed_m_s=0.05),
            m.step(mouth_xy=(0.01, 0.), prey_xy=(0.2, 0.), true_bearing_deg=0.0,
                   eye_bearing_deg=80.0, eye_salience=0.3, speed_m_s=0.05),
            m.step(mouth_xy=(0.02, 0.), prey_xy=(0.2, 0.), true_bearing_deg=0.0,
                   eye_bearing_deg=2.0, eye_salience=0.3, speed_m_s=0.05),
        ]
        modes = m.summarise(rows)["failure_modes"]
        self.assertEqual(set(modes), {"not detected", "detected, bearing wrong",
                                      "detected, bearing usable"})


class TheApproachCriterionIsThePublishedOne(unittest.TestCase):
    def test_all_three_conditions_are_required(self):
        m = HuntMetrics()
        self.assertTrue(m.is_approaching(10.0, 0.05, True))
        self.assertFalse(m.is_approaching(120.0, 0.05, True), "bearing")
        self.assertFalse(m.is_approaching(10.0, 0.001, True), "speed")
        self.assertFalse(m.is_approaching(10.0, 0.05, False), "range not closing")

    def test_the_bearing_window_is_the_published_ninety_degrees(self):
        self.assertEqual(HuntMetrics.APPROACH_BEARING_DEG, 90.0)


if __name__ == "__main__":
    unittest.main()
