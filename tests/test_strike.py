"""Tests for brain/strike.py.

The strike exists because the arithmetic said pursuit cannot work: the walker
moves at 0.055 m/s, the cricket escapes at 0.093-0.143, and a real gecko
strikes at 0.851. These tests guard that arithmetic, the published success
rate, and -- as much as anything else -- the provenance, because almost every
number here was measured in a different species.
"""

import importlib.util
import pathlib
import sys
import unittest

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from brain.strike import Strike                              # noqa: E402
from common.provenance import parameter_value                # noqa: E402


class TheArithmeticThatMakesItNecessary(unittest.TestCase):
    """Why a walking-only animal could never have caught anything."""

    def test_the_walker_cannot_outrun_the_prey(self):
        walk = 0.055
        escape_low = 0.093
        self.assertLess(walk, escape_low,
                        "if the walker were faster than the prey, pursuit would "
                        "work and the strike would be optional")

    def test_the_strike_can(self):
        strike = Strike()
        self.assertGreater(strike.peak_speed_m_s, 0.143,
                           "the strike must beat the fastest published escape")
        self.assertGreater(strike.peak_speed_m_s / 0.055, 10.0,
                           "and it is roughly fifteen times the walk")


class ItFiresOnlyFromStrikingDistance(unittest.TestCase):
    def test_out_of_range_is_refused(self):
        s = Strike()
        self.assertIsNone(s.fire(0.30))
        self.assertIsNone(s.fire(s.trigger_distance_m * 1.01))
        self.assertIsNotNone(s.fire(s.trigger_distance_m * 0.99))

    def test_it_will_not_fire_twice_at_once(self):
        s = Strike()
        self.assertIsNotNone(s.fire(0.01))
        self.assertIsNone(s.fire(0.01), "a strike in flight cannot be relaunched")

    def test_a_bad_distance_is_refused(self):
        s = Strike()
        self.assertFalse(s.in_range(float("nan")))
        with self.assertRaises(ValueError):
            s.speed_for(float("nan"))


class TheOnePublishedControlLaw(unittest.TestCase):
    """Attack distance and peak speed rise together, r = 0.47, P = 0.009."""

    def test_a_longer_strike_is_a_faster_strike(self):
        s = Strike(rng=np.random.default_rng(0))
        near = np.mean([s.speed_for(0.002) for _ in range(400)])
        far = np.mean([s.speed_for(0.020) for _ in range(400)])
        self.assertGreater(far, near,
                           "this is the direction the published correlation has")

    def test_the_scaling_is_moderate_not_total(self):
        """r = 0.47 explains about 22 % of the variance. Modelling it as a
        strong deterministic law would overstate what was measured."""
        s = Strike(rng=np.random.default_rng(0))
        near = np.mean([s.speed_for(0.0) for _ in range(2000)])
        far = np.mean([s.speed_for(s.trigger_distance_m) for _ in range(2000)])
        self.assertLess((far - near) / s.peak_speed_m_s, 0.75,
                        "the whole span must stay a tendency, not a rule")

    def test_the_published_mean_sits_in_the_middle(self):
        s = Strike(rng=np.random.default_rng(1))
        mid = np.mean([s.speed_for(s.trigger_distance_m / 2.0) for _ in range(2000)])
        self.assertAlmostEqual(mid / s.peak_speed_m_s, 1.0, delta=0.05)


class ItMissesAsOftenAsTheAnimalDoes(unittest.TestCase):
    def test_the_success_rate_reproduces_the_published_one(self):
        """82.9 % on evasive crickets, 116/140. A model that never misses has
        been fitted to the harness rather than the animal."""
        s = Strike(rng=np.random.default_rng(3))
        for _ in range(4000):
            s.fire(0.015)
            s.step(1.0)
        st = s.state()
        self.assertAlmostEqual(st["success_rate"], 0.829, delta=0.02)

    def test_the_outcome_is_decided_at_launch(self):
        """A strike is ballistic: peak velocity is reached 9 ms before contact
        and the animal is decelerating on arrival, so it cannot correct."""
        s = Strike(rng=np.random.default_rng(0))
        plan = s.fire(0.01)
        self.assertIn("will_hit", plan)
        _, done, hit = s.step(s.duration_s)
        self.assertTrue(done)
        self.assertEqual(hit, plan["will_hit"])

    def test_it_never_hits_without_being_fired(self):
        s = Strike()
        fraction, done, hit = s.step(0.05)
        self.assertEqual((fraction, done, hit), (0.0, False, False))


class TheShapeOfTheStrike(unittest.TestCase):
    def test_it_lasts_the_published_duration(self):
        s = Strike(rng=np.random.default_rng(0))
        s.fire(0.01)
        steps = 0
        while True:
            _, done, _ = s.step(0.004)
            steps += 1
            if done or steps > 500:
                break
        self.assertAlmostEqual(steps * 0.004, s.duration_s, delta=0.005)

    def test_speed_peaks_before_contact(self):
        """Published: peak velocity 9.03 ms before contact, peak acceleration
        16.34 ms before. The animal arrives slowing down."""
        s = Strike()
        samples = [s.velocity_profile(f) for f in np.linspace(0, 1, 101)]
        peak_at = float(np.argmax(samples)) / 100.0
        self.assertLess(peak_at, 1.0, "the peak must not be at contact")
        self.assertGreater(peak_at, 0.5, "nor in the first half")

    def test_the_two_named_strikes_are_both_reachable(self):
        """`snap` and `bag jump` are separate categories in the published
        E. macularius ethogram, n=18."""
        s = Strike()
        modes = {s.mode_for(d) for d in np.linspace(0, s.trigger_distance_m, 40)}
        self.assertEqual(modes, {"snap", "bag_jump"})


class ProvenanceIsOnTheRecord(unittest.TestCase):
    def test_almost_nothing_here_is_the_target_species(self):
        """The uncomfortable fact, pinned so it cannot quietly drift into
        being described as a leopard gecko measurement."""
        from common.provenance import load_registry
        entries = load_registry()["entries"]
        for key in ("strike_trigger_distance_m", "strike_peak_speed_m_s",
                    "strike_peak_acceleration_m_s2", "strike_capture_success",
                    "strike_distance_speed_correlation_r"):
            self.assertEqual(entries[key]["species"], "Coleonyx variegatus", key)

    def test_the_one_target_species_number_is_marked_uncertain(self):
        """The duration is the only figure from E. macularius and it is the
        weakest: read off a figure, from a paper whose capture data was too
        sparse to tabulate, filmed at 64 fps against an 80 ms event."""
        from common.provenance import load_registry
        entry = load_registry()["entries"]["strike_duration_s"]
        self.assertEqual(entry["species"], "E_macularius")
        self.assertEqual(entry["confidence"], "uncertain")
        self.assertIn("64 fps", entry["source"])

    def test_the_chameleon_numbers_are_named_as_wrong(self):
        """"486 m/s^2, 5.8 m/s, 35 cm" circulates as leopard gecko strike data.
        It is chameleon tongue projection. The module must keep saying so."""
        import brain.strike as mod
        self.assertIn("chameleon", mod.__doc__.lower())
        self.assertIn("486", mod.__doc__)

    def test_there_is_no_tongue(self):
        """Geckos take prey with the jaws. A tongue strike here would be a
        chameleon wearing a gecko's name."""
        s = Strike()
        self.assertNotIn("tongue", dir(s))
        self.assertIn("No tongue", brain_doc := __import__("brain.strike",
                      fromlist=["x"]).__doc__)

    def test_the_invented_parts_are_declared(self):
        s = Strike()
        invented = s.state()["invented"]
        self.assertTrue(invented)
        self.assertTrue(any("boundary" in i for i in invented))


if __name__ == "__main__":
    unittest.main()
