"""The hypothalamus must reproduce published physiology, not merely look plausible.

`brain/drives.py` is plausible: hunger rises, eating lowers it, energy trades
against movement. Every constant in it is invented, and nothing it does can be
checked against an animal. These tests exist so the replacement can be wrong in
public.

The load-bearing check is that the module reproduces the scorecard's own arithmetic
from the raw measurement. The scorecard states 0.075 mL O2/g/h measured and, after
scaling, 51.1 J/h and ~140 days of reserve for a 40 g adult. If the module cannot
land on those from the registry alone, its energy budget is not the published one.
"""
import math
import unittest

import numpy as np

from brain.hypothalamus import Homeostasis, Physiology


class ReproducesThePublishedArithmetic(unittest.TestCase):
    """Independent re-derivation of numbers the corpus states outright."""

    def setUp(self):
        self.forty = Physiology.from_registry(body_mass_kg=0.040)

    def test_resting_metabolism_of_a_40g_adult_matches_the_scorecard(self):
        """Scorecard: 2.54 mL O2/h = 51.1 J/h = 14.2 mW."""
        joules_per_hour = self.forty.resting_power_W() * 3600.0
        self.assertAlmostEqual(joules_per_hour, 51.1, delta=1.0)
        self.assertAlmostEqual(self.forty.resting_power_W() * 1000.0, 14.2, delta=0.4)

    def test_the_starvation_reserve_matches_the_scorecard(self):
        """Scorecard: ~140 days of resting metabolism in the tail."""
        days = self.forty.reserve_J / (self.forty.resting_power_W() * 86400.0)
        self.assertAlmostEqual(days, 140.0, delta=10.0)

    def test_the_oxycalorific_equivalent_lands_on_the_physical_constant(self):
        """Recovered from the scorecard's own two numbers; must be ~20.1 J/mL."""
        self.assertAlmostEqual(self.forty.oxycalorific_J_per_mL, 20.1, delta=0.2)

    def test_the_allometry_is_anchored_at_the_mass_it_was_measured_on(self):
        """Anchoring at 40 g instead inflates resting metabolism by about 17%."""
        grams = 40.0
        specific = (self.forty.rmr_mL_O2_per_g_per_h
                    * (grams / self.forty.rmr_reference_mass_g) ** self.forty.rmr_mass_exponent)
        self.assertAlmostEqual(specific, 0.064, delta=0.002)

    def test_one_meal_covers_roughly_the_published_feeding_interval(self):
        """Two independent measurements that nobody arranged to agree.

        Metabolism and the husbandry schedule were recorded by different people
        for different reasons. One meal should last about as long as the gap to
        the next one, with a small surplus for an animal maintaining condition.
        """
        from common.provenance import parameter_value
        mass = 0.040
        physiology = Physiology.from_registry(body_mass_kg=mass)
        offering = mass * float(parameter_value("meal_offering_fraction_bodyweight"))
        eaten = offering * float(parameter_value("meal_consumed_fraction_of_offering"))
        meal_hours = physiology.meal_energy_J(eaten) / (physiology.resting_power_W() * 3600.0)
        interval_hours = 7 * 24 / float(parameter_value("feeding_events_per_week"))
        self.assertGreater(meal_hours, interval_hours * 0.7,
                           "a meal that does not reach the next feeding is a starving animal")
        self.assertLess(meal_hours, interval_hours * 2.5,
                        "a meal worth several intervals means the energy model is too generous")


class HungerIsSlow(unittest.TestCase):
    """The finding this module exists to make unavoidable."""

    def setUp(self):
        self.physiology = Physiology.from_registry(body_mass_kg=0.038)

    def test_an_episode_barely_moves_hunger(self):
        for seconds in (20.0, 300.0):
            state = Homeostasis(self.physiology)
            state.step(seconds)
            self.assertLess(state.energy_deficit, .005,
                            f"{seconds:.0f} s of a real gecko's metabolism is nothing")

    def test_hunger_reaches_the_published_inter_meal_interval(self):
        """THE calibration of this module.

        The mean inter-meal interval is 2.33 days. An animal fed on that schedule
        should be substantially hungry when the next meal is due -- not sated, and
        not starving. Nothing was fitted to make this true: it falls out of the
        measured metabolism, the measured meal size and the measured schedule.
        """
        from common.provenance import parameter_value
        interval_s = float(parameter_value("inter_meal_interval_days")) * 86400.0
        state = Homeostasis(self.physiology)
        state.step(interval_s)
        self.assertGreater(state.energy_deficit, .40, "should be hungry when the meal is due")
        self.assertLess(state.energy_deficit, .95, "should not be desperate on a normal schedule")

    def test_a_meal_at_that_interval_very_nearly_resets_hunger(self):
        from common.provenance import parameter_value
        interval_s = float(parameter_value("inter_meal_interval_days")) * 86400.0
        state = Homeostasis(self.physiology)
        state.step(interval_s)
        state.step(1.0, meal_wet_mass_kg=0.038 * 0.05 * 0.547)
        self.assertLess(state.energy_deficit, .10)

    def test_hunger_and_starvation_are_different_timescales(self):
        """Hunger saturates in about a week; the reserve lasts months."""
        week = Homeostasis(self.physiology)
        week.step(7 * 86400.0)
        self.assertGreaterEqual(week.energy_deficit, .95, "a week unfed is maximally hungry")
        self.assertLess(week.starvation_fraction, .10, "but nowhere near starving")
        months = Homeostasis(self.physiology)
        months.step(140 * 86400.0)
        self.assertGreater(months.starvation_fraction, .90,
                           "the tail reserve is ~140 days and must actually run out")

    def test_the_old_module_saturates_where_this_one_does_not(self):
        """A direct comparison, so the difference is on the record."""
        from brain.drives import DriveState
        old = DriveState()
        for _ in range(67):
            old.update(1.0)
        new = Homeostasis(self.physiology)
        new.step(67.0)
        self.assertAlmostEqual(old.hunger, 1.0, places=6)
        self.assertLess(new.energy_deficit, .002)


class HomeostaticReinforcement(unittest.TestCase):
    """Reward is the reduction in drive, not a hand-written bonus."""

    def setUp(self):
        self.physiology = Physiology.from_registry(body_mass_kg=0.038)
        self.meal = 0.038 * 0.05 * 0.547

    def test_eating_while_hungry_is_rewarded(self):
        state = Homeostasis(self.physiology, energy_J=self.physiology.reserve_J * .80)
        reward = state.step(1.0, meal_wet_mass_kg=self.meal)
        self.assertGreater(reward, 0.0)

    def test_starving_is_punished(self):
        state = Homeostasis(self.physiology)
        reward = state.step(86400.0)
        self.assertLess(reward, 0.0, "losing energy must reduce reward")

    def test_a_starving_animal_is_still_rewarded_for_eating(self):
        """Clipping the drive would zero the reward exactly when food matters most."""
        starving = Homeostasis(self.physiology,
                               energy_J=self.physiology.reserve_J * .80)
        self.assertEqual(starving.energy_deficit, 1.0, "observation saturates")
        self.assertGreater(starving.deficit_in_meals, 1.0, "drive does not")
        self.assertGreater(starving.step(1.0, meal_wet_mass_kg=self.meal), 0.0)

    def test_eating_when_already_full_earns_nothing(self):
        """A full animal gains no drive reduction, so overeating is not rewarded.

        This is the property that makes HRRL homeostatic rather than acquisitive:
        the reward is the deficit closed, so there is none left to close.
        """
        state = Homeostasis(self.physiology)
        reward = state.step(1.0, meal_wet_mass_kg=self.meal)
        self.assertAlmostEqual(reward, 0.0, delta=1e-6)

    def test_a_bigger_deficit_makes_the_same_meal_worth_more(self):
        small = Homeostasis(self.physiology, energy_J=self.physiology.reserve_J * .95)
        large = Homeostasis(self.physiology, energy_J=self.physiology.reserve_J * .60)
        self.assertGreater(large.step(1.0, meal_wet_mass_kg=self.meal),
                           small.step(1.0, meal_wet_mass_kg=self.meal))

    def test_the_drive_is_convex_which_is_why_hunger_changes_what_food_is_worth(self):
        """n=m would make D linear and a meal worth the same at any hunger."""
        state = Homeostasis(self.physiology)
        self.assertGreater(state.drive_exponent_n, state.drive_exponent_m)
        with self.assertRaisesRegex(ValueError, "convex"):
            Homeostasis(self.physiology, drive_exponent_n=2.0, drive_exponent_m=2.0)

    def test_the_reserve_cannot_go_negative_or_overflow(self):
        state = Homeostasis(self.physiology, energy_J=10.0)
        state.step(86400.0 * 365)
        self.assertGreaterEqual(state.energy_J, 0.0)
        state.step(1.0, meal_wet_mass_kg=1.0)
        self.assertLessEqual(state.energy_J, self.physiology.reserve_J)


class ThermostatIsWiredButInert(unittest.TestCase):
    """It must say so rather than appear to work."""

    def setUp(self):
        self.physiology = Physiology.from_registry(body_mass_kg=0.038)

    def test_at_the_preferred_temperature_the_error_is_exactly_zero(self):
        state = Homeostasis(self.physiology)
        self.assertEqual(state.thermal_error_C, (0.0, 0.0))
        self.assertTrue(state.state()["thermostat_inert"])

    def test_it_responds_correctly_once_a_temperature_field_exists(self):
        low, high = self.physiology.preferred_temperature_C
        cold = Homeostasis(self.physiology, body_temperature_C=low - 5.0)
        warm = Homeostasis(self.physiology, body_temperature_C=high + 5.0)
        self.assertAlmostEqual(cold.thermal_error_C[0], 5.0)
        self.assertEqual(cold.thermal_error_C[1], 0.0)
        self.assertAlmostEqual(warm.thermal_error_C[1], 5.0)
        self.assertEqual(warm.thermal_error_C[0], 0.0)
        self.assertGreater(cold.drive(), Homeostasis(self.physiology).drive())

    def test_the_preferred_band_is_the_published_one(self):
        self.assertEqual(self.physiology.preferred_temperature_C, (29.5, 31.9))


class CostOfMoving(unittest.TestCase):
    """The activity term is published too, and its temperature independence is
    itself the published finding."""

    def setUp(self):
        self.physiology = Physiology.from_registry(body_mass_kg=0.038)

    def test_walking_costs_about_twice_resting(self):
        walking = self.physiology.locomotion_power_W(0.055)
        resting = self.physiology.resting_power_W()
        self.assertGreater(walking / resting, 1.5)
        self.assertLess(walking / resting, 4.0)

    def test_cost_of_transport_is_linear_in_speed_and_zero_at_rest(self):
        self.assertEqual(self.physiology.locomotion_power_W(0.0), 0.0)
        self.assertAlmostEqual(self.physiology.locomotion_power_W(0.10),
                               2 * self.physiology.locomotion_power_W(0.05), places=12)

    def test_negative_speed_costs_nothing_rather_than_refunding_energy(self):
        self.assertEqual(self.physiology.locomotion_power_W(-1.0), 0.0)

    def test_moving_drains_the_reserve_faster_than_resting(self):
        resting = Homeostasis(self.physiology)
        moving = Homeostasis(self.physiology)
        resting.step(3600.0)
        moving.step(3600.0, activity_power_W=self.physiology.locomotion_power_W(0.055))
        self.assertGreater(moving.energy_deficit, resting.energy_deficit)


class HonestDriveSet(unittest.TestCase):
    def setUp(self):
        self.physiology = Physiology.from_registry(body_mass_kg=0.038)

    def test_only_drives_with_a_published_basis_survive(self):
        """curiosity and target_interest had none; fear belongs to the tectum."""
        state = Homeostasis(self.physiology)
        self.assertEqual(state.vector().shape, (4,))
        self.assertEqual(len(state.vector()), 4)

    def test_starvation_is_reported_separately_from_hunger(self):
        state = Homeostasis(self.physiology)
        self.assertIn("starvation_fraction", state.state())
        self.assertIn("energy_debt_J", state.state())
        self.assertIn("meal_scale_J", state.state())

    def test_the_interoception_vector_is_bounded(self):
        for energy in (0.0, .5, 1.0):
            state = Homeostasis(self.physiology, energy_J=self.physiology.reserve_J * energy)
            vector = state.vector()
            self.assertTrue(np.all(vector >= 0.0) and np.all(vector <= 1.0))
            self.assertTrue(np.all(np.isfinite(vector)))

    def test_energy_persists_across_an_episode_reset(self):
        """A gecko does not become full because an episode ended."""
        state = Homeostasis(self.physiology)
        state.step(30 * 86400.0)
        hungry = state.energy_deficit
        state.reset()
        self.assertAlmostEqual(state.energy_deficit, hungry, places=12)
        self.assertEqual(state.elapsed_s, 0.0)

    def test_resetting_the_body_is_available_but_explicit(self):
        state = Homeostasis(self.physiology)
        state.step(30 * 86400.0)
        state.reset(energy_J=self.physiology.reserve_J)
        self.assertAlmostEqual(state.energy_deficit, 0.0)

    def test_time_compression_defaults_to_none_and_is_always_reported(self):
        state = Homeostasis(self.physiology)
        self.assertEqual(state.time_compression, 1.0)
        self.assertIn("time_compression", state.state())
        fast = Homeostasis(self.physiology, time_compression=1000.0)
        fast.step(60.0)
        slow = Homeostasis(self.physiology)
        slow.step(60.0)
        self.assertGreater(fast.energy_deficit, slow.energy_deficit)

    def test_an_invalid_compression_is_refused(self):
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                Homeostasis(self.physiology, time_compression=bad)

    def test_the_summary_names_what_is_not_modelled(self):
        missing = " ".join(self.physiology.summary()["not_modelled"]).lower()
        self.assertIn("postprandial", missing)
        self.assertIn("q10", missing)


class WiredIntoTheBrainEnvironment(unittest.TestCase):
    """Opt-in, because the honest drive vector is a different shape."""

    def env(self, **kwargs):
        from envs.gecko_brain_env import GeckoBrainEnv
        return GeckoBrainEnv(seed=0, max_steps=10, **kwargs)

    def test_the_drive_vector_shrinks_from_six_to_four(self):
        old, new = self.env(), self.env(homeostasis=True)
        self.assertEqual(old.observation_space["drives"].shape, (6,))
        self.assertEqual(new.observation_space["drives"].shape, (4,))
        self.assertEqual(new.reset(seed=0)[0]["drives"].shape, (4,))

    def test_it_is_off_by_default_so_nothing_existing_changes(self):
        self.assertIsNone(self.env().homeostasis)

    def test_physiology_reads_the_real_body_mass_from_the_model(self):
        env = self.env(homeostasis=True)
        model_mass = float(np.sum(env.walk_env.model.body_mass[1:]))
        self.assertAlmostEqual(env.homeostasis.physiology.body_mass_kg, model_mass, places=9)

    def test_a_step_produces_an_hrrl_reward_and_costs_energy(self):
        env = self.env(homeostasis=True)
        env.reset(seed=0)
        before = env.homeostasis.energy_J
        env.step(np.zeros(4, dtype=np.float32))
        self.assertLess(env.homeostasis.energy_J, before, "living costs energy")
        self.assertLessEqual(env.homeostasis_reward, 0.0, "losing energy is not rewarded")

    def test_energy_survives_an_episode_reset(self):
        env = self.env(homeostasis=True)
        env.reset(seed=0)
        env.homeostasis.step(30 * 86400.0)
        hungry = env.homeostasis.energy_deficit
        env.reset(seed=1)
        self.assertAlmostEqual(env.homeostasis.energy_deficit, hungry, places=12)


if __name__ == "__main__":
    unittest.main()
