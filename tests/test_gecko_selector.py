"""Tests for brain/gecko_selector.py -- the animal's action selection.

These are BEHAVIOURAL: they assert what the gecko does, not what the numbers
are. The numbers are guarded in tests/test_prescott_bg.py against the published
table; here the question is only whether a gecko carrying those numbers behaves
like an animal.
"""

import importlib.util
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


_gs = _load("_gecko_selector_test", "brain/gecko_selector.py")
GeckoSelector = _gs.GeckoSelector
salience_from_drives = _gs.salience_from_drives
CHANNELS = _gs.CHANNELS


def settle(sel, steps=20, **drives):
    for _ in range(steps):
        sel.step(**drives)
    return sel


class TheAnimalDecides(unittest.TestCase):
    def test_a_starving_gecko_that_can_see_prey_hunts(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        self.assertEqual(sel.selected(), "hunt")
        self.assertTrue(sel.committed(), "it should commit, not half-hunt")

    def test_a_sated_gecko_ignores_visible_prey(self):
        """Sigma-pi: hunting needs hunger AND prey, not either alone."""
        sel = settle(GeckoSelector(), hunger=0.05, prey_visible=1.0)
        self.assertNotEqual(sel.selected(), "hunt")

    def test_a_hungry_gecko_with_nothing_in_sight_explores(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=0.0)
        self.assertEqual(sel.selected(), "explore")

    def test_a_predator_outranks_a_meal(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0, threat=0.95)
        self.assertEqual(sel.selected(), "flee")

    def test_exhaustion_produces_rest_and_cold_produces_basking(self):
        self.assertEqual(settle(GeckoSelector(), fatigue=0.9).selected(), "rest")
        self.assertEqual(settle(GeckoSelector(), cold=0.7).selected(), "bask")

    def test_a_contented_gecko_does_nothing_in_particular(self):
        """None is a real answer. There is no fallback behaviour, and adding
        one would hide a broken competition behind something that looked fine."""
        self.assertIsNone(settle(GeckoSelector()).selected())


class Persistence(unittest.TestCase):
    """A behaviour with a duration, from published structure rather than an
    invented stickiness constant."""

    def test_a_running_behaviour_resists_a_marginally_better_rival(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        self.assertEqual(sel.selected(), "hunt")
        for _ in range(5):
            sel.step(hunger=0.8, prey_visible=1.0, cold=0.62)
        self.assertEqual(sel.selected(), "hunt",
                         "a slightly better option should not cause a switch")

    def test_a_decisively_better_rival_does_win(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        for _ in range(20):
            sel.step(hunger=0.8, prey_visible=1.0, cold=0.99)
        self.assertEqual(sel.selected(), "bask",
                         "persistence must not become paralysis")

    def test_reset_forgets_the_running_behaviour(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        sel.reset()
        self.assertIsNone(sel.selected())

    def test_persistence_is_not_an_invented_constant_any_more(self):
        """The old salience fed its own output back through an INVENTED gain of
        0.15. Persistence now comes from the model's cortico-thalamic loop.
        Guarded so it cannot creep back in."""
        import inspect
        src = inspect.getsource(salience_from_drives)
        self.assertNotIn("persistence_gain", src)
        self.assertNotIn("persistence", inspect.signature(
            salience_from_drives).parameters)


class DopamineDoesWhatItDoesInAnimals(unittest.TestCase):
    def test_depletion_freezes_the_animal(self):
        """The published akinesia result: at zero tonic dopamine nothing is
        selected however loud the drives are."""
        sel = GeckoSelector(dopamine=0.0)
        settle(sel, hunger=0.9, prey_visible=1.0, threat=0.9, cold=0.9)
        self.assertIsNone(sel.selected())

    def test_excess_dopamine_releases_losing_behaviours(self):
        """Distortion: more than one behaviour partly expressed at once."""
        calm = settle(GeckoSelector(dopamine=0.3),
                      hunger=0.7, prey_visible=0.9, cold=0.6)
        wild = settle(GeckoSelector(dopamine=0.6),
                      hunger=0.7, prey_visible=0.9, cold=0.6)
        self.assertGreaterEqual(wild.bg.distortion(), calm.bg.distortion())

    def test_a_dopamine_that_breaks_the_transfer_is_refused(self):
        with self.assertRaises(ValueError):
            GeckoSelector(dopamine=5.0)


class TheSeamWithTheHypothalamus(unittest.TestCase):
    def test_it_accepts_the_hypothalamus_drive_vector_directly(self):
        sel = GeckoSelector()
        for _ in range(20):
            gates = sel.step_from_homeostasis([0.8, 0.0, 0.0, 0.0],
                                              prey_visible=1.0)
        self.assertEqual(len(gates), len(CHANNELS))
        self.assertEqual(sel.selected(), "hunt")

    def test_a_wrong_shaped_drive_vector_is_refused(self):
        with self.assertRaises(ValueError):
            GeckoSelector().step_from_homeostasis([0.1, 0.2])

    def test_non_finite_drives_are_refused(self):
        with self.assertRaises(ValueError):
            GeckoSelector().step(hunger=float("nan"))


class TheContract(unittest.TestCase):
    def test_the_six_behaviours_and_their_order_are_fixed(self):
        self.assertEqual(CHANNELS,
                         ("hunt", "flee", "explore", "bask", "rest", "groom"))
        self.assertEqual(len(GeckoSelector().gates()), 6)

    def test_gates_always_agree_with_what_was_selected(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        gates = sel.gates()
        self.assertEqual(sel.channels[int(np.argmax(gates))], sel.selected())

    def test_the_salience_weights_are_declared_invented(self):
        """They decide what the animal wants, and nothing published constrains
        them. The docstring must keep saying so."""
        self.assertIn("INVENTED", salience_from_drives.__doc__)

    def test_state_reports_whether_the_network_settled(self):
        sel = settle(GeckoSelector(), hunger=0.8, prey_visible=1.0)
        self.assertIn("settled", sel.state())
        self.assertTrue(sel.state()["settled"])


class RunsInsideTheLiveEnvironment(unittest.TestCase):
    """Brain 1 -> brain 2, on simulated data, in the real loop.

    This is the seam that matters: the hypothalamus's drive vector reaching the
    selector every control step, with threat and prey visibility from the world
    rather than from a test fixture.
    """

    def _env(self, **kw):
        try:
            from envs.gecko_brain_env import GeckoBrainEnv
        except Exception as exc:                      # pragma: no cover
            self.skipTest(f"environment unavailable: {exc}")
        return GeckoBrainEnv(homeostasis=True, action_selection=True, **kw)

    def test_action_selection_requires_the_hypothalamus(self):
        """Its salience IS the drive vector, so the dependency is refused
        rather than silently satisfied with zeros."""
        from envs.gecko_brain_env import GeckoBrainEnv
        with self.assertRaises(ValueError):
            GeckoBrainEnv(homeostasis=False, action_selection=True)

    def test_the_choice_is_reported_and_steers_nothing(self):
        env = self._env()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
            self.assertIn("behaviour", info)
            self.assertEqual(len(info["behaviour_gates"]), len(CHANNELS))
            # Only one behaviour has a controller. A six-way chooser wired to a
            # one-behaviour body would look like integration and mean nothing.
            self.assertTrue(info["behaviour_controls_nothing"])
        finally:
            env.close()

    def test_a_rested_unhungry_gecko_chooses_nothing(self):
        """No fallback behaviour. At the start of an episode every drive is
        near zero and the honest answer is that the animal wants nothing."""
        env = self._env()
        try:
            env.reset(seed=0)
            _, _, _, _, info = env.step(env.action_space.sample())
            self.assertIsNone(info["behaviour"])
        finally:
            env.close()

    def test_drives_that_build_produce_a_behaviour(self):
        """With the clock compressed the animal gets tired and hungry, and then
        it does something. Without this the seam is never exercised."""
        env = self._env(homeostasis_time_compression=20000.0)
        try:
            env.reset(seed=0)
            chosen = set()
            for _ in range(120):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                chosen.add(info["behaviour"])
                if term or trunc:
                    env.reset()
            self.assertTrue(chosen - {None},
                            "a hungry, tired gecko should want something")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
